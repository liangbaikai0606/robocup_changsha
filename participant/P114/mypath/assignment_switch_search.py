"""E030：联合 assignment 的整体切换惩罚搜索。

每一步读取全局训练状态，为 3! 个一对一匹配计算距离、两帧趋势、动态可达性
和 ``beta * N_switch``，再用同一四阶段控制器执行。该脚本是 privileged
diagnostic，不是合法提交策略。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MYPATH_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MYPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MYPATH_ROOT))

from coverage_bench.config import TaskConfig, load_task_config
from coverage_bench.envs.factory import make_training_env
from coverage_bench.suites import ScenarioCase, load_suite
from oracle_four_stage import ControlPreset, _four_stage_actions
from oracle_search import _decode


MASTER_SEED = 20260930
PUBLIC_SEEDS = {1001, 1002, 2001, 2002}
BETAS = (0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 1.00)
TREND_WEIGHTS = (0.0, 1.0, 2.0, 4.0)
M_UNREACHABLE = 10.0
CONTROL = ControlPreset("center-early", 0.00, 0.40, 0.025)


@dataclass(frozen=True)
class AssignmentRule:
    name: str
    beta: float
    trend_weight: float
    unreachable_switch_factor: float = 0.0


RULES = tuple(
    AssignmentRule(f"b{beta:.2f}-l{trend:.1f}", beta, trend)
    for trend in TREND_WEIGHTS
    for beta in BETAS
) + tuple(
    AssignmentRule(f"b{beta:.2f}-l2.0-no-waive", beta, 2.0, 1.0)
    for beta in (0.02, 0.05, 0.12, 0.30, 0.50, 1.00)
)+ tuple(
    AssignmentRule(
        f"b{beta:.2f}-l2.0-u{factor:.2f}", beta, 2.0, factor
    )
    for beta in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
    for factor in (0.25, 0.50, 0.75)
)


@lru_cache(maxsize=2)
def _config_for_layout(layout: str) -> TaskConfig:
    base = load_task_config(REPO_ROOT / "configs" / "task-v1.yaml")
    scenario = base.scenario.model_copy(update={"layout_kind": layout})
    return base.model_copy(update={"scenario": scenario})


def _generate_seeds(count: int, master_seed: int) -> list[int]:
    rng = random.Random(master_seed)
    seeds: list[int] = []
    used = set(PUBLIC_SEEDS)
    while len(seeds) < count:
        seed = rng.getrandbits(64)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


def _optimistic_closure(
    relative: np.ndarray,
    robot_velocity: np.ndarray,
    steps: int,
    config: TaskConfig,
) -> float:
    distance = float(np.linalg.norm(relative))
    if distance <= 1e-12 or steps <= 0:
        return 0.0
    direction = relative / distance
    radial_speed = float(np.dot(robot_velocity, direction))
    dt = float(config.public.dt)
    damping = float(config.public.damping)
    acceleration = float(config.public.drive_force / config.public.robot_mass) * dt
    max_speed = float(config.public.robot_max_speed)
    closure = 0.0
    best = 0.0
    for _ in range(steps):
        closure += radial_speed * dt
        best = max(best, closure)
        radial_speed = (1.0 - damping) * radial_speed + acceleration
        radial_speed = min(radial_speed, max_speed)
    return max(best, 0.0)


class JointAssignment:
    """保存 pair 距离历史，并联合选择一个一对一匹配。"""

    def __init__(self, config: TaskConfig, rule: AssignmentRule) -> None:
        self.config = config
        self.rule = rule
        self.n = int(config.num_agents)
        self.m = int(config.num_targets)
        self.horizon = int(config.horizon)
        self.cover_radius = float(config.public.target_radius)
        self.assignments = list(permutations(range(self.m)))
        self.history: list[np.ndarray] = []
        self.physical_bad_streak = np.zeros((self.n, self.m), dtype=np.int64)
        self.trend_bad_streak = np.zeros((self.n, self.m), dtype=np.int64)
        self.previous: tuple[int, ...] | None = None
        self.switches = 0
        self.reassign_steps = 0
        self.waived_switches = 0

    def choose(self, state: np.ndarray, step_index: int) -> tuple[int, ...]:
        robot_pos, robot_vel, target_pos, _target_vel = _decode(
            state, self.n, self.m
        )
        delta = target_pos[None, :, :] - robot_pos[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        self.history.append(distance.copy())
        del self.history[:-3]
        steps_left = self.horizon - step_index

        if len(self.history) >= 3:
            trend = (self.history[-1] - self.history[-3]) / 2.0
            closing = -trend
        else:
            trend = np.zeros_like(distance)
            closing = np.zeros_like(distance)

        unreachable = np.zeros((self.n, self.m), dtype=np.bool_)
        if len(self.history) >= 3:
            for i in range(self.n):
                for j in range(self.m):
                    gap = max(0.0, float(distance[i, j]) - self.cover_radius)
                    optimistic = _optimistic_closure(
                        delta[i, j], robot_vel[i], steps_left, self.config
                    )
                    physical_bad = optimistic + 0.08 < gap
                    expected = steps_left * max(float(closing[i, j]), 0.0)
                    trend_bad = (
                        steps_left <= 4
                        and expected + 0.06 < gap
                        and float(closing[i, j]) <= 0.0
                    )
                    self.physical_bad_streak[i, j] = (
                        self.physical_bad_streak[i, j] + 1 if physical_bad else 0
                    )
                    self.trend_bad_streak[i, j] = (
                        self.trend_bad_streak[i, j] + 1 if trend_bad else 0
                    )
                    unreachable[i, j] = bool(
                        self.physical_bad_streak[i, j] >= 2
                        or self.trend_bad_streak[i, j] >= 2
                    )

        pair_cost = distance + self.rule.trend_weight * trend
        pair_cost = np.where(unreachable, M_UNREACHABLE, pair_cost)
        best_assignment = self.assignments[0]
        best_key = (float("inf"), float("inf"), self.assignments[0])
        for assignment in self.assignments:
            base = sum(float(pair_cost[i, assignment[i]]) for i in range(self.n))
            penalty_units = 0.0
            waived = 0
            if self.previous is not None:
                for i in range(self.n):
                    if assignment[i] == self.previous[i]:
                        continue
                    if unreachable[i, self.previous[i]]:
                        waived += 1
                        penalty_units += self.rule.unreachable_switch_factor
                    else:
                        penalty_units += 1.0
            total = base + self.rule.beta * penalty_units
            key = (total, base, assignment)
            if key < best_key:
                best_key = key
                best_assignment = assignment
                best_waived = waived

        if self.previous is not None:
            changed = sum(
                int(best_assignment[i] != self.previous[i]) for i in range(self.n)
            )
            self.switches += changed
            self.reassign_steps += int(changed > 0)
            self.waived_switches += best_waived
        self.previous = best_assignment
        return best_assignment


def _rollout(case: ScenarioCase, rule: AssignmentRule) -> dict[str, Any]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=case.scenario_seed)
    assignment = JointAssignment(config, rule)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for step in range(int(config.horizon)):
            state = env.state()
            chosen = assignment.choose(state, step)
            actions = _four_stage_actions(state, chosen, config, 0.0, CONTROL)
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    return {
        "j": total / int(config.horizon),
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "switches": assignment.switches,
        "reassign_steps": assignment.reassign_steps,
        "waived_switches": assignment.waived_switches,
    }


def _run_scenario(
    task: tuple[int, str, str], rules: tuple[AssignmentRule, ...] = RULES
) -> dict[str, Any]:
    seed, layout, split = task
    config = _config_for_layout(layout)
    group = "basic" if layout == "uniform" else "cooperation"
    case = ScenarioCase(f"random-{group}-{seed}", group, config, seed)
    return {
        "seed": seed,
        "layout": layout,
        "group": group,
        "split": split,
        "results": {rule.name: _rollout(case, rule) for rule in rules},
    }


def _paired_scores(rows: list[dict[str, Any]], name: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    return [
        500.0
        * (
            float(lookup[(seed, "basic")]["results"][name]["j"])
            + float(lookup[(seed, "cooperation")]["results"][name]["j"])
        )
        for seed in seeds
    ]


def _ci95(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return [mean, mean]
    half = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return [mean - half, mean + half]


def _summarize(rows: list[dict[str, Any]], name: str, baseline: str) -> dict[str, Any]:
    score = _paired_scores(rows, name)
    base = _paired_scores(rows, baseline)
    gap = [value - old for value, old in zip(score, base)]
    episodes = [row["results"][name] for row in rows]
    groups: dict[str, Any] = {}
    for group in ("basic", "cooperation"):
        records = [row["results"][name] for row in rows if row["group"] == group]
        groups[group] = {
            "score_scale": 1000.0 * float(np.mean([item["j"] for item in records])),
            "coverage": float(np.mean([item["coverage"] for item in records])),
            "collision": float(np.mean([item["collision"] for item in records])),
        }
    return {
        "score": float(np.mean(score)),
        "score_ci95_normal": _ci95(score),
        "minus_baseline": float(np.mean(gap)),
        "gap_ci95_normal": _ci95(gap),
        "wins": int(sum(x > 1e-12 for x in gap)),
        "ties": int(sum(abs(x) <= 1e-12 for x in gap)),
        "losses": int(sum(x < -1e-12 for x in gap)),
        "mean_switches_per_episode": float(np.mean([item["switches"] for item in episodes])),
        "mean_reassign_steps": float(np.mean([item["reassign_steps"] for item in episodes])),
        "waived_switches": int(sum(item["waived_switches"] for item in episodes)),
        "groups": groups,
    }


def _public_results(selected: AssignmentRule) -> dict[str, Any]:
    suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    compare = (
        AssignmentRule("distance-no-penalty", 0.0, 0.0),
        AssignmentRule("same-trend-no-penalty", 0.0, selected.trend_weight),
        selected,
    )
    rows = []
    for group in suite.groups:
        for case in group.cases:
            rows.append({
                "case_id": case.case_id,
                "group": group.group_id,
                "seed": case.scenario_seed,
                "results": {rule.name: _rollout(case, rule) for rule in compare},
            })
    result: dict[str, Any] = {"cases": rows}
    for rule in compare:
        by_group = {
            group: [row["results"][rule.name]["j"] for row in rows if row["group"] == group]
            for group in ("basic", "cooperation")
        }
        result[rule.name] = 500.0 * (
            float(np.mean(by_group["basic"])) + float(np.mean(by_group["cooperation"]))
        )
    return result


def _write_selected_csv(
    path: Path,
    rows: list[dict[str, Any]],
    selected: AssignmentRule,
    no_penalty_name: str,
) -> None:
    fields = (
        "split", "seed", "group", "layout", "no_penalty_j", "selected_j", "gap_j",
        "selected_switches", "selected_reassign_steps", "selected_waived_switches",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["split"], int(x["seed"]), x["group"])):
            old = row["results"][no_penalty_name]
            new = row["results"][selected.name]
            writer.writerow({
                "split": row["split"], "seed": row["seed"], "group": row["group"],
                "layout": row["layout"], "no_penalty_j": old["j"], "selected_j": new["j"],
                "gap_j": new["j"] - old["j"], "selected_switches": new["switches"],
                "selected_reassign_steps": new["reassign_steps"],
                "selected_waived_switches": new["waived_switches"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--only-rules", help="Comma-separated rule names for confirmation runs")
    parser.add_argument("--selected-rule", choices=tuple(rule.name for rule in RULES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tune_count, args.test_count, args.workers) <= 0:
        parser.error("counts and workers must be positive")

    seeds = _generate_seeds(args.tune_count + args.test_count, args.master_seed)
    tune_seeds = set(seeds[: args.tune_count])
    tasks = [
        (seed, layout, "tune" if seed in tune_seeds else "test")
        for seed in seeds
        for layout in ("uniform", "crossing")
    ]
    if args.only_rules:
        wanted = {name.strip() for name in args.only_rules.split(",") if name.strip()}
        run_rules = tuple(rule for rule in RULES if rule.name in wanted)
        missing = wanted - {rule.name for rule in run_rules}
        if missing:
            parser.error(f"unknown --only-rules: {sorted(missing)}")
    else:
        run_rules = RULES
    if not run_rules:
        parser.error("no rules selected")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_scenario, task, run_rules): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)}", flush=True)

    tune_rows = [row for row in rows if row["split"] == "tune"]
    test_rows = [row for row in rows if row["split"] == "test"]
    zero_rules = [rule for rule in run_rules if rule.beta == 0.0]
    if not zero_rules:
        parser.error("selected rule set must include a beta=0 baseline")
    distance_baseline = min(zero_rules, key=lambda rule: rule.trend_weight)
    tune_summary = {
        rule.name: _summarize(tune_rows, rule.name, distance_baseline.name)
        for rule in run_rules
    }
    # 分数优先；近似同分时偏好更大的 beta 和更少的切换。
    if args.selected_rule:
        selected = next(
            (rule for rule in run_rules if rule.name == args.selected_rule), None
        )
        if selected is None:
            parser.error("--selected-rule must also be present in --only-rules")
    else:
        selected = max(
            run_rules,
            key=lambda rule: (
                tune_summary[rule.name]["score"],
                rule.beta,
                -tune_summary[rule.name]["mean_switches_per_episode"],
            ),
        )
    no_penalty = next(
        rule for rule in run_rules
        if rule.beta == 0.0 and rule.trend_weight == selected.trend_weight
    )
    test_summary = {
        distance_baseline.name: _summarize(test_rows, distance_baseline.name, distance_baseline.name),
        no_penalty.name: _summarize(test_rows, no_penalty.name, no_penalty.name),
        selected.name: _summarize(test_rows, selected.name, no_penalty.name),
    }
    summary = {
        "experiment": "E030",
        "warning": "Privileged global-state diagnostic; not a legal policy.",
        "master_seed": args.master_seed,
        "tune_seed_count": args.tune_count,
        "test_seed_count": args.test_count,
        "layouts_per_seed": 2,
        "selected_rule": asdict(selected),
        "same_trend_no_penalty": asdict(no_penalty),
        "tune": tune_summary,
        "test": test_summary,
        "public": _public_results(selected),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_selected_csv(output / "episodes-selected.csv", rows, selected, no_penalty.name)
    print(json.dumps({
        "selected_rule": asdict(selected),
        "same_trend_no_penalty": asdict(no_penalty),
        "test": test_summary,
        "public_scores": {key: value for key, value in summary["public"].items() if key != "cases"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
