"""E029：局部动态可达性规则搜索。

保持 E028 的目标排序和控制不变，只在排名第一的目标被保守地判为不可达、且
还有未判死的候选时跳到下一个目标。先在调参种子选规则，再在未参与选择的
保留种子上报告结果。策略判断只使用正式局部观测；全局环境只用于计分。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
P114_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(P114_ROOT) not in sys.path:
    sys.path.insert(0, str(P114_ROOT))

from coverage_bench.config import TaskConfig, load_task_config
from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import AgentObservation, EpisodeContext, PublicTaskParams
from entry import CoverFirstPolicy


MASTER_SEED = 20260928
PUBLIC_SEEDS = {1001, 1002, 2001, 2002}


@dataclass(frozen=True)
class ReachabilityRule:
    name: str
    physical_margin: float
    physical_bad_steps: int
    edge_margin: float
    trend_horizon: int
    closing_ceiling: float
    trend_bad_steps: int
    switch_mode: str = "any"


# 范围故意较小；避免用大量阈值把随机调参集拟合掉。
RULES = (
    ReachabilityRule("physical-0", 0.00, 1, 0.04, 0, 0.00, 99),
    ReachabilityRule("physical-3", 0.03, 1, 0.04, 0, 0.00, 99),
    ReachabilityRule("physical-6", 0.06, 1, 0.04, 0, 0.00, 99),
    ReachabilityRule("late3-bad1", 0.06, 99, 0.03, 3, 0.00, 1),
    ReachabilityRule("late3-bad2", 0.06, 99, 0.04, 3, 0.00, 2),
    ReachabilityRule("late4-bad2", 0.06, 99, 0.04, 4, 0.00, 2),
    ReachabilityRule("late4-slow2", 0.06, 99, 0.04, 4, 0.01, 2),
    ReachabilityRule("hybrid-strict", 0.03, 2, 0.04, 4, 0.00, 2),
    ReachabilityRule("hybrid-safe", 0.06, 2, 0.05, 3, 0.00, 2),
    ReachabilityRule("hybrid-wide", 0.08, 2, 0.06, 4, 0.00, 2),
    ReachabilityRule("physical-6-strong", 0.06, 1, 0.04, 0, 0.00, 99, "strong"),
    ReachabilityRule("hybrid-safe-strong", 0.06, 2, 0.05, 3, 0.00, 2, "strong"),
    ReachabilityRule("hybrid-wide-strong", 0.08, 2, 0.06, 4, 0.00, 2, "strong"),
    ReachabilityRule("hybrid-safe-tiered", 0.06, 2, 0.05, 3, 0.00, 2, "tiered"),
    ReachabilityRule("hybrid-wide-tiered", 0.08, 2, 0.06, 4, 0.00, 2, "tiered"),
    ReachabilityRule("sticky-only", 1.00, 99, 1.00, 0, -1.00, 99, "sticky"),
    ReachabilityRule("physical-6-sticky", 0.06, 1, 0.04, 0, 0.00, 99, "sticky"),
    ReachabilityRule("late4-slow2-sticky", 0.06, 99, 0.04, 4, 0.01, 2, "sticky"),
    ReachabilityRule("hybrid-safe-sticky", 0.06, 2, 0.05, 3, 0.00, 2, "sticky"),
    ReachabilityRule("hybrid-wide-sticky", 0.08, 2, 0.06, 4, 0.00, 2, "sticky"),
)


@lru_cache(maxsize=2)
def _config_for_layout(layout: str) -> TaskConfig:
    base = load_task_config(REPO_ROOT / "configs" / "task-v1.yaml")
    scenario = base.scenario.model_copy(update={"layout_kind": layout})
    return base.model_copy(update={"scenario": scenario})


def _public_params(config: TaskConfig) -> PublicTaskParams:
    task = config.public
    return PublicTaskParams(
        map_half_extent=task.map_half_extent,
        dt=task.dt,
        robot_radius=task.robot_radius,
        robot_mass=task.robot_mass,
        drive_force=task.drive_force,
        damping=task.damping,
        robot_max_speed=task.robot_max_speed,
        contact_force=task.contact_force,
        contact_margin=task.contact_margin,
        target_radius=task.target_radius,
        target_max_speed=task.target_max_speed,
        sense_radius=task.sense_radius,
        motion_kind=task.motion_kind,
        turn_interval_steps=tuple(task.turn_interval_steps),
        target_speed_fraction=tuple(task.target_speed_fraction),
        robot_boundary=task.robot_boundary,
        target_boundary=task.target_boundary,
    )


class DynamicReachabilityPolicy(CoverFirstPolicy):
    """E028 加一层目标可达性过滤；其他追圈、避碰和助推逻辑不变。"""

    def __init__(self, rule: ReachabilityRule) -> None:
        super().__init__()
        self.rule = rule
        self._distance_history: dict[int, list[tuple[int, float]]] = {}
        self._physical_bad_streak: dict[int, int] = {}
        self._trend_bad_streak: dict[int, int] = {}
        self._active_target: int | None = None
        self.diagnostics: Counter[str] = Counter()

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self._distance_history = {}
        self._physical_bad_streak = {}
        self._trend_bad_streak = {}
        self._active_target = None
        self.diagnostics = Counter()

    def act(self, observation: AgentObservation) -> np.ndarray:
        self._record_distances(observation)
        return super().act(observation)

    def _record_distances(self, observation: AgentObservation) -> None:
        step = int(observation["step_index"])
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            distance = float(np.linalg.norm(targets[index, :2] * self._position_scale))
            history = self._distance_history.setdefault(index, [])
            if history and history[-1][0] != step - 1:
                history.clear()
                self._physical_bad_streak[index] = 0
                self._trend_bad_streak[index] = 0
            if not history or history[-1][0] != step:
                history.append((step, distance))
                del history[:-3]

    def _optimistic_closure(
        self,
        relative: np.ndarray,
        velocity: np.ndarray,
        steps: int,
    ) -> float:
        """目标暂时静止、全部驱动力沿连线时，剩余步数最多可缩短多少。"""
        distance = float(np.linalg.norm(relative))
        if distance <= 1e-12 or steps <= 0:
            return 0.0
        direction = np.asarray(relative, dtype=np.float64) / distance
        radial_speed = float(np.dot(np.asarray(velocity, dtype=np.float64), direction))
        closure = 0.0
        best = 0.0
        acceleration = self._dt / self._mass
        for _ in range(steps):
            closure += radial_speed * self._dt
            best = max(best, closure)
            radial_speed = (1.0 - self._damping) * radial_speed + acceleration
            radial_speed = min(radial_speed, self._max_speed)
        return max(best, 0.0)

    def _classify(
        self,
        target_index: int,
        relative: np.ndarray,
        self_velocity: np.ndarray,
        steps_left: int,
    ) -> str:
        distance = float(np.linalg.norm(relative))
        gap = max(0.0, distance - self._cover_radius)
        if gap <= 1e-9:
            self._physical_bad_streak[target_index] = 0
            self._trend_bad_streak[target_index] = 0
            return "inside"

        history = self._distance_history.get(target_index, [])
        if len(history) < 3:
            return "edge-history"

        closing = (history[-3][1] - history[-1][1]) / 2.0
        expected = steps_left * max(closing, 0.0)
        optimistic = self._optimistic_closure(relative, self_velocity, steps_left)
        physical_bad = optimistic + self.rule.physical_margin < gap
        trend_bad = (
            steps_left <= self.rule.trend_horizon
            and expected + self.rule.edge_margin < gap
            and closing <= self.rule.closing_ceiling
        )
        self._physical_bad_streak[target_index] = (
            self._physical_bad_streak.get(target_index, 0) + 1 if physical_bad else 0
        )
        self._trend_bad_streak[target_index] = (
            self._trend_bad_streak.get(target_index, 0) + 1 if trend_bad else 0
        )
        if (
            self._physical_bad_streak[target_index] >= self.rule.physical_bad_steps
            or self._trend_bad_streak[target_index] >= self.rule.trend_bad_steps
        ):
            return "unreachable"
        slack = expected - gap
        if slack >= self.rule.edge_margin:
            return "reachable"
        return "edge"

    def _chase(self, observation: AgentObservation) -> np.ndarray:
        empty, occupied = self._visible_targets(observation)
        pool = empty if empty else occupied
        if not pool:
            return np.zeros(2, dtype=np.float32)

        state = np.asarray(observation["self_state"], dtype=np.float64)
        self_velocity = state[2:4] * self._velocity_scale
        steps_left = self._horizon - int(observation["step_index"])
        classified: list[tuple[int, np.ndarray, str]] = []
        for target_index, relative in pool:
            status = self._classify(target_index, relative, self_velocity, steps_left)
            self.diagnostics[f"status_{status}"] += 1
            classified.append((target_index, relative, status))

        chosen = classified[0]
        if self.rule.switch_mode == "sticky":
            active = next(
                (item for item in classified if item[0] == self._active_target), None
            )
            if active is not None and active[2] != "unreachable":
                chosen = active
                self.diagnostics["retained_active"] += 1
            elif active is not None:
                replacement = next(
                    (item for item in classified if item[2] != "unreachable"), None
                )
                if replacement is not None:
                    chosen = replacement
                    if replacement[0] != active[0]:
                        self.diagnostics["switches"] += 1
                else:
                    chosen = active
                    self.diagnostics["all_unreachable_fallback"] += 1
            if self._active_target is not None and chosen[0] != self._active_target:
                self.diagnostics["active_changed"] += 1
            self._active_target = chosen[0]
        elif self.rule.switch_mode == "tiered":
            rank = {
                "inside": 0,
                "reachable": 1,
                "edge-history": 2,
                "edge": 2,
                "unreachable": 3,
            }
            _best_position, best = min(
                enumerate(classified), key=lambda pair: (rank[pair[1][2]], pair[0])
            )
            if best is not chosen and rank[best[2]] < rank[chosen[2]]:
                chosen = best
                self.diagnostics["switches"] += 1
        elif chosen[2] == "unreachable":
            allowed = (
                {"inside", "reachable"}
                if self.rule.switch_mode == "strong"
                else {"inside", "reachable", "edge-history", "edge"}
            )
            replacement = next((item for item in classified[1:] if item[2] in allowed), None)
            if replacement is not None:
                chosen = replacement
                self.diagnostics["switches"] += 1
            else:
                self.diagnostics["all_unreachable_fallback"] += 1

        target_index, relative, _status = chosen
        target_velocity = self._target_velocity_estimates.get(
            target_index, np.zeros(2, dtype=np.float64)
        )
        return self._thrust_to_target(relative, self_velocity, target_velocity)


def _run_policy(config: TaskConfig, seed: int, rule: ReachabilityRule | None) -> dict[str, Any]:
    env = make_training_env(config)
    observations, _infos = env.reset(seed=seed)
    policies: list[CoverFirstPolicy] = [
        CoverFirstPolicy() if rule is None else DynamicReachabilityPolicy(rule)
        for _ in range(config.num_agents)
    ]
    params = _public_params(config)
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=config.num_agents,
                num_targets=config.num_targets,
                horizon=config.horizon,
                task=params,
                policy_seed=(seed + index) % (2**64),
            )
        )
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for _ in range(config.horizon):
            actions = {
                agent_id: policies[index].act(observations[agent_id])
                for index, agent_id in enumerate(env.agents)
            }
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent_id = next(iter(rewards))
            total += float(rewards[agent_id])
            metrics = infos[agent_id]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        for policy in policies:
            policy.close()
        env.close()

    diagnostics: Counter[str] = Counter()
    for policy in policies:
        if isinstance(policy, DynamicReachabilityPolicy):
            diagnostics.update(policy.diagnostics)
    return {
        "j": total / config.horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "diagnostics": dict(diagnostics),
    }


def _run_scenario(
    task: tuple[int, str, str], rules: tuple[ReachabilityRule, ...] = RULES
) -> dict[str, Any]:
    seed, layout, split = task
    config = _config_for_layout(layout)
    group = "basic" if layout == "uniform" else "cooperation"
    results = {"baseline": _run_policy(config, seed, None)}
    for rule in rules:
        results[rule.name] = _run_policy(config, seed, rule)
    return {"seed": seed, "layout": layout, "group": group, "split": split, "results": results}


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


def _paired_scores(rows: list[dict[str, Any]], policy_name: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    return [
        500.0
        * (
            float(lookup[(seed, "basic")]["results"][policy_name]["j"])
            + float(lookup[(seed, "cooperation")]["results"][policy_name]["j"])
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


def _summarize_policy(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    scores = _paired_scores(rows, name)
    baseline = _paired_scores(rows, "baseline")
    gaps = [value - base for value, base in zip(scores, baseline)]
    diagnostics: Counter[str] = Counter()
    for row in rows:
        diagnostics.update(row["results"][name]["diagnostics"])
    by_group: dict[str, Any] = {}
    for group in ("basic", "cooperation"):
        group_rows = [row for row in rows if row["group"] == group]
        by_group[group] = {
            "score_scale": 1000.0 * float(np.mean([row["results"][name]["j"] for row in group_rows])),
            "coverage": float(np.mean([row["results"][name]["coverage"] for row in group_rows])),
            "collision": float(np.mean([row["results"][name]["collision"] for row in group_rows])),
        }
    return {
        "score": float(np.mean(scores)),
        "score_ci95_normal": _ci95(scores),
        "minus_baseline": float(np.mean(gaps)),
        "paired_gap_ci95_normal": _ci95(gaps),
        "wins": int(sum(gap > 1e-12 for gap in gaps)),
        "ties": int(sum(abs(gap) <= 1e-12 for gap in gaps)),
        "losses": int(sum(gap < -1e-12 for gap in gaps)),
        "groups": by_group,
        "diagnostics": dict(diagnostics),
    }


def _write_episode_csv(path: Path, rows: list[dict[str, Any]], selected: str) -> None:
    fields = (
        "split", "seed", "group", "layout", "baseline_j", "selected_j", "gap_j",
        "baseline_coverage", "selected_coverage", "baseline_collision", "selected_collision",
        "switches", "unreachable", "edge", "edge_history", "reachable", "inside",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["split"], int(item["seed"]), item["group"])):
            base = row["results"]["baseline"]
            chosen = row["results"][selected]
            diag = chosen["diagnostics"]
            writer.writerow({
                "split": row["split"],
                "seed": row["seed"],
                "group": row["group"],
                "layout": row["layout"],
                "baseline_j": base["j"],
                "selected_j": chosen["j"],
                "gap_j": chosen["j"] - base["j"],
                "baseline_coverage": base["coverage"],
                "selected_coverage": chosen["coverage"],
                "baseline_collision": base["collision"],
                "selected_collision": chosen["collision"],
                "switches": diag.get("switches", 0),
                "unreachable": diag.get("status_unreachable", 0),
                "edge": diag.get("status_edge", 0),
                "edge_history": diag.get("status_edge-history", 0),
                "reachable": diag.get("status_reachable", 0),
                "inside": diag.get("status_inside", 0),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--only-rule", choices=tuple(rule.name for rule in RULES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tune_count, args.test_count, args.workers) <= 0:
        parser.error("counts and workers must be positive")

    seeds = _generate_seeds(args.tune_count + args.test_count, args.master_seed)
    tune = set(seeds[: args.tune_count])
    tasks = [
        (seed, layout, "tune" if seed in tune else "test")
        for seed in seeds
        for layout in ("uniform", "crossing")
    ]
    run_rules = (
        tuple(rule for rule in RULES if rule.name == args.only_rule)
        if args.only_rule
        else RULES
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_scenario, task, run_rules): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)}", flush=True)

    tune_rows = [row for row in rows if row["split"] == "tune"]
    test_rows = [row for row in rows if row["split"] == "test"]
    tune_summary = {"baseline": _summarize_policy(tune_rows, "baseline")}
    for rule in run_rules:
        tune_summary[rule.name] = _summarize_policy(tune_rows, rule.name)
    selected_rule = max(run_rules, key=lambda rule: tune_summary[rule.name]["score"])
    test_summary = {
        "baseline": _summarize_policy(test_rows, "baseline"),
        selected_rule.name: _summarize_policy(test_rows, selected_rule.name),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "experiment": "E029",
        "master_seed": args.master_seed,
        "tune_seed_count": args.tune_count,
        "test_seed_count": args.test_count,
        "layouts_per_seed": 2,
        "selection": "highest tune score; test split untouched until selection",
        "selected_rule": asdict(selected_rule),
        "tune": tune_summary,
        "test": test_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_episode_csv(output / "episodes-selected.csv", rows, selected_rule.name)
    print(json.dumps({"selected_rule": asdict(selected_rule), "test": test_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
