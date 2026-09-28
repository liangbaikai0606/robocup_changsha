"""可达性防抖的全局 Oracle 对照。

两套都用同一条 PD 控制，读取 env.state()，不是合法提交策略。
差别只在每一步的 6 种一对一分配：

- plain：代价只有当前距离。
- debounce：沿用 E029 的 hybrid-wide 判不可达，不可达的配对加大代价；
  换目标再加 0.03，旧目标已不可达时只保留 25% 的切换代价。

增益用 E032 调参选出的 Kp=20、Kd=4，这里不再挑参数。
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
from coverage_bench.suites import ScenarioCase
from oracle_search import _decode
from pd_oracle import _pd_action


MASTER_SEED = 20261003
PUBLIC_SEEDS = {1001, 1002, 2001, 2002}
KP = 20.0
KD = 4.0
BETA = 0.03
UNREACHABLE_FACTOR = 0.25
M_UNREACHABLE = 10.0


def _optimistic_closure(relative: np.ndarray, velocity: np.ndarray, steps: int, config: TaskConfig) -> float:
    """目标暂时不动、油门全朝连线时，剩余步数最多能缩短多少。"""
    distance = float(np.linalg.norm(relative))
    if distance <= 1e-12 or steps <= 0:
        return 0.0
    direction = relative / distance
    radial_speed = float(np.dot(velocity, direction))
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


class _Assigner:
    """每一步在 6 种分配里选总代价最低的。"""

    def __init__(self, config: TaskConfig, debounce: bool) -> None:
        self.config = config
        self.debounce = debounce
        self.n = int(config.num_agents)
        self.m = int(config.num_targets)
        self.horizon = int(config.horizon)
        self.cover_radius = float(config.public.target_radius)
        self.assignments = list(permutations(range(self.m)))
        self.history: list[np.ndarray] = []
        self.physical_bad = np.zeros((self.n, self.m), dtype=np.int64)
        self.trend_bad = np.zeros((self.n, self.m), dtype=np.int64)
        self.previous: tuple[int, ...] | None = None
        self.switches = 0
        self.reassign_steps = 0

    def choose(self, state: np.ndarray, step_index: int) -> tuple[int, ...]:
        robot_pos, robot_vel, target_pos, _target_vel = _decode(state, self.n, self.m)
        delta = target_pos[None, :, :] - robot_pos[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        self.history.append(distance.copy())
        del self.history[:-3]
        steps_left = self.horizon - step_index
        unreachable = np.zeros((self.n, self.m), dtype=np.bool_)
        if self.debounce and len(self.history) >= 3:
            closing = (self.history[-3] - self.history[-1]) / 2.0
            for i in range(self.n):
                for j in range(self.m):
                    gap = max(0.0, float(distance[i, j]) - self.cover_radius)
                    optimistic = _optimistic_closure(delta[i, j], robot_vel[i], steps_left, self.config)
                    expected = steps_left * max(float(closing[i, j]), 0.0)
                    physical_bad = optimistic + 0.08 < gap
                    trend_bad = steps_left <= 4 and expected + 0.06 < gap and float(closing[i, j]) <= 0.0
                    self.physical_bad[i, j] = self.physical_bad[i, j] + 1 if physical_bad else 0
                    self.trend_bad[i, j] = self.trend_bad[i, j] + 1 if trend_bad else 0
                    unreachable[i, j] = bool(self.physical_bad[i, j] >= 2 or self.trend_bad[i, j] >= 2)
        pair_cost = np.where(unreachable, M_UNREACHABLE, distance) if self.debounce else distance
        best_assignment = self.assignments[0]
        best_key = (float("inf"), self.assignments[0])
        for assignment in self.assignments:
            base = sum(float(pair_cost[i, assignment[i]]) for i in range(self.n))
            penalty_units = 0.0
            if self.debounce and self.previous is not None:
                for i in range(self.n):
                    if assignment[i] == self.previous[i]:
                        continue
                    if unreachable[i, self.previous[i]]:
                        penalty_units += UNREACHABLE_FACTOR
                    else:
                        penalty_units += 1.0
            key = (base + BETA * penalty_units, assignment)
            if key < best_key:
                best_key = key
                best_assignment = assignment
        if self.previous is not None:
            changed = sum(int(best_assignment[i] != self.previous[i]) for i in range(self.n))
            self.switches += changed
            self.reassign_steps += int(changed > 0)
        self.previous = best_assignment
        return best_assignment


def _rollout(
    case: ScenarioCase,
    debounce: bool,
    kp: float = KP,
    kd: float = KD,
) -> dict[str, Any]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, debounce)
    n = int(config.num_agents)
    m = int(config.num_targets)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for step in range(int(config.horizon)):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            actions: dict[str, np.ndarray] = {}
            for robot_index, target_index in enumerate(chosen):
                delta_p = target_pos[target_index] - robot_pos[robot_index]
                delta_v = target_vel[target_index] - robot_vel[robot_index]
                actions[f"agent_{robot_index}"] = _pd_action(delta_p, delta_v, kp, kd)
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    horizon = int(config.horizon)
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "switches": assigner.switches,
        "reassign_steps": assigner.reassign_steps,
    }


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    plain = _rollout(case, False)
    debounce = _rollout(case, True)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
    }
    for name, result in (("plain", plain), ("debounce", debounce)):
        for key, value in result.items():
            row[f"{name}_{key}"] = value
    row["gap_j"] = float(debounce["j"]) - float(plain["j"])
    return row


def _layout_config(layout: str) -> TaskConfig:
    base = load_task_config(REPO_ROOT / "configs" / "task-v1.yaml")
    scenario = base.scenario.model_copy(update={"layout_kind": layout})
    return base.model_copy(update={"scenario": scenario})


def _random_seeds(count: int, master_seed: int) -> list[int]:
    rng = random.Random(master_seed)
    seeds: list[int] = []
    used = set(PUBLIC_SEEDS)
    while len(seeds) < count:
        seed = rng.getrandbits(64)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


def _mean_ci(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    if len(arr) < 2:
        return [mean, mean]
    half = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return [mean - half, mean + half]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows if str(row["case_id"]).startswith("random-")}
    plain_scores: list[float] = []
    debounce_scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        plain_scores.append(500.0 * (float(basic["plain_j"]) + float(coop["plain_j"])))
        debounce_scores.append(500.0 * (float(basic["debounce_j"]) + float(coop["debounce_j"])))
    gaps = [new - old for old, new in zip(plain_scores, debounce_scores)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)
    random_rows = [row for row in rows if str(row["case_id"]).startswith("random-")]

    def mean_of(key: str) -> float:
        return float(np.mean([float(row[key]) for row in random_rows]))

    return {
        "oracle_warning": "读取全局状态，每步重选分配。不是合法策略，也不是数学上界。",
        "control": "两边相同：a = clip(20 * Δp + 4 * Δv, -1, 1)",
        "debounce": "不可达配对代价 10；换目标加 0.03；旧目标不可达时切换代价保留 25%",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "plain_score": float(np.mean(plain_scores)),
        "debounce_score": float(np.mean(debounce_scores)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "plain_switches_per_episode": mean_of("plain_switches"),
        "debounce_switches_per_episode": mean_of("debounce_switches"),
        "plain_coverage": mean_of("plain_coverage"),
        "debounce_coverage": mean_of("debounce_coverage"),
        "plain_collision": mean_of("plain_collision"),
        "debounce_collision": mean_of("debounce_collision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("--count 和 --workers 必须为正")

    seeds = _random_seeds(args.count, args.master_seed)
    basic = _layout_config("uniform")
    cooperation = _layout_config("crossing")
    cases = [
        case
        for seed in seeds
        for case in (
            ScenarioCase(f"random-basic-{seed}", "basic", basic, seed),
            ScenarioCase(f"random-cooperation-{seed}", "cooperation", cooperation, seed),
        )
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_evaluate_case, case) for case in cases]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)
    rows.sort(key=lambda row: (str(row["group"]), int(row["seed"])))
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, seeds)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
