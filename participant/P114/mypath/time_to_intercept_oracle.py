"""用最早覆盖步数 T_ij 做分配代价。

开车固定为 a = clip(10*Δp,-1,1)，和 E034 的比例控制相同。
对照分配是可达性防抖。新分配只改代价：

- t_min：每步选总 T 最小的一对一分配
- t_save2：只有新分配的总 T 至少少 2 步才换，否则留在旧分配

T_ij 用离散截击估计：目标按当前速度直行，车朝预测位置满力，
第一次进入半径 0.15 的步数。剩余步内进不去记为 H+1。
读取全局状态，不是合法策略。种子与 E034 相同。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

from coverage_bench.envs.factory import make_training_env
from coverage_bench.suites import ScenarioCase
from oracle_search import _decode
from reachability_debounce_oracle import (
    MASTER_SEED,
    _layout_config,
    _mean_ci,
    _random_seeds,
    _rollout,
)


def _time_to_cover(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    steps_left: int,
    config: Any,
) -> int:
    """满力朝常速度预测点开，第一次进圈的步数。进不去则是 steps_left+1。"""
    radius = float(config.public.target_radius)
    if float(np.linalg.norm(target_pos - robot_pos)) <= radius:
        return 0
    if steps_left <= 0:
        return 1
    pub = config.public
    dt = float(pub.dt)
    damping = float(pub.damping)
    accel = float(pub.drive_force) / float(pub.robot_mass) * dt
    max_speed = float(pub.robot_max_speed)
    pos = np.array(robot_pos, dtype=np.float64)
    vel = np.array(robot_vel, dtype=np.float64)
    tp = np.array(target_pos, dtype=np.float64)
    tv = np.array(target_vel, dtype=np.float64)
    for k in range(1, steps_left + 1):
        delta = tp - pos
        action = np.clip(10.0 * delta, -1.0, 1.0)
        pos = pos + vel * dt
        vel = (1.0 - damping) * vel + action * accel
        speed = float(np.linalg.norm(vel))
        if speed > max_speed:
            vel *= max_speed / speed
        tp = tp + tv * dt
        if float(np.linalg.norm(tp - pos)) <= radius:
            return k
    return steps_left + 1


def _time_matrix(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    steps_left: int,
    config: Any,
) -> np.ndarray:
    n, m = len(robot_pos), len(target_pos)
    times = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            times[i, j] = _time_to_cover(
                robot_pos[i], robot_vel[i], target_pos[j], target_vel[j], steps_left, config
            )
    return times


def _choose_assignment(
    times: np.ndarray,
    previous: tuple[int, ...] | None,
    save_steps: int,
) -> tuple[int, ...]:
    """6 种分配里总 T 最小。save_steps>0 时，省不够这么多步就留在旧分配。"""
    n = times.shape[0]
    best = tuple(range(n))
    best_cost = float("inf")
    for assignment in permutations(range(n)):
        cost = sum(float(times[i, assignment[i]]) for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best = assignment
    if previous is None or save_steps <= 0:
        return best
    stay = sum(float(times[i, previous[i]]) for i in range(n))
    if best_cost + save_steps > stay:
        return previous
    return best


def _rollout_time(case: ScenarioCase, save_steps: int) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    m = int(config.num_targets)
    horizon = int(config.horizon)
    previous: tuple[int, ...] | None = None
    switches = 0
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for step in range(horizon):
            robot_pos, robot_vel, target_pos, target_vel = _decode(env.state(), n, m)
            times = _time_matrix(
                robot_pos, robot_vel, target_pos, target_vel, horizon - step, config
            )
            chosen = _choose_assignment(times, previous, save_steps)
            if previous is not None:
                switches += sum(int(chosen[i] != previous[i]) for i in range(n))
            previous = chosen
            actions = {
                f"agent_{i}": np.clip(
                    10.0 * (target_pos[chosen[i]] - robot_pos[i]), -1.0, 1.0
                ).astype(np.float32)
                for i in range(n)
            }
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "switches": float(switches),
    }


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    baseline = _rollout(case, True, 10.0, 0.0)
    t_min = _rollout_time(case, 0)
    t_save2 = _rollout_time(case, 2)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
    }
    for name, result in (("debounce", baseline), ("t_min", t_min), ("t_save2", t_save2)):
        for key, value in result.items():
            if key in ("j", "coverage", "collision", "switches"):
                row[f"{name}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], law: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{law}_j"]) + float(lookup[(seed, "cooperation")][f"{law}_j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base = _paired(rows, seeds, "debounce")
    laws: dict[str, Any] = {}
    for law in ("debounce", "t_min", "t_save2"):
        scores = _paired(rows, seeds, law)
        gaps = [new - old for old, new in zip(base, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        laws[law] = {
            "score": float(np.mean(scores)),
            "gap_vs_debounce": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{law}_coverage"]) for row in rows])),
            "collision": float(np.mean([float(row[f"{law}_collision"]) for row in rows])),
            "switches": float(np.mean([float(row[f"{law}_switches"]) for row in rows])),
        }
    return {
        "oracle_warning": "T_ij 用了全局位置和速度。不是合法策略，也不是 evaluate_one。",
        "control": "三套都是 a = clip(10*Δp,-1,1)",
        "debounce": "E034 的可达性防抖分配",
        "t_min": "每步最小总截击步数",
        "t_save2": "新分配的总 T 至少少 2 步才换",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
