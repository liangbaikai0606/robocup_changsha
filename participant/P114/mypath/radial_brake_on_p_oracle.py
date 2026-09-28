"""E034 的二维追踪加上靠近时的径向刹车。

平时：

    a = clip(10 * Δp, -1, 1)

只有离圈心小于 d_brake，并且相对径向速度 c > 0 时：

    a = clip(10 * Δp - Kd * c * n, -1, 1)

n = Δp / ||Δp||，c = (v_R - v_T) · n。切向力仍留在 10*Δp 里。
分配仍是可达性防抖。读取全局状态，不是合法策略。
种子与 E034 相同：主种子 20261003 的 300 个，分数按全部种子合并。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from reachability_debounce_oracle import MASTER_SEED, _Assigner, _layout_config, _mean_ci, _random_seeds


SETTINGS = ((0.0, 0.0),) + tuple(
    (distance, kd) for distance in (0.25, 0.35, 0.50) for kd in (0.5, 1.0, 2.0)
)


def _rollout(case: ScenarioCase, brake_distance: float, kd: float) -> dict[str, Any]:
    """brake_distance 为 0 时就是 E034 的二维比例追踪。"""
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    n = int(config.num_agents)
    m = int(config.num_targets)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    brake_steps = 0
    try:
        for step in range(int(config.horizon)):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            actions: dict[str, np.ndarray] = {}
            for robot_index, target_index in enumerate(chosen):
                delta = target_pos[target_index] - robot_pos[robot_index]
                command = 10.0 * delta
                distance = float(np.linalg.norm(delta))
                if kd > 0.0 and distance < brake_distance and distance > 1e-8:
                    normal = delta / distance
                    closing = float(np.dot(robot_vel[robot_index] - target_vel[target_index], normal))
                    if closing > 0.0:
                        command = command - kd * closing * normal
                        brake_steps += 1
                actions[f"agent_{robot_index}"] = np.clip(command, -1.0, 1.0).astype(np.float32)
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
        "brake_steps": brake_steps,
    }


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for distance, kd in SETTINGS:
        result = _rollout(case, distance, kd)
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "brake_distance": distance,
                "kd": kd,
                **result,
            }
        )
    return rows


def _paired(rows: list[dict[str, Any]], seeds: list[int], distance: float, kd: float) -> list[float]:
    chosen = [row for row in rows if row["brake_distance"] == distance and row["kd"] == kd]
    lookup = {(int(row["seed"]), str(row["group"])): row for row in chosen}
    return [
        500.0 * (float(lookup[(seed, "basic")]["j"]) + float(lookup[(seed, "cooperation")]["j"]))
        for seed in seeds
    ]


def _compare(rows: list[dict[str, Any]], seeds: list[int], distance: float, kd: float) -> dict[str, Any]:
    baseline = np.asarray(_paired(rows, seeds, 0.0, 0.0), dtype=np.float64)
    candidate = np.asarray(_paired(rows, seeds, distance, kd), dtype=np.float64)
    gaps = candidate - baseline
    wins = int(np.sum(gaps > 1e-12))
    losses = int(np.sum(gaps < -1e-12))
    return {
        "brake_distance": distance,
        "kd": kd,
        "score": float(np.mean(candidate)),
        "baseline": float(np.mean(baseline)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps.tolist()),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    table = [_compare(rows, seeds, distance, kd) for distance, kd in SETTINGS if kd > 0.0]
    table.sort(key=lambda item: float(item["score"]), reverse=True)
    return {
        "oracle_warning": "读取全局状态，每步重选分配。不是合法策略，也不是数学上界。网格在同一批 300 个种子上看过。",
        "baseline": "a = clip(10 * Δp, -1, 1)，可达性防抖",
        "brake": "d < d_brake 且 c > 0 时再减 Kd * c * n",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "baseline_score": _compare(rows, seeds, 0.0, 0.0)["score"],
        "ranking": table,
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
            rows.extend(future.result())
            if index % 50 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)
    summary = _summarize(rows, seeds)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
