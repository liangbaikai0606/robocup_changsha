"""可达性防抖不变，比较两种车速内环。

外环两边相同：想要的速度是目标速度加上位置差，再限到 [-1, 1]。

    v* = clip(v_target + 2 * (p_target - p_robot), -1, 1)

简单 P 刹车：

    a = clip(4 * (v* - v), -1, 1)

PD 控速多一项速度误差的变化：

    a = clip(4 * (v* - v) + 0.5 * (e - e_prev) / dt, -1, 1)

读取全局状态，不是合法策略。种子与 E033 相同。
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

from coverage_bench.suites import ScenarioCase
from oracle_search import _decode
from reachability_debounce_oracle import (
    MASTER_SEED,
    _Assigner,
    _layout_config,
    _mean_ci,
    _random_seeds,
)


OUTER_GAIN = 2.0
SPEED_KP = 4.0
SPEED_KD = 0.5


def _desired_velocity(delta_p: np.ndarray, target_velocity: np.ndarray) -> np.ndarray:
    """外环：位置差按比例变成想要的速度，并限在速度上限内。"""
    return np.clip(target_velocity + OUTER_GAIN * delta_p, -1.0, 1.0)


def _rollout(case: ScenarioCase, pd_speed: bool) -> dict[str, Any]:
    config = case.task_config
    from coverage_bench.envs.factory import make_training_env

    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    n = int(config.num_agents)
    m = int(config.num_targets)
    dt = float(config.public.dt)
    previous_error = np.zeros((n, 2), dtype=np.float64)
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
                desired = _desired_velocity(delta_p, target_vel[target_index])
                error = desired - robot_vel[robot_index]
                command = SPEED_KP * error
                if pd_speed and step > 0:
                    command = command + SPEED_KD * (error - previous_error[robot_index]) / dt
                previous_error[robot_index] = error
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
        "switches": assigner.switches,
    }


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    proportional = _rollout(case, False)
    pd = _rollout(case, True)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
    }
    for name, result in (("p_brake", proportional), ("pd_speed", pd)):
        for key, value in result.items():
            row[f"{name}_{key}"] = value
    row["gap_j"] = float(pd["j"]) - float(proportional["j"])
    return row


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    p_scores: list[float] = []
    pd_scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        p_scores.append(500.0 * (float(basic["p_brake_j"]) + float(coop["p_brake_j"])))
        pd_scores.append(500.0 * (float(basic["pd_speed_j"]) + float(coop["pd_speed_j"])))
    gaps = [new - old for old, new in zip(p_scores, pd_scores)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)

    def mean_of(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    return {
        "oracle_warning": "读取全局状态，每步重选分配。不是合法策略，也不是数学上界。增益没有再搜索。",
        "assignment": "两边都是可达性防抖",
        "outer": "v* = clip(v_target + 2 * Δp, -1, 1)",
        "p_brake": "a = clip(4 * (v* - v), -1, 1)",
        "pd_speed": "a = clip(4 * (v* - v) + 0.5 * (e - e_prev) / dt, -1, 1)",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "p_brake_score": float(np.mean(p_scores)),
        "pd_speed_score": float(np.mean(pd_scores)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "p_brake_coverage": mean_of("p_brake_coverage"),
        "pd_speed_coverage": mean_of("pd_speed_coverage"),
        "p_brake_collision": mean_of("p_brake_collision"),
        "pd_speed_collision": mean_of("pd_speed_collision"),
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
