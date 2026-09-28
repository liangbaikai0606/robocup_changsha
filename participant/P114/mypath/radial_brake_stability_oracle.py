"""二维追踪保持 E034，刹车减在已经限幅的力上。

d_brake 固定为 0.25。只扫 Kd。c = v_R · n，不含目标速度。
另外统计进圈后离开同一圈的次数，以及第一次进圈的步数。
读取全局状态，不是合法策略。
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


BRAKE_DISTANCE = 0.25
GAINS = (0.0, 0.25, 0.5, 1.0)


def _rollout(case: ScenarioCase, kd: float) -> dict[str, Any]:
    """kd 为 0 时就是 clip(10 * Δp)。"""
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    exits = 0
    first_entry = np.full(n, -1, dtype=np.int64)
    inside_target = np.full(n, -1, dtype=np.int64)
    brake_applied = 0
    brake_changed = 0
    try:
        for step in range(horizon):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, _target_vel = _decode(state, n, m)
            actions: dict[str, np.ndarray] = {}
            for robot_index, target_index in enumerate(chosen):
                delta = target_pos[target_index] - robot_pos[robot_index]
                base = np.clip(10.0 * delta, -1.0, 1.0)
                command = base
                distance = float(np.linalg.norm(delta))
                if kd > 0.0 and BRAKE_DISTANCE > distance > 1e-8:
                    normal = delta / distance
                    closing = float(np.dot(robot_vel[robot_index], normal))
                    if closing > 0.0:
                        command = np.clip(base - kd * closing * normal, -1.0, 1.0)
                        brake_applied += 1
                        if float(np.max(np.abs(command - base))) > 1e-6:
                            brake_changed += 1
                actions[f"agent_{robot_index}"] = np.asarray(command, dtype=np.float32)
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
            robot_pos, _robot_vel, target_pos, _target_vel = _decode(env.state(), n, m)
            for robot_index, target_index in enumerate(chosen):
                previous = int(inside_target[robot_index])
                if previous >= 0:
                    stayed = float(np.linalg.norm(target_pos[previous] - robot_pos[robot_index]))
                    if stayed > radius:
                        exits += 1
                        inside_target[robot_index] = -1
                distance = float(np.linalg.norm(target_pos[target_index] - robot_pos[robot_index]))
                if distance <= radius:
                    if first_entry[robot_index] < 0:
                        first_entry[robot_index] = step + 1
                    inside_target[robot_index] = int(target_index)
    finally:
        env.close()
    entered = first_entry[first_entry >= 0]
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "exits": exits,
        "entered_robots": int(len(entered)),
        "first_entry_sum": float(np.sum(entered)) if len(entered) else 0.0,
        "brake_applied": brake_applied,
        "brake_changed": brake_changed,
    }


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kd in GAINS:
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "kd": kd,
                **_rollout(case, kd),
            }
        )
    return rows


def _by_gain(rows: list[dict[str, Any]], kd: float) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["seed"]), str(row["group"])): row for row in rows if row["kd"] == kd}


def _episode_pairs(rows: list[dict[str, Any]], seeds: list[int], kd: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline = _by_gain(rows, 0.0)
    candidate = _by_gain(rows, kd)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for seed in seeds:
        for group in ("basic", "cooperation"):
            pairs.append((baseline[(seed, group)], candidate[(seed, group)]))
    return pairs


def _compare(rows: list[dict[str, Any]], seeds: list[int], kd: float) -> dict[str, Any]:
    scores: list[float] = []
    base_scores: list[float] = []
    exits: list[float] = []
    base_exits: list[float] = []
    firsts: list[float] = []
    base_firsts: list[float] = []
    entered = 0
    base_entered = 0
    changed = 0
    applied = 0
    episodes = 0
    baseline = _by_gain(rows, 0.0)
    candidate = _by_gain(rows, kd)
    for seed in seeds:
        basic = candidate[(seed, "basic")]
        coop = candidate[(seed, "cooperation")]
        old_basic = baseline[(seed, "basic")]
        old_coop = baseline[(seed, "cooperation")]
        scores.append(500.0 * (float(basic["j"]) + float(coop["j"])))
        base_scores.append(500.0 * (float(old_basic["j"]) + float(old_coop["j"])))
        for new_row, old_row in ((basic, old_basic), (coop, old_coop)):
            exits.append(float(new_row["exits"]))
            base_exits.append(float(old_row["exits"]))
            episodes += 1
            entered += int(new_row["entered_robots"])
            base_entered += int(old_row["entered_robots"])
            changed += int(new_row["brake_changed"])
            applied += int(new_row["brake_applied"])
            if int(new_row["entered_robots"]):
                firsts.append(float(new_row["first_entry_sum"]) / int(new_row["entered_robots"]))
            if int(old_row["entered_robots"]):
                base_firsts.append(float(old_row["first_entry_sum"]) / int(old_row["entered_robots"]))
    gaps = [new - old for old, new in zip(base_scores, scores)]
    exit_gaps = [new - old for old, new in zip(base_exits, exits)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)
    robot_slots = episodes * 3
    return {
        "kd": kd,
        "brake_distance": BRAKE_DISTANCE,
        "score": float(np.mean(scores)),
        "baseline_score": float(np.mean(base_scores)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "exits_per_episode": float(np.mean(exits)),
        "baseline_exits_per_episode": float(np.mean(base_exits)),
        "exit_gap": float(np.mean(exit_gaps)),
        "exit_gap_ci95_normal": _mean_ci(exit_gaps),
        "first_entry_step": float(np.mean(firsts)) if firsts else None,
        "baseline_first_entry_step": float(np.mean(base_firsts)) if base_firsts else None,
        "entered_robot_fraction": entered / robot_slots,
        "baseline_entered_robot_fraction": base_entered / robot_slots,
        "brake_applied_per_episode": applied / episodes,
        "brake_changed_per_episode": changed / episodes,
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    return {
        "oracle_warning": "读取全局状态。Kd 在同一批 300 个种子上比较，不是先选再冻结。",
        "control": "a_base = clip(10*Δp,-1,1)；d<0.25 且 v_R·n>0 时 a = clip(a_base - Kd*c*n,-1,1)",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "baseline": _compare(rows, seeds, 0.0),
        "variants": [_compare(rows, seeds, kd) for kd in GAINS if kd > 0.0],
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
