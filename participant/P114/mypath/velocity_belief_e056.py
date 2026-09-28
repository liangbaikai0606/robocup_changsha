"""把 E055 里「真实目标速度 +2.09」拆成选圈和开车，并试一个本地过滤。

四档都只看当前 entry.py 看得见的圈：

- est：截击步数和开车都用两帧估计。
- true_assign：只有截击步数用真实速度，开车仍用估计。
- true_thrust：只有开车用真实速度，选圈仍用估计。
- true_both：两边都用真实速度，应对上 E055 的 167.50。
- true_if_missing：已经有两帧估计时仍用估计；还没有估计的那一帧才填真实速度。
- drop_spike：估计速度比目标最快速度还大，就当成看不见的那几步被算进了一帧，改用 0。

真实速度那三档读了全局状态。drop_spike 只用局部估计。不是 evaluate_one，不改 entry.py。
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
PARTICIPANT_ROOT = Path(__file__).resolve().parents[1]
MYPATH_ROOT = Path(__file__).resolve().parent
for folder in (REPO_ROOT, PARTICIPANT_ROOT, MYPATH_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import AgentObservation, EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase
from entry import CoverFirstPolicy
from oracle_search import _decode
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds


MODES = ("est", "true_assign", "true_thrust", "true_both", "true_if_missing", "drop_spike")


def _visible_indexes(observation: AgentObservation) -> list[int]:
    visible = np.asarray(observation["target_visible"], dtype=np.bool_)
    return [index for index, flag in enumerate(visible) if bool(flag)]


def _act(
    policy: CoverFirstPolicy,
    observation: AgentObservation,
    mode: str,
    robot_vel: np.ndarray,
    target_vel: np.ndarray,
    target_max_speed: float,
) -> tuple[np.ndarray, int | None, dict[str, float]]:
    policy._update_target_velocity(observation)
    stats = {"visible": 0.0, "missing": 0.0, "spike": 0.0, "error_sum": 0.0, "error_count": 0.0}
    if policy._commit is not None:
        committed = policy._follow_commit(observation)
        if committed is not None:
            return committed, None, stats
    boost = policy._start_boost(observation)
    if boost is not None:
        return boost, None, stats

    visible = _visible_indexes(observation)
    estimates = {
        index: value.copy()
        for index, value in policy._target_velocity_estimates.items()
    }
    for index in visible:
        stats["visible"] += 1.0
        estimate = estimates.get(index)
        if estimate is None:
            stats["missing"] += 1.0
            continue
        truth = target_vel[index]
        error = float(np.linalg.norm(estimate - truth))
        stats["error_sum"] += error
        stats["error_count"] += 1.0
        if float(np.linalg.norm(estimate)) > target_max_speed + 1e-6:
            stats["spike"] += 1.0

    assign_speed = dict(estimates)
    thrust_speed = dict(estimates)
    if mode in ("true_assign", "true_both"):
        for index in visible:
            assign_speed[index] = target_vel[index].copy()
    if mode in ("true_thrust", "true_both"):
        for index in visible:
            thrust_speed[index] = target_vel[index].copy()
    if mode == "true_if_missing":
        for index in visible:
            if index not in estimates:
                assign_speed[index] = target_vel[index].copy()
                thrust_speed[index] = target_vel[index].copy()
    if mode == "drop_spike":
        for index in visible:
            estimate = estimates.get(index)
            if estimate is not None and float(np.linalg.norm(estimate)) > target_max_speed + 1e-6:
                assign_speed[index] = np.zeros(2, dtype=np.float64)
                thrust_speed[index] = np.zeros(2, dtype=np.float64)

    policy._target_velocity_estimates = assign_speed
    chosen = policy._choose_intercept_target(observation)
    policy._target_velocity_estimates = thrust_speed
    if chosen is None:
        policy._target_velocity_estimates = estimates
        return np.zeros(2, dtype=np.float32), None, stats
    target_index, relative = chosen
    speed = thrust_speed.get(target_index, np.zeros(2, dtype=np.float64))
    action = policy._thrust_to_target(relative, robot_vel, speed)
    policy._target_velocity_estimates = estimates
    return action, int(target_index), stats


def _rollout(case: ScenarioCase, mode: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    target_max_speed = float(config.public.target_max_speed)
    policies = [CoverFirstPolicy() for _robot in range(n)]
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=n,
                num_targets=int(config.num_targets),
                horizon=horizon,
                task=task,
                policy_seed=0,
            )
        )
    total = 0.0
    visible = 0.0
    missing = 0.0
    spike = 0.0
    error_sum = 0.0
    error_count = 0.0
    try:
        for _step in range(horizon):
            _robot_pos, robot_vel, _target_pos, target_vel = _decode(env.state(), n, int(config.num_targets))
            actions: dict[str, np.ndarray] = {}
            for index, agent_id in enumerate(env.agents):
                action, _target_index, stats = _act(
                    policies[index],
                    observations[agent_id],
                    mode,
                    robot_vel[index],
                    target_vel,
                    target_max_speed,
                )
                actions[agent_id] = np.asarray(action, dtype=np.float32)
                if mode == "est":
                    visible += stats["visible"]
                    missing += stats["missing"]
                    spike += stats["spike"]
                    error_sum += stats["error_sum"]
                    error_count += stats["error_count"]
            observations, rewards, _terms, _truncs, _infos = env.step(actions)
            total += float(next(iter(rewards.values())))
    finally:
        for policy in policies:
            policy.close()
        env.close()
    return {
        "j": total / horizon,
        "visible": visible,
        "missing": missing,
        "spike": spike,
        "error_sum": error_sum,
        "error_count": error_count,
    }


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "mode": mode,
                **_rollout(case, mode),
            }
        )
    return rows


def _paired(rows: list[dict[str, Any]], seeds: list[int], mode: str) -> list[float]:
    chosen = {(int(row["seed"]), str(row["group"])): row for row in rows if row["mode"] == mode}
    return [
        500.0 * (float(chosen[(seed, "basic")]["j"]) + float(chosen[(seed, "cooperation")]["j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    scores = {mode: _paired(rows, seeds, mode) for mode in MODES}
    base = scores["est"]
    gaps: dict[str, Any] = {}
    for mode in MODES:
        if mode == "est":
            continue
        delta = [new - old for old, new in zip(base, scores[mode])]
        wins = sum(value > 1e-12 for value in delta)
        losses = sum(value < -1e-12 for value in delta)
        gaps[mode] = {
            "gap": float(np.mean(delta)),
            "gap_ci95_normal": _mean_ci(delta),
            "wins": wins,
            "ties": len(delta) - wins - losses,
            "losses": losses,
        }
    est_rows = [row for row in rows if row["mode"] == "est"]
    visible = sum(float(row["visible"]) for row in est_rows)
    missing = sum(float(row["missing"]) for row in est_rows)
    spike = sum(float(row["spike"]) for row in est_rows)
    error_count = sum(float(row["error_count"]) for row in est_rows)
    error_sum = sum(float(row["error_sum"]) for row in est_rows)
    return {
        "note": "脚本自算，不是 evaluate_one。true_* 读了全局目标速度。drop_spike 只看本地估计有没有超过目标最快速度。未改 entry.py。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "scores": {mode: float(np.mean(values)) for mode, values in scores.items()},
        "gaps_vs_est": gaps,
        "estimate_on_visible": {
            "visible_slots": visible,
            "missing_before_two_frames": missing,
            "missing_fraction": (missing / visible) if visible else 0.0,
            "faster_than_target_max": spike,
            "spike_fraction_of_estimates": (spike / error_count) if error_count else 0.0,
            "mean_speed_error": (error_sum / error_count) if error_count else 0.0,
        },
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
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
