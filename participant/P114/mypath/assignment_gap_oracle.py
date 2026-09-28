"""把 E044 的 169.20 和局部 E2 的 165.41 拆开。

每一档只比上一档多一种信息或一种分配，开车先保持现在的 entry.py。
最后一档才改成 E044 的 clip(10*Δp)。差加起来等于两头的分差。
读取全局状态的档不是合法策略。
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
from entry import CoverFirstPolicy, _YIELD_MARGIN
from oracle_search import _decode
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds
from time_to_intercept_oracle import _choose_assignment, _time_matrix


MODES = ("local", "true_speed", "see_peers", "see_targets", "joint_e028", "joint_p")


def _visible_target_indexes(observation: AgentObservation) -> list[int]:
    visible = np.asarray(observation["target_visible"], dtype=np.bool_)
    return [index for index, flag in enumerate(visible) if flag]


def _greedy_target(
    policy: CoverFirstPolicy,
    observation: AgentObservation,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    peer_indexes: list[int],
    target_indexes: list[int],
) -> int | None:
    """和 entry.py 的让圈相同，只是可见的人和圈由调用方给定。"""
    steps_left = policy._horizon - int(observation["step_index"])
    my_index = int(observation["agent_index"])
    rows: list[dict[str, Any]] = []
    for target_index in target_indexes:
        relative = target_pos[target_index] - robot_pos
        occupied = False
        for peer_index in peer_indexes:
            # peer position is filled by the caller through robot_bank on the policy
            peer_pos = policy._gap_peer_pos[peer_index]
            if float(np.linalg.norm(target_pos[target_index] - peer_pos)) <= policy._cover_radius:
                occupied = True
                break
        rows.append(
            {
                "index": target_index,
                "relative": relative,
                "distance": float(np.linalg.norm(relative)),
                "occupied": occupied,
                "time": policy._intercept_time(
                    relative, robot_vel, target_vel[target_index], steps_left
                ),
                "yielded": False,
            }
        )
    for row in rows:
        if row["occupied"]:
            continue
        best_peer: tuple[int, int] | None = None
        for peer_index in peer_indexes:
            peer_pos = policy._gap_peer_pos[peer_index]
            peer_vel = policy._gap_peer_vel[peer_index]
            if float(np.linalg.norm(target_pos[int(row["index"])] - peer_pos)) > policy._sense_radius:
                continue
            peer_time = policy._intercept_time(
                target_pos[int(row["index"])] - peer_pos,
                peer_vel,
                target_vel[int(row["index"])],
                steps_left,
            )
            if best_peer is None or (peer_time, peer_index) < best_peer:
                best_peer = (peer_time, peer_index)
        if best_peer is None:
            continue
        peer_time, peer_index = best_peer
        if peer_time + _YIELD_MARGIN < int(row["time"]) or (
            peer_time == int(row["time"]) and peer_index < my_index
        ):
            row["yielded"] = True
    free = [row for row in rows if not row["occupied"] and not row["yielded"]]
    pool = free or [row for row in rows if row["occupied"]]
    if not pool:
        return None
    pool.sort(key=lambda row: (row["time"], row["distance"], row["index"]))
    return int(pool[0]["index"])


def _act_local(
    policy: CoverFirstPolicy,
    observation: AgentObservation,
    mode: str,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    all_robot_pos: np.ndarray,
    all_robot_vel: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    policy._update_target_velocity(observation)
    if policy._commit is not None:
        committed = policy._follow_commit(observation)
        if committed is not None:
            return committed, None
    boost = policy._start_boost(observation)
    if boost is not None:
        return boost, None
    my_index = int(observation["agent_index"])
    if mode == "true_speed":
        for target_index in _visible_target_indexes(observation):
            policy._target_velocity_estimates[target_index] = target_vel[target_index].copy()
        chosen = policy._choose_intercept_target(observation)
        if chosen is None:
            return np.zeros(2, dtype=np.float32), None
        target_index, relative = chosen
        return (
            policy._thrust_to_target(relative, robot_vel, target_vel[target_index]),
            target_index,
        )
    if mode == "local":
        chosen = policy._choose_intercept_target(observation)
        if chosen is None:
            return np.zeros(2, dtype=np.float32), None
        target_index, relative = chosen
        speed = policy._target_velocity_estimates.get(target_index, np.zeros(2))
        return policy._thrust_to_target(relative, robot_vel, speed), target_index

    sense = policy._sense_radius
    peer_indexes = [
        index
        for index in range(len(all_robot_pos))
        if index != my_index
        and (
            mode in ("see_peers", "see_targets")
            or float(np.linalg.norm(all_robot_pos[index] - robot_pos)) <= sense
        )
    ]
    if mode == "see_targets":
        target_indexes = list(range(len(target_pos)))
    else:
        target_indexes = _visible_target_indexes(observation)
    policy._gap_peer_pos = all_robot_pos
    policy._gap_peer_vel = all_robot_vel
    target_index = _greedy_target(
        policy,
        observation,
        robot_pos,
        robot_vel,
        target_pos,
        target_vel,
        peer_indexes,
        target_indexes,
    )
    if target_index is None:
        return np.zeros(2, dtype=np.float32), None
    relative = target_pos[target_index] - robot_pos
    return policy._thrust_to_target(relative, robot_vel, target_vel[target_index]), target_index


def _rollout(case: ScenarioCase, mode: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    m = int(config.num_targets)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    policies = [CoverFirstPolicy() for _robot in range(n)]
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=n,
                num_targets=m,
                horizon=horizon,
                task=task,
                policy_seed=0,
            )
        )
    total = 0.0
    duplicate_steps = 0
    chosen_steps = 0
    try:
        for step in range(horizon):
            robot_pos, robot_vel, target_pos, target_vel = _decode(env.state(), n, m)
            if mode == "joint_p":
                times = _time_matrix(robot_pos, robot_vel, target_pos, target_vel, horizon - step, config)
                chosen = _choose_assignment(times, None, 0)
                actions = {
                    f"agent_{index}": np.clip(10.0 * (target_pos[chosen[index]] - robot_pos[index]), -1.0, 1.0).astype(
                        np.float32
                    )
                    for index in range(n)
                }
                picks = list(chosen)
            elif mode == "joint_e028":
                times = _time_matrix(robot_pos, robot_vel, target_pos, target_vel, horizon - step, config)
                chosen = _choose_assignment(times, None, 0)
                actions = {}
                picks = []
                for index, agent_id in enumerate(env.agents):
                    policies[index]._update_target_velocity(observations[agent_id])
                    action = None
                    if policies[index]._commit is not None:
                        action = policies[index]._follow_commit(observations[agent_id])
                    if action is None:
                        action = policies[index]._start_boost(observations[agent_id])
                    if action is None:
                        relative = target_pos[chosen[index]] - robot_pos[index]
                        action = policies[index]._thrust_to_target(
                            relative, robot_vel[index], target_vel[chosen[index]]
                        )
                        picks.append(int(chosen[index]))
                    actions[agent_id] = np.asarray(action, dtype=np.float32)
            else:
                actions = {}
                picks = []
                for index, agent_id in enumerate(env.agents):
                    action, target_index = _act_local(
                        policies[index],
                        observations[agent_id],
                        mode,
                        robot_pos[index],
                        robot_vel[index],
                        target_pos,
                        target_vel,
                        robot_pos,
                        robot_vel,
                    )
                    actions[agent_id] = np.asarray(action, dtype=np.float32)
                    if target_index is not None:
                        picks.append(target_index)
            if len(picks) >= 2 and len(set(picks)) < len(picks):
                duplicate_steps += 1
            if picks:
                chosen_steps += 1
            observations, rewards, _terms, _truncs, _infos = env.step(actions)
            total += float(next(iter(rewards.values())))
    finally:
        for policy in policies:
            policy.close()
        env.close()
    return {
        "j": total / horizon,
        "duplicate_steps": float(duplicate_steps),
        "chosen_steps": float(chosen_steps),
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


def _mean_duplicate(rows: list[dict[str, Any]], mode: str) -> float:
    values = [float(row["duplicate_steps"]) for row in rows if row["mode"] == mode]
    return float(np.mean(values)) if values else 0.0


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    scores = {mode: _paired(rows, seeds, mode) for mode in MODES}
    names = {
        "true_speed": "看得见的圈用真实目标速度，不再用两帧估计",
        "see_peers": "再知道视野外的队友在哪",
        "see_targets": "再知道视野外的圈在哪",
        "joint_e028": "再改成六种分配里总截击步数最少，开车不变",
        "joint_p": "分配不变，开车改成 clip(10*Δp)",
    }
    steps: list[dict[str, Any]] = []
    previous = "local"
    for mode in MODES[1:]:
        gaps = [new - old for old, new in zip(scores[previous], scores[mode])]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        steps.append(
            {
                "from": previous,
                "to": mode,
                "meaning": names[mode],
                "gap": float(np.mean(gaps)),
                "gap_ci95_normal": _mean_ci(gaps),
                "wins": wins,
                "ties": len(gaps) - wins - losses,
                "losses": losses,
            }
        )
        previous = mode
    return {
        "oracle_warning": "除 local 外都读了全局状态。不是 evaluate_one，未改 entry.py。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "scores": {mode: float(np.mean(values)) for mode, values in scores.items()},
        "total_gap_joint_p_minus_local": float(np.mean(scores["joint_p"]) - np.mean(scores["local"])),
        "steps": steps,
        "duplicate_steps_per_episode": {mode: _mean_duplicate(rows, mode) for mode in MODES},
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
