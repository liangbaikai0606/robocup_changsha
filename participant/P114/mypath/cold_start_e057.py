"""E058：第一次看见目标、还没有两帧速度时，合法做法能追回多少。

zero 是当前 entry.py，没有估计就当速度是 0。
hold：新圈还没有估计时不改去追它。
probe：在 hold 上，第一拍的力改成只跟位置，不把未知速度放进相对速度项。
interval：沿连线用目标最快速度算最好和最坏截击步数，只有最坏仍小于对方最好才换。
oracle_first：还没有估计的那一帧填上真实速度，用来量信息价值。

不是 evaluate_one，不改 entry.py。
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


MODES = ("zero", "hold", "interval", "probe", "oracle_first")


def _unit(relative: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(relative))
    if norm <= 1e-9:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(relative, dtype=np.float64) / norm


def _bounds(
    policy: CoverFirstPolicy,
    relative: np.ndarray,
    robot_vel: np.ndarray,
    known_velocity: np.ndarray | None,
    steps_left: int,
    v_max: float,
) -> tuple[int, int, bool]:
    """返回最好步数、最坏步数，以及这一拍速度是不是已经估出来了。"""
    if known_velocity is not None:
        time = policy._intercept_time(relative, robot_vel, known_velocity, steps_left)
        return time, time, True
    direction = _unit(relative)
    best = policy._intercept_time(relative, robot_vel, -v_max * direction, steps_left)
    worst = policy._intercept_time(relative, robot_vel, v_max * direction, steps_left)
    return best, worst, False


def _rows(
    policy: CoverFirstPolicy,
    observation: AgentObservation,
    v_max: float,
    robust: bool,
    truth: np.ndarray | None,
) -> list[dict[str, Any]]:
    targets = np.asarray(observation["targets"], dtype=np.float64)
    visible = np.asarray(observation["target_visible"], dtype=np.bool_)
    peers = np.asarray(observation["peers"], dtype=np.float64)
    peer_visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
    self_pos, self_vel = policy._self_motion(observation)
    steps_left = policy._horizon - int(observation["step_index"])
    built: list[dict[str, Any]] = []
    for index, is_visible in enumerate(visible):
        if not is_visible:
            continue
        relative = targets[index, :2] * policy._position_scale
        estimate = policy._target_velocity_estimates.get(index)
        if estimate is None and truth is not None:
            estimate = np.asarray(truth[index], dtype=np.float64)
        best, worst, known = _bounds(
            policy,
            relative,
            self_vel,
            None if estimate is None else np.asarray(estimate, dtype=np.float64),
            steps_left,
            v_max,
        )
        if not robust:
            best = worst = policy._intercept_time(
                relative,
                self_vel,
                np.zeros(2, dtype=np.float64) if estimate is None else np.asarray(estimate, dtype=np.float64),
                steps_left,
            )
        built.append(
            {
                "index": index,
                "relative": relative.astype(np.float64),
                "distance": float(np.linalg.norm(relative)),
                "occupied": policy._occupied(relative, peers, peer_visible),
                "best": best,
                "worst": worst,
                "known": known,
                "yielded": False,
            }
        )
    return built


def _mark_yields(
    policy: CoverFirstPolicy,
    observation: AgentObservation,
    rows: list[dict[str, Any]],
    v_max: float,
    robust: bool,
    truth: np.ndarray | None,
) -> None:
    steps_left = policy._horizon - int(observation["step_index"])
    self_pos, self_vel = policy._self_motion(observation)
    peers = policy._visible_peers(observation, self_pos, self_vel)
    my_index = int(observation["agent_index"])
    for row in rows:
        if row["occupied"]:
            continue
        target_abs = self_pos + np.asarray(row["relative"], dtype=np.float64)
        best_peer: tuple[int, int, int] | None = None
        for peer_index, peer_pos, peer_vel, _peer_dist in peers:
            if float(np.linalg.norm(target_abs - peer_pos)) > policy._sense_radius:
                continue
            peer_relative = target_abs - peer_pos
            estimate = policy._target_velocity_estimates.get(int(row["index"]))
            if estimate is None and truth is not None:
                estimate = np.asarray(truth[int(row["index"])], dtype=np.float64)
            if robust and estimate is None:
                peer_best, peer_worst, _known = _bounds(
                    policy, peer_relative, peer_vel, None, steps_left, v_max
                )
            else:
                speed = np.zeros(2, dtype=np.float64) if estimate is None else np.asarray(estimate, dtype=np.float64)
                peer_best = peer_worst = policy._intercept_time(peer_relative, peer_vel, speed, steps_left)
            if best_peer is None or (peer_worst, peer_best, peer_index) < best_peer:
                best_peer = (peer_worst, peer_best, peer_index)
        if best_peer is None:
            continue
        peer_worst, peer_best, peer_index = best_peer
        clearer = peer_worst + _YIELD_MARGIN < int(row["best"])
        tied = (
            peer_best == peer_worst == int(row["best"]) == int(row["worst"])
            and peer_index < my_index
        )
        if clearer or tied:
            row["yielded"] = True


def _pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    free = [row for row in rows if not row["occupied"] and not row["yielded"]]
    occupied = [row for row in rows if row["occupied"]]
    return free or occupied


def _choose(
    policy: CoverFirstPolicy,
    observation: AgentObservation,
    mode: str,
    held: int | None,
    v_max: float,
    truth: np.ndarray | None = None,
) -> tuple[int, np.ndarray] | None:
    robust = mode == "interval"
    filled = truth if mode == "oracle_first" else None
    rows = _rows(policy, observation, v_max, robust, filled)
    if not rows:
        return None
    _mark_yields(policy, observation, rows, v_max, robust, filled)
    pool = _pool(rows)
    if not pool:
        return None
    if mode in ("zero", "oracle_first"):
        pool.sort(key=lambda row: (row["worst"], row["distance"], row["index"]))
        chosen = pool[0]
    elif mode in ("hold", "probe"):
        pool.sort(key=lambda row: (row["worst"], row["distance"], row["index"]))
        chosen = pool[0]
        if held is not None and int(chosen["index"]) != held and not bool(chosen["known"]):
            kept = next((row for row in rows if int(row["index"]) == held), None)
            if kept is not None:
                chosen = kept
    else:
        undominated = [
            row
            for row in pool
            if not any(
                int(other["worst"]) < int(row["best"])
                for other in pool
                if int(other["index"]) != int(row["index"])
            )
        ]
        kept = next((row for row in undominated if int(row["index"]) == held), None)
        if kept is not None:
            chosen = kept
        else:
            undominated.sort(key=lambda row: (row["worst"], row["best"], row["distance"], row["index"]))
            chosen = undominated[0]
    return int(chosen["index"]), np.asarray(chosen["relative"], dtype=np.float64)


def _rollout(case: ScenarioCase, mode: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    v_max = float(config.public.target_max_speed)
    policies = [CoverFirstPolicy() for _robot in range(n)]
    held: list[int | None] = [None for _robot in range(n)]
    kept_steps = 0
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
    try:
        for _step in range(horizon):
            _robot_pos, _robot_vel, _target_pos, target_vel = _decode(
                env.state(), n, int(config.num_targets)
            )
            actions: dict[str, np.ndarray] = {}
            for index, agent_id in enumerate(env.agents):
                observation = observations[agent_id]
                policy = policies[index]
                policy._update_target_velocity(observation)
                action = None
                if policy._commit is not None:
                    action = policy._follow_commit(observation)
                if action is None:
                    action = policy._start_boost(observation)
                if action is None:
                    before = held[index]
                    chosen = _choose(policy, observation, mode, before, v_max, target_vel)
                    if chosen is None:
                        held[index] = None
                        action = np.zeros(2, dtype=np.float32)
                    else:
                        target_index, relative = chosen
                        if before is not None and target_index == before:
                            zero_choice = _choose(policy, observation, "zero", None, v_max)
                            if zero_choice is not None and int(zero_choice[0]) != target_index:
                                kept_steps += 1
                        held[index] = target_index
                        self_vel = policy._self_motion(observation)[1]
                        estimate = policy._target_velocity_estimates.get(target_index)
                        if mode == "probe" and estimate is None:
                            action = np.clip(20.0 * relative, -1.0, 1.0).astype(np.float32)
                        else:
                            if estimate is None and mode == "oracle_first":
                                speed = np.asarray(target_vel[target_index], dtype=np.float64)
                            else:
                                speed = (
                                    np.zeros(2, dtype=np.float64)
                                    if estimate is None
                                    else np.asarray(estimate, dtype=np.float64)
                                )
                            action = policy._thrust_to_target(relative, self_vel, speed)
                actions[agent_id] = np.asarray(action, dtype=np.float32)
            observations, rewards, _terms, _truncs, _infos = env.step(actions)
            total += float(next(iter(rewards.values())))
    finally:
        for policy in policies:
            policy.close()
        env.close()
    return {"j": total / horizon, "kept_steps": float(kept_steps)}


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "group": case.group_id,
            "seed": int(case.scenario_seed),
            "mode": mode,
            **_rollout(case, mode),
        }
        for mode in MODES
    ]


def _paired(rows: list[dict[str, Any]], seeds: list[int], mode: str) -> list[float]:
    chosen = {(int(row["seed"]), str(row["group"])): row for row in rows if row["mode"] == mode}
    return [
        500.0 * (float(chosen[(seed, "basic")]["j"]) + float(chosen[(seed, "cooperation")]["j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    scores = {mode: _paired(rows, seeds, mode) for mode in MODES}
    base = scores["zero"]
    gaps: dict[str, Any] = {}
    for mode in MODES:
        if mode == "zero":
            continue
        delta = [new - old for old, new in zip(base, scores[mode])]
        wins = sum(value > 1e-12 for value in delta)
        losses = sum(value < -1e-12 for value in delta)
        gap = float(np.mean(delta))
        gaps[mode] = {
            "gap": gap,
            "gap_ci95_normal": _mean_ci(delta),
            "wins": wins,
            "ties": len(delta) - wins - losses,
            "losses": losses,
            "kept_steps_per_episode": float(
                np.mean([float(row["kept_steps"]) for row in rows if row["mode"] == mode])
            ),
        }
    oracle_gap = float(gaps["oracle_first"]["gap"])
    for mode, item in gaps.items():
        item["fraction_of_oracle_first"] = (item["gap"] / oracle_gap) if abs(oracle_gap) > 1e-12 else 0.0
    return {
        "note": "脚本自算，不是 evaluate_one。zero 是当前 entry.py：没有速度估计就当 0。hold 和 interval、probe 不读真实速度。oracle_first 只在还没有估计的那一帧填真实速度。未改 entry.py。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "scores": {mode: float(np.mean(values)) for mode, values in scores.items()},
        "gaps_vs_zero": gaps,
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
