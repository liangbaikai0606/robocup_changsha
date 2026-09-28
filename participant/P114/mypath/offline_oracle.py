"""离线先知：开局就知道未来 10 步的真实目标轨迹。

每一步先试 6 种一对一分配，再把后面两步也展开，其余步用真实位置的最近分配。
力是 a = clip(20*Δp + 4*(v_T - v_R), -1, 1)，Δp 和 v_T 用下一步的真实圈位置。
积分走官方 advance_robots / advance_targets。

这是做得到的回报，最优值至少这么高。连续力没有搜完，所以不是 max 的证明。
速度上限画出来的圆才是 sum r 的上界，而且偏松。不写入 entry.py。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[3]
MYPATH_ROOT = Path(__file__).resolve().parent
ENTRY_ROOT = REPO_ROOT / "participant" / "P114"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MYPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MYPATH_ROOT))
if str(ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(ENTRY_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.envs.motion import advance_targets
from coverage_bench.envs.physics import advance_robots
from coverage_bench.envs.scenario import snapshot
from coverage_bench.envs.types import ScenarioState
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.protocol import EpisodeContext, PublicTaskParams
from coverage_bench.rewards import compute_reward
from coverage_bench.suites import ScenarioCase
from entry import build_policy
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds

LOOKAHEAD = 2
ASSIGNMENTS = list(permutations((0, 1, 2)))


def _capture(state: ScenarioState) -> dict[str, Any]:
    """记下车、圈和目标随机源，便于分支之后退回。"""
    return {
        "robot_positions": state.robot_positions.copy(),
        "robot_velocities": state.robot_velocities.copy(),
        "target_positions": state.target_positions.copy(),
        "target_velocities": state.target_velocities.copy(),
        "target_next_turn": state.target_next_turn.copy(),
        "step_index": int(state.step_index),
        "rng": copy.deepcopy(state.target_rng.bit_generator.state),
    }


def _restore(state: ScenarioState, saved: dict[str, Any]) -> None:
    state.robot_positions[:] = saved["robot_positions"]
    state.robot_velocities[:] = saved["robot_velocities"]
    state.target_positions[:] = saved["target_positions"]
    state.target_velocities[:] = saved["target_velocities"]
    state.target_next_turn[:] = saved["target_next_turn"]
    state.step_index = int(saved["step_index"])
    state.target_rng.bit_generator.state = copy.deepcopy(saved["rng"])


def _future_targets(state: ScenarioState) -> np.ndarray:
    """这一步积分之后，圈会在哪。算完退回，不消耗随机源。"""
    saved = _capture(state)
    advance_targets(state)
    future = state.target_positions.copy()
    _restore(state, saved)
    return future


def _preview_action(state: ScenarioState, assignment: tuple[int, ...]) -> dict[str, np.ndarray]:
    """朝下一步的真实圈位置做位置加相对速度。"""
    dt = float(state.config.public.dt)
    current = state.target_positions
    nxt = _future_targets(state)
    target_velocity = (nxt - current) / dt
    actions: dict[str, np.ndarray] = {}
    for robot_index, target_index in enumerate(assignment):
        offset = nxt[target_index] - state.robot_positions[robot_index]
        relative_velocity = target_velocity[target_index] - state.robot_velocities[robot_index]
        actions[f"agent_{robot_index}"] = np.clip(
            20.0 * offset + 4.0 * relative_velocity, -1.0, 1.0
        ).astype(np.float32)
    return actions


def _nearest_assignment(state: ScenarioState) -> tuple[int, ...]:
    """用这一刻的真实位置，选距离和最小的一对一。"""
    robots = state.robot_positions
    targets = state.target_positions
    best = ASSIGNMENTS[0]
    best_cost = float("inf")
    for assignment in ASSIGNMENTS:
        cost = sum(
            float(np.linalg.norm(robots[robot_index] - targets[target_index]))
            for robot_index, target_index in enumerate(assignment)
        )
        if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and assignment < best):
            best = assignment
            best_cost = cost
    return best


def _apply(state: ScenarioState, assignment: tuple[int, ...]) -> float:
    actions = _preview_action(state, assignment)
    advance_robots(state, actions)
    advance_targets(state)
    state.step_index += 1
    metrics = compute_step_metrics(snapshot(state))
    return float(compute_reward(metrics, float(state.config.collision_weight))["team_reward"])


def _simulate(state: ScenarioState, forced: tuple[tuple[int, ...], ...]) -> float:
    """先走给定的前几步分配，后面每步改成最近分配。算完退回。"""
    saved = _capture(state)
    total = 0.0
    horizon = int(state.config.horizon)
    for assignment in forced:
        if state.step_index >= horizon:
            break
        total += _apply(state, assignment)
    while state.step_index < horizon:
        total += _apply(state, _nearest_assignment(state))
    _restore(state, saved)
    return total


def _plan_first(state: ScenarioState) -> tuple[int, ...]:
    """展开前两步的 6×6 种分配，用真实未来轨迹打分，只执行第一步。"""
    best = ASSIGNMENTS[0]
    best_value = -1e18
    if int(state.step_index) + 1 >= int(state.config.horizon):
        return _nearest_assignment(state)
    for first in ASSIGNMENTS:
        for second in ASSIGNMENTS:
            value = _simulate(state, (first, second))
            if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and first < best):
                best = first
                best_value = value
    return best


def _rollout_oracle(state: ScenarioState) -> dict[str, float]:
    horizon = int(state.config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    for _ in range(horizon):
        assignment = _plan_first(state)
        actions = _preview_action(state, assignment)
        advance_robots(state, actions)
        advance_targets(state)
        state.step_index += 1
        metrics = compute_step_metrics(snapshot(state))
        total += float(compute_reward(metrics, float(state.config.collision_weight))["team_reward"])
        coverage.append(float(metrics.coverage_rate))
        collision.append(float(metrics.collision_rate))
    return {
        "return_sum": total,
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
    }


def _axis_offsets(state: ScenarioState) -> list[float]:
    """油门打满、从静止出发，每一拍单轴最多离开起点多远。"""
    pub = state.config.public
    dt = float(pub.dt)
    damping = float(pub.damping)
    accel = float(pub.drive_force) / float(pub.robot_mass) * dt
    velocity = 0.0
    traveled = 0.0
    offsets: list[float] = []
    for _ in range(int(state.config.horizon)):
        traveled += velocity * dt
        offsets.append(traveled)
        velocity = (1.0 - damping) * velocity + accel
    return offsets


def _thrust_square_upper(state: ScenarioState) -> float:
    """不用碰撞助力时，每步位置落在正方形里。这是 sum r 的上界，但各步没有连成一条轨迹。"""
    radius = float(state.config.public.target_radius)
    horizon = int(state.config.horizon)
    n = int(state.config.num_agents)
    m = int(state.config.num_targets)
    origin = state.robot_positions.copy()
    saved = _capture(state)
    total = 0.0
    for reach in _axis_offsets(state):
        advance_targets(state)
        adj = np.zeros((n, m), dtype=np.int32)
        for robot_index in range(n):
            for target_index in range(m):
                overflow = np.abs(state.target_positions[target_index] - origin[robot_index]) - reach
                overflow = np.maximum(overflow, 0.0)
                if float(np.hypot(overflow[0], overflow[1])) <= radius:
                    adj[robot_index, target_index] = 1
        matched = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
        total += float(np.sum(matched >= 0)) / float(m)
    _restore(state, saved)
    return total


def _speed_cap_upper(state: ScenarioState) -> float:
    """每步位移不超过最大速度乘 dt。圆内能配上的覆盖率之和，是 sum r 的上界。"""
    pub = state.config.public
    dt = float(pub.dt)
    max_speed = float(pub.robot_max_speed)
    radius = float(pub.target_radius)
    horizon = int(state.config.horizon)
    n = int(state.config.num_agents)
    m = int(state.config.num_targets)
    origin = state.robot_positions.copy()
    speed0 = np.linalg.norm(state.robot_velocities, axis=1)
    saved = _capture(state)
    total = 0.0
    for step in range(1, horizon + 1):
        advance_targets(state)
        reach = speed0 * dt + (step - 1) * max_speed * dt
        adj = np.zeros((n, m), dtype=np.int32)
        for robot_index in range(n):
            for target_index in range(m):
                gap = float(np.linalg.norm(state.target_positions[target_index] - origin[robot_index]))
                if gap <= float(reach[robot_index]) + radius:
                    adj[robot_index, target_index] = 1
        matched = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
        total += float(np.sum(matched >= 0)) / float(m)
    _restore(state, saved)
    return total


def _context(case: ScenarioCase, agent_index: int) -> EpisodeContext:
    task = case.task_config.public
    return EpisodeContext(
        agent_index=agent_index,
        num_agents=int(case.task_config.num_agents),
        num_targets=int(case.task_config.num_targets),
        horizon=int(case.task_config.horizon),
        task=PublicTaskParams(
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
        ),
        policy_seed=0,
    )


def _rollout_entry(case: ScenarioCase) -> dict[str, float]:
    """同一局上跑当前 entry.py，用来和先知配对。"""
    env = make_training_env(case.task_config)
    observations, _infos = env.reset(seed=int(case.scenario_seed))
    policies = [build_policy(None) for _ in env.agents]
    for index, policy in enumerate(policies):
        policy.reset(_context(case, index))
    horizon = int(case.task_config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for _ in range(horizon):
            actions = {
                agent: policies[index].act(observations[agent])
                for index, agent in enumerate(env.agents)
            }
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = env.agents[0] if env.agents else next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    return {
        "return_sum": total,
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
    }


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    env = make_training_env(case.task_config)
    env.reset(seed=int(case.scenario_seed))
    assert env._scenario_state is not None
    upper = _speed_cap_upper(env._scenario_state)
    thrust_upper = _thrust_square_upper(env._scenario_state)
    oracle = _rollout_oracle(env._scenario_state)
    env.close()
    entry = _rollout_entry(case)
    return {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
        "oracle_return_sum": oracle["return_sum"],
        "oracle_j": oracle["j"],
        "oracle_coverage": oracle["coverage"],
        "oracle_collision": oracle["collision"],
        "entry_return_sum": entry["return_sum"],
        "entry_j": entry["j"],
        "entry_coverage": entry["coverage"],
        "entry_collision": entry["collision"],
        "upper_return_sum": upper,
        "thrust_upper_return_sum": thrust_upper,
        "gap_j": float(oracle["j"]) - float(entry["j"]),
    }


def _check_matches_env() -> None:
    """内部积分和 env.step 必须是同一套物理。"""
    case = ScenarioCase("check", "basic", _layout_config("uniform"), 1001)
    env = make_training_env(case.task_config)
    env.reset(seed=1001)
    assert env._scenario_state is not None
    state = env._scenario_state
    saved = _capture(state)
    internal = 0.0
    for _ in range(int(case.task_config.horizon)):
        assignment = ASSIGNMENTS[0]
        actions = _preview_action(state, assignment)
        internal += _apply(state, assignment)
    _restore(state, saved)
    env.reset(seed=1001)
    external = 0.0
    for _ in range(int(case.task_config.horizon)):
        actions = _preview_action(env._scenario_state, ASSIGNMENTS[0])
        _obs, rewards, _terms, _truncs, _infos = env.step(actions)
        external += float(next(iter(rewards.values())))
    env.close()
    if abs(internal - external) > 1e-6:
        raise RuntimeError(f"内部积分和 env.step 不一致: {internal} vs {external}")


def _paired(rows: list[dict[str, Any]], seeds: list[int], prefix: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        scores.append(500.0 * (float(basic[f"{prefix}_j"]) + float(coop[f"{prefix}_j"])))
    return scores


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    oracle_scores = _paired(rows, seeds, "oracle")
    entry_scores = _paired(rows, seeds, "entry")
    gaps = [new - old for old, new in zip(entry_scores, oracle_scores)]
    wins = sum(gap > 1e-9 for gap in gaps)
    losses = sum(gap < -1e-9 for gap in gaps)
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}

    def mean_return(prefix: str) -> float:
        values = [float(row[f"{prefix}_return_sum"]) for row in rows]
        return float(np.mean(values))

    upper_returns = [float(row["upper_return_sum"]) for row in rows]
    thrust_returns = [float(row["thrust_upper_return_sum"]) for row in rows]
    upper_scores: list[float] = []
    thrust_scores: list[float] = []
    for seed in seeds:
        basic = float(lookup[(seed, "basic")]["upper_return_sum"])
        coop = float(lookup[(seed, "cooperation")]["upper_return_sum"])
        upper_scores.append(50.0 * (basic + coop))
        thrust_scores.append(
            50.0
            * (
                float(lookup[(seed, "basic")]["thrust_upper_return_sum"])
                + float(lookup[(seed, "cooperation")]["thrust_upper_return_sum"])
            )
        )
    return {
        "note": "先知分数是真实物理里做到的，最优值至少这么高，不是 max 的证明。速度圆是 sum r 的上界，用了最大速度 1，比稳速 0.4 松。不是 evaluate_one。未改 entry.py。",
        "lookahead": LOOKAHEAD,
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "entry_score": float(np.mean(entry_scores)),
        "oracle_score": float(np.mean(oracle_scores)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "entry_return_sum": mean_return("entry"),
        "oracle_return_sum": mean_return("oracle"),
        "upper_return_sum": float(np.mean(upper_returns)),
        "upper_score": float(np.mean(upper_scores)),
        "thrust_upper_return_sum": float(np.mean(thrust_returns)),
        "thrust_upper_score": float(np.mean(thrust_scores)),
        "entry_coverage": float(np.mean([float(row["entry_coverage"]) for row in rows])),
        "oracle_coverage": float(np.mean([float(row["oracle_coverage"]) for row in rows])),
        "entry_collision": float(np.mean([float(row["entry_collision"]) for row in rows])),
        "oracle_collision": float(np.mean([float(row["oracle_collision"]) for row in rows])),
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
    _check_matches_env()
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
            if index % 20 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)
    rows.sort(key=lambda row: (str(row["group"]), int(row["seed"])))
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, seeds)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
