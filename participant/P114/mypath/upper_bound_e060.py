"""上界阶梯。同一批主种子 20261003 的 300 个种子。

local：当前 entry.py。只读局部观测。
global_tti：每步用全局真实位置和速度，按常速度外推算截击步数，选总步数最少的一对一。
future_tti：截击步数改用真实未来轨迹，开车仍是这一拍的真实位置和速度。
action_beam：每车 7 个动作模板，未来 3 步束搜索，只执行第一拍。覆盖次数相同就选更靠近圈的。搜索模型没有车和车的力。
disk：从开局出发，每步位移不超过最大速度乘 dt，再做最大匹配。碰撞当 0。这是松弛上界。
square：同一时刻改用油门正方形。碰撞把车弹快时可以越过它。
any：速度圆里每个圈只要有一辆车够得到就算，不配一对一。

169.20 和 208.33 都不是这里的上界。不是 evaluate_one。不改 entry.py。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[3]
MYPATH_ROOT = Path(__file__).resolve().parent
ENTRY_ROOT = REPO_ROOT / "participant" / "P114"
for folder in (REPO_ROOT, MYPATH_ROOT, ENTRY_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.envs.motion import advance_targets
from coverage_bench.envs.types import ScenarioState
from coverage_bench.protocol import EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase
from entry import CoverFirstPolicy
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds


BEAM_DEPTH = 3
BEAM_WIDTH = 8
TEMPLATE_COUNT = 7
JOINT = np.array(list(product(range(TEMPLATE_COUNT), repeat=3)), dtype=np.int64)
PERMS = np.array(list(permutations(range(3))), dtype=np.int64)
LAWS = ("local", "global_tti", "future_tti", "action_beam", "disk", "square", "any")


def _capture(state: ScenarioState) -> dict[str, Any]:
    """记下圈的运动，算完未来轨迹后可以退回。"""
    return {
        "target_positions": state.target_positions.copy(),
        "target_velocities": state.target_velocities.copy(),
        "target_next_turn": state.target_next_turn.copy(),
        "step_index": int(state.step_index),
        "rng": copy.deepcopy(state.target_rng.bit_generator.state),
    }


def _restore(state: ScenarioState, saved: dict[str, Any]) -> None:
    state.target_positions[:] = saved["target_positions"]
    state.target_velocities[:] = saved["target_velocities"]
    state.target_next_turn[:] = saved["target_next_turn"]
    state.step_index = int(saved["step_index"])
    state.target_rng.bit_generator.state = copy.deepcopy(saved["rng"])


def _physics(config: Any) -> dict[str, float]:
    pub = config.public
    return {
        "dt": float(pub.dt),
        "damping": float(pub.damping),
        "accel": float(pub.drive_force) / float(pub.robot_mass) * float(pub.dt),
        "cap": float(pub.robot_max_speed),
        "limit": float(pub.map_half_extent) - float(pub.robot_radius),
        "radius": float(pub.target_radius),
        "collide": float(pub.robot_radius) * 2.0,
        "weight": float(config.collision_weight),
    }


def _pd(offset: np.ndarray, robot_vel: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
    """和当前 entry.py 同一条开车公式，这里喂的是真实速度。"""
    return np.clip(20.0 * offset + 4.0 * (target_vel - robot_vel), -1.0, 1.0)


def _integrate(pos: np.ndarray, vel: np.ndarray, action: np.ndarray, phys: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """单车、没有碰撞力。位置先走，再阻尼、加力、限速、碰边停下。"""
    pos = pos + vel * phys["dt"]
    vel = (1.0 - phys["damping"]) * vel + action * phys["accel"]
    speed = float(np.linalg.norm(vel))
    if speed > phys["cap"]:
        vel = vel * (phys["cap"] / speed)
    limit = phys["limit"]
    for axis in (0, 1):
        if pos[axis] > limit:
            pos[axis] = limit
            if vel[axis] > 0.0:
                vel[axis] = 0.0
        elif pos[axis] < -limit:
            pos[axis] = -limit
            if vel[axis] < 0.0:
                vel[axis] = 0.0
    return pos, vel


def _remaining_path(state: ScenarioState) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """从这一拍起，圈的真实位置。算完退回，不消耗随机源。"""
    saved = _capture(state)
    pres: list[np.ndarray] = []
    pre_vels: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    horizon = int(state.config.horizon)
    try:
        while state.step_index < horizon:
            pres.append(state.target_positions.copy())
            pre_vels.append(state.target_velocities.copy())
            advance_targets(state)
            posts.append(state.target_positions.copy())
            state.step_index += 1
    finally:
        _restore(state, saved)
    return pres, pre_vels, posts


def _time_constant(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    steps_left: int,
    phys: dict[str, float],
) -> int:
    """目标按当前速度直行。第一次进圈的步数，进不去记成还剩的步数加 1。"""
    if float(np.linalg.norm(target_pos - robot_pos)) <= phys["radius"]:
        return 0
    pos = np.array(robot_pos, dtype=np.float64)
    vel = np.array(robot_vel, dtype=np.float64)
    point = np.array(target_pos, dtype=np.float64)
    speed = np.array(target_vel, dtype=np.float64)
    for step in range(1, steps_left + 1):
        pos, vel = _integrate(pos, vel, _pd(point - pos, vel, speed), phys)
        point = point + speed * phys["dt"]
        if float(np.linalg.norm(point - pos)) <= phys["radius"]:
            return step
    return steps_left + 1


def _time_true(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    pres: list[np.ndarray],
    pre_vels: list[np.ndarray],
    posts: list[np.ndarray],
    target_index: int,
    phys: dict[str, float],
) -> int:
    """目标走记录下来的真实轨迹。开车仍是每拍朝当时的真实位置。"""
    if float(np.linalg.norm(pres[0][target_index] - robot_pos)) <= phys["radius"]:
        return 0
    pos = np.array(robot_pos, dtype=np.float64)
    vel = np.array(robot_vel, dtype=np.float64)
    for step, post in enumerate(posts):
        action = _pd(pres[step][target_index] - pos, vel, pre_vels[step][target_index])
        pos, vel = _integrate(pos, vel, action, phys)
        if float(np.linalg.norm(post[target_index] - pos)) <= phys["radius"]:
            return step + 1
    return len(posts) + 1


def _best_assignment(times: np.ndarray) -> tuple[int, ...]:
    best = tuple(range(times.shape[0]))
    best_cost = float("inf")
    for assignment in permutations(range(times.shape[0])):
        cost = sum(float(times[robot, assignment[robot]]) for robot in range(times.shape[0]))
        if cost < best_cost:
            best = assignment
            best_cost = cost
    return best


def _matching(adj: np.ndarray) -> int:
    if adj.size == 0 or int(np.sum(adj)) == 0:
        return 0
    matched = maximum_bipartite_matching(csr_matrix(adj.astype(np.int32)), perm_type="column")
    return int(np.sum(matched >= 0))


def _square_covers(origin: np.ndarray, velocity: np.ndarray, target: np.ndarray, steps: int, phys: dict[str, float]) -> bool:
    """每个轴单独打满，忽略碰撞。正方形盖住圈心减半径，就算这一拍够得到。"""
    alpha = 1.0 - phys["damping"]
    coast_scale = phys["dt"] * (1.0 - alpha**steps) / phys["damping"]
    force_scale = phys["dt"] * phys["accel"] / phys["damping"]
    center = origin + velocity * coast_scale
    width = force_scale * float(sum(1.0 - alpha**q for q in range(1, steps)))
    closest = np.clip(target, center - width, center + width)
    return float(np.linalg.norm(closest - target)) <= phys["radius"] + 1e-9


def _bounds(state: ScenarioState) -> dict[str, float]:
    """开局各自算每一拍。车可以在不同拍出现在互达不到的地方，所以偏高。"""
    saved = _capture(state)
    phys = _physics(state.config)
    horizon = int(state.config.horizon)
    origin = state.robot_positions.copy()
    origin_vel = state.robot_velocities.copy()
    speed0 = np.linalg.norm(origin_vel, axis=1)
    n = int(state.config.num_agents)
    m = int(state.config.num_targets)
    disk_cover: list[float] = []
    square_cover: list[float] = []
    any_cover: list[float] = []
    inflation: list[float] = []
    try:
        for step in range(1, horizon + 1):
            advance_targets(state)
            state.step_index += 1
            disk = np.zeros((n, m), dtype=np.int32)
            square = np.zeros((n, m), dtype=np.int32)
            reach = speed0 * phys["dt"] + (step - 1) * phys["cap"] * phys["dt"]
            targets = state.target_positions
            for robot in range(n):
                for target in range(m):
                    gap = float(np.linalg.norm(targets[target] - origin[robot]))
                    if gap <= float(reach[robot]) + phys["radius"]:
                        disk[robot, target] = 1
                    if _square_covers(origin[robot], origin_vel[robot], targets[target], step, phys):
                        square[robot, target] = 1
            disk_n = _matching(disk)
            square_n = _matching(square)
            any_n = int(np.sum(np.any(disk, axis=0)))
            disk_cover.append(disk_n / m)
            square_cover.append(square_n / m)
            any_cover.append(any_n / m)
            inflation.append(float(any_n - disk_n))
    finally:
        _restore(state, saved)
    return {
        "disk_j": float(np.mean(disk_cover)),
        "square_j": float(np.mean(square_cover)),
        "any_j": float(np.mean(any_cover)),
        "disk_inflation": float(np.mean(inflation)),
        "disk_coverage": float(np.mean(disk_cover)),
        "square_coverage": float(np.mean(square_cover)),
        "any_coverage": float(np.mean(any_cover)),
    }


def _choose_actions(state: ScenarioState, mode: str, phys: dict[str, float]) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    robots = state.robot_positions
    robot_vel = state.robot_velocities
    targets = state.target_positions
    target_vel = state.target_velocities
    n = len(robots)
    steps_left = int(state.config.horizon) - int(state.step_index)
    times = np.zeros((n, n), dtype=np.float64)
    if mode == "global":
        for robot in range(n):
            for target in range(n):
                times[robot, target] = _time_constant(
                    robots[robot], robot_vel[robot], targets[target], target_vel[target], steps_left, phys
                )
    else:
        pres, pre_vels, posts = _remaining_path(state)
        for robot in range(n):
            for target in range(n):
                times[robot, target] = _time_true(
                    robots[robot], robot_vel[robot], pres, pre_vels, posts, target, phys
                )
    chosen = _best_assignment(times)
    actions = {
        f"agent_{robot}": _pd(
            targets[chosen[robot]] - robots[robot],
            robot_vel[robot],
            target_vel[chosen[robot]],
        ).astype(np.float32)
        for robot in range(n)
    }
    return actions, chosen


def _rollout_assignment(case: ScenarioCase, mode: str) -> dict[str, float]:
    env = make_training_env(case.task_config)
    env.reset(seed=int(case.scenario_seed))
    assert env._scenario_state is not None
    phys = _physics(case.task_config)
    horizon = int(case.task_config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    switches = 0
    previous: tuple[int, ...] | None = None
    try:
        for _step in range(horizon):
            actions, chosen = _choose_actions(env._scenario_state, mode, phys)
            if previous is not None:
                switches += sum(int(chosen[i] != previous[i]) for i in range(len(chosen)))
            previous = chosen
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


def _step_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    actions: np.ndarray,
    phys: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    new_pos = pos + vel * phys["dt"]
    new_vel = (1.0 - phys["damping"]) * vel + actions * phys["accel"]
    speed = np.linalg.norm(new_vel, axis=2, keepdims=True)
    fast = speed[..., 0] > phys["cap"]
    capped = new_vel * (phys["cap"] / np.maximum(speed, 1e-12))
    new_vel = np.where(fast[..., None], capped, new_vel)
    limit = phys["limit"]
    hit_hi = new_pos > limit
    hit_lo = new_pos < -limit
    new_pos = np.clip(new_pos, -limit, limit)
    new_vel = np.where(hit_hi & (new_vel > 0.0), 0.0, new_vel)
    new_vel = np.where(hit_lo & (new_vel < 0.0), 0.0, new_vel)
    return new_pos, new_vel


def _reward_batch(pos: np.ndarray, targets: np.ndarray, phys: dict[str, float]) -> np.ndarray:
    delta = pos[:, :, None, :] - targets[None, None, :, :]
    dist = np.linalg.norm(delta, axis=3)
    covered = dist <= phys["radius"]
    counts = [covered[:, np.arange(3), perm].sum(axis=1) for perm in PERMS]
    matched = np.maximum.reduce(counts).astype(np.float64)
    distance_01 = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    distance_02 = np.linalg.norm(pos[:, 0] - pos[:, 2], axis=1)
    distance_12 = np.linalg.norm(pos[:, 1] - pos[:, 2], axis=1)
    hit_01 = distance_01 < phys["collide"]
    hit_02 = distance_02 < phys["collide"]
    hit_12 = distance_12 < phys["collide"]
    involved = (
        (hit_01 | hit_02).astype(np.float64)
        + (hit_01 | hit_12).astype(np.float64)
        + (hit_02 | hit_12).astype(np.float64)
    )
    # 覆盖次数是整数。一样近的时候，用离最近圈的距离把平局打开，避免随便挑一个满力。
    near = dist.min(axis=2).mean(axis=1)
    return matched / 3.0 - phys["weight"] * (involved / 3.0) - 0.01 * near


def _templates(
    pos: np.ndarray,
    vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    lead: np.ndarray,
) -> np.ndarray:
    """7 个模板：朝 3 个圈的当前开车、停下、反向、朝最近圈打满两轴、朝两拍后的真实位置。"""
    out = np.zeros((3, TEMPLATE_COUNT, 2), dtype=np.float64)
    for robot in range(3):
        for target in range(3):
            out[robot, target] = _pd(target_pos[target] - pos[robot], vel[robot], target_vel[target])
        out[robot, 4] = np.clip(-10.0 * vel[robot], -1.0, 1.0)
        nearest = int(np.argmin(np.linalg.norm(target_pos - pos[robot], axis=1)))
        out[robot, 5] = np.sign(target_pos[nearest] - pos[robot])
        out[robot, 6] = np.clip(10.0 * (lead[nearest] - pos[robot]), -1.0, 1.0)
    return out


def _beam_action(state: ScenarioState, phys: dict[str, float]) -> np.ndarray:
    pres, pre_vels, posts = _remaining_path(state)
    depth_limit = min(BEAM_DEPTH, len(posts))
    parent_pos = state.robot_positions.reshape(1, 3, 2).copy()
    parent_vel = state.robot_velocities.reshape(1, 3, 2).copy()
    parent_score = np.zeros(1, dtype=np.float64)
    parent_first = np.zeros((1, 3, 2), dtype=np.float64)
    for depth in range(depth_limit):
        blocks: list[np.ndarray] = []
        lead = posts[min(depth + 1, depth_limit - 1)]
        for index in range(parent_pos.shape[0]):
            templates = _templates(parent_pos[index], parent_vel[index], pres[depth], pre_vels[depth], lead)
            blocks.append(templates[np.arange(3)[None, :], JOINT, :])
        actions = np.concatenate(blocks, axis=0)
        flat_pos = np.repeat(parent_pos, JOINT.shape[0], axis=0)
        flat_vel = np.repeat(parent_vel, JOINT.shape[0], axis=0)
        new_pos, new_vel = _step_batch(flat_pos, flat_vel, actions, phys)
        total = np.repeat(parent_score, JOINT.shape[0]) + _reward_batch(new_pos, posts[depth], phys)
        first = actions if depth == 0 else np.repeat(parent_first, JOINT.shape[0], axis=0)
        width = min(BEAM_WIDTH, total.shape[0])
        top = np.argpartition(total, -width)[-width:]
        order = top[np.argsort(total[top])[::-1]]
        parent_pos = new_pos[order]
        parent_vel = new_vel[order]
        parent_score = total[order]
        parent_first = first[order]
    return parent_first[0]


def _rollout_beam(case: ScenarioCase) -> dict[str, float]:
    env = make_training_env(case.task_config)
    env.reset(seed=int(case.scenario_seed))
    assert env._scenario_state is not None
    phys = _physics(case.task_config)
    horizon = int(case.task_config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for _step in range(horizon):
            action = _beam_action(env._scenario_state, phys)
            actions = {f"agent_{robot}": action[robot].astype(np.float32) for robot in range(3)}
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
        "switches": 0.0,
    }


def _rollout_local(case: ScenarioCase) -> dict[str, float]:
    env = make_training_env(case.task_config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    horizon = int(case.task_config.horizon)
    task = _public_task_params(case)
    policies = [CoverFirstPolicy() for _robot in env.agents]
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=int(case.task_config.num_agents),
                num_targets=int(case.task_config.num_targets),
                horizon=horizon,
                task=task,
                policy_seed=0,
            )
        )
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for _step in range(horizon):
            actions = {
                agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                for index, agent_id in enumerate(env.agents)
            }
            observations, rewards, _terms, _truncs, infos = env.step(actions)
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
        "switches": 0.0,
    }


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    env = make_training_env(case.task_config)
    env.reset(seed=int(case.scenario_seed))
    assert env._scenario_state is not None
    bounds = _bounds(env._scenario_state)
    env.close()
    local = _rollout_local(case)
    global_tti = _rollout_assignment(case, "global")
    future_tti = _rollout_assignment(case, "future")
    beam = _rollout_beam(case)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
        "disk_inflation": bounds["disk_inflation"],
    }
    results = {
        "local": local,
        "global_tti": global_tti,
        "future_tti": future_tti,
        "action_beam": beam,
        "disk": {
            "j": bounds["disk_j"],
            "coverage": bounds["disk_coverage"],
            "collision": 0.0,
            "switches": 0.0,
        },
        "square": {
            "j": bounds["square_j"],
            "coverage": bounds["square_coverage"],
            "collision": 0.0,
            "switches": 0.0,
        },
        "any": {
            "j": bounds["any_j"],
            "coverage": bounds["any_coverage"],
            "collision": 0.0,
            "switches": 0.0,
        },
    }
    for name, result in results.items():
        for key in ("j", "coverage", "collision", "switches"):
            row[f"{name}_{key}"] = result[key]
    return row


def _check_target_path() -> None:
    """只推圈，和零力 env.step 的圈位置必须一样。"""
    case = ScenarioCase("check", "basic", _layout_config("uniform"), 1001)
    env = make_training_env(case.task_config)
    env.reset(seed=1001)
    assert env._scenario_state is not None
    state = env._scenario_state
    saved = _capture(state)
    posts: list[np.ndarray] = []
    horizon = int(case.task_config.horizon)
    for _step in range(horizon):
        advance_targets(state)
        posts.append(state.target_positions.copy())
        state.step_index += 1
    _restore(state, saved)
    zeros = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
    for step in range(horizon):
        env.step(zeros)
        if not np.allclose(env._scenario_state.target_positions, posts[step]):
            raise RuntimeError(f"目标轨迹和 env.step 不一致，第 {step + 1} 步")
    env.close()


def _paired(rows: list[dict[str, Any]], seeds: list[int], law: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{law}_j"]) + float(lookup[(seed, "cooperation")][f"{law}_j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    local = _paired(rows, seeds, "local")
    laws: dict[str, Any] = {}
    for law in LAWS:
        scores = _paired(rows, seeds, law)
        gaps = [new - old for old, new in zip(local, scores)]
        wins = sum(gap > 1e-9 for gap in gaps)
        losses = sum(gap < -1e-9 for gap in gaps)
        laws[law] = {
            "score": float(np.mean(scores)),
            "gap_vs_local": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{law}_coverage"]) for row in rows])),
            "collision": float(np.mean([float(row[f"{law}_collision"]) for row in rows])),
            "switches": float(np.mean([float(row[f"{law}_switches"]) for row in rows])),
        }
    over_disk = {
        law: int(sum(float(row[f"{law}_coverage"]) > float(row["disk_coverage"]) + 1e-9 for row in rows))
        for law in ("local", "global_tti", "future_tti", "action_beam")
    }
    over_square = {
        law: int(sum(float(row[f"{law}_coverage"]) > float(row["square_coverage"]) + 1e-9 for row in rows))
        for law in ("local", "global_tti", "future_tti", "action_beam")
    }
    return {
        "note": "disk 是松弛上界：知道未来圈的位置，每步单独配，碰撞当 0，位移用最大速度。不是 evaluate_one。未改 entry.py。",
        "not_upper": "local、global_tti、future_tti、action_beam 都是做出来的分数，不是上界。square 忽略碰撞助推。any 不配一对一。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
        "disk_inflation_targets": float(np.mean([float(row["disk_inflation"]) for row in rows])),
        "episodes_over_disk": over_disk,
        "episodes_over_square": over_square,
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
    _check_target_path()
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
