"""全局搜索上界：用官方环境真算回报，不作为正式提交策略。"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.suites import load_suite

_SPEC = get_protocol_spec()


def _decode(state: np.ndarray, n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从全局状态取出车和圈的位置、速度。"""
    capacity_agents = int(_SPEC.agent_capacity)
    capacity_targets = int(_SPEC.target_capacity)
    length = float(_SPEC.position_scale)
    speed_scale = float(_SPEC.velocity_scale)
    robots = state[: 5 * capacity_agents].reshape(capacity_agents, 5)
    target_offset = 6 * capacity_agents
    targets = state[target_offset : target_offset + 5 * capacity_targets].reshape(capacity_targets, 5)
    robot_pos = robots[:n, :2] * length
    robot_vel = robots[:n, 2:4] * speed_scale
    target_pos = targets[:m, :2] * length
    target_vel = targets[:m, 2:4] * speed_scale
    return robot_pos, robot_vel, target_pos, target_vel


def _aim(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    dt: float,
    damping: float,
) -> np.ndarray:
    """让下一拍速度指向目标，使再下一拍的位置靠近 goal。"""
    committed = position + velocity * dt
    desired_velocity = (goal - committed) / dt
    action = (desired_velocity - (1.0 - damping) * velocity) / dt
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _actions_for_assignment(
    state: np.ndarray,
    assignment: tuple[int, ...],
    n: int,
    m: int,
    dt: float,
    damping: float,
    lead_steps: float,
    cover_radius: float,
) -> dict[str, np.ndarray]:
    """每台车按分配去提前量上的圈；已经罩住就改跟圈的速度。"""
    robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
    actions: dict[str, np.ndarray] = {}
    for robot_index, target_index in enumerate(assignment):
        goal = target_pos[target_index] + target_vel[target_index] * (lead_steps * dt)
        committed = robot_pos[robot_index] + robot_vel[robot_index] * dt
        if float(np.linalg.norm(committed - target_pos[target_index])) <= cover_radius:
            desired_velocity = target_vel[target_index]
            action = (desired_velocity - (1.0 - damping) * robot_vel[robot_index]) / dt
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        else:
            action = _aim(robot_pos[robot_index], robot_vel[robot_index], goal, dt, damping)
        actions[f"agent_{robot_index}"] = action
    return actions


def _rollout(case, assignment: tuple[int, ...], lead_steps: float) -> tuple[float, float, float, float]:
    """在官方环境里跑完一局，返回回报、平均 J、平均覆盖、平均碰撞。"""
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=case.scenario_seed)
    dt = float(config.public.dt)
    damping = float(config.public.damping)
    cover_radius = float(config.public.target_radius)
    n = int(config.num_agents)
    m = int(config.num_targets)
    total = 0.0
    coverages: list[float] = []
    collisions: list[float] = []
    for _ in range(int(config.horizon)):
        actions = _actions_for_assignment(env.state(), assignment, n, m, dt, damping, lead_steps, cover_radius)
        _obs, rewards, _terms, _truncs, infos = env.step(actions)
        agent = next(iter(rewards))
        total += float(rewards[agent])
        metrics = infos[agent]["metrics"]
        coverages.append(float(metrics.coverage_rate))
        collisions.append(float(metrics.collision_rate))
    env.close()
    steps = int(config.horizon)
    return total, total / steps, float(np.mean(coverages)), float(np.mean(collisions))


def _rollout_dynamic(case, lead_steps: float) -> tuple[float, float, float, float]:
    """每一步在 6 种分配里选离提前点更近的那组，再用官方环境执行。"""
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=case.scenario_seed)
    dt = float(config.public.dt)
    damping = float(config.public.damping)
    cover_radius = float(config.public.target_radius)
    n = int(config.num_agents)
    m = int(config.num_targets)
    total = 0.0
    coverages: list[float] = []
    collisions: list[float] = []
    permutations = list(itertools.permutations(range(m)))
    for _ in range(int(config.horizon)):
        state = env.state()
        robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
        best_assignment = permutations[0]
        best_cost = float("inf")
        for assignment in permutations:
            cost = 0.0
            for robot_index, target_index in enumerate(assignment):
                goal = target_pos[target_index] + target_vel[target_index] * (lead_steps * dt)
                committed = robot_pos[robot_index] + robot_vel[robot_index] * dt
                cost += float(np.linalg.norm(committed - goal))
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
        actions = _actions_for_assignment(state, best_assignment, n, m, dt, damping, lead_steps, cover_radius)
        _obs, rewards, _terms, _truncs, infos = env.step(actions)
        agent = next(iter(rewards))
        total += float(rewards[agent])
        metrics = infos[agent]["metrics"]
        coverages.append(float(metrics.coverage_rate))
        collisions.append(float(metrics.collision_rate))
    env.close()
    steps = int(config.horizon)
    return total, total / steps, float(np.mean(coverages)), float(np.mean(collisions))


def main() -> None:
    suite = load_suite(Path("configs/public-suite-v1.yaml"))
    leads = (1.0, 2.0, 3.0, 4.0, 5.0)
    rows: list[tuple[str, str, int, float, float, float, float, str]] = []
    for group in suite.groups:
        for case in group.cases:
            permutations = list(itertools.permutations(range(case.task_config.num_targets)))
            best: tuple[float, float, float, float, str] | None = None
            for assignment in permutations:
                for lead in leads:
                    total, mean_j, coverage, collision = _rollout(case, assignment, lead)
                    label = f"assign={assignment} lead={lead}"
                    if best is None or total > best[0]:
                        best = (total, mean_j, coverage, collision, label)
            assert best is not None
            for repeat_index in range(group.policy_repeats):
                rows.append((group.group_id, case.case_id, repeat_index, *best))
                print(
                    f"{case.case_id} rep={repeat_index} R={best[0]:.3f} J={best[1]:.3f} "
                    f"cov={best[2]:.3f} col={best[3]:.3f} {best[4]}",
                    flush=True,
                )
    group_j: dict[str, list[float]] = {}
    for group_id, _case, _rep, _total, mean_j, _cov, _col, _label in rows:
        group_j.setdefault(group_id, []).append(mean_j)
    score = 1000.0 * float(np.mean([float(np.mean(values)) for values in group_j.values()]))
    print(f"fixed performance_score ~= {score:.2f}")
    dynamic_rows: list[tuple[str, float]] = []
    print("--- dynamic ---")
    for group in suite.groups:
        for case in group.cases:
            best: tuple[float, float, float, float, float] | None = None
            for lead in leads:
                total, mean_j, coverage, collision = _rollout_dynamic(case, lead)
                if best is None or total > best[0]:
                    best = (total, mean_j, coverage, collision, lead)
            assert best is not None
            for repeat_index in range(group.policy_repeats):
                dynamic_rows.append((group.group_id, best[1]))
                print(
                    f"{case.case_id} rep={repeat_index} R={best[0]:.3f} J={best[1]:.3f} "
                    f"cov={best[2]:.3f} col={best[3]:.3f} lead={best[4]}",
                    flush=True,
                )
    dynamic_groups: dict[str, list[float]] = {}
    for group_id, mean_j in dynamic_rows:
        dynamic_groups.setdefault(group_id, []).append(mean_j)
    dynamic_score = 1000.0 * float(np.mean([float(np.mean(values)) for values in dynamic_groups.values()]))
    print(f"dynamic performance_score ~= {dynamic_score:.2f}")


if __name__ == "__main__":
    main()
