"""P114 全局 Oracle：仅用于公开套件的上界/差距分析，不作为正式提交策略。

特点：
- 直接读取训练环境 env.state() 的全局状态，因此不符合正式执行时的局部观测约束；
- 3v3 时直接枚举 3! 个 assignment，不依赖 Hungarian 库；
- 提供 greedy / predictive 两种全局控制器；
- 输出逐回合 return、mean J、coverage、collision，以及按官方锚点口径计算的 performance score。

运行示例：
    python participant/P114/oracle_eval.py --mode all
    python participant/P114/oracle_eval.py --mode predictive --trace
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.suites import load_suite


@dataclass
class DecodedState:
    robot_pos: np.ndarray
    robot_vel: np.ndarray
    robot_radius: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray
    target_radius: np.ndarray
    steps_remaining: int


@dataclass
class EpisodeResult:
    group_id: str
    case_id: str
    repeat_index: int
    return_sum: float
    mean_j: float
    mean_coverage: float
    mean_collision: float
    full_coverage_fraction: float


def _decode_state(state: np.ndarray, config) -> DecodedState:
    spec = get_protocol_spec()
    A = int(spec.agent_capacity)
    B = int(spec.target_capacity)
    L = float(spec.position_scale)
    V = float(spec.velocity_scale)
    n = int(config.num_agents)
    m = int(config.num_targets)

    robot_table = state[: 5 * A].reshape(A, 5)
    robot_flags = state[5 * A : 6 * A]
    target_offset = 6 * A
    target_table = state[target_offset : target_offset + 5 * B].reshape(B, 5)
    target_flags = state[target_offset + 5 * B : target_offset + 6 * B]

    if int(np.sum(robot_flags > 0.5)) != n or int(np.sum(target_flags > 0.5)) != m:
        raise RuntimeError("global state entity flags do not match task config")

    robot_pos = robot_table[:n, :2].astype(np.float64) * L
    robot_vel = robot_table[:n, 2:4].astype(np.float64) * V
    robot_radius = robot_table[:n, 4].astype(np.float64) * L
    target_pos = target_table[:m, :2].astype(np.float64) * L
    target_vel = target_table[:m, 2:4].astype(np.float64) * V
    target_radius = target_table[:m, 4].astype(np.float64) * L

    frac_remaining = float(state[-1])
    steps_remaining = int(round(frac_remaining * int(config.horizon)))
    return DecodedState(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_radius=robot_radius,
        target_pos=target_pos,
        target_vel=target_vel,
        target_radius=target_radius,
        steps_remaining=max(0, steps_remaining),
    )


def _reflect_axis(pos: float, vel: float, dt: float, low: float, high: float) -> Tuple[float, float]:
    """与官方目标反射边界一致的单轴预测；不预测未来随机转向事件。"""
    if abs(vel) < 1e-12:
        return pos, 0.0
    remaining = float(dt)
    p = float(pos)
    v = float(vel)
    while remaining > 1e-12:
        wall = high if v > 0.0 else low
        time_to_wall = (wall - p) / v
        if time_to_wall > remaining:
            p += v * remaining
            remaining = 0.0
        else:
            p = wall
            v = -v
            remaining -= max(0.0, time_to_wall)
    return p, v


def _predict_target(pos: np.ndarray, vel: np.ndarray, steps: int, config) -> Tuple[np.ndarray, np.ndarray]:
    p = np.array(pos, dtype=np.float64, copy=True)
    v = np.array(vel, dtype=np.float64, copy=True)
    dt = float(config.public.dt)
    bound = float(config.public.map_half_extent - config.public.target_radius)
    for _ in range(max(0, int(steps))):
        p[0], v[0] = _reflect_axis(p[0], v[0], dt, -bound, bound)
        p[1], v[1] = _reflect_axis(p[1], v[1], dt, -bound, bound)
    return p, v


def _clip_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    if speed > max_speed and speed > 0.0:
        return v * (max_speed / speed)
    return v


def _inverse_velocity_action(current_vel: np.ndarray, desired_vel: np.ndarray, config) -> np.ndarray:
    """反解下一时刻速度所需动作；忽略接触力，随后 clip 到合法动作范围。"""
    pub = config.public
    retained = (1.0 - float(pub.damping)) * current_vel
    scale = float(pub.robot_mass) / (float(pub.drive_force) * float(pub.dt))
    action = (desired_vel - retained) * scale
    return np.clip(action, -1.0, 1.0)


def _control_to_target(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    cover_radius: float,
    config,
) -> np.ndarray:
    """短回合追踪器：远处快速闭合，进入覆盖圈后尽量匹配目标速度。"""
    pub = config.public
    delta = target_pos - robot_pos
    dist = float(np.linalg.norm(delta))
    if dist > 1e-9:
        direction = delta / dist
    else:
        direction = np.zeros(2, dtype=np.float64)

    # 只需要进入覆盖圈，不需要追到目标中心。
    outside = max(0.0, dist - float(cover_radius))
    if dist <= 0.75 * float(cover_radius):
        desired = np.array(target_vel, dtype=np.float64)
    else:
        # horizon 很短，远处应尽快闭合；越靠近圈边越提前制动。
        closing_speed = min(float(pub.robot_max_speed), 4.0 * outside + 0.10)
        desired = np.array(target_vel, dtype=np.float64) + direction * closing_speed
    desired = _clip_speed(desired, float(pub.robot_max_speed))
    return _inverse_velocity_action(robot_vel, desired, config)


def _simulate_pair_hit_time(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    cover_radius: float,
    remaining_steps: int,
    config,
) -> Tuple[int | None, float]:
    """忽略机器人间接触，估计当前 pair 在剩余步数内最早何时能进覆盖圈。"""
    rp = np.array(robot_pos, dtype=np.float64, copy=True)
    rv = np.array(robot_vel, dtype=np.float64, copy=True)
    tp = np.array(target_pos, dtype=np.float64, copy=True)
    tv = np.array(target_vel, dtype=np.float64, copy=True)
    pub = config.public
    dt = float(pub.dt)
    final_gap = max(0.0, float(np.linalg.norm(tp - rp)) - cover_radius)

    for k in range(1, max(1, remaining_steps) + 1):
        act = _control_to_target(rp, rv, tp, tv, cover_radius, config)

        # 与官方无接触情况下的顺序一致：先用旧速度更新位置，再更新速度。
        rp = rp + rv * dt
        rv = (1.0 - float(pub.damping)) * rv + (
            float(pub.drive_force) * act / float(pub.robot_mass)
        ) * dt
        rv = _clip_speed(rv, float(pub.robot_max_speed))

        tp, tv = _predict_target(tp, tv, 1, config)
        final_gap = max(0.0, float(np.linalg.norm(tp - rp)) - cover_radius)
        if final_gap <= 1e-9:
            return k, 0.0
    return None, final_gap


def _assignment_costs(decoded: DecodedState, config, mode: str) -> np.ndarray:
    n = len(decoded.robot_pos)
    m = len(decoded.target_pos)
    costs = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            if mode == "greedy":
                d = float(np.linalg.norm(decoded.target_pos[j] - decoded.robot_pos[i]))
                costs[i, j] = max(0.0, d - float(decoded.target_radius[j]))
            else:
                hit, residual = _simulate_pair_hit_time(
                    decoded.robot_pos[i],
                    decoded.robot_vel[i],
                    decoded.target_pos[j],
                    decoded.target_vel[j],
                    float(decoded.target_radius[j]),
                    decoded.steps_remaining,
                    config,
                )
                # 可达 pair 优先按最早进入圈的步数；不可达 pair 给大罚项，
                # residual 用来区分“都到不了时谁最接近”。
                if hit is None:
                    costs[i, j] = 100.0 + 10.0 * residual
                else:
                    costs[i, j] = float(hit) + 0.05 * residual
    return costs


def _best_assignment(
    costs: np.ndarray,
    previous: Sequence[int] | None,
    switch_penalty: float,
) -> Tuple[int, ...]:
    n, m = costs.shape
    if n != m:
        raise ValueError("current oracle expects equal numbers of robots and targets")

    best_perm: Tuple[int, ...] | None = None
    best_cost = math.inf
    for perm in itertools.permutations(range(m), n):
        c = sum(float(costs[i, perm[i]]) for i in range(n))
        if previous is not None:
            c += switch_penalty * sum(int(perm[i] != previous[i]) for i in range(n))
        if c < best_cost:
            best_cost = c
            best_perm = tuple(int(x) for x in perm)
    assert best_perm is not None
    return best_perm


def _avoidance_correction(i: int, decoded: DecodedState) -> np.ndarray:
    """很轻的全局避碰修正，避免为了不碰撞绕太远。"""
    correction = np.zeros(2, dtype=np.float64)
    pi = decoded.robot_pos[i]
    ri = float(decoded.robot_radius[i])
    for k in range(len(decoded.robot_pos)):
        if k == i:
            continue
        delta = pi - decoded.robot_pos[k]
        d = float(np.linalg.norm(delta))
        safe = ri + float(decoded.robot_radius[k]) + 0.06
        if 1e-9 < d < safe:
            correction += (delta / d) * min(0.35, 3.0 * (safe - d))
    return correction


def _actions_for_assignment(
    decoded: DecodedState,
    assignment: Sequence[int],
    config,
    mode: str,
) -> Dict[str, np.ndarray]:
    actions: Dict[str, np.ndarray] = {}
    for i, j in enumerate(assignment):
        if mode == "predictive":
            hit, _ = _simulate_pair_hit_time(
                decoded.robot_pos[i],
                decoded.robot_vel[i],
                decoded.target_pos[j],
                decoded.target_vel[j],
                float(decoded.target_radius[j]),
                decoded.steps_remaining,
                config,
            )
            lead_steps = 1 if hit is None else max(1, min(int(hit), decoded.steps_remaining))
            aim_pos, aim_vel = _predict_target(
                decoded.target_pos[j], decoded.target_vel[j], lead_steps, config
            )
        else:
            aim_pos = decoded.target_pos[j]
            aim_vel = decoded.target_vel[j]

        action = _control_to_target(
            decoded.robot_pos[i],
            decoded.robot_vel[i],
            aim_pos,
            aim_vel,
            float(decoded.target_radius[j]),
            config,
        )
        action = np.clip(action + _avoidance_correction(i, decoded), -1.0, 1.0)
        actions[f"agent_{i}"] = action.astype(np.float32)
    return actions


def run_episode(case, repeat_index: int, mode: str, trace: bool) -> EpisodeResult:
    env = make_training_env(case.task_config)
    observations, _ = env.reset(seed=int(case.scenario_seed))
    del observations

    previous_assignment: Tuple[int, ...] | None = None
    rewards: List[float] = []
    coverages: List[float] = []
    collisions: List[float] = []
    fulls: List[float] = []

    try:
        for step in range(int(case.task_config.horizon)):
            decoded = _decode_state(env.state(), case.task_config)
            costs = _assignment_costs(decoded, case.task_config, mode)
            assignment = _best_assignment(
                costs,
                previous_assignment,
                switch_penalty=0.20 if mode == "predictive" else 0.05,
            )
            actions = _actions_for_assignment(decoded, assignment, case.task_config, mode)
            _, reward_dict, _, trunc_dict, infos = env.step(actions)

            # 所有 agent 收到相同团队奖励，只记一次。
            team_reward = float(next(iter(reward_dict.values())))
            info0 = infos["agent_0"]
            metrics = info0["metrics"]
            rewards.append(team_reward)
            coverages.append(float(metrics.coverage_rate))
            collisions.append(float(metrics.collision_rate))
            fulls.append(float(metrics.full_coverage))

            if trace:
                print(
                    f"{case.case_id} rep={repeat_index} step={step + 1:02d} "
                    f"assign={assignment} reward={team_reward:.3f} "
                    f"cov={metrics.coverage_rate:.3f} col={metrics.collision_rate:.3f}"
                )

            previous_assignment = assignment
            if trunc_dict and all(bool(x) for x in trunc_dict.values()):
                break
    finally:
        env.close()

    return_sum = float(np.sum(rewards))
    horizon = int(case.task_config.horizon)
    return EpisodeResult(
        group_id=case.group_id,
        case_id=case.case_id,
        repeat_index=int(repeat_index),
        return_sum=return_sum,
        mean_j=return_sum / horizon,
        mean_coverage=float(np.mean(coverages)) if coverages else 0.0,
        mean_collision=float(np.mean(collisions)) if collisions else 0.0,
        full_coverage_fraction=float(np.mean(fulls)) if fulls else 0.0,
    )


def evaluate_mode(suite_path: Path, mode: str, trace: bool) -> Tuple[List[EpisodeResult], float]:
    suite = load_suite(suite_path)
    results: List[EpisodeResult] = []

    for group in suite.groups:
        for case in group.cases:
            for repeat_index in range(group.policy_repeats):
                results.append(run_episode(case, repeat_index, mode, trace))

    group_mean_j: Dict[str, float] = {}
    for group in suite.groups:
        vals = [r.mean_j for r in results if r.group_id == group.group_id]
        group_mean_j[group.group_id] = float(np.mean(vals))

    # 当前 public suite 使用等权组、random_anchor=0、learning_anchor=1。
    performance_score = 1000.0 * float(np.mean(list(group_mean_j.values())))
    return results, performance_score


def _print_summary(mode: str, results: Iterable[EpisodeResult], score: float) -> None:
    rows = list(results)
    print(f"\n=== Oracle mode: {mode} ===")
    print("case       rep   return      J     coverage collision full_cov")
    for r in rows:
        print(
            f"{r.case_id:<10} {r.repeat_index:>3d} "
            f"{r.return_sum:>7.3f} {r.mean_j:>7.3f} "
            f"{r.mean_coverage:>8.3f} {r.mean_collision:>9.3f} "
            f"{r.full_coverage_fraction:>8.3f}"
        )
    print(f"performance_score ~= {score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P114 global-state oracle evaluator")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/public-suite-v1.yaml"),
        help="suite yaml path",
    )
    parser.add_argument(
        "--mode",
        choices=("greedy", "predictive", "all"),
        default="all",
        help="oracle variant",
    )
    parser.add_argument("--trace", action="store_true", help="print every step")
    args = parser.parse_args()

    modes = ("greedy", "predictive") if args.mode == "all" else (args.mode,)
    for mode in modes:
        results, score = evaluate_mode(args.suite, mode, args.trace)
        _print_summary(mode, results, score)


if __name__ == "__main__":
    main()
