"""把 E039 四个大桶拆细。只诊断 E028，不改 entry.py，不发明新策略。

同一批主种子 20261003、300 个种子、均匀 + 交叉。
板块：
1. STATIC_UNREACHABLE → HARD / TARGET_ASSISTED / MARGINAL / TARGET_ESCAPED
2. near → ONE_STEP_LATE / HORIZON_END / TARGET_ESCAPE / WRONG_HEADING /
          INSUFFICIENT_PROGRESS / ACTION_SATURATION
3. overlap → GEOMETRIC / AVOIDABLE_ALLOCATION / MULTI_TARGET_OVERLAP / RESOURCE_SHORTAGE
4. travel → LATE_START / TARGET_SWITCH / TARGET_MOVED_AWAY / CONTROL_LIMITED /
            BAD_INITIAL_ASSIGNMENT / OBSERVATION_DELAY
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[3]
PARTICIPANT_ROOT = Path(__file__).resolve().parents[1]
MYPATH_ROOT = Path(__file__).resolve().parent
for folder in (REPO_ROOT, PARTICIPANT_ROOT, MYPATH_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase
from entry import CoverFirstPolicy
from loss_attribution import NEAR_MARGIN, REACH_SLACK, _start_reachable
from oracle_search import _decode
from reachability_debounce_oracle import (
    MASTER_SEED,
    _layout_config,
    _optimistic_closure,
    _random_seeds,
)


SHARE = 1.0 / 3.0
MARGINAL_HI = 0.05
SENSE = 0.6

PANEL1 = (
    "HARD_UNREACHABLE",
    "TARGET_ASSISTED_REACHABLE",
    "MARGINAL_UNREACHABLE",
    "TARGET_ESCAPED",
)
PANEL2 = (
    "ONE_STEP_LATE",
    "HORIZON_END",
    "TARGET_ESCAPE",
    "WRONG_HEADING",
    "INSUFFICIENT_PROGRESS",
    "ACTION_SATURATION",
)
PANEL3 = (
    "GEOMETRIC_CONFLICT",
    "AVOIDABLE_ALLOCATION_CONFLICT",
    "MULTI_TARGET_OVERLAP",
    "RESOURCE_SHORTAGE",
)
PANEL4 = (
    "LATE_START",
    "TARGET_SWITCH",
    "TARGET_MOVED_AWAY",
    "CONTROL_LIMITED",
    "BAD_INITIAL_ASSIGNMENT",
    "OBSERVATION_DELAY",
)
COARSE = ("unreachable", "near", "overlap", "travel", "crowd", "collision")


def _dmax(relative: np.ndarray, velocity: np.ndarray, steps: int, config: Any) -> float:
    return float(_optimistic_closure(relative, velocity, steps, config))


def _matching(robot_pos: np.ndarray, target_pos: np.ndarray, radius: float) -> np.ndarray:
    n, m = len(robot_pos), len(target_pos)
    adj = np.zeros((n, m), dtype=np.int32)
    dist = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
    adj[dist <= radius] = 1
    return np.asarray(maximum_bipartite_matching(csr_matrix(adj), perm_type="column"), dtype=np.int64)


def _choose_local(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    radius: float,
    sense: float,
) -> np.ndarray:
    """用全局位置近似 E022/E028 的局部认圈：空圈优先，否则最近可见圈。"""
    n, m = len(robot_pos), len(target_pos)
    assign = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        empty: list[tuple[float, int]] = []
        occupied: list[tuple[float, int]] = []
        for j in range(m):
            delta = target_pos[j] - robot_pos[i]
            dist = float(np.linalg.norm(delta))
            if dist > sense:
                continue
            taken = False
            for k in range(n):
                if k == i:
                    continue
                if float(np.linalg.norm(robot_pos[k] - target_pos[j])) <= radius:
                    if float(np.linalg.norm(robot_pos[k] - robot_pos[i])) <= sense:
                        taken = True
                        break
            (occupied if taken else empty).append((dist, j))
        bucket = empty or occupied
        if bucket:
            bucket.sort(key=lambda item: item[0])
            assign[i] = bucket[0][1]
    return assign


def _dynamic_best_gap(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    future_targets: list[np.ndarray],
    radius: float,
    config: Any,
) -> tuple[float, bool]:
    """用真实未来圈位，乐观估计任意车是否还能进圈。返回最小净 gap，以及是否可达。"""
    if not future_targets:
        return 1e9, False
    best = 1e9
    reachable = False
    for ahead, target_pos in enumerate(future_targets, start=1):
        for i in range(len(robot_pos)):
            delta = target_pos - robot_pos[i]
            gap = max(0.0, float(np.linalg.norm(delta)) - radius)
            closure = _dmax(delta, robot_vel[i], ahead, config)
            rem = gap - closure
            best = min(best, rem)
            if closure + REACH_SLACK >= gap:
                reachable = True
    return best, reachable


def _classify_unreachable(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_index: int,
    remaining: int,
    future_targets: list[np.ndarray],
    config: Any,
    radius: float,
) -> tuple[str, dict[str, float]]:
    deltas = target_pos[target_index] - robot_pos
    dists = np.linalg.norm(deltas, axis=1)
    nearest = int(np.argmin(dists))
    gap = max(0.0, float(dists[nearest]) - radius)
    dmax = _dmax(deltas[nearest], robot_vel[nearest], max(remaining, 0), config)
    margin = dmax - gap
    best_gap, dyn_ok = _dynamic_best_gap(
        robot_pos, robot_vel, [row[target_index] for row in future_targets], radius, config
    )
    toward = target_pos[target_index] - robot_pos[nearest]
    toward_n = float(np.linalg.norm(toward))
    radial_out = 0.0
    if toward_n > 1e-9:
        radial_out = float(np.dot(target_vel[target_index], toward / toward_n))
    if dyn_ok:
        cause = "TARGET_ASSISTED_REACHABLE"
    elif 0.0 < best_gap <= MARGINAL_HI:
        cause = "MARGINAL_UNREACHABLE"
    else:
        # 真实未来轨迹上，目标是否持续把最近车甩远
        escaped = False
        if future_targets:
            start_d = float(np.linalg.norm(target_pos[target_index] - robot_pos[nearest]))
            end_d = float(np.linalg.norm(future_targets[-1][target_index] - robot_pos[nearest]))
            if end_d > start_d + 0.05 or radial_out > 0.05:
                escaped = True
        if escaped and margin > -0.25:
            cause = "TARGET_ESCAPED"
        else:
            cause = "HARD_UNREACHABLE"
    return cause, {
        "gap": gap,
        "margin": margin,
        "dmax": dmax,
        "best_dynamic_gap": float(best_gap),
        "radial_out": radial_out,
        "remaining_steps": float(remaining),
        "nearest_robot": float(nearest),
    }


def _classify_near(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_index: int,
    action: np.ndarray,
    remaining: int,
    next_inside: bool,
    radius: float,
) -> tuple[str, dict[str, float]]:
    deltas = target_pos[target_index] - robot_pos
    dists = np.linalg.norm(deltas, axis=1)
    nearest = int(np.argmin(dists))
    gap = float(dists[nearest]) - radius
    direction = deltas[nearest]
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-9 else np.zeros(2)
    radial_v = float(np.dot(robot_vel[nearest], unit))
    act = np.asarray(action[nearest], dtype=np.float64)
    act_radial = float(np.dot(act, unit))
    target_radial = float(np.dot(target_vel[target_index], unit))
    sat = float(np.max(np.abs(act))) >= 0.99
    meta = {
        "gap": gap,
        "remaining_steps": float(remaining),
        "radial_speed": radial_v,
        "action_radial": act_radial,
        "target_radial": target_radial,
        "saturated": float(sat),
        "nearest_robot": float(nearest),
    }
    if next_inside:
        return "ONE_STEP_LATE", meta
    if remaining <= 0:
        return "HORIZON_END", meta
    if target_radial > 0.02:
        return "TARGET_ESCAPE", meta
    if radial_v < -0.02 or act_radial < -0.2:
        return "WRONG_HEADING", meta
    if sat and act_radial > 0.5:
        return "ACTION_SATURATION", meta
    return "INSUFFICIENT_PROGRESS", meta


def _classify_overlap(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_index: int,
    assign: np.ndarray,
    matching: np.ndarray,
    remaining: int,
    config: Any,
    radius: float,
) -> tuple[str, dict[str, float]]:
    n, m = len(robot_pos), len(target_pos)
    inside = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2) <= radius
    coverers = [i for i in range(n) if bool(inside[i, target_index])]
    empty_targets = [j for j in range(m) if not bool(np.any(inside[:, j]))]
    matched_size = int(np.sum(matching >= 0))
    # 其他车是否乐观可达这个未计入的圈（从圈外车算）
    outsiders = [i for i in range(n) if i not in coverers]
    reachable_by_other = False
    for i in outsiders:
        delta = target_pos[target_index] - robot_pos[i]
        gap = max(0.0, float(np.linalg.norm(delta)) - radius)
        if _dmax(delta, robot_vel[i], max(remaining, 0), config) + REACH_SLACK >= gap:
            reachable_by_other = True
            break
    # 空圈是否有人能去
    empty_reachable = False
    for j in empty_targets:
        for i in coverers:
            delta = target_pos[j] - robot_pos[i]
            gap = max(0.0, float(np.linalg.norm(delta)) - radius)
            if _dmax(delta, robot_vel[i], max(remaining, 0), config) + REACH_SLACK >= gap:
                empty_reachable = True
                break
        if empty_reachable:
            break
    multi = int(np.sum(inside[coverers[0]])) >= 2 if coverers else False
    meta = {
        "matched_size": float(matched_size),
        "coverers": float(len(coverers)),
        "empty_targets": float(len(empty_targets)),
        "reachable_by_other": float(reachable_by_other),
        "empty_reachable": float(empty_reachable),
        "assigned": float(assign[coverers[0]]) if coverers else -1.0,
    }
    if multi:
        return "MULTI_TARGET_OVERLAP", meta
    if empty_reachable and empty_targets:
        return "AVOIDABLE_ALLOCATION_CONFLICT", meta
    if not reachable_by_other and not empty_reachable:
        return "RESOURCE_SHORTAGE", meta
    return "GEOMETRIC_CONFLICT", meta


def _classify_travel(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_index: int,
    assign_history: list[np.ndarray],
    step: int,
    start_assign: np.ndarray,
    start_visible: np.ndarray,
    remaining: int,
    config: Any,
    radius: float,
) -> tuple[str, dict[str, float]]:
    dists = np.linalg.norm(target_pos[target_index] - robot_pos, axis=1)
    nearest = int(np.argmin(dists))
    # 有效追踪时长：最近车连续把该目标当 assign 的步数
    age = 0
    for past in reversed(assign_history[: step + 1]):
        if int(past[nearest]) == target_index:
            age += 1
        else:
            break
    switched = False
    for past in assign_history[: step + 1]:
        if int(past[nearest]) not in (-1, target_index):
            switched = True
            break
    toward = target_pos[target_index] - robot_pos[nearest]
    norm = float(np.linalg.norm(toward))
    unit = toward / norm if norm > 1e-9 else np.zeros(2)
    target_away = float(np.dot(target_vel[target_index], unit)) > 0.02
    late = age <= max(1, step // 3) and step >= 3
    start_best = int(np.argmin(np.linalg.norm(target_pos[target_index] - robot_pos, axis=1)))
    # 开局指派：用 start_assign 在 step0；这里用当前最近是否开局就该追
    bad_init = int(start_assign[nearest]) not in (-1, target_index) and int(start_assign[nearest]) >= 0
    saw_late = not bool(start_visible[nearest, target_index]) and age <= 2 and step >= 2
    # 满力追赶：径向速度为正且 gap 仍大
    gap = float(dists[nearest]) - radius
    radial = float(np.dot(robot_vel[nearest], unit))
    limited = age >= max(3, step - 1) and radial > 0.05 and gap > NEAR_MARGIN
    meta = {
        "pursuit_age": float(age),
        "gap": gap,
        "target_away": float(target_away),
        "nearest_robot": float(nearest),
        "remaining_steps": float(remaining),
    }
    if saw_late:
        return "OBSERVATION_DELAY", meta
    if switched:
        return "TARGET_SWITCH", meta
    if late:
        return "LATE_START", meta
    if bad_init and int(start_assign[start_best]) == target_index and start_best != nearest:
        return "BAD_INITIAL_ASSIGNMENT", meta
    if target_away and limited:
        return "TARGET_MOVED_AWAY", meta
    if limited:
        return "CONTROL_LIMITED", meta
    if target_away:
        return "TARGET_MOVED_AWAY", meta
    return "LATE_START", meta


def _rollout_case(case: ScenarioCase) -> dict[str, Any]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    sense = float(config.public.sense_radius)
    task = _public_task_params(case)
    policies = [CoverFirstPolicy() for _ in range(n)]
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
    start_reachable = _start_reachable(env.state(), config)
    start_state = env.state().copy()
    s_rp, s_rv, s_tp, _s_tv = _decode(start_state, n, m)
    start_assign = _choose_local(s_rp, s_tp, radius, sense)
    start_visible = (
        np.linalg.norm(s_tp[None, :, :] - s_rp[:, None, :], axis=2) <= sense
    )

    # 先整局滚完，记下轨迹，再离线分类（需要真实未来圈位）。
    traj_rp: list[np.ndarray] = []
    traj_rv: list[np.ndarray] = []
    traj_tp: list[np.ndarray] = []
    traj_tv: list[np.ndarray] = []
    traj_actions: list[np.ndarray] = []
    traj_assign: list[np.ndarray] = []
    traj_reward: list[float] = []
    traj_collision: list[float] = []
    traj_matching: list[np.ndarray] = []

    try:
        for _step in range(horizon):
            rp, rv, tp, tv = _decode(env.state(), n, m)
            assign = _choose_local(rp, tp, radius, sense)
            actions_dict = {
                agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                for index, agent_id in enumerate(env.agents)
            }
            action_mat = np.stack([actions_dict[f"agent_{i}"] for i in range(n)], axis=0)
            observations, rewards, _terms, _truncs, infos = env.step(actions_dict)
            agent = next(iter(rewards))
            rp2, rv2, tp2, tv2 = _decode(env.state(), n, m)
            matching = _matching(rp2, tp2, radius)
            traj_rp.append(rp2.copy())
            traj_rv.append(rv2.copy())
            traj_tp.append(tp2.copy())
            traj_tv.append(tv2.copy())
            traj_actions.append(action_mat.copy())
            traj_assign.append(assign.copy())
            traj_reward.append(float(rewards[agent]))
            traj_collision.append(float(infos[agent]["metrics"].collision_rate))
            traj_matching.append(matching.copy())
    finally:
        for policy in policies:
            policy.close()
        env.close()

    panel_sums = {name: 0.0 for name in (*PANEL1, *PANEL2, *PANEL3, *PANEL4)}
    coarse_sums = {name: 0.0 for name in COARSE}
    events: list[dict[str, Any]] = []
    margins: list[float] = []

    for step in range(horizon):
        rp = traj_rp[step]
        rv = traj_rv[step]
        tp = traj_tp[step]
        tv = traj_tv[step]
        actions = traj_actions[step]
        assign = traj_assign[step]
        matching = traj_matching[step]
        collision_rate = traj_collision[step]
        remaining = horizon - (step + 1)
        dist = np.linalg.norm(tp[None, :, :] - rp[:, None, :], axis=2)
        inside = dist <= radius
        future_targets = traj_tp[step + 1 :]
        next_inside = None
        if step + 1 < horizon:
            next_dist = np.linalg.norm(
                traj_tp[step + 1][None, :, :] - traj_rp[step + 1][:, None, :], axis=2
            )
            next_inside = next_dist <= radius

        for j in range(m):
            if matching[j] >= 0:
                continue
            nearest = int(np.argmin(dist[:, j]))
            gap = float(dist[nearest, j]) - radius
            base = {
                "seed": int(case.scenario_seed),
                "layout": case.group_id,
                "episode": case.case_id,
                "step": step,
                "target_id": j,
                "robot_id": nearest,
                "score_loss": SHARE,
                "distance": float(dist[nearest, j]),
                "gap": gap,
                "remaining_steps": remaining,
                "assigned_target": int(assign[nearest]),
            }
            if bool(np.any(inside[:, j])):
                cause, meta = _classify_overlap(
                    rp, rv, tp, j, assign, matching, remaining, config, radius
                )
                coarse = "overlap"
            elif bool(np.any(np.sum(inside, axis=0) >= 2)):
                cause = "CROWD"
                coarse = "crowd"
                meta = {"coverers_elsewhere": 1.0}
            elif float(np.min(dist[:, j])) <= radius + NEAR_MARGIN:
                ni = bool(np.any(next_inside[:, j])) if next_inside is not None else False
                cause, meta = _classify_near(
                    rp, rv, tp, tv, j, actions, remaining, ni, radius
                )
                coarse = "near"
            elif not bool(start_reachable[j]):
                cause, meta = _classify_unreachable(
                    rp, rv, tp, tv, j, remaining, future_targets, config, radius
                )
                coarse = "unreachable"
                margins.append(float(meta["margin"]))
            else:
                cause, meta = _classify_travel(
                    rp,
                    rv,
                    tp,
                    tv,
                    j,
                    traj_assign,
                    step,
                    start_assign,
                    start_visible,
                    remaining,
                    config,
                    radius,
                )
                coarse = "travel"

            if cause in panel_sums:
                panel_sums[cause] += SHARE
            coarse_sums[coarse] += SHARE
            events.append({**base, "cause": cause, "coarse": coarse, **meta})

        coarse_sums["collision"] += 0.2 * collision_rate

    # 按局平均（与 E039 相同：/horizon）
    panel_means = {name: panel_sums[name] / horizon for name in panel_sums}
    coarse_means = {name: coarse_sums[name] / horizon for name in coarse_sums}
    reward_mean = float(np.sum(traj_reward) / horizon)
    return {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
        "reward": reward_mean,
        "panel": panel_means,
        "coarse": coarse_means,
        "events": events,
        "margins": margins,
    }


def _paired_points(rows: list[dict[str, Any]], seeds: list[int], key_path: tuple[str, ...]) -> float:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    values: list[float] = []
    for seed in seeds:
        parts = []
        for group in ("basic", "cooperation"):
            node: Any = lookup[(seed, group)]
            for key in key_path:
                node = node[key]
            parts.append(float(node))
        values.append(500.0 * (parts[0] + parts[1]))
    return float(np.mean(values))


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    score = _paired_points(rows, seeds, ("reward",))
    lost = 1000.0 - score
    coarse_points = {name: _paired_points(rows, seeds, ("coarse", name)) for name in COARSE}
    panel_points = {
        name: _paired_points(rows, seeds, ("panel", name))
        for name in (*PANEL1, *PANEL2, *PANEL3, *PANEL4)
    }
    # crowd 细类未单独建 PANEL，保持 coarse
    panel1_sum = sum(panel_points[name] for name in PANEL1)
    panel2_sum = sum(panel_points[name] for name in PANEL2)
    panel3_sum = sum(panel_points[name] for name in PANEL3)
    panel4_sum = sum(panel_points[name] for name in PANEL4)
    margins = [float(v) for row in rows for v in row["margins"]]
    margin_hist = {
        "count": len(margins),
        "mean": float(np.mean(margins)) if margins else float("nan"),
        "p10": float(np.percentile(margins, 10)) if margins else float("nan"),
        "p50": float(np.percentile(margins, 50)) if margins else float("nan"),
        "p90": float(np.percentile(margins, 90)) if margins else float("nan"),
        "frac_margin_gt_0": float(np.mean(np.asarray(margins) > 0)) if margins else 0.0,
        "frac_margin_gt_neg005": float(np.mean(np.asarray(margins) > -0.05)) if margins else 0.0,
    }
    return {
        "note": "E028 失分四板块拆细。脚本自算，不是 evaluate_one，未改 entry.py。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "score": score,
        "lost": lost,
        "coarse_points": coarse_points,
        "panels": {
            "1_dynamic_reachability": {
                "parent_bucket_points": coarse_points["unreachable"],
                "subtotal": panel1_sum,
                "parent_gap": panel1_sum - coarse_points["unreachable"],
                "causes": {name: panel_points[name] for name in PANEL1},
                "margin_distribution": margin_hist,
            },
            "2_near_miss": {
                "parent_bucket_points": coarse_points["near"],
                "subtotal": panel2_sum,
                "parent_gap": panel2_sum - coarse_points["near"],
                "causes": {name: panel_points[name] for name in PANEL2},
            },
            "3_matching_conflict": {
                "parent_bucket_points": coarse_points["overlap"],
                "subtotal": panel3_sum,
                "parent_gap": panel3_sum - coarse_points["overlap"],
                "causes": {name: panel_points[name] for name in PANEL3},
            },
            "4_en_route_delay": {
                "parent_bucket_points": coarse_points["travel"],
                "subtotal": panel4_sum,
                "parent_gap": panel4_sum - coarse_points["travel"],
                "causes": {name: panel_points[name] for name in PANEL4},
            },
        },
        "refined_total": {
            "hard_unreachable": panel_points["HARD_UNREACHABLE"],
            "reachability_error": (
                panel_points["TARGET_ASSISTED_REACHABLE"]
                + panel_points["MARGINAL_UNREACHABLE"]
                + panel_points["TARGET_ESCAPED"]
            ),
            "near_miss": panel2_sum,
            "assignment_conflict": panel3_sum,
            "en_route_delay": panel4_sum,
            "crowd": coarse_points["crowd"],
            "collision": coarse_points["collision"],
        },
        "account_check": {
            "coarse_sum": sum(coarse_points.values()),
            "refined_sum": (
                panel1_sum
                + panel2_sum
                + panel3_sum
                + panel4_sum
                + coarse_points["crowd"]
                + coarse_points["collision"]
            ),
            "lost": lost,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=200000)
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
        futures = [executor.submit(_rollout_case, case) for case in cases]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)
    summary = _summarize(rows, seeds)
    # episode 级汇总
    episode_rows = []
    for row in rows:
        flat = {
            "case_id": row["case_id"],
            "group": row["group"],
            "seed": row["seed"],
            "reward": row["reward"],
            **{f"coarse_{k}": v for k, v in row["coarse"].items()},
            **{f"panel_{k}": v for k, v in row["panel"].items()},
        }
        episode_rows.append(flat)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    # 事件表（截断）
    all_events: list[dict[str, Any]] = []
    for row in rows:
        all_events.extend(row["events"])
        if len(all_events) >= args.max_events:
            all_events = all_events[: args.max_events]
            break
    if all_events:
        # 统一字段
        keys: list[str] = sorted({key for event in all_events for key in event})
        with (output / "events.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_events)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
