"""把 E039 的「开局就够不着」按真实未来轨迹再拆开。

E039 在开局把目标当成不动，用一维缩短量加 0.08 判死。这里只重看那些被记进
unreachable 的 target-step，不改策略。

这一步的静态余量沿用同一条一维公式：

    m = D_available - (d - r)

真实未来用环境里已经发生的目标轨迹。车的受力在每个轴上独立，范围是 [-1, 1]，
忽略碰撞和速度上限时，未来第 h 步的可达位置是一个正方形。正方形碰到圈，就说明
存在一组力能在那一步进圈。

读全局状态，只跑上场 E028。不是 evaluate_one。
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
from coverage_bench.protocol import EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase
from entry import CoverFirstPolicy
from loss_attribution import NEAR_MARGIN, _attribute_step, _covered_targets, _start_reachable
from oracle_search import _decode
from reachability_debounce_oracle import (
    MASTER_SEED,
    _layout_config,
    _optimistic_closure,
    _random_seeds,
)


MARGIN_BINS = (-np.inf, -0.5, -0.3, -0.1, -0.05, 0.0, 0.05, np.inf)
MISS_BINS = (-np.inf, 0.0, 0.01, 0.05, 0.10, 0.30, np.inf)
MARGINAL_MISS = 0.05
CLASSES = (
    "hard_unreachable",
    "marginal_unreachable",
    "target_assisted_reachable",
    "frozen_2d_already_reachable",
)


def _coast_and_width(velocity: np.ndarray, steps: int) -> tuple[np.ndarray, float]:
    """不做力时的位置，以及每个轴上还能被力推动的最大距离。"""
    alpha = 0.75
    coast = velocity * (0.4 * (1.0 - alpha**steps))
    width = 0.04 * float(sum(1.0 - alpha**q for q in range(1, steps)))
    return coast, width


def _box_miss(position: np.ndarray, velocity: np.ndarray, future_targets: np.ndarray, radius: float) -> float:
    """未来每个落点到可达正方形的距离，再减圈半径。负数表示正方形已经盖住圈。"""
    if len(future_targets) == 0:
        return float("inf")
    misses: list[float] = []
    for steps, target in enumerate(future_targets, start=1):
        coast_offset, width = _coast_and_width(velocity, steps)
        center = position + coast_offset
        closest = np.clip(target, center - width, center + width)
        misses.append(float(np.linalg.norm(closest - target) - radius))
    return min(misses)


def _integrate(position: np.ndarray, velocity: np.ndarray, action: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """和 MPE2 一样：先用旧速度走位置，再阻尼、加力、限速，最后夹在场地里。"""
    position = position + velocity * 0.1
    velocity = 0.75 * velocity + 0.1 * action
    speed = float(np.linalg.norm(velocity))
    if speed > 1.0:
        velocity = velocity * (1.0 / speed)
    for axis in (0, 1):
        if position[axis] > limit:
            position[axis] = limit
            if velocity[axis] > 0.0:
                velocity[axis] = 0.0
        elif position[axis] < -limit:
            position[axis] = -limit
            if velocity[axis] < 0.0:
                velocity[axis] = 0.0
    return position, velocity


def _witness_distance(position: np.ndarray, velocity: np.ndarray, future_targets: np.ndarray, limit: float) -> float:
    """饱和追踪的最好距离。进了圈就证明真的够得着；没进不能证明够不着。"""
    if len(future_targets) == 0:
        return float("inf")
    best = float("inf")
    pos = position.copy()
    vel = velocity.copy()
    for target in future_targets:
        pos, vel = _integrate(pos, vel, np.sign(target - pos), limit)
        best = min(best, float(np.linalg.norm(pos - target)))
    for end in range(len(future_targets)):
        pos = position.copy()
        vel = velocity.copy()
        aim = future_targets[end]
        for step in range(end + 1):
            pos, vel = _integrate(pos, vel, np.sign(aim - pos), limit)
            best = min(best, float(np.linalg.norm(pos - future_targets[step])))
    return best


def _check_box_contains_simulated_points() -> None:
    """随机力序列的落点必须落在正方形里，否则后面的「够不着」不可信。"""
    rng = np.random.default_rng(0)
    for _ in range(200):
        position = rng.uniform(-0.8, 0.8, size=2)
        velocity = rng.uniform(-0.4, 0.4, size=2)
        steps = int(rng.integers(1, 10))
        actions = rng.uniform(-1.0, 1.0, size=(steps, 2))
        pos = position.copy()
        vel = velocity.copy()
        for action in actions:
            pos = pos + vel * 0.1
            vel = 0.75 * vel + 0.1 * action
        coast, width = _coast_and_width(velocity, steps)
        center = position + coast
        if np.any(pos < center - width - 1e-9) or np.any(pos > center + width + 1e-9):
            raise RuntimeError("可达正方形没有包住模拟落点")


def _static_pair(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    steps: int,
    radius: float,
    config: Any,
) -> dict[str, float]:
    """最近的车，以及一维余量最大的那台车。"""
    nearest_distance = float("inf")
    best_margin = -float("inf")
    nearest: dict[str, float] = {}
    best: dict[str, float] = {}
    for index in range(len(robot_pos)):
        delta = target_pos - robot_pos[index]
        distance = float(np.linalg.norm(delta))
        gap = max(0.0, distance - radius)
        closure = float(_optimistic_closure(delta, robot_vel[index], steps, config)) if steps > 0 else 0.0
        margin = closure - gap
        radial = float(np.dot(robot_vel[index], delta / distance)) if distance > 1e-8 else 0.0
        features = {
            "distance": distance,
            "gap": gap,
            "closure": closure,
            "margin": margin,
            "speed": float(np.linalg.norm(robot_vel[index])),
            "radial_speed": radial,
        }
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = features
        if margin > best_margin:
            best_margin = margin
            best = features
    return {
        "nearest_distance": nearest["distance"],
        "nearest_gap": nearest["gap"],
        "nearest_closure": nearest["closure"],
        "nearest_margin": nearest["margin"],
        "nearest_speed": nearest["speed"],
        "nearest_radial_speed": nearest["radial_speed"],
        "best_gap": best["gap"],
        "best_closure": best["closure"],
        "best_margin": best["margin"],
        "best_speed": best["speed"],
        "best_radial_speed": best["radial_speed"],
    }


def _classify(moving_miss: float, frozen_miss: float) -> str:
    if moving_miss <= 0.0 and frozen_miss > 0.0:
        return "target_assisted_reachable"
    if moving_miss <= 0.0:
        return "frozen_2d_already_reachable"
    if moving_miss <= MARGINAL_MISS:
        return "marginal_unreachable"
    return "hard_unreachable"


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    limit = float(config.public.map_half_extent - config.public.robot_radius)
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
    robot_pos, robot_vel, target_pos, _target_vel = _decode(env.state(), n, m)
    start_reachable = _start_reachable(env.state(), config)
    start_features = [
        _static_pair(robot_pos, robot_vel, target_pos[target_index], horizon, radius, config)
        for target_index in range(m)
    ]
    previously_covered = np.zeros(m, dtype=np.bool_)
    robot_trace: list[np.ndarray] = []
    velocity_trace: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []
    unreachable_steps: list[tuple[int, int]] = []
    try:
        for step in range(horizon):
            actions = {
                agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                for index, agent_id in enumerate(env.agents)
            }
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            robot_pos, robot_vel, target_pos, _target_vel = _decode(env.state(), n, m)
            robot_trace.append(robot_pos.copy())
            velocity_trace.append(robot_vel.copy())
            target_trace.append(target_pos.copy())
            shares = _attribute_step(
                robot_pos,
                target_pos,
                radius,
                float(infos[agent]["metrics"].collision_rate),
                start_reachable,
                previously_covered,
            )
            distance = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
            inside = distance <= radius
            matching = _covered_targets(robot_pos, target_pos, radius)
            share = 0.0
            for target_index, robot_index in enumerate(matching):
                if robot_index >= 0 or bool(np.any(inside[:, target_index])):
                    continue
                if bool(np.any(np.sum(inside, axis=0) >= 2)):
                    continue
                if float(np.min(distance[:, target_index])) <= radius + NEAR_MARGIN:
                    continue
                if bool(start_reachable[target_index]):
                    continue
                unreachable_steps.append((step, target_index))
                share += 1.0 / m
            if abs(share - shares["unreachable"]) > 1e-9:
                raise RuntimeError(f"unreachable 对不上 E039: {share} vs {shares['unreachable']}")
            previously_covered = matching >= 0
    finally:
        for policy in policies:
            policy.close()
        env.close()

    rows: list[dict[str, Any]] = []
    for step, target_index in unreachable_steps:
        remaining = horizon - step - 1
        current_target = target_trace[step][target_index]
        if remaining == 0:
            future = np.zeros((0, 2), dtype=np.float64)
        else:
            future = np.stack(target_trace[step + 1 :])[:, target_index, :]
        frozen_future = np.repeat(current_target[None, :], remaining, axis=0)
        moving_miss = float("inf")
        frozen_miss = float("inf")
        witness = float("inf")
        best_robot = 0
        for robot_index in range(n):
            if remaining == 0:
                robot_moving = float(np.linalg.norm(robot_trace[step][robot_index] - current_target) - radius)
                robot_frozen = robot_moving
                robot_witness = float("inf")
            else:
                robot_moving = _box_miss(
                    robot_trace[step][robot_index], velocity_trace[step][robot_index], future, radius
                )
                robot_frozen = _box_miss(
                    robot_trace[step][robot_index], velocity_trace[step][robot_index], frozen_future, radius
                )
                robot_witness = _witness_distance(
                    robot_trace[step][robot_index], velocity_trace[step][robot_index], future, limit
                )
            if robot_moving < moving_miss:
                moving_miss = robot_moving
                best_robot = robot_index
            frozen_miss = min(frozen_miss, robot_frozen)
            witness = min(witness, robot_witness)
        features = _static_pair(
            robot_trace[step],
            velocity_trace[step],
            current_target,
            remaining,
            radius,
            config,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "step": step,
                "target": target_index,
                "remaining_steps": remaining,
                "start_margin": start_features[target_index]["best_margin"],
                "start_margin_with_slack": start_features[target_index]["best_margin"] + 0.08,
                "moving_miss": moving_miss,
                "frozen_miss": frozen_miss,
                "witness_distance": witness,
                "witness_enters": int(witness <= radius),
                "class_name": _classify(moving_miss, frozen_miss),
                "best_robot": best_robot,
                **features,
            }
        )
    return rows


def _points(count: int, seeds: int, horizon: int) -> float:
    """一个 target-step 占 1/3 步奖励，再按 500*(两种布局) 对种子平均。"""
    return count * 500.0 / (3.0 * horizon * seeds)


def _histogram(values: np.ndarray, edges: tuple[float, ...]) -> list[dict[str, float]]:
    counts, _bins = np.histogram(values, bins=edges)
    total = max(int(values.size), 1)
    rows: list[dict[str, float]] = []
    for index, count in enumerate(counts):
        rows.append(
            {
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": int(count),
                "fraction": float(count) / total,
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], seeds: list[int], horizon: int) -> dict[str, Any]:
    classes = {name: 0 for name in CLASSES}
    remaining_bins = {"0": 0, "1_to_3": 0, "4_to_6": 0, "7_to_9": 0}
    for row in rows:
        classes[str(row["class_name"])] += 1
        remaining = int(row["remaining_steps"])
        if remaining == 0:
            remaining_bins["0"] += 1
        elif remaining <= 3:
            remaining_bins["1_to_3"] += 1
        elif remaining <= 6:
            remaining_bins["4_to_6"] += 1
        else:
            remaining_bins["7_to_9"] += 1
    margins = np.asarray([float(row["best_margin"]) for row in rows], dtype=np.float64)
    start_margins = np.asarray([float(row["start_margin"]) for row in rows], dtype=np.float64)
    misses = np.asarray([float(row["moving_miss"]) for row in rows], dtype=np.float64)
    early = [row for row in rows if int(row["remaining_steps"]) >= 5]
    early_margins = np.asarray([float(row["best_margin"]) for row in early], dtype=np.float64)
    witness_hits = sum(int(row["witness_enters"]) for row in rows)
    point = lambda count: _points(count, len(seeds), horizon)
    class_points = {name: point(count) for name, count in classes.items()}
    truly_lost = class_points["hard_unreachable"] + class_points["marginal_unreachable"]
    return {
        "oracle_warning": "只重看 E028 在 E039 里标成开局够不着的 target-step。读取全局状态和真实未来轨迹，不是 evaluate_one。",
        "static_margin": "m = 一维可缩短距离 - (d - r)，用这一步剩下的步数，目标先当不动。负数表示按这条公式进不去。",
        "moving_miss": "真实未来轨迹下，可达正方形离圈边还差多少。小于等于 0 表示存在一组力能进圈。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "target_steps": len(rows),
        "unreachable_points": point(len(rows)),
        "class_points": class_points,
        "truly_unreachable_points": truly_lost,
        "not_truly_unreachable_points": point(len(rows)) - truly_lost,
        "witness_confirmed_points": point(witness_hits),
        "remaining_step_points": {name: point(count) for name, count in remaining_bins.items()},
        "best_margin_histogram": _histogram(margins, MARGIN_BINS),
        "best_margin_histogram_remaining_at_least_5": _histogram(early_margins, MARGIN_BINS),
        "start_margin_histogram": _histogram(start_margins, MARGIN_BINS),
        "moving_miss_histogram": _histogram(misses, MISS_BINS),
        "margin_below_minus_0_3": float(np.mean(margins < -0.3)) if len(rows) else None,
        "margin_between_minus_0_05_and_0": float(np.mean((margins > -0.05) & (margins < 0.0))) if len(rows) else None,
        "early_margin_below_minus_0_3": float(np.mean(early_margins < -0.3)) if early else None,
        "early_margin_between_minus_0_05_and_0": float(np.mean((early_margins > -0.05) & (early_margins < 0.0))) if early else None,
        "median_best_margin": float(np.median(margins)) if len(rows) else None,
        "median_start_margin": float(np.median(start_margins)) if len(rows) else None,
        "median_moving_miss": float(np.median(misses)) if len(rows) else None,
    }


def main() -> None:
    _check_box_contains_simulated_points()
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
    horizon = int(basic.horizon)
    summary = _summarize(rows, seeds, horizon)
    if rows:
        with (output / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
