"""短时域动作序列搜索。

分配固定为 E041 的可达性防抖。追谁不变，只改这一步的力。
目标速度用全局状态，常速度外推，不预测转向和撞墙反射。
不是合法策略，也不是 evaluate_one。

pure:     a = clip(10 * Δp, -1, 1)
lead_0.3: a = clip(10 * (Δp + 0.3 * v_T), -1, 1)
pos_pd:   a = clip(20 * Δp + 4 * (v_T - v_R), -1, 1)
grid:     每步力取 {-1,0,1}^2，未来 3 步共 729 串，只执行第一拍
template: 每步 7 个瞄准模板，未来 3 步共 343 串，只执行第一拍

模型代价：越早进圈越好；同一进圈次数下，距离和更小更好。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
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
from prediction_oracle import _actions_from_aim
from reachability_debounce_oracle import (
    MASTER_SEED,
    _Assigner,
    _layout_config,
    _mean_ci,
    _random_seeds,
)


LAWS = ("pure", "lead_0.3", "pos_pd", "grid", "template")
HORIZON = 3
COVER_WEIGHT = 100.0

_GRID = np.array(list(product((-1.0, 0.0, 1.0), repeat=2)), dtype=np.float64)
_GRID_SEQ = _GRID[np.array(list(product(range(len(_GRID)), repeat=HORIZON)))]
_TEMPLATE_SEQ = np.array(list(product(range(7), repeat=HORIZON)), dtype=np.int64)


def _physics(config: Any) -> dict[str, float]:
    pub = config.public
    return {
        "dt": float(pub.dt),
        "damping": float(pub.damping),
        "accel": float(pub.drive_force) / float(pub.robot_mass) * float(pub.dt),
        "cap": float(pub.robot_max_speed),
        "limit": float(pub.map_half_extent) - float(pub.robot_radius),
        "radius": float(pub.target_radius),
    }


def _step_many(
    pos: np.ndarray,
    vel: np.ndarray,
    actions: np.ndarray,
    phys: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    new_pos = pos + vel * phys["dt"]
    new_vel = (1.0 - phys["damping"]) * vel + actions * phys["accel"]
    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    fast = speed[:, 0] > phys["cap"]
    scaled = new_vel * (phys["cap"] / np.maximum(speed, 1e-12))
    new_vel = np.where(fast[:, None], scaled, new_vel)
    limit = phys["limit"]
    hit_hi = new_pos > limit
    hit_lo = new_pos < -limit
    new_pos = np.clip(new_pos, -limit, limit)
    new_vel = np.where(hit_hi & (new_vel > 0.0), 0.0, new_vel)
    new_vel = np.where(hit_lo & (new_vel < 0.0), 0.0, new_vel)
    return new_pos, new_vel


def _rollout_cost(
    pos: np.ndarray,
    vel: np.ndarray,
    target: np.ndarray,
    target_vel: np.ndarray,
    actions: np.ndarray,
    phys: dict[str, float],
) -> np.ndarray:
    """actions 形状是 (条数, 步数, 2)。返回每条序列的模型代价。"""
    count, horizon, _ = actions.shape
    state_pos = np.repeat(pos.reshape(1, 2), count, axis=0)
    state_vel = np.repeat(vel.reshape(1, 2), count, axis=0)
    point = np.asarray(target, dtype=np.float64).copy()
    speed = np.asarray(target_vel, dtype=np.float64)
    cost = np.zeros(count, dtype=np.float64)
    radius = phys["radius"]
    for step in range(horizon):
        state_pos, state_vel = _step_many(state_pos, state_vel, actions[:, step], phys)
        point = point + speed * phys["dt"]
        dist = np.linalg.norm(point - state_pos, axis=1)
        covered = dist <= radius
        cost += dist - COVER_WEIGHT * float(horizon - step) * covered
    return cost


def _template_actions(
    pos: np.ndarray,
    target: np.ndarray,
    target_vel: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    delta = target.reshape(1, 2) - pos
    turn = np.deg2rad(30.0)
    cosine, sine = float(np.cos(turn)), float(np.sin(turn))
    left = np.column_stack(
        (
            cosine * delta[:, 0] - sine * delta[:, 1],
            sine * delta[:, 0] + cosine * delta[:, 1],
        )
    )
    right = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        )
    )
    pursuit = np.clip(10.0 * delta, -1.0, 1.0)
    table = np.stack(
        (
            pursuit,
            np.clip(10.0 * (delta + 0.3 * target_vel.reshape(1, 2)), -1.0, 1.0),
            0.5 * pursuit,
            np.zeros_like(pursuit),
            np.clip(10.0 * left, -1.0, 1.0),
            np.clip(10.0 * right, -1.0, 1.0),
            np.clip(-10.0 * delta, -1.0, 1.0),
        ),
        axis=0,
    )
    return table[ids, np.arange(ids.shape[0])]


def _best_grid(
    pos: np.ndarray,
    vel: np.ndarray,
    target: np.ndarray,
    target_vel: np.ndarray,
    steps_left: int,
    phys: dict[str, float],
) -> np.ndarray:
    horizon = min(HORIZON, max(steps_left, 1))
    actions = np.repeat(_GRID_SEQ[:, :horizon], 1, axis=0)
    cost = _rollout_cost(pos, vel, target, target_vel, actions, phys)
    return actions[int(np.argmin(cost)), 0]


def _best_template(
    pos: np.ndarray,
    vel: np.ndarray,
    target: np.ndarray,
    target_vel: np.ndarray,
    steps_left: int,
    phys: dict[str, float],
) -> np.ndarray:
    horizon = min(HORIZON, max(steps_left, 1))
    ids = _TEMPLATE_SEQ[:, :horizon]
    count = ids.shape[0]
    state_pos = np.repeat(pos.reshape(1, 2), count, axis=0)
    state_vel = np.repeat(vel.reshape(1, 2), count, axis=0)
    point = np.asarray(target, dtype=np.float64).copy()
    speed = np.asarray(target_vel, dtype=np.float64)
    cost = np.zeros(count, dtype=np.float64)
    first = np.zeros((count, 2), dtype=np.float64)
    radius = phys["radius"]
    for step in range(horizon):
        act = _template_actions(state_pos, point, speed, ids[:, step])
        if step == 0:
            first = act
        state_pos, state_vel = _step_many(state_pos, state_vel, act, phys)
        point = point + speed * phys["dt"]
        dist = np.linalg.norm(point - state_pos, axis=1)
        cost += dist - COVER_WEIGHT * float(horizon - step) * (dist <= radius)
    return first[int(np.argmin(cost))]


def _pd_actions(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    assignment: tuple[int, ...],
) -> np.ndarray:
    actions = np.zeros((len(robot_pos), 2), dtype=np.float64)
    for index, target_index in enumerate(assignment):
        delta = target_pos[target_index] - robot_pos[index]
        relative = target_vel[target_index] - robot_vel[index]
        actions[index] = np.clip(20.0 * delta + 4.0 * relative, -1.0, 1.0)
    return actions


def _rollout(case: ScenarioCase, law: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    phys = _physics(config)
    n = int(config.num_agents)
    m = int(config.num_targets)
    horizon = int(config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    differ = 0
    steps = 0
    try:
        for step in range(horizon):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            steps_left = horizon - step
            if law == "pure":
                actions_mat = _actions_from_aim(
                    robot_pos, target_pos, target_vel, chosen, lambda _i: 0.0
                )
            elif law == "lead_0.3":
                actions_mat = _actions_from_aim(
                    robot_pos, target_pos, target_vel, chosen, lambda _i: 0.3
                )
            elif law == "pos_pd":
                actions_mat = _pd_actions(robot_pos, robot_vel, target_pos, target_vel, chosen)
            elif law in ("grid", "template"):
                pure = _actions_from_aim(
                    robot_pos, target_pos, target_vel, chosen, lambda _i: 0.0
                )
                actions_mat = np.zeros((n, 2), dtype=np.float64)
                for index, target_index in enumerate(chosen):
                    if law == "grid":
                        action = _best_grid(
                            robot_pos[index],
                            robot_vel[index],
                            target_pos[target_index],
                            target_vel[target_index],
                            steps_left,
                            phys,
                        )
                    else:
                        action = _best_template(
                            robot_pos[index],
                            robot_vel[index],
                            target_pos[target_index],
                            target_vel[target_index],
                            steps_left,
                            phys,
                        )
                    actions_mat[index] = action
                    if float(np.linalg.norm(action - pure[index])) > 1e-3:
                        differ += 1
                    steps += 1
            else:
                raise ValueError(law)
            actions = {f"agent_{index}": actions_mat[index].astype(np.float32) for index in range(n)}
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
        "differ": float(differ) / float(steps) if steps else 0.0,
    }


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
    }
    for law in LAWS:
        result = _rollout(case, law)
        for key, value in result.items():
            row[f"{law}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], law: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{law}_j"]) + float(lookup[(seed, "cooperation")][f"{law}_j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base = _paired(rows, seeds, "pure")
    laws: dict[str, Any] = {}
    for law in LAWS:
        scores = _paired(rows, seeds, law)
        gaps = [new - old for old, new in zip(base, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        laws[law] = {
            "score": float(np.mean(scores)),
            "gap_vs_pure": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{law}_coverage"]) for row in rows])),
            "collision": float(np.mean([float(row[f"{law}_collision"]) for row in rows])),
            "differ_vs_pure": float(np.mean([float(row[f"{law}_differ"]) for row in rows])),
        }
    lead = _paired(rows, seeds, "lead_0.3")
    for law in ("grid", "template"):
        scores = _paired(rows, seeds, law)
        gaps = [new - old for old, new in zip(lead, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        laws[law]["gap_vs_lead_0.3"] = float(np.mean(gaps))
        laws[law]["gap_vs_lead_ci95_normal"] = _mean_ci(gaps)
        laws[law]["vs_lead_wins"] = wins
        laws[law]["vs_lead_ties"] = len(gaps) - wins - losses
        laws[law]["vs_lead_losses"] = losses
    return {
        "oracle_warning": "目标速度来自全局状态。不是 evaluate_one，未改 entry.py。",
        "assignment": "可达性防抖，与 E041 相同",
        "model": "常速度外推 3 步；车的积分含阻尼、力、速度上限和边界夹紧；不含车与车的斥力，也不含目标转向",
        "grid": "{-1,0,1}^2 的 729 条开环序列，只执行第一拍",
        "template": "追踪、提前 0.3 秒、半力、停下、左右偏 30 度、反向，共 343 条，只执行第一拍",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
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
