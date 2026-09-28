"""把一步奖励相对满分拆开，各块加起来等于丢掉的分。

满分一步的奖励是 1。实际是

    r = 覆盖率 - 0.2 * 碰撞参与率

没罩住的每个圈占 1/3，只归入一个原因：

- overlap：有车已经在这个圈里，但一台车只能算一个圈
- crowd：这个圈里没车，别的圈里叠了至少两台
- near：最近的车在圈外 0.10 以内
- unreachable：开局时目标先当不动，没有一台车能在 10 步内乐观地进圈
- travel：开局够得着，这一步最近的车仍在 0.10 以外

碰撞扣分单独一块。出圈只另记次数，不再加进总分，避免和上面的原因重复。

对照三条已有决策，不发明新控制器：上场 E028；防抖加 clip(10*Δp)；防抖加 clip(20*Δp+4*Δv)。
后两条读取全局状态，不是合法策略。
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
from oracle_search import _decode
from pd_oracle import _pd_action
from reachability_debounce_oracle import (
    MASTER_SEED,
    _Assigner,
    _layout_config,
    _optimistic_closure,
    _random_seeds,
)


NEAR_MARGIN = 0.10
REACH_SLACK = 0.08
BUCKETS = ("overlap", "crowd", "near", "unreachable", "travel", "collision")
LAWS = ("e028", "proportional", "pd")


def _start_reachable(state: np.ndarray, config: Any) -> np.ndarray:
    """开局、目标先当不动时，是否有一台车能在整局内乐观地进这个圈。"""
    n = int(config.num_agents)
    m = int(config.num_targets)
    robot_pos, robot_vel, target_pos, _target_vel = _decode(state, n, m)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    reachable = np.zeros(m, dtype=np.bool_)
    for target_index in range(m):
        for robot_index in range(n):
            delta = target_pos[target_index] - robot_pos[robot_index]
            gap = max(0.0, float(np.linalg.norm(delta)) - radius)
            closure = _optimistic_closure(delta, robot_vel[robot_index], horizon, config)
            if closure + REACH_SLACK >= gap:
                reachable[target_index] = True
                break
    return reachable


def _covered_targets(robot_pos: np.ndarray, target_pos: np.ndarray, radius: float) -> np.ndarray:
    """和官方一样做最大匹配。返回长度为目标数的数组，-1 表示这个圈没被算上。"""
    robot_count, target_count = len(robot_pos), len(target_pos)
    adjacency = np.zeros((robot_count, target_count), dtype=np.int32)
    distance = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
    adjacency[distance <= radius] = 1
    matching = maximum_bipartite_matching(csr_matrix(adjacency), perm_type="column")
    return np.asarray(matching, dtype=np.int64)


def _attribute_step(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    radius: float,
    collision_rate: float,
    start_reachable: np.ndarray,
    previously_covered: np.ndarray,
) -> dict[str, float]:
    """这一步各类失分。覆盖类加碰撞类，再加实际奖励，应当等于 1。"""
    shares = {name: 0.0 for name in BUCKETS}
    shares["exits"] = 0.0
    distance = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
    inside = distance <= radius
    matching = _covered_targets(robot_pos, target_pos, radius)
    share = 1.0 / len(target_pos)
    for target_index, robot_index in enumerate(matching):
        if robot_index >= 0:
            continue
        if bool(previously_covered[target_index]):
            shares["exits"] += share
        if bool(np.any(inside[:, target_index])):
            shares["overlap"] += share
        elif bool(np.any(np.sum(inside, axis=0) >= 2)):
            shares["crowd"] += share
        elif float(np.min(distance[:, target_index])) <= radius + NEAR_MARGIN:
            shares["near"] += share
        elif not bool(start_reachable[target_index]):
            shares["unreachable"] += share
        else:
            shares["travel"] += share
    shares["collision"] = 0.2 * float(collision_rate)
    return shares


def _empty_totals() -> dict[str, float]:
    totals = {name: 0.0 for name in (*BUCKETS, "exits", "reward")}
    totals.update({f"cover_{count}": 0.0 for count in range(4)})
    return totals


def _rollout_oracle(case: ScenarioCase, kp: float, kd: float) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    start_reachable = _start_reachable(env.state(), config)
    previously_covered = np.zeros(m, dtype=np.bool_)
    totals = _empty_totals()
    try:
        for step in range(horizon):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            actions = {
                f"agent_{robot_index}": _pd_action(
                    target_pos[target_index] - robot_pos[robot_index],
                    target_vel[target_index] - robot_vel[robot_index],
                    kp,
                    kd,
                )
                for robot_index, target_index in enumerate(chosen)
            }
            _observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            reward = float(rewards[agent])
            collision_rate = float(infos[agent]["metrics"].collision_rate)
            robot_pos, _robot_vel, target_pos, _target_vel = _decode(env.state(), n, m)
            shares = _attribute_step(
                robot_pos, target_pos, radius, collision_rate, start_reachable, previously_covered
            )
            for name in (*BUCKETS, "exits"):
                totals[name] += shares[name]
            totals["reward"] += reward
            covered_count = int(round((1.0 - sum(shares[name] for name in BUCKETS if name != "collision")) * m))
            totals[f"cover_{covered_count}"] += 1.0
            previously_covered = _covered_targets(robot_pos, target_pos, radius) >= 0
    finally:
        env.close()
    return {key: value / horizon for key, value in totals.items()}


def _rollout_e028(case: ScenarioCase) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
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
    start_reachable = _start_reachable(env.state(), config)
    previously_covered = np.zeros(m, dtype=np.bool_)
    totals = _empty_totals()
    try:
        for _step in range(horizon):
            actions = {
                agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                for index, agent_id in enumerate(env.agents)
            }
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            reward = float(rewards[agent])
            collision_rate = float(infos[agent]["metrics"].collision_rate)
            robot_pos, _robot_vel, target_pos, _target_vel = _decode(env.state(), n, m)
            shares = _attribute_step(
                robot_pos, target_pos, radius, collision_rate, start_reachable, previously_covered
            )
            for name in (*BUCKETS, "exits"):
                totals[name] += shares[name]
            totals["reward"] += reward
            covered_count = int(round((1.0 - sum(shares[name] for name in BUCKETS if name != "collision")) * m))
            totals[f"cover_{covered_count}"] += 1.0
            previously_covered = _covered_targets(robot_pos, target_pos, radius) >= 0
    finally:
        for policy in policies:
            policy.close()
        env.close()
    return {key: value / horizon for key, value in totals.items()}


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    results = {
        "e028": _rollout_e028(case),
        "proportional": _rollout_oracle(case, 10.0, 0.0),
        "pd": _rollout_oracle(case, 20.0, 4.0),
    }
    rows: list[dict[str, Any]] = []
    for law, totals in results.items():
        accounted = sum(totals[name] for name in BUCKETS) + totals["reward"]
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "law": law,
                "accounted": accounted,
                **totals,
            }
        )
    return rows


def _law_rows(rows: list[dict[str, Any]], law: str) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["seed"]), str(row["group"])): row for row in rows if row["law"] == law}


def _mean_points(rows: list[dict[str, Any]], seeds: list[int], law: str) -> dict[str, Any]:
    """配对分按 500*(basic+cooperation) 计。各块的配对分加总等于 1000 减实际分。"""
    chosen = _law_rows(rows, law)
    score_parts: dict[str, list[float]] = {name: [] for name in (*BUCKETS, "exits", "reward")}
    cover_parts: dict[str, list[float]] = {f"cover_{count}": [] for count in range(4)}
    residuals: list[float] = []
    for seed in seeds:
        for name in score_parts:
            basic = float(chosen[(seed, "basic")][name])
            cooperation = float(chosen[(seed, "cooperation")][name])
            score_parts[name].append(500.0 * (basic + cooperation))
        for name in cover_parts:
            basic = float(chosen[(seed, "basic")][name])
            cooperation = float(chosen[(seed, "cooperation")][name])
            cover_parts[name].append(0.5 * (basic + cooperation))
        residuals.append(float(chosen[(seed, "basic")]["accounted"]) - 1.0)
        residuals.append(float(chosen[(seed, "cooperation")]["accounted"]) - 1.0)
    points = {name: float(np.mean(values)) for name, values in score_parts.items()}
    lost = 1000.0 - points["reward"]
    explained = sum(points[name] for name in BUCKETS)
    by_group: dict[str, dict[str, float]] = {}
    for group in ("basic", "cooperation"):
        group_rows = [chosen[(seed, group)] for seed in seeds]
        by_group[group] = {
            "score": 500.0 * float(np.mean([float(row["reward"]) for row in group_rows])),
            **{
                name: 500.0 * float(np.mean([float(row[name]) for row in group_rows]))
                for name in (*BUCKETS, "exits")
            },
        }
    return {
        "law": law,
        "score": points["reward"],
        "lost": lost,
        "points": {name: points[name] for name in BUCKETS},
        "explained": explained,
        "explain_gap": explained - lost,
        "exit_points_already_inside": points["exits"],
        "by_group": by_group,
        "cover_step_fraction": {name: float(np.mean(values)) for name, values in cover_parts.items()},
        "max_abs_account_error": float(np.max(np.abs(residuals))),
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    return {
        "oracle_warning": "防抖两条读取全局状态。E028 只用局部观测。都不是 evaluate_one，也不是组织方核验。",
        "buckets": {
            "overlap": "有车在圈内，但匹配没把这个圈算上",
            "crowd": "本圈没车，另一个圈里至少两台车",
            "near": "最近的车在圈外 0.10 以内",
            "unreachable": "开局、目标不动时，10 步乐观缩短量加 0.08 仍进不了圈",
            "travel": "开局够得着，这一步最近的车仍更远",
            "collision": "0.2 乘碰撞参与率",
        },
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": [_mean_points(rows, seeds, law) for law in LAWS],
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
