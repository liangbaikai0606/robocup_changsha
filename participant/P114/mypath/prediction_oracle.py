"""预测链诊断：纯追踪、提前量、截击、短时域 MPC。

分配都用 E034 的可达性防抖。只改瞄准点。读取全局状态，不是合法策略。
种子与 E034 相同：主种子 20261003 的 300 个。

    pure:     a = clip(10 * Δp, -1, 1)
    lead:     a = clip(10 * (Δp + τ * v_T), -1, 1)
    intercept: 在剩余步里选一个 τ，使预测圈位最可能被乐观缩短量盖住
    mpc:      每步在 τ∈{0,0.1,0.2,0.3} 和截击动作里，用 3 步常速度模型选回报最高的第一步
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
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[3]
MYPATH_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MYPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MYPATH_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.suites import ScenarioCase
from oracle_search import _decode
from reachability_debounce_oracle import (
    MASTER_SEED,
    _Assigner,
    _layout_config,
    _mean_ci,
    _optimistic_closure,
    _random_seeds,
)


LAWS = ("pure", "lead_0.1", "lead_0.2", "lead_0.3", "lead_fd_0.2", "intercept", "mpc")
TAUS = (0.0, 0.1, 0.2, 0.3)
MPC_HORIZON = 3


def _clip_aim(delta: np.ndarray) -> np.ndarray:
    return np.clip(10.0 * delta, -1.0, 1.0).astype(np.float32)


def _lead_delta(delta_p: np.ndarray, target_vel: np.ndarray, tau: float) -> np.ndarray:
    return delta_p + tau * target_vel


def _intercept_tau(
    delta_p: np.ndarray,
    robot_vel: np.ndarray,
    target_vel: np.ndarray,
    steps_left: int,
    config: Any,
    radius: float,
) -> float:
    """选最短的预测时间，使乐观缩短量能盖住预测后的圈外距离。盖不住就选净缺口最小的。"""
    if steps_left <= 0:
        return 0.0
    dt = float(config.public.dt)
    best_tau = 0.0
    best_short = 1e9
    found = False
    for k in range(0, steps_left + 1):
        tau = k * dt
        aim = delta_p + tau * target_vel
        gap = max(0.0, float(np.linalg.norm(aim)) - radius)
        closure = _optimistic_closure(aim, robot_vel, max(k, 1), config) if k > 0 else 0.0
        short = gap - closure
        if short <= 0.08 and not found:
            return tau
        if short < best_short:
            best_short = short
            best_tau = tau
            found = short <= 0.08
    return best_tau


def _euler(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    actions: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pub = config.public
    dt = float(pub.dt)
    new_pos = robot_pos + robot_vel * dt
    new_vel = (1.0 - float(pub.damping)) * robot_vel + actions * (float(pub.drive_force) / float(pub.robot_mass)) * dt
    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    cap = float(pub.robot_max_speed)
    scale = np.ones_like(speed)
    fast = speed[:, 0] > cap
    scale[fast, 0] = cap / speed[fast, 0]
    new_vel = new_vel * scale
    new_target = target_pos + target_vel * dt
    half = float(pub.map_half_extent)
    new_target = np.clip(new_target, -half, half)
    return new_pos, new_vel, new_target, target_vel


def _cover_rate(robot_pos: np.ndarray, target_pos: np.ndarray, radius: float) -> float:
    n, m = len(robot_pos), len(target_pos)
    adj = np.zeros((n, m), dtype=np.int32)
    dist = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
    adj[dist <= radius] = 1
    matching = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
    return float(np.sum(np.asarray(matching) >= 0)) / float(m)


def _actions_from_aim(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    assignment: tuple[int, ...],
    tau_of: Callable[[int], float],
) -> np.ndarray:
    actions = np.zeros((len(robot_pos), 2), dtype=np.float64)
    for i, j in enumerate(assignment):
        delta = _lead_delta(target_pos[j] - robot_pos[i], target_vel[j], tau_of(i))
        actions[i] = _clip_aim(delta)
    return actions


def _mpc_action(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    assignment: tuple[int, ...],
    steps_left: int,
    config: Any,
) -> np.ndarray:
    radius = float(config.public.target_radius)
    n = len(robot_pos)
    intercept_taus = [
        _intercept_tau(
            target_pos[assignment[i]] - robot_pos[i],
            robot_vel[i],
            target_vel[assignment[i]],
            steps_left,
            config,
            radius,
        )
        for i in range(n)
    ]
    candidates: list[np.ndarray] = []
    for taus in product(TAUS, repeat=n):
        candidates.append(
            _actions_from_aim(
                robot_pos, target_pos, target_vel, assignment, lambda i, taus=taus: taus[i]
            )
        )
    candidates.append(
        _actions_from_aim(
            robot_pos,
            target_pos,
            target_vel,
            assignment,
            lambda i, intercept_taus=intercept_taus: intercept_taus[i],
        )
    )
    best_score = -1e18
    best = candidates[0]
    horizon = min(MPC_HORIZON, max(steps_left, 1))
    for first in candidates:
        rp, rv, tp, tv = robot_pos.copy(), robot_vel.copy(), target_pos.copy(), target_vel.copy()
        score = 0.0
        act = first
        for _ in range(horizon):
            rp, rv, tp, tv = _euler(rp, rv, tp, tv, act, config)
            score += _cover_rate(rp, tp, radius)
            act = _actions_from_aim(rp, tp, tv, assignment, lambda _i: 0.2)
        if score > best_score:
            best_score = score
            best = first
    return best


def _rollout(case: ScenarioCase, law: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    prev_target: np.ndarray | None = None
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for step in range(horizon):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            fd_vel = target_vel
            if prev_target is not None:
                fd_vel = (target_pos - prev_target) / float(config.public.dt)
            prev_target = target_pos.copy()
            steps_left = horizon - step
            actions_mat = np.zeros((n, 2), dtype=np.float64)
            if law == "pure":
                actions_mat = _actions_from_aim(robot_pos, target_pos, target_vel, chosen, lambda _i: 0.0)
            elif law.startswith("lead_0."):
                tau = float(law.split("_")[1])
                actions_mat = _actions_from_aim(robot_pos, target_pos, target_vel, chosen, lambda _i, tau=tau: tau)
            elif law == "lead_fd_0.2":
                actions_mat = _actions_from_aim(robot_pos, target_pos, fd_vel, chosen, lambda _i: 0.2)
            elif law == "intercept":
                taus = [
                    _intercept_tau(
                        target_pos[chosen[i]] - robot_pos[i],
                        robot_vel[i],
                        target_vel[chosen[i]],
                        steps_left,
                        config,
                        radius,
                    )
                    for i in range(n)
                ]
                actions_mat = _actions_from_aim(
                    robot_pos, target_pos, target_vel, chosen, lambda i, taus=taus: taus[i]
                )
            elif law == "mpc":
                actions_mat = _mpc_action(
                    robot_pos, robot_vel, target_pos, target_vel, chosen, steps_left, config
                )
            else:
                raise ValueError(law)
            actions = {f"agent_{i}": actions_mat[i].astype(np.float32) for i in range(n)}
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
    scores: list[float] = []
    for seed in seeds:
        basic = float(lookup[(seed, "basic")][f"{law}_j"])
        coop = float(lookup[(seed, "cooperation")][f"{law}_j"])
        scores.append(500.0 * (basic + coop))
    return scores


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
        }
    return {
        "oracle_warning": "提前量用了全局目标速度。不是合法策略，也不是 evaluate_one。",
        "assignment": "全部是可达性防抖，与 E034 的比例控制同一条分配",
        "pure": "a = clip(10*Δp,-1,1)",
        "lead": "a = clip(10*(Δp+τ*v_T),-1,1)，v_T 来自全局状态",
        "lead_fd": "τ=0.2，速度用相邻两帧绝对位置差",
        "intercept": "每个车选一个 τ，使常速度预测点能被乐观缩短量盖住",
        "mpc": "3 步常速度模型，在固定 τ 和截击动作里选第一步",
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
