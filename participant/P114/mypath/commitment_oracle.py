"""切换成本跟着连续追踪时长变，旧目标判死就解锁。

β(A) = β0 + α * A。β0 固定为 0.03，只扫 α。
分配仍是可达性防抖，开车仍是 clip(20*Δp + 4*Δv)。
启动晚沿用四桶拆细里的同一条分类，只统计赶路桶里的 LATE_START。
读取全局状态，不是合法策略。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
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
from coverage_bench.suites import ScenarioCase
from e039_bucket_refine import _classify_travel
from loss_attribution import NEAR_MARGIN, _covered_targets, _start_reachable
from oracle_search import _decode
from pd_oracle import _pd_action
from reachability_debounce_oracle import (
    BETA,
    M_UNREACHABLE,
    MASTER_SEED,
    _layout_config,
    _mean_ci,
    _optimistic_closure,
    _random_seeds,
)


ALPHAS = (0.0, 0.01, 0.02, 0.04)
KP = 20.0
KD = 4.0


class _CommitmentAssigner:
    """防抖分配。离开仍可达的旧目标要付 β0 + α*连续步数；旧目标已不可达则不付。"""

    def __init__(self, config: Any, alpha: float) -> None:
        self.config = config
        self.alpha = float(alpha)
        self.n = int(config.num_agents)
        self.m = int(config.num_targets)
        self.horizon = int(config.horizon)
        self.cover_radius = float(config.public.target_radius)
        self.assignments = list(permutations(range(self.m)))
        self.history: list[np.ndarray] = []
        self.physical_bad = np.zeros((self.n, self.m), dtype=np.int64)
        self.trend_bad = np.zeros((self.n, self.m), dtype=np.int64)
        self.previous: tuple[int, ...] | None = None
        self.age = np.zeros(self.n, dtype=np.int64)
        self.switches = 0

    def choose(self, state: np.ndarray, step_index: int) -> tuple[int, ...]:
        robot_pos, robot_vel, target_pos, _target_vel = _decode(state, self.n, self.m)
        delta = target_pos[None, :, :] - robot_pos[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        self.history.append(distance.copy())
        del self.history[:-3]
        steps_left = self.horizon - step_index
        unreachable = np.zeros((self.n, self.m), dtype=np.bool_)
        if len(self.history) >= 3:
            closing = (self.history[-3] - self.history[-1]) / 2.0
            for robot_index in range(self.n):
                for target_index in range(self.m):
                    gap = max(0.0, float(distance[robot_index, target_index]) - self.cover_radius)
                    optimistic = _optimistic_closure(
                        delta[robot_index, target_index], robot_vel[robot_index], steps_left, self.config
                    )
                    expected = steps_left * max(float(closing[robot_index, target_index]), 0.0)
                    physical_bad = optimistic + 0.08 < gap
                    trend_bad = (
                        steps_left <= 4
                        and expected + 0.06 < gap
                        and float(closing[robot_index, target_index]) <= 0.0
                    )
                    self.physical_bad[robot_index, target_index] = (
                        self.physical_bad[robot_index, target_index] + 1 if physical_bad else 0
                    )
                    self.trend_bad[robot_index, target_index] = (
                        self.trend_bad[robot_index, target_index] + 1 if trend_bad else 0
                    )
                    unreachable[robot_index, target_index] = bool(
                        self.physical_bad[robot_index, target_index] >= 2
                        or self.trend_bad[robot_index, target_index] >= 2
                    )
        pair_cost = np.where(unreachable, M_UNREACHABLE, distance)
        best_assignment = self.assignments[0]
        best_key = (float("inf"), self.assignments[0])
        for assignment in self.assignments:
            base = sum(float(pair_cost[robot_index, assignment[robot_index]]) for robot_index in range(self.n))
            penalty = 0.0
            if self.previous is not None:
                for robot_index in range(self.n):
                    if assignment[robot_index] == self.previous[robot_index]:
                        continue
                    if unreachable[robot_index, self.previous[robot_index]]:
                        continue
                    penalty += BETA + self.alpha * float(self.age[robot_index])
            key = (base + penalty, assignment)
            if key < best_key:
                best_key = key
                best_assignment = assignment
        if self.previous is None:
            self.age[:] = 1
        else:
            changed = 0
            for robot_index in range(self.n):
                if best_assignment[robot_index] == self.previous[robot_index]:
                    self.age[robot_index] += 1
                else:
                    self.age[robot_index] = 1
                    changed += 1
            self.switches += changed
        self.previous = best_assignment
        return best_assignment


def _episode_clean(case: ScenarioCase, alpha: float) -> dict[str, float]:
    """开局重新选目标。连续追踪步数从这一局的第一次分配开始算。"""
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _CommitmentAssigner(config, alpha)
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    horizon = int(config.horizon)
    sense = float(config.public.sense_radius)
    robot_pos, _robot_vel, target_pos, _target_vel = _decode(env.state(), n, m)
    start_reachable = _start_reachable(env.state(), config)
    start_visible = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2) <= sense
    total = 0.0
    late_share = 0.0
    switch_share = 0.0
    assignments: list[np.ndarray] = []
    try:
        for step in range(horizon):
            state = env.state()
            chosen = assigner.choose(state, step)
            assignments.append(np.asarray(chosen, dtype=np.int64))
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            actions = {
                f"agent_{robot_index}": _pd_action(
                    target_pos[chosen[robot_index]] - robot_pos[robot_index],
                    target_vel[chosen[robot_index]] - robot_vel[robot_index],
                    KP,
                    KD,
                )
                for robot_index in range(n)
            }
            _observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            robot_pos, robot_vel, target_pos, target_vel = _decode(env.state(), n, m)
            distance = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
            inside = distance <= radius
            matching = _covered_targets(robot_pos, target_pos, radius)
            for target_index, robot_index in enumerate(matching):
                if robot_index >= 0 or bool(np.any(inside[:, target_index])):
                    continue
                if bool(np.any(np.sum(inside, axis=0) >= 2)):
                    continue
                if float(np.min(distance[:, target_index])) <= radius + NEAR_MARGIN:
                    continue
                if not bool(start_reachable[target_index]):
                    continue
                cause, _meta = _classify_travel(
                    robot_pos,
                    robot_vel,
                    target_pos,
                    target_vel,
                    target_index,
                    assignments,
                    step,
                    assignments[0],
                    start_visible,
                    horizon - step - 1,
                    config,
                    radius,
                )
                if cause == "LATE_START":
                    late_share += 1.0 / m
                elif cause == "TARGET_SWITCH":
                    switch_share += 1.0 / m
    finally:
        env.close()
    return {
        "j": total / horizon,
        "switches": float(assigner.switches),
        "late_j": late_share / horizon,
        "switch_loss_j": switch_share / horizon,
    }


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "alpha": alpha,
                **_episode_clean(case, alpha),
            }
        )
    return rows


def _lookup(rows: list[dict[str, Any]], alpha: float) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["seed"]), str(row["group"])): row for row in rows if row["alpha"] == alpha}


def _paired(rows: list[dict[str, Any]], seeds: list[int], alpha: float, field: str) -> list[float]:
    chosen = _lookup(rows, alpha)
    values: list[float] = []
    for seed in seeds:
        basic = float(chosen[(seed, "basic")][field])
        cooperation = float(chosen[(seed, "cooperation")][field])
        if field == "j":
            values.append(500.0 * (basic + cooperation))
        elif field == "switches":
            values.append(0.5 * (basic + cooperation))
        else:
            values.append(500.0 * (basic + cooperation))
    return values


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    baseline_score = _paired(rows, seeds, 0.0, "j")
    variants: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        scores = _paired(rows, seeds, alpha, "j")
        switches = _paired(rows, seeds, alpha, "switches")
        late = _paired(rows, seeds, alpha, "late_j")
        switch_loss = _paired(rows, seeds, alpha, "switch_loss_j")
        gaps = [new - old for old, new in zip(baseline_score, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        variants.append(
            {
                "alpha": alpha,
                "beta0": BETA,
                "score": float(np.mean(scores)),
                "gap_vs_alpha0": float(np.mean(gaps)),
                "gap_ci95_normal": _mean_ci(gaps),
                "wins": wins,
                "ties": len(gaps) - wins - losses,
                "losses": losses,
                "switches_per_episode": float(np.mean(switches)),
                "late_start_loss": float(np.mean(late)),
                "target_switch_loss": float(np.mean(switch_loss)),
            }
        )
    return {
        "oracle_warning": "全局分配和全局目标速度。不是 evaluate_one，也没有改 entry.py。",
        "rule": "旧目标仍可达时，换它要付 0.03 + α*连续追踪步数；旧目标已不可达则解锁。开车是 clip(20*Δp+4*Δv)。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "variants": variants,
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
