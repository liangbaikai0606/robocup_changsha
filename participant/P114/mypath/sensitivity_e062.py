"""开发集上看上场参数是一片平地，还是一个尖点。

对照是当前 entry.py：Kp=20，Kd=4，让圈门槛 1。
门槛写在代码里是 peer_time + margin < my_time。
margin=1 时，整数步数上队友至少要快 2 步才让。

只在主种子 20261003 上拨开这些数。
格子里的最高分不写进 entry.py：候选一多，最高的那格会偏高。
"""

from __future__ import annotations

import argparse
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

import entry as entry_module
from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase, load_suite
from entry import CoverFirstPolicy
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds


BASE = (20.0, 4.0, 1)
CELLS: tuple[tuple[float, float, int], ...] = (
    (20.0, 4.0, 1),
    (10.0, 4.0, 1),
    (15.0, 4.0, 1),
    (30.0, 4.0, 1),
    (40.0, 4.0, 1),
    (20.0, 0.0, 1),
    (20.0, 2.0, 1),
    (20.0, 8.0, 1),
    (20.0, 12.0, 1),
    (10.0, 0.0, 1),
    (10.0, 12.0, 1),
    (40.0, 0.0, 1),
    (40.0, 12.0, 1),
    (20.0, 4.0, 0),
    (20.0, 4.0, 2),
    (20.0, 4.0, 3),
)


def _cell_name(kp: float, kd: float, margin: int) -> str:
    return f"k{int(kp)}_d{int(kd)}_m{margin}"


class GainPolicy(CoverFirstPolicy):
    """选圈仍是当前规则，只改位置增益、速度增益和让圈门槛。"""

    def __init__(self, kp: float, kd: float) -> None:
        super().__init__()
        self.kp = float(kp)
        self.kd = float(kd)

    def _thrust_to_target(
        self,
        relative: np.ndarray,
        self_velocity: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        offset = np.asarray(relative, dtype=np.float64)
        velocity = np.asarray(self_velocity, dtype=np.float64)
        target_speed = np.asarray(target_velocity, dtype=np.float64)
        command = self.kp * offset + self.kd * (target_speed - velocity)
        return np.clip(command, -1.0, 1.0).astype(np.float32)


def _rollout(case: ScenarioCase, kp: float, kd: float, margin: int) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    robots = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    old_margin = int(entry_module._YIELD_MARGIN)
    entry_module._YIELD_MARGIN = int(margin)
    policies = [GainPolicy(kp, kd) for _robot in range(robots)]
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=robots,
                num_targets=int(config.num_targets),
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
        entry_module._YIELD_MARGIN = old_margin
        env.close()
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
    }


def _evaluate(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id, "group": case.group_id, "seed": int(case.scenario_seed)}
    for kp, kd, margin in CELLS:
        result = _rollout(case, kp, kd, margin)
        name = _cell_name(kp, kd, margin)
        for key, value in result.items():
            row[f"{name}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], name: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{name}_j"]) + float(lookup[(seed, "cooperation")][f"{name}_j"]))
        for seed in seeds
    ]


def _cell_summary(scores: list[float], base: list[float]) -> dict[str, Any]:
    gaps = [new - old for old, new in zip(base, scores)]
    wins = sum(gap > 1e-9 for gap in gaps)
    losses = sum(gap < -1e-9 for gap in gaps)
    arr = np.asarray(scores, dtype=np.float64)
    p10, p50, p90 = (float(value) for value in np.percentile(arr, [10, 50, 90]))
    ci = _mean_ci(gaps)
    return {
        "score": float(np.mean(arr)),
        "gap_vs_base": float(np.mean(gaps)),
        "gap_ci95_normal": ci,
        "ci_width": float(ci[1] - ci[0]),
        "ci_crosses_zero": bool(ci[0] <= 0.0 <= ci[1]),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "minimum": float(np.min(arr)),
        "below_100": int(np.sum(arr < 100.0)),
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base_name = _cell_name(*BASE)
    base_scores = _paired(rows, seeds, base_name)
    cells: dict[str, Any] = {}
    for kp, kd, margin in CELLS:
        name = _cell_name(kp, kd, margin)
        summary = _cell_summary(_paired(rows, seeds, name), base_scores)
        summary["kp"] = kp
        summary["kd"] = kd
        summary["yield_margin"] = margin
        summary["coverage"] = float(np.mean([float(row[f"{name}_coverage"]) for row in rows]))
        summary["collision"] = float(np.mean([float(row[f"{name}_collision"]) for row in rows]))
        cells[name] = summary
    return {
        "note": "开发集 20261003。对照是 Kp=20、Kd=4、让圈门槛 1。不从格子里挑最高分。不是 evaluate_one。未改 entry.py。",
        "yield_rule": "peer_time + margin < my_time。margin=1 表示整数上至少快 2 步。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "base": base_name,
        "cells": cells,
    }


def _public_scores() -> dict[str, float]:
    suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    scores: dict[str, float] = {}
    for kp, kd, margin in CELLS:
        grouped: dict[str, list[float]] = {}
        for group in suite.groups:
            for case in group.cases:
                grouped.setdefault(case.group_id, []).append(_rollout(case, kp, kd, margin)["j"])
        name = _cell_name(kp, kd, margin)
        scores[name] = 500.0 * (float(np.mean(grouped["basic"])) + float(np.mean(grouped["cooperation"])))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("--count 和 --workers 必须为正")
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
        futures = [executor.submit(_evaluate, case) for case in cases]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)
    summary = _summarize(rows, seeds)
    summary["public_four"] = _public_scores()
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
