"""E030：在 E025 的 500 个配对种子上扫描 distance + lambda * trend。

动态全局 Oracle 每步构造 3x3 代价矩阵并枚举一对一分配。trend 使用两步
平均距离变化；第 0 步为 0，第 1 步退化为单步变化。动作控制统一使用固定
1 步提前量，实验中不加入切换惩罚、可达性过滤或碰撞代价。
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
MYPATH_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MYPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MYPATH_ROOT))

from coverage_bench.envs.factory import make_training_env
from oracle_search import _actions_for_assignment, _decode
from random500_compare import _config_for_layout, _mean_ci


LAMBDAS = (0.0, 1.0, 2.0, 3.0, 4.0)
LEAD_STEPS = 1.0
CSV_FIELDS = (
    "seed",
    "group",
    "layout",
    "lambda",
    "total_return",
    "mean_j",
    "coverage",
    "collision",
    "assignment_switches",
)


def _best_assignment(cost: np.ndarray) -> tuple[int, ...]:
    candidates = permutations(range(cost.shape[1]))
    return min(candidates, key=lambda assignment: sum(float(cost[i, j]) for i, j in enumerate(assignment)))


def _rollout(seed: int, layout: str, trend_weight: float) -> dict[str, Any]:
    group = "basic" if layout == "uniform" else "cooperation"
    config = _config_for_layout(layout)
    env = make_training_env(config)
    env.reset(seed=seed)
    dt = float(config.public.dt)
    damping = float(config.public.damping)
    cover_radius = float(config.public.target_radius)
    n = int(config.num_agents)
    m = int(config.num_targets)
    distance_history: list[np.ndarray] = []
    previous_assignment: tuple[int, ...] | None = None
    switches = 0
    total = 0.0
    coverages: list[float] = []
    collisions: list[float] = []
    try:
        for _step in range(config.horizon):
            state = env.state()
            robot_pos, _robot_vel, target_pos, _target_vel = _decode(state, n, m)
            distances = np.linalg.norm(robot_pos[:, None, :] - target_pos[None, :, :], axis=2)
            if len(distance_history) >= 2:
                trend = (distances - distance_history[-2]) / 2.0
            elif distance_history:
                trend = distances - distance_history[-1]
            else:
                trend = np.zeros_like(distances)
            assignment = _best_assignment(distances + trend_weight * trend)
            if previous_assignment is not None and assignment != previous_assignment:
                switches += 1
            previous_assignment = assignment
            distance_history.append(distances)
            if len(distance_history) > 2:
                distance_history.pop(0)

            actions = _actions_for_assignment(
                state,
                assignment,
                n,
                m,
                dt,
                damping,
                LEAD_STEPS,
                cover_radius,
            )
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent_id = next(iter(rewards))
            total += float(rewards[agent_id])
            metrics = infos[agent_id]["metrics"]
            coverages.append(float(metrics.coverage_rate))
            collisions.append(float(metrics.collision_rate))
    finally:
        env.close()

    return {
        "seed": seed,
        "group": group,
        "layout": layout,
        "lambda": trend_weight,
        "total_return": total,
        "mean_j": total / config.horizon,
        "coverage": float(np.mean(coverages)),
        "collision": float(np.mean(collisions)),
        "assignment_switches": switches,
    }


def _run_task(task: tuple[int, str]) -> list[dict[str, Any]]:
    seed, layout = task
    return [_rollout(seed, layout, trend_weight) for trend_weight in LAMBDAS]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (float(row["lambda"]), int(row["seed"]), str(row["group"])))
        )
    temp.replace(path)


def _metric_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    j = np.asarray([float(row["mean_j"]) for row in rows], dtype=np.float64)
    coverage = np.asarray([float(row["coverage"]) for row in rows], dtype=np.float64)
    collision = np.asarray([float(row["collision"]) for row in rows], dtype=np.float64)
    switches = np.asarray([float(row["assignment_switches"]) for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "mean_j": float(np.mean(j)),
        "mean_j_ci95_normal": _mean_ci(j),
        "mean_coverage": float(np.mean(coverage)),
        "mean_collision": float(np.mean(collision)),
        "mean_assignment_switches": float(np.mean(switches)),
        "zero_switch_fraction": float(np.mean(switches == 0.0)),
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for trend_weight in LAMBDAS:
        selected = [row for row in rows if float(row["lambda"]) == trend_weight]
        by_group = {
            group: [row for row in selected if row["group"] == group]
            for group in ("basic", "cooperation")
        }
        row_map = {(int(row["seed"]), str(row["group"])): row for row in selected}
        paired_scores = np.asarray(
            [
                500.0
                * (
                    float(row_map[(seed, "basic")]["mean_j"])
                    + float(row_map[(seed, "cooperation")]["mean_j"])
                )
                for seed in seeds
            ],
            dtype=np.float64,
        )
        summaries[str(int(trend_weight))] = {
            "performance_score": float(np.mean(paired_scores)),
            "score_ci95_normal": _mean_ci(paired_scores),
            "basic": _metric_stats(by_group["basic"]),
            "cooperation": _metric_stats(by_group["cooperation"]),
        }

    best_key = max(summaries, key=lambda key: float(summaries[key]["performance_score"]))
    return {
        "experiment_id": "E030",
        "description": "Dynamic global assignment by distance + lambda * two-step distance trend",
        "seed_source": "outputs/P114/random500-e025/seeds.json",
        "unique_scenario_seeds": len(seeds),
        "layouts_per_seed": 2,
        "lambdas": list(LAMBDAS),
        "lead_steps_fixed": LEAD_STEPS,
        "total_oracle_episodes": len(rows),
        "assignment": "minimum-cost one-to-one permutation recalculated every step",
        "trend": "(d_t - d_t-2) / 2; one-step fallback at t=1; zero at t=0",
        "excluded_features": ["switch penalty", "reachability filter", "collision cost"],
        "reference_scores": {
            "E024_random500": 165.17333333333332,
            "E025_fixed_assignment_oracle": 167.54,
        },
        "best_lambda": float(best_key),
        "best_performance_score": float(summaries[best_key]["performance_score"]),
        "results": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=Path,
        default=REPO_ROOT / "outputs" / "P114" / "random500-e025" / "seeds.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "P114" / "random500-e030-trend-oracle",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    seed_payload = json.loads(args.seeds.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in seed_payload["seeds"]]
    if len(seeds) != len(set(seeds)):
        parser.error("seed list contains duplicates")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "episodes.csv"
    tasks = [(seed, layout) for seed in seeds for layout in ("uniform", "crossing")]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 20 == 0 or completed == len(tasks):
                _write_rows(rows_path, rows)
                print(f"completed {completed}/{len(tasks)} seed-layout tasks", flush=True)

    summary = _summarize(rows, seeds)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        key: {
            "score": value["performance_score"],
            "basic_switches": value["basic"]["mean_assignment_switches"],
            "coop_switches": value["cooperation"]["mean_assignment_switches"],
        }
        for key, value in summary["results"].items()
    }
    print(json.dumps({"best_lambda": summary["best_lambda"], "results": compact}, indent=2))


if __name__ == "__main__":
    main()
