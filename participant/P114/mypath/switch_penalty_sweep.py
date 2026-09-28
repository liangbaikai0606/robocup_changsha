"""E031：扫描逐机器人固定 switch penalty 的动态全局分配。

沿用 E030 的纯距离 3x3 一对一分配。若机器人 i 的候选目标 j 不等于上一步
分配，则给该边增加 beta。只扫描 beta，不加入趋势、可达性或碰撞代价。
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


BETAS = (0.0, 0.03, 0.05, 0.08)
LEAD_STEPS = 1.0
CSV_FIELDS = (
    "seed",
    "group",
    "layout",
    "beta",
    "total_return",
    "mean_j",
    "coverage",
    "collision",
    "assignment_switches",
    "robots_switched",
)


def _best_assignment(cost: np.ndarray) -> tuple[int, ...]:
    return min(
        permutations(range(cost.shape[1])),
        key=lambda assignment: sum(float(cost[i, j]) for i, j in enumerate(assignment)),
    )


def _rollout(seed: int, layout: str, beta: float) -> dict[str, Any]:
    group = "basic" if layout == "uniform" else "cooperation"
    config = _config_for_layout(layout)
    env = make_training_env(config)
    env.reset(seed=seed)
    dt = float(config.public.dt)
    damping = float(config.public.damping)
    cover_radius = float(config.public.target_radius)
    n = int(config.num_agents)
    m = int(config.num_targets)
    previous_assignment: tuple[int, ...] | None = None
    assignment_switches = 0
    robots_switched = 0
    total = 0.0
    coverages: list[float] = []
    collisions: list[float] = []
    try:
        for _step in range(config.horizon):
            state = env.state()
            robot_pos, _robot_vel, target_pos, _target_vel = _decode(state, n, m)
            cost = np.linalg.norm(robot_pos[:, None, :] - target_pos[None, :, :], axis=2)
            if previous_assignment is not None and beta > 0.0:
                for robot_index in range(n):
                    for target_index in range(m):
                        if target_index != previous_assignment[robot_index]:
                            cost[robot_index, target_index] += beta
            assignment = _best_assignment(cost)
            if previous_assignment is not None and assignment != previous_assignment:
                assignment_switches += 1
                robots_switched += sum(
                    int(old_target != new_target)
                    for old_target, new_target in zip(previous_assignment, assignment)
                )
            previous_assignment = assignment

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
        "beta": beta,
        "total_return": total,
        "mean_j": total / config.horizon,
        "coverage": float(np.mean(coverages)),
        "collision": float(np.mean(collisions)),
        "assignment_switches": assignment_switches,
        "robots_switched": robots_switched,
    }


def _run_task(task: tuple[int, str]) -> list[dict[str, Any]]:
    seed, layout = task
    return [_rollout(seed, layout, beta) for beta in BETAS]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (float(row["beta"]), int(row["seed"]), str(row["group"])))
        )
    temp.replace(path)


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    j = np.asarray([float(row["mean_j"]) for row in rows], dtype=np.float64)
    coverage = np.asarray([float(row["coverage"]) for row in rows], dtype=np.float64)
    collision = np.asarray([float(row["collision"]) for row in rows], dtype=np.float64)
    switches = np.asarray([float(row["assignment_switches"]) for row in rows], dtype=np.float64)
    robots = np.asarray([float(row["robots_switched"]) for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "mean_j": float(np.mean(j)),
        "mean_coverage": float(np.mean(coverage)),
        "mean_collision": float(np.mean(collision)),
        "mean_assignment_switches": float(np.mean(switches)),
        "mean_robots_switched": float(np.mean(robots)),
        "zero_switch_fraction": float(np.mean(switches == 0.0)),
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    score_vectors: dict[float, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    for beta in BETAS:
        selected = [row for row in rows if float(row["beta"]) == beta]
        row_map = {(int(row["seed"]), str(row["group"])): row for row in selected}
        scores = np.asarray(
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
        score_vectors[beta] = scores
        summaries[f"{beta:.2f}"] = {
            "performance_score": float(np.mean(scores)),
            "score_ci95_normal": _mean_ci(scores),
            "basic": _group_stats([row for row in selected if row["group"] == "basic"]),
            "cooperation": _group_stats(
                [row for row in selected if row["group"] == "cooperation"]
            ),
        }

    baseline = score_vectors[0.0]
    for beta in BETAS:
        delta = score_vectors[beta] - baseline
        target = summaries[f"{beta:.2f}"]
        target["delta_vs_beta0"] = float(np.mean(delta))
        target["delta_vs_beta0_ci95_normal"] = _mean_ci(delta)
        target["paired_better_tie_worse"] = {
            "better": int(np.sum(delta > 1e-12)),
            "tie": int(np.sum(np.abs(delta) <= 1e-12)),
            "worse": int(np.sum(delta < -1e-12)),
        }

    best_key = max(summaries, key=lambda key: float(summaries[key]["performance_score"]))
    return {
        "experiment_id": "E031",
        "description": "Per-robot fixed switch penalty in dynamic global distance assignment",
        "seed_source": "outputs/P114/random500-e025/seeds.json",
        "unique_scenario_seeds": len(seeds),
        "layouts_per_seed": 2,
        "betas": list(BETAS),
        "lead_steps_fixed": LEAD_STEPS,
        "total_oracle_episodes": len(rows),
        "assignment": "minimum-cost one-to-one permutation recalculated every step",
        "penalty": "add beta to edge (robot i, target j) when j differs from i's previous target",
        "excluded_features": ["distance trend", "reachability filter", "collision cost"],
        "reference_scores": {
            "E024_random500": 165.17333333333332,
            "E025_fixed_assignment_oracle": 167.54,
            "E030_dynamic_distance": 147.48666666666668,
        },
        "best_beta": float(best_key),
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
        default=REPO_ROOT / "outputs" / "P114" / "random500-e031-switch-penalty",
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
            "delta": value["delta_vs_beta0"],
            "basic_switches": value["basic"]["mean_assignment_switches"],
            "coop_switches": value["cooperation"]["mean_assignment_switches"],
            "paired": value["paired_better_tie_worse"],
        }
        for key, value in summary["results"].items()
    }
    print(json.dumps({"best_beta": summary["best_beta"], "results": compact}, indent=2))


if __name__ == "__main__":
    main()
