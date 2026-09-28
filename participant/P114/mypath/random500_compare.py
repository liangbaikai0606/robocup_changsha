"""E025：在 500 个随机种子上比较 E024 与固定分配全局 Oracle。

每个种子分别生成 uniform（basic）和 crossing（cooperation）场景，因此总共
运行 1000 个 E024 回合。Oracle 对每个场景枚举 3! 种固定分配和 1～5 步
提前量，使用官方环境实跑后取最高回报。Oracle 读取 env.state()，只用于诊断。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
P114_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(P114_ROOT) not in sys.path:
    sys.path.insert(0, str(P114_ROOT))

from coverage_bench.config import TaskConfig, load_task_config
from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import EpisodeContext, PublicTaskParams
from coverage_bench.suites import ScenarioCase
from entry import CoverFirstPolicy
from oracle_search import _rollout as oracle_rollout


DEFAULT_MASTER_SEED = 20260927
DEFAULT_COUNT = 500
LEADS = (1.0, 2.0, 3.0, 4.0, 5.0)
PUBLIC_SEEDS = {1001, 1002, 2001, 2002}
CSV_FIELDS = (
    "seed",
    "group",
    "layout",
    "e024_return",
    "e024_j",
    "e024_coverage",
    "e024_collision",
    "e024_boost_starts",
    "oracle_return",
    "oracle_j",
    "oracle_coverage",
    "oracle_collision",
    "gap_j",
    "oracle_assignment",
    "oracle_lead_steps",
)


@lru_cache(maxsize=2)
def _config_for_layout(layout: str) -> TaskConfig:
    base = load_task_config(REPO_ROOT / "configs" / "task-v1.yaml")
    scenario = base.scenario.model_copy(update={"layout_kind": layout})
    return base.model_copy(update={"scenario": scenario})


def _public_params(config: TaskConfig) -> PublicTaskParams:
    task = config.public
    return PublicTaskParams(
        map_half_extent=task.map_half_extent,
        dt=task.dt,
        robot_radius=task.robot_radius,
        robot_mass=task.robot_mass,
        drive_force=task.drive_force,
        damping=task.damping,
        robot_max_speed=task.robot_max_speed,
        contact_force=task.contact_force,
        contact_margin=task.contact_margin,
        target_radius=task.target_radius,
        target_max_speed=task.target_max_speed,
        sense_radius=task.sense_radius,
        motion_kind=task.motion_kind,
        turn_interval_steps=tuple(task.turn_interval_steps),
        target_speed_fraction=tuple(task.target_speed_fraction),
        robot_boundary=task.robot_boundary,
        target_boundary=task.target_boundary,
    )


def _run_e024(config: TaskConfig, seed: int) -> tuple[float, float, float, float, int]:
    env = make_training_env(config)
    observations, _infos = env.reset(seed=seed)
    policies = [CoverFirstPolicy() for _ in range(config.num_agents)]
    task = _public_params(config)
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=config.num_agents,
                num_targets=config.num_targets,
                horizon=config.horizon,
                task=task,
                policy_seed=(seed + index) % (2**64),
            )
        )

    total = 0.0
    coverages: list[float] = []
    collisions: list[float] = []
    boost_starts = 0
    try:
        for _step in range(config.horizon):
            actions: dict[str, np.ndarray] = {}
            for index, agent_id in enumerate(env.agents):
                before = policies[index]._commit is not None
                actions[agent_id] = policies[index].act(observations[agent_id])
                after = policies[index]._commit is not None
                boost_starts += int(not before and after)
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent_id = next(iter(rewards))
            total += float(rewards[agent_id])
            metrics = infos[agent_id]["metrics"]
            coverages.append(float(metrics.coverage_rate))
            collisions.append(float(metrics.collision_rate))
    finally:
        for policy in policies:
            policy.close()
        env.close()

    return (
        total,
        total / config.horizon,
        float(np.mean(coverages)),
        float(np.mean(collisions)),
        boost_starts,
    )


def _best_oracle(config: TaskConfig, seed: int, group: str) -> tuple[float, float, float, float, str, float]:
    case = ScenarioCase(
        case_id=f"random-{group}-{seed}",
        group_id=group,
        task_config=config,
        scenario_seed=seed,
    )
    best: tuple[float, float, float, float, str, float] | None = None
    for assignment in permutations(range(config.num_targets)):
        for lead in LEADS:
            total, mean_j, coverage, collision = oracle_rollout(case, assignment, lead)
            candidate = (total, mean_j, coverage, collision, "-".join(map(str, assignment)), lead)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best


def _run_pair(task: tuple[int, str]) -> dict[str, Any]:
    seed, layout = task
    group = "basic" if layout == "uniform" else "cooperation"
    config = _config_for_layout(layout)
    e024 = _run_e024(config, seed)
    oracle = _best_oracle(config, seed, group)
    return {
        "seed": seed,
        "group": group,
        "layout": layout,
        "e024_return": e024[0],
        "e024_j": e024[1],
        "e024_coverage": e024[2],
        "e024_collision": e024[3],
        "e024_boost_starts": e024[4],
        "oracle_return": oracle[0],
        "oracle_j": oracle[1],
        "oracle_coverage": oracle[2],
        "oracle_collision": oracle[3],
        "gap_j": oracle[1] - e024[1],
        "oracle_assignment": oracle[4],
        "oracle_lead_steps": oracle[5],
    }


def _generate_seeds(count: int, master_seed: int) -> list[int]:
    rng = random.Random(master_seed)
    seeds: list[int] = []
    used = set(PUBLIC_SEEDS)
    while len(seeds) < count:
        seed = rng.getrandbits(64)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row["seed"]), str(row["group"]))))
    temp.replace(path)


def _mean_ci(values: np.ndarray) -> list[float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return [mean, mean]
    half = 1.96 * float(np.std(values, ddof=1)) / float(np.sqrt(len(values)))
    return [mean - half, mean + half]


def _stats(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    j = np.asarray([float(row[f"{prefix}_j"]) for row in rows], dtype=np.float64)
    coverage = np.asarray([float(row[f"{prefix}_coverage"]) for row in rows], dtype=np.float64)
    collision = np.asarray([float(row[f"{prefix}_collision"]) for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "mean_j": float(np.mean(j)),
        "score_scale": 1000.0 * float(np.mean(j)),
        "mean_j_ci95_normal": _mean_ci(j),
        "mean_coverage": float(np.mean(coverage)),
        "mean_collision": float(np.mean(collision)),
        "j_quantiles": {
            "p05": float(np.quantile(j, 0.05)),
            "p25": float(np.quantile(j, 0.25)),
            "p50": float(np.quantile(j, 0.50)),
            "p75": float(np.quantile(j, 0.75)),
            "p95": float(np.quantile(j, 0.95)),
        },
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int], master_seed: int) -> dict[str, Any]:
    by_group = {
        group: [row for row in rows if row["group"] == group]
        for group in ("basic", "cooperation")
    }
    group_summary: dict[str, Any] = {}
    for group, group_rows in by_group.items():
        gaps = np.asarray([float(row["gap_j"]) for row in group_rows], dtype=np.float64)
        group_summary[group] = {
            "e024": _stats(group_rows, "e024"),
            "oracle": _stats(group_rows, "oracle"),
            "oracle_minus_e024_mean_j": float(np.mean(gaps)),
            "oracle_minus_e024_ci95_normal": _mean_ci(gaps),
            "e024_beats_oracle_fraction": float(np.mean(gaps < -1e-12)),
            "ties_fraction": float(np.mean(np.abs(gaps) <= 1e-12)),
        }

    paired_seed_scores: list[tuple[float, float]] = []
    rows_by_key = {(int(row["seed"]), str(row["group"])): row for row in rows}
    for seed in seeds:
        basic = rows_by_key[(seed, "basic")]
        cooperation = rows_by_key[(seed, "cooperation")]
        e024_score = 500.0 * (float(basic["e024_j"]) + float(cooperation["e024_j"]))
        oracle_score = 500.0 * (float(basic["oracle_j"]) + float(cooperation["oracle_j"]))
        paired_seed_scores.append((e024_score, oracle_score))
    paired = np.asarray(paired_seed_scores, dtype=np.float64)
    gap = paired[:, 1] - paired[:, 0]
    boost_starts = int(sum(int(row["e024_boost_starts"]) for row in rows))

    return {
        "experiment_id": "E025",
        "description": "E024 versus fixed-assignment privileged Oracle on paired random seeds",
        "master_seed": master_seed,
        "unique_scenario_seeds": len(seeds),
        "layouts_per_seed": 2,
        "total_e024_episodes": len(rows),
        "total_oracle_candidate_episodes": len(rows) * 30,
        "public_seeds_excluded": sorted(PUBLIC_SEEDS),
        "oracle_definition": {
            "global_state": True,
            "assignments": 6,
            "lead_steps": list(LEADS),
            "selection": "best realized return per seed and layout",
            "warning": "Diagnostic comparator, not a legal policy or mathematical upper bound.",
        },
        "overall": {
            "e024_performance_score": float(np.mean(paired[:, 0])),
            "e024_score_ci95_normal": _mean_ci(paired[:, 0]),
            "oracle_performance_score": float(np.mean(paired[:, 1])),
            "oracle_score_ci95_normal": _mean_ci(paired[:, 1]),
            "oracle_minus_e024_score": float(np.mean(gap)),
            "paired_gap_ci95_normal": _mean_ci(gap),
            "e024_score_as_oracle_fraction": (
                float(np.mean(paired[:, 0])) / float(np.mean(paired[:, 1]))
                if float(np.mean(paired[:, 1])) != 0.0
                else None
            ),
            "e024_boost_starts": boost_starts,
        },
        "groups": group_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "P114" / "random500-e025",
    )
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("--count and --workers must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "episodes.csv"
    seeds = _generate_seeds(args.count, args.master_seed)
    (output / "seeds.json").write_text(
        json.dumps({"master_seed": args.master_seed, "seeds": seeds}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tasks = [(seed, layout) for seed in seeds for layout in ("uniform", "crossing")]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_pair, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 10 == 0 or completed == len(tasks):
                _write_rows(rows_path, rows)
                print(f"completed {completed}/{len(tasks)}", flush=True)

    summary = _summarize(rows, seeds, args.master_seed)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
