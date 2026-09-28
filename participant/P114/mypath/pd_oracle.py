"""PD 全局 Oracle：对照比例控制 clip(10 * Δp)。

读取 env.state()，只用于诊断，不是合法提交策略。
每个场景在 6 种固定分配里取回报最高的一种。提前量固定为 0，
这样和「位置误差乘 10 再限幅」只差控制律本身。

控制律（x、y 分别限幅）：

    a = clip(Kp * (p_target - p_robot) + Kd * (v_target - v_robot), -1, 1)

Kp=10、Kd=0 就是当前远处的比例控制。参数只在调参半区上选择，
隔离半区不再改参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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

from coverage_bench.config import TaskConfig, load_task_config
from coverage_bench.envs.factory import make_training_env
from coverage_bench.suites import ScenarioCase, load_suite
from oracle_search import _decode


MASTER_SEED = 20261002
PUBLIC_SEEDS = {1001, 1002, 2001, 2002}
BASELINE = (10.0, 0.0)
GAINS = tuple(
    (kp, kd)
    for kp in (4.0, 10.0, 20.0)
    for kd in (0.0, 1.0, 2.0, 4.0, 8.0)
)


def _pd_action(delta_p: np.ndarray, delta_v: np.ndarray, kp: float, kd: float) -> np.ndarray:
    """位置误差的比例项加相对速度的微分项，再按分量限幅。"""
    command = kp * delta_p + kd * delta_v
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def _rollout(
    env: Any,
    seed: int,
    assignment: tuple[int, ...],
    kp: float,
    kd: float,
    horizon: int,
    n: int,
    m: int,
) -> tuple[float, float, float]:
    """跑完一局，返回总回报、平均覆盖、平均碰撞。"""
    env.reset(seed=seed)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    for _ in range(horizon):
        robot_pos, robot_vel, target_pos, target_vel = _decode(env.state(), n, m)
        actions: dict[str, np.ndarray] = {}
        for robot_index, target_index in enumerate(assignment):
            delta_p = target_pos[target_index] - robot_pos[robot_index]
            delta_v = target_vel[target_index] - robot_vel[robot_index]
            actions[f"agent_{robot_index}"] = _pd_action(delta_p, delta_v, kp, kd)
        _obs, rewards, _terms, _truncs, infos = env.step(actions)
        agent = next(iter(rewards))
        total += float(rewards[agent])
        metrics = infos[agent]["metrics"]
        coverage.append(float(metrics.coverage_rate))
        collision.append(float(metrics.collision_rate))
    return total, float(np.mean(coverage)), float(np.mean(collision))


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    """一个场景上，每种增益取 6 种固定分配里回报最高的。"""
    config = case.task_config
    env = make_training_env(config)
    n = int(config.num_agents)
    m = int(config.num_targets)
    horizon = int(config.horizon)
    assignments = list(permutations(range(m)))
    rows: list[dict[str, Any]] = []
    try:
        for kp, kd in GAINS:
            best: tuple[float, float, float, tuple[int, ...]] | None = None
            for assignment in assignments:
                total, coverage, collision = _rollout(
                    env, int(case.scenario_seed), assignment, kp, kd, horizon, n, m
                )
                if best is None or total > best[0]:
                    best = (total, coverage, collision, assignment)
            assert best is not None
            rows.append(
                {
                    "case_id": case.case_id,
                    "group": case.group_id,
                    "seed": int(case.scenario_seed),
                    "kp": kp,
                    "kd": kd,
                    "return": best[0],
                    "j": best[0] / horizon,
                    "coverage": best[1],
                    "collision": best[2],
                    "assignment": str(best[3]),
                }
            )
    finally:
        env.close()
    return rows


def _layout_config(layout: str) -> TaskConfig:
    base = load_task_config(REPO_ROOT / "configs" / "task-v1.yaml")
    scenario = base.scenario.model_copy(update={"layout_kind": layout})
    return base.model_copy(update={"scenario": scenario})


def _random_seeds(count: int, master_seed: int) -> list[int]:
    rng = random.Random(master_seed)
    seeds: list[int] = []
    used = set(PUBLIC_SEEDS)
    while len(seeds) < count:
        seed = rng.getrandbits(64)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


def _paired_scores(rows: list[dict[str, Any]], seeds: list[int], kp: float, kd: float) -> list[float]:
    """每个种子的两组等权分：500 * (J_basic + J_cooperation)。"""
    chosen = [row for row in rows if row["kp"] == kp and row["kd"] == kd and str(row["case_id"]).startswith("random-")]
    lookup = {(int(row["seed"]), str(row["group"])): row for row in chosen}
    scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        scores.append(500.0 * (float(basic["j"]) + float(coop["j"])))
    return scores


def _mean_ci(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    if len(arr) < 2:
        return [mean, mean]
    half = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return [mean - half, mean + half]


def _public_score(rows: list[dict[str, Any]], kp: float, kd: float) -> float:
    chosen = [
        row
        for row in rows
        if row["kp"] == kp and row["kd"] == kd and not str(row["case_id"]).startswith("random-")
    ]
    groups = {
        group: [float(row["j"]) for row in chosen if row["group"] == group]
        for group in ("basic", "cooperation")
    }
    return 500.0 * (float(np.mean(groups["basic"])) + float(np.mean(groups["cooperation"])))


def _split_summary(rows: list[dict[str, Any]], seeds: list[int], kp: float, kd: float) -> dict[str, Any]:
    baseline = _paired_scores(rows, seeds, *BASELINE)
    candidate = _paired_scores(rows, seeds, kp, kd)
    gaps = [new - old for old, new in zip(baseline, candidate)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)
    ties = len(gaps) - wins - losses
    return {
        "kp": kp,
        "kd": kd,
        "score": float(np.mean(candidate)) if candidate else None,
        "baseline_kp10_kd0": float(np.mean(baseline)) if baseline else None,
        "gap": float(np.mean(gaps)) if gaps else None,
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _summarize(rows: list[dict[str, Any]], tune_seeds: list[int], holdout_seeds: list[int]) -> dict[str, Any]:
    tune_table = [_split_summary(rows, tune_seeds, kp, kd) for kp, kd in GAINS]
    tune_table.sort(key=lambda item: float(item["score"]), reverse=True)
    winner = tune_table[0]
    holdout = _split_summary(rows, holdout_seeds, float(winner["kp"]), float(winner["kd"]))
    return {
        "oracle_warning": "读取全局状态，并且每个场景事后选择固定分配。不是合法策略，也不是数学上界。",
        "control": "a = clip(Kp * (p_target - p_robot) + Kd * (v_target - v_robot), -1, 1)",
        "baseline": {"kp": BASELINE[0], "kd": BASELINE[1]},
        "master_seed": MASTER_SEED,
        "tune_seeds": len(tune_seeds),
        "holdout_seeds": len(holdout_seeds),
        "tune_ranking": tune_table,
        "selected_on_tune": {"kp": winner["kp"], "kd": winner["kd"], "tune": winner},
        "holdout_frozen": holdout,
        "public_baseline": _public_score(rows, *BASELINE),
        "public_selected": _public_score(rows, float(winner["kp"]), float(winner["kd"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 2 or args.count % 2 != 0 or args.workers <= 0:
        parser.error("--count 必须是不小于 2 的偶数，--workers 必须为正")

    public_suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    cases = [case for group in public_suite.groups for case in group.cases]
    random_seeds = _random_seeds(args.count, args.master_seed)
    tune_seeds = random_seeds[: args.count // 2]
    holdout_seeds = random_seeds[args.count // 2 :]
    basic = _layout_config("uniform")
    cooperation = _layout_config("crossing")
    for seed in random_seeds:
        cases.append(ScenarioCase(f"random-basic-{seed}", "basic", basic, seed))
        cases.append(ScenarioCase(f"random-cooperation-{seed}", "cooperation", cooperation, seed))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate_case, case): case.case_id for case in cases}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 10 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)

    rows.sort(key=lambda row: (str(row["case_id"]), float(row["kp"]), float(row["kd"])))
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, tune_seeds, holdout_seeds)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
