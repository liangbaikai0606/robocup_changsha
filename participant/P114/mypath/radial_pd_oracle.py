"""沿连线的 PD：分配仍是可达性防抖。

力只加在车指向圈的方向上。误差是离圈边还有多远，阻尼是相对径向速度：

    a_r = K_p * (d - r) - K_d * ((v_R - v_hat_T) · n)
    a = clip(a_r * n, -1, 1)

n = (p_T - p_R) / ||p_T - p_R||。K_d = 0 就是同一条公式的 P。
目标速度做指数平滑，不用有效/无效判断。参数只在前一半种子上选，后一半冻结。
读取全局状态，不是合法策略。
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
MYPATH_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MYPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MYPATH_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.suites import ScenarioCase
from oracle_search import _decode
from reachability_debounce_oracle import MASTER_SEED, _Assigner, _layout_config, _mean_ci, _random_seeds


KP = 4.0
GAINS = ((0.0, 0.0),) + tuple((kd, alpha) for kd in (0.5, 1.0, 2.0) for alpha in (0.0, 0.5, 0.8))


def _rollout(case: ScenarioCase, kd: float, alpha: float) -> dict[str, Any]:
    """跑一局。alpha 是旧速度估计的权重，越大越平滑。"""
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=int(case.scenario_seed))
    assigner = _Assigner(config, True)
    n = int(config.num_agents)
    m = int(config.num_targets)
    radius = float(config.public.target_radius)
    estimate = np.zeros((m, 2), dtype=np.float64)
    seen = np.zeros(m, dtype=np.bool_)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for step in range(int(config.horizon)):
            state = env.state()
            chosen = assigner.choose(state, step)
            robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
            for index in range(m):
                if not seen[index]:
                    estimate[index] = target_vel[index]
                    seen[index] = True
                else:
                    estimate[index] = alpha * estimate[index] + (1.0 - alpha) * target_vel[index]
            actions: dict[str, np.ndarray] = {}
            for robot_index, target_index in enumerate(chosen):
                delta = target_pos[target_index] - robot_pos[robot_index]
                distance = float(np.linalg.norm(delta))
                if distance < 1e-8:
                    actions[f"agent_{robot_index}"] = np.zeros(2, dtype=np.float32)
                    continue
                normal = delta / distance
                radial_speed = float(np.dot(robot_vel[robot_index] - estimate[target_index], normal))
                radial_force = KP * (distance - radius) - kd * radial_speed
                actions[f"agent_{robot_index}"] = np.clip(radial_force * normal, -1.0, 1.0).astype(np.float32)
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    horizon = int(config.horizon)
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
    }


def _evaluate_case(case: ScenarioCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kd, alpha in GAINS:
        result = _rollout(case, kd, alpha)
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group_id,
                "seed": int(case.scenario_seed),
                "kd": kd,
                "alpha": alpha,
                **result,
            }
        )
    return rows


def _paired(rows: list[dict[str, Any]], seeds: list[int], kd: float, alpha: float) -> list[float]:
    chosen = [row for row in rows if row["kd"] == kd and row["alpha"] == alpha]
    lookup = {(int(row["seed"]), str(row["group"])): row for row in chosen}
    scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        scores.append(500.0 * (float(basic["j"]) + float(coop["j"])))
    return scores


def _compare(rows: list[dict[str, Any]], seeds: list[int], kd: float, alpha: float) -> dict[str, Any]:
    baseline = _paired(rows, seeds, 0.0, 0.0)
    candidate = _paired(rows, seeds, kd, alpha)
    gaps = [new - old for old, new in zip(baseline, candidate)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)
    return {
        "kd": kd,
        "alpha": alpha,
        "score": float(np.mean(candidate)),
        "p_score": float(np.mean(baseline)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
    }


def _summarize(rows: list[dict[str, Any]], tune_seeds: list[int], holdout_seeds: list[int]) -> dict[str, Any]:
    tune = [_compare(rows, tune_seeds, kd, alpha) for kd, alpha in GAINS if kd > 0.0]
    tune.sort(key=lambda item: float(item["score"]), reverse=True)
    winner = tune[0]
    return {
        "oracle_warning": "读取全局状态，每步重选分配。不是合法策略，也不是数学上界。",
        "control": "a_r = 4*(d-r) - Kd*((v_R - v_hat_T)·n)，力沿连线再按分量限幅",
        "smoothing": "v_hat = alpha*v_hat_old + (1-alpha)*v_T",
        "master_seed": MASTER_SEED,
        "tune_seeds": len(tune_seeds),
        "holdout_seeds": len(holdout_seeds),
        "tune_p": _compare(rows, tune_seeds, 0.0, 0.0),
        "tune_ranking": tune,
        "selected": {"kd": winner["kd"], "alpha": winner["alpha"]},
        "holdout": _compare(rows, holdout_seeds, float(winner["kd"]), float(winner["alpha"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 2 or args.count % 2 != 0:
        parser.error("--count 必须是不小于 2 的偶数")
    seeds = _random_seeds(args.count, args.master_seed)
    tune_seeds = seeds[: args.count // 2]
    holdout_seeds = seeds[args.count // 2 :]
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
    summary = _summarize(rows, tune_seeds, holdout_seeds)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
