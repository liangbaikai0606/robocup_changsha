"""可达性防抖保持不变，只比较开车方式。

分配都用 E033 的 debounce。对照是 a = clip(10 * Δp, -1, 1)。
PD 是 a = clip(20 * Δp + 4 * Δv, -1, 1)。读取全局状态，不是合法策略。
种子与 E033 相同：主种子 20261003 的 300 个。
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

from coverage_bench.suites import ScenarioCase
from reachability_debounce_oracle import (
    MASTER_SEED,
    _layout_config,
    _mean_ci,
    _random_seeds,
    _rollout,
)


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    proportional = _rollout(case, True, 10.0, 0.0)
    pd = _rollout(case, True, 20.0, 4.0)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
    }
    for name, result in (("proportional", proportional), ("pd", pd)):
        for key, value in result.items():
            row[f"{name}_{key}"] = value
    row["gap_j"] = float(pd["j"]) - float(proportional["j"])
    return row


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    old_scores: list[float] = []
    new_scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        old_scores.append(500.0 * (float(basic["proportional_j"]) + float(coop["proportional_j"])))
        new_scores.append(500.0 * (float(basic["pd_j"]) + float(coop["pd_j"])))
    gaps = [new - old for old, new in zip(old_scores, new_scores)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)

    def mean_of(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    return {
        "oracle_warning": "读取全局状态，每步重选分配。不是合法策略，也不是数学上界。",
        "assignment": "两边都是可达性防抖",
        "proportional": "a = clip(10 * Δp, -1, 1)",
        "pd": "a = clip(20 * Δp + 4 * Δv, -1, 1)",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "proportional_score": float(np.mean(old_scores)),
        "pd_score": float(np.mean(new_scores)),
        "gap": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "proportional_coverage": mean_of("proportional_coverage"),
        "pd_coverage": mean_of("pd_coverage"),
        "proportional_collision": mean_of("proportional_collision"),
        "pd_collision": mean_of("pd_collision"),
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
