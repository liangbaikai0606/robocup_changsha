"""在「可见队友更快就让」上面，扫「至少省几步才换」。

省 0 步就是一有更短的截击时间就换。
同一规则再跑三批种子，看省 2 步的降分是不是这一批数据碰巧。
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

MYPATH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MYPATH_ROOT.parents[2]
for folder in (REPO_ROOT, MYPATH_ROOT.parents[0], MYPATH_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from coverage_bench.suites import ScenarioCase
from local_assignment_oracle import _rollout_local
from reachability_debounce_oracle import _layout_config, _mean_ci, _random_seeds


SAVES = (0, 1, 2, 3, 4)
MASTERS = (20261003, 20261011, 20261028)


def _one(case: ScenarioCase, save_steps: int) -> dict[str, Any]:
    result = _rollout_local(case, "e2", save_steps=save_steps)
    return {
        "group": case.group_id,
        "seed": int(case.scenario_seed),
        "save_steps": save_steps,
        "j": result["j"],
        "coverage": result["coverage"],
        "collision": result["collision"],
        "switches": result["switches"],
    }


def _score_table(rows: list[dict[str, Any]], seeds: list[int], save_steps: int) -> dict[str, Any]:
    picked = [row for row in rows if int(row["save_steps"]) == save_steps]
    lookup = {(int(row["seed"]), str(row["group"])): row for row in picked}
    direct = [row for row in rows if int(row["save_steps"]) == 0]
    direct_lookup = {(int(row["seed"]), str(row["group"])): row for row in direct}
    scores = [
        500.0 * (float(lookup[(seed, "basic")]["j"]) + float(lookup[(seed, "cooperation")]["j"]))
        for seed in seeds
    ]
    base = [
        500.0 * (float(direct_lookup[(seed, "basic")]["j"]) + float(direct_lookup[(seed, "cooperation")]["j"]))
        for seed in seeds
    ]
    gaps = [new - old for old, new in zip(base, scores)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)
    return {
        "score": float(np.mean(scores)),
        "gap_vs_direct_switch": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "coverage": float(np.mean([float(row["coverage"]) for row in picked])),
        "collision": float(np.mean([float(row["collision"]) for row in picked])),
        "switches": float(np.mean([float(row["switches"]) for row in picked])),
    }


def _run_master(master: int, count: int, workers: int) -> dict[str, Any]:
    seeds = _random_seeds(count, master)
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
    jobs = [(case, save_steps) for case in cases for save_steps in SAVES]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_one, case, save_steps) for case, save_steps in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    return {
        "master_seed": master,
        "seeds": len(seeds),
        "by_save_steps": {str(save_steps): _score_table(rows, seeds, save_steps) for save_steps in SAVES},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "note": "基础是 E2：可见队友至少快 1 步就让。save_steps=0 表示一有更短截击时间就换。对照都相对这一档。不是 evaluate_one。",
        "masters": [_run_master(master, args.count, args.workers) for master in MASTERS],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
