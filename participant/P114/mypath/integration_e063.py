"""反向消融：从当前上场策略拿掉一块，看还剩多少。

full 就是 entry.py。另外五档每次只改一件事：

- no_yield：不让给更快的队友
- no_tti：选圈和让圈都改回距离
- no_pd：力改成 clip(10*Δp)
- no_velocity：目标速度始终当 0，选圈和开车都不用估计
- no_occupied：已被站上的圈和空圈一起排
- hold2：新圈至少少 2 步截击才换

开发集是主种子 20261003。新种子是 20261102，这批以前没用来定参数。
不是 evaluate_one。不改 entry.py。
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
ENTRY_ROOT = REPO_ROOT / "participant" / "P114"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MYPATH_ROOT) not in sys.path:
    sys.path.insert(0, str(MYPATH_ROOT))
if str(ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(ENTRY_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import AgentObservation, EpisodeContext, PublicTaskParams
from coverage_bench.suites import ScenarioCase
from entry import CoverFirstPolicy
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds

FRESH_MASTER = 20261102
MODES = ("full", "no_yield", "no_tti", "no_pd", "no_velocity", "no_occupied", "hold2")


class _Variant(CoverFirstPolicy):
    """只改当前上场策略的一块。"""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self._held_target: int | None = None

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self._held_target = None

    def _update_target_velocity(self, observation: AgentObservation) -> None:
        if self.mode == "no_velocity":
            self._target_velocity_estimates = {}
            return
        super()._update_target_velocity(observation)

    def _thrust_to_target(
        self,
        relative: np.ndarray,
        self_velocity: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        if self.mode == "no_pd":
            return np.clip(np.asarray(relative, dtype=np.float64) * 10.0, -1.0, 1.0).astype(np.float32)
        return super()._thrust_to_target(relative, self_velocity, target_velocity)

    def _visible_intercept_rows(self, observation: AgentObservation) -> list[dict[str, object]]:
        rows = super()._visible_intercept_rows(observation)
        if self.mode == "no_occupied":
            for row in rows:
                row["occupied"] = False
        if self.mode == "no_tti":
            for row in rows:
                row["time"] = row["distance"]
        return rows

    def _mark_peer_yields(self, observation: AgentObservation, rows: list[dict[str, object]]) -> None:
        if self.mode == "no_yield":
            return
        if self.mode != "no_tti":
            super()._mark_peer_yields(observation, rows)
            return
        self_pos, _self_vel = self._self_motion(observation)
        peers = self._visible_peers(observation, self_pos, _self_vel)
        my_index = int(observation["agent_index"])
        for row in rows:
            if row["occupied"]:
                continue
            target_abs = self_pos + np.asarray(row["relative"], dtype=np.float64)
            best_peer: tuple[float, int] | None = None
            for peer_index, peer_pos, _peer_vel, _peer_dist in peers:
                if float(np.linalg.norm(target_abs - peer_pos)) > self._sense_radius:
                    continue
                peer_distance = float(np.linalg.norm(target_abs - peer_pos))
                if best_peer is None or (peer_distance, peer_index) < best_peer:
                    best_peer = (peer_distance, peer_index)
            if best_peer is None:
                continue
            peer_distance, peer_index = best_peer
            mine = float(row["distance"])
            if peer_distance < mine or (peer_distance == mine and peer_index < my_index):
                row["yielded"] = True

    def _choose_intercept_target(self, observation: AgentObservation) -> tuple[int, np.ndarray] | None:
        chosen = super()._choose_intercept_target(observation)
        if self.mode != "hold2":
            return chosen
        if chosen is None:
            self._held_target = None
            return None
        rows = self._visible_intercept_rows(observation)
        self._mark_peer_yields(observation, rows)
        free = [row for row in rows if not row["occupied"] and not row["yielded"]]
        pool = free or [row for row in rows if row["occupied"]]
        held = next((row for row in pool if row["index"] == self._held_target), None)
        new = next((row for row in pool if row["index"] == chosen[0]), None)
        if (
            held is not None
            and new is not None
            and not (float(new["time"]) + 2.0 <= float(held["time"]))
        ):
            chosen = (int(held["index"]), np.asarray(held["relative"]))
        self._held_target = int(chosen[0])
        return chosen


def _context(case: ScenarioCase, agent_index: int) -> EpisodeContext:
    task = case.task_config.public
    return EpisodeContext(
        agent_index=agent_index,
        num_agents=int(case.task_config.num_agents),
        num_targets=int(case.task_config.num_targets),
        horizon=int(case.task_config.horizon),
        task=PublicTaskParams(
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
        ),
        policy_seed=0,
    )


def _rollout(case: ScenarioCase, mode: str) -> dict[str, float]:
    env = make_training_env(case.task_config)
    observations, _infos = env.reset(seed=int(case.scenario_seed))
    policies = [_Variant(mode) for _agent in env.agents]
    for index, policy in enumerate(policies):
        policy.reset(_context(case, index))
    horizon = int(case.task_config.horizon)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for _step in range(horizon):
            actions = {
                agent: policies[index].act(observations[agent]) for index, agent in enumerate(env.agents)
            }
            observations, rewards, _terms, _truncs, infos = env.step(actions)
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
    for mode in MODES:
        result = _rollout(case, mode)
        for key, value in result.items():
            row[f"{mode}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], mode: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    scores: list[float] = []
    for seed in seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        scores.append(500.0 * (float(basic[f"{mode}_j"]) + float(coop[f"{mode}_j"])))
    return scores


def _pack(rows: list[dict[str, Any]], seeds: list[int], master: int, split: str) -> dict[str, Any]:
    full_scores = _paired(rows, seeds, "full")
    packed: dict[str, Any] = {"split": split, "master_seed": master, "seeds": len(seeds)}
    modes: dict[str, Any] = {}
    for mode in MODES:
        scores = _paired(rows, seeds, mode)
        gaps = [score - base for score, base in zip(scores, full_scores)]
        wins = sum(gap > 1e-9 for gap in gaps)
        losses = sum(gap < -1e-9 for gap in gaps)
        subset = rows
        ordered = np.sort(np.asarray(scores, dtype=np.float64))
        modes[mode] = {
            "score": float(np.mean(scores)),
            "score_ci95_normal": _mean_ci(scores),
            "gap_vs_full": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{mode}_coverage"]) for row in subset])),
            "collision": float(np.mean([float(row[f"{mode}_collision"]) for row in subset])),
            "p10": float(np.quantile(ordered, 0.10)),
            "p50": float(np.quantile(ordered, 0.50)),
            "p90": float(np.quantile(ordered, 0.90)),
            "below_100": int(np.sum(ordered < 100.0)),
            "min_score": float(ordered[0]) if len(ordered) else float("nan"),
        }
    packed["modes"] = modes
    return packed


def _run_split(master: int, count: int, workers: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = _random_seeds(count, master)
    basic = _layout_config("uniform")
    cooperation = _layout_config("crossing")
    cases = [
        case
        for seed in seeds
        for case in (
            ScenarioCase(f"{split}-basic-{seed}", "basic", basic, seed),
            ScenarioCase(f"{split}-cooperation-{seed}", "cooperation", cooperation, seed),
        )
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_evaluate_case, case) for case in cases]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(cases):
                print(f"{split} {index}/{len(cases)}", flush=True)
    rows.sort(key=lambda row: (str(row["group"]), int(row["seed"])))
    return rows, _pack(rows, seeds, master, split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("--count 和 --workers 必须为正")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dev_rows, dev_summary = _run_split(MASTER_SEED, args.count, args.workers, "dev")
    fresh_rows, fresh_summary = _run_split(FRESH_MASTER, args.count, args.workers, "fresh")
    fieldnames = list(dev_rows[0])
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", *fieldnames])
        writer.writeheader()
        for row in dev_rows:
            writer.writerow({"split": "dev", **row})
        for row in fresh_rows:
            writer.writerow({"split": "fresh", **row})
    summary = {
        "note": "full 是当前 entry.py。差是这一档减去 full。负差表示拿掉以后更低。不是 evaluate_one。未改 entry.py。",
        "fresh_master": FRESH_MASTER,
        "dev": dev_summary,
        "fresh": fresh_summary,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
