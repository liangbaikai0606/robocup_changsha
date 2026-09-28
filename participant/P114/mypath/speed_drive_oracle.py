"""收紧局部目标速度，并试几种开车公式。

速度：只在连续两步都看见时做差分，速率不超过任务公布的目标最大速度。
再试把新差分和旧估计各取一半。

开车：
- base：现在的远距满油门、近圈刹车、圈内跟速
- p：全程 a = clip(10*Δp)
- lead：a = clip(10*(Δp + 0.2*v_hat))
- pd：a = clip(20*Δp + 4*(v_hat - v))

都只读局部观测。对照是当前 entry.py。
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

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import AgentObservation, EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase, load_suite
from entry import CoverFirstPolicy
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds


LAWS = (
    ("base", "raw", "base"),
    ("cap", "cap", "base"),
    ("ema", "ema", "base"),
    ("p", "raw", "p"),
    ("lead", "raw", "lead"),
    ("pd", "raw", "pd"),
    ("cap_p", "cap", "p"),
    ("cap_lead", "cap", "lead"),
    ("ema_lead", "ema", "lead"),
)


class TunedPolicy(CoverFirstPolicy):
    """局部速度估计和开车公式可换，选圈仍是现在的让圈规则。"""

    def __init__(self, velocity_mode: str, drive_mode: str) -> None:
        super().__init__()
        self.velocity_mode = velocity_mode
        self.drive_mode = drive_mode
        self._target_speed_limit = 0.2
        self._seen_step: dict[int, int] = {}

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self._target_speed_limit = float(context.task.target_max_speed)
        self._seen_step = {}

    def _update_target_velocity(self, observation: AgentObservation) -> None:
        if self.velocity_mode == "raw":
            super()._update_target_velocity(observation)
            return
        self_state = np.asarray(observation["self_state"], dtype=np.float64)
        self_pos = self_state[:2] * self._position_scale
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        step = int(observation["step_index"])
        if step <= 0:
            self._last_target_abs = {}
            self._target_velocity_estimates = {}
            self._seen_step = {}
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            target_abs = self_pos + targets[index, :2] * self._position_scale
            previous = self._last_target_abs.get(index)
            previous_step = self._seen_step.get(index)
            if previous is not None and previous_step == step - 1 and self._dt > 0.0:
                fresh = (target_abs - previous) / self._dt
                speed = float(np.linalg.norm(fresh))
                if speed > self._target_speed_limit > 0.0:
                    fresh *= self._target_speed_limit / speed
                old = self._target_velocity_estimates.get(index)
                if self.velocity_mode == "ema" and old is not None:
                    fresh = 0.5 * fresh + 0.5 * old
                self._target_velocity_estimates[index] = fresh
            self._last_target_abs[index] = target_abs.copy()
            self._seen_step[index] = step

    def _thrust_to_target(
        self,
        relative: np.ndarray,
        self_velocity: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        offset = np.asarray(relative, dtype=np.float64)
        velocity = np.asarray(self_velocity, dtype=np.float64)
        target_speed = np.asarray(target_velocity, dtype=np.float64)
        if self.drive_mode == "p":
            return np.clip(10.0 * offset, -1.0, 1.0).astype(np.float32)
        if self.drive_mode == "lead":
            return np.clip(10.0 * (offset + 0.2 * target_speed), -1.0, 1.0).astype(np.float32)
        if self.drive_mode == "pd":
            return np.clip(20.0 * offset + 4.0 * (target_speed - velocity), -1.0, 1.0).astype(np.float32)
        return super()._thrust_to_target(relative, self_velocity, target_velocity)


def _rollout(case: ScenarioCase, velocity_mode: str, drive_mode: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    robots = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    policies = [
        CoverFirstPolicy() if velocity_mode == "raw" and drive_mode == "base" else TunedPolicy(velocity_mode, drive_mode)
        for _robot in range(robots)
    ]
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
        env.close()
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
    }


def _evaluate(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id, "group": case.group_id, "seed": int(case.scenario_seed)}
    for name, velocity_mode, drive_mode in LAWS:
        result = _rollout(case, velocity_mode, drive_mode)
        for key, value in result.items():
            row[f"{name}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], law: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{law}_j"]) + float(lookup[(seed, "cooperation")][f"{law}_j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base = _paired(rows, seeds, "base")
    laws: dict[str, Any] = {}
    for name, _velocity, _drive in LAWS:
        scores = _paired(rows, seeds, name)
        gaps = [new - old for old, new in zip(base, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        laws[name] = {
            "score": float(np.mean(scores)),
            "gap_vs_base": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{name}_coverage"]) for row in rows])),
            "collision": float(np.mean([float(row[f"{name}_collision"]) for row in rows])),
        }
    return {
        "note": "只读局部观测。对照是当前 entry.py。不是 evaluate_one。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
    }


def _public_scores() -> dict[str, float]:
    suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    scores: dict[str, float] = {}
    for name, velocity_mode, drive_mode in LAWS:
        grouped: dict[str, list[float]] = {}
        for group in suite.groups:
            for case in group.cases:
                grouped.setdefault(case.group_id, []).append(_rollout(case, velocity_mode, drive_mode)["j"])
        scores[name] = 500.0 * (float(np.mean(grouped["basic"])) + float(np.mean(grouped["cooperation"])))
    return scores


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
