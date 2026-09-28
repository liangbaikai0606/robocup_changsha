"""角色分配，以及不用分配的势场。

角色是短时身份，不是每一步重选：

- 追踪者盯住一个圈，直到进圈、让给更快的队友、圈里已经有人，或剩余步数不够。
- 驻守者守住已经站上的圈，出圈才重新选。
- 补位者在没有可追的空圈时停着。2 步以内且不该让的圈才会离开补位。

势场不用这套身份。一档只追最近的可见圈，同时躲开过近的队友。
另一档把所有可见圈的吸引力加在一起。

对照是当前 entry.py。都只读局部观测。
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


_BACKUP_TIME = 2
_REPULSE_RANGE = 0.25


class RolePolicy(CoverFirstPolicy):
    """追踪和驻守会留在原圈上。backup=True 时，没有空圈就停，不去挤已经站上的圈。"""

    def __init__(self, backup: bool) -> None:
        super().__init__()
        self.backup = backup
        self._role = "FREE"
        self._role_target: int | None = None
        self.role_steps = {"FREE": 0, "CHASER": 0, "HOLDER": 0, "BACKUP": 0}
        self.last_target: int | None = None

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self._role = "FREE"
        self._role_target = None
        self.role_steps = {"FREE": 0, "CHASER": 0, "HOLDER": 0, "BACKUP": 0}
        self.last_target = None

    def _choose_intercept_target(self, observation: AgentObservation) -> tuple[int, np.ndarray] | None:
        rows = self._visible_intercept_rows(observation)
        if rows:
            self._mark_peer_yields(observation, rows)
        by_index = {int(row["index"]): row for row in rows}
        steps_left = self._horizon - int(observation["step_index"])
        if self._role in ("HOLDER", "CHASER") and self._role_target is not None:
            row = by_index.get(self._role_target)
            if self._keep_role(row, steps_left):
                if self._role == "CHASER" and float(row["distance"]) <= self._cover_radius:
                    self._role = "HOLDER"
                self._count(int(row["index"]))
                return int(row["index"]), np.asarray(row["relative"], dtype=np.float64)
            self._role = "FREE"
            self._role_target = None
        chosen = self._pick_new(rows)
        if chosen is None:
            self._role = "BACKUP" if self.backup else "FREE"
            self._role_target = None
            self._count(None)
            return None
        self._role = "HOLDER" if float(chosen["distance"]) <= self._cover_radius else "CHASER"
        self._role_target = int(chosen["index"])
        self._count(self._role_target)
        return self._role_target, np.asarray(chosen["relative"], dtype=np.float64)

    def _keep_role(self, row: dict[str, object] | None, steps_left: int) -> bool:
        if row is None:
            return False
        distance = float(row["distance"])
        if self._role == "HOLDER":
            return distance <= self._cover_radius
        if distance <= self._cover_radius:
            return True
        if bool(row["occupied"]) or bool(row["yielded"]):
            return False
        return int(row["time"]) <= steps_left

    def _pick_new(self, rows: list[dict[str, object]]) -> dict[str, object] | None:
        free = [row for row in rows if not row["occupied"] and not row["yielded"]]
        if free:
            free.sort(key=lambda row: (int(row["time"]), float(row["distance"]), int(row["index"])))
            return free[0]
        near = [
            row
            for row in rows
            if int(row["time"]) <= _BACKUP_TIME and not row["occupied"] and not row["yielded"]
        ]
        if near:
            near.sort(key=lambda row: (int(row["time"]), float(row["distance"]), int(row["index"])))
            return near[0]
        if self.backup:
            return None
        occupied = [row for row in rows if row["occupied"]]
        if not occupied:
            return None
        occupied.sort(key=lambda row: (int(row["time"]), float(row["distance"]), int(row["index"])))
        return occupied[0]

    def _count(self, target: int | None) -> None:
        self.role_steps[self._role] += 1
        self.last_target = target


class PotentialFieldPolicy(CoverFirstPolicy):
    """靠近目标、躲开队友。sum_targets 为真时，所有可见圈一起拉。"""

    def __init__(self, sum_targets: bool) -> None:
        super().__init__()
        self.sum_targets = sum_targets
        self.last_target: int | None = None

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self.last_target = None

    def act(self, observation: AgentObservation) -> np.ndarray:
        force, target_index = self._attract(observation)
        force = force + self._repel(observation)
        self.last_target = target_index
        return np.clip(force, -1.0, 1.0).astype(np.float32)

    def _attract(self, observation: AgentObservation) -> tuple[np.ndarray, int | None]:
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        force = np.zeros(2, dtype=np.float64)
        nearest: tuple[float, int, np.ndarray] | None = None
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            relative = targets[index, :2] * self._position_scale
            distance = float(np.linalg.norm(relative))
            if self.sum_targets:
                force += relative / max(distance, 0.05)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, index, relative)
        if nearest is None:
            return force, None
        if not self.sum_targets:
            force = 10.0 * nearest[2]
        return force, nearest[1]

    def _repel(self, observation: AgentObservation) -> np.ndarray:
        peers = np.asarray(observation["peers"], dtype=np.float64)
        visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
        force = np.zeros(2, dtype=np.float64)
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            relative = peers[index, :2] * self._position_scale
            distance = float(np.linalg.norm(relative))
            if distance <= 1e-8 or distance >= _REPULSE_RANGE:
                continue
            force -= (( _REPULSE_RANGE - distance) / _REPULSE_RANGE) * (relative / distance)
        return force


def _make_policies(law: str, count: int) -> list[CoverFirstPolicy]:
    if law == "e2":
        return [CoverFirstPolicy() for _robot in range(count)]
    if law == "role_stick":
        return [RolePolicy(False) for _robot in range(count)]
    if law == "role_backup":
        return [RolePolicy(True) for _robot in range(count)]
    if law == "pf_nearest":
        return [PotentialFieldPolicy(False) for _robot in range(count)]
    if law == "pf_sum":
        return [PotentialFieldPolicy(True) for _robot in range(count)]
    raise ValueError(law)


def _rollout(case: ScenarioCase, law: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    policies = _make_policies(law, n)
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=n,
                num_targets=int(config.num_targets),
                horizon=horizon,
                task=task,
                policy_seed=0,
            )
        )
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    switches = 0
    previous: list[int | None] = [None for _robot in range(n)]
    try:
        for _step in range(horizon):
            actions = {
                agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                for index, agent_id in enumerate(env.agents)
            }
            for index, policy in enumerate(policies):
                current = getattr(policy, "last_target", None)
                if previous[index] is not None and current != previous[index]:
                    switches += 1
                previous[index] = current
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    role_steps = {"FREE": 0, "CHASER": 0, "HOLDER": 0, "BACKUP": 0}
    for policy in policies:
        counts = getattr(policy, "role_steps", None)
        if counts is None:
            continue
        for name, value in counts.items():
            role_steps[name] += int(value)
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "switches": float(switches),
        "role_free": float(role_steps["FREE"]),
        "role_chaser": float(role_steps["CHASER"]),
        "role_holder": float(role_steps["HOLDER"]),
        "role_backup": float(role_steps["BACKUP"]),
    }


LAWS = ("e2", "role_stick", "role_backup", "pf_nearest", "pf_sum")


def _evaluate(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id, "group": case.group_id, "seed": int(case.scenario_seed)}
    for law in LAWS:
        result = _rollout(case, law)
        for key, value in result.items():
            row[f"{law}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], law: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{law}_j"]) + float(lookup[(seed, "cooperation")][f"{law}_j"]))
        for seed in seeds
    ]


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base = _paired(rows, seeds, "e2")
    laws: dict[str, Any] = {}
    for law in LAWS:
        scores = _paired(rows, seeds, law)
        gaps = [new - old for old, new in zip(base, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        laws[law] = {
            "score": float(np.mean(scores)),
            "gap_vs_e2": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{law}_coverage"]) for row in rows])),
            "collision": float(np.mean([float(row[f"{law}_collision"]) for row in rows])),
            "switches": float(np.mean([float(row[f"{law}_switches"]) for row in rows])),
            "role_steps_per_episode": {
                name: float(np.mean([float(row[f"{law}_role_{name}"]) for row in rows]))
                for name in ("free", "chaser", "holder", "backup")
            },
        }
    return {
        "note": "角色和势场都只读局部观测。对照是当前 entry.py。不是 evaluate_one。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
    }


def _public_score(law: str) -> float:
    suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    grouped: dict[str, list[float]] = {}
    for group in suite.groups:
        for case in group.cases:
            grouped.setdefault(case.group_id, []).append(_rollout(case, law)["j"])
    return 500.0 * (float(np.mean(grouped["basic"])) + float(np.mean(grouped["cooperation"])))


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
    summary["public_four"] = {law: _public_score(law) for law in LAWS}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
