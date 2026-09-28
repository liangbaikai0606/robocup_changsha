"""团队认领：看得见的时候，用报价决定这个圈归谁。

报价是

    Q = -T + alpha * A

T 是自己估计的截击步数，A 是已经连续追这个圈的步数。
队友的 A 只能从「他连续几步都最适合这个圈」里推断。
别人的报价要高于自己 delta，才把圈让出去。报价相同则编号小的留。
已经站进圈的队友仍然优先，不和报价抢。

对照是当前 entry.py 的 E052，也就是 E2。只读局部观测。
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


ALPHAS = (0.0, 1.0, 2.0)
DELTAS = (0.0, 1.0)


class ClaimPolicy(CoverFirstPolicy):
    """用认领强度替换「队友至少快 1 步才让」。开车仍走父类。"""

    def __init__(self, alpha: float, delta: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.delta = float(delta)
        self._my_age: dict[int, int] = {}
        self._peer_focus: dict[int, tuple[int, int]] = {}
        self.last_target: int | None = None

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self._my_age = {}
        self._peer_focus = {}
        self.last_target = None

    def _choose_intercept_target(self, observation: AgentObservation) -> tuple[int, np.ndarray] | None:
        rows = self._visible_intercept_rows(observation)
        if not rows:
            self._my_age = {}
            self.last_target = None
            return None
        self._refresh_peer_focus(observation, rows)
        self._mark_claim_yields(observation, rows)
        free = [row for row in rows if not row["occupied"] and not row["yielded"]]
        occupied = [row for row in rows if row["occupied"]]
        pool = free or occupied
        if not pool:
            self._my_age = {}
            self.last_target = None
            return None
        pool.sort(key=lambda row: (-self._self_claim(row), int(row["time"]), float(row["distance"]), int(row["index"])))
        chosen = pool[0]
        index = int(chosen["index"])
        self._my_age = {index: self._my_age.get(index, 0) + 1}
        self.last_target = index
        relative = np.asarray(chosen["relative"], dtype=np.float64)
        return index, relative

    def _self_claim(self, row: dict[str, object]) -> float:
        return -float(row["time"]) + self.alpha * float(self._my_age.get(int(row["index"]), 0))

    def _refresh_peer_focus(self, observation: AgentObservation, rows: list[dict[str, object]]) -> None:
        """队友连续几步都是某个圈最快，就把这几步记成他的追踪时长。"""
        steps_left = self._horizon - int(observation["step_index"])
        self_pos, _self_vel = self._self_motion(observation)
        peers = self._visible_peers(observation, self_pos, _self_vel)
        seen: set[int] = set()
        for peer_index, peer_pos, peer_vel, _peer_dist in peers:
            best: tuple[int, int] | None = None
            for row in rows:
                target_abs = self_pos + np.asarray(row["relative"], dtype=np.float64)
                if float(np.linalg.norm(target_abs - peer_pos)) > self._sense_radius:
                    continue
                peer_time = self._intercept_time(
                    target_abs - peer_pos,
                    peer_vel,
                    np.asarray(row["velocity"], dtype=np.float64),
                    steps_left,
                )
                candidate = (peer_time, int(row["index"]))
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                continue
            seen.add(peer_index)
            target_index = best[1]
            previous = self._peer_focus.get(peer_index)
            age = previous[1] + 1 if previous is not None and previous[0] == target_index else 1
            self._peer_focus[peer_index] = (target_index, age)
        for peer_index in list(self._peer_focus):
            if peer_index not in seen:
                del self._peer_focus[peer_index]

    def _mark_claim_yields(self, observation: AgentObservation, rows: list[dict[str, object]]) -> None:
        steps_left = self._horizon - int(observation["step_index"])
        self_pos, _self_vel = self._self_motion(observation)
        peers = self._visible_peers(observation, self_pos, _self_vel)
        my_index = int(observation["agent_index"])
        for row in rows:
            if row["occupied"]:
                continue
            target_index = int(row["index"])
            own_claim = self._self_claim(row)
            target_abs = self_pos + np.asarray(row["relative"], dtype=np.float64)
            best_key: tuple[float, int] | None = None
            best_claim = 0.0
            best_peer = -1
            for peer_index, peer_pos, peer_vel, _peer_dist in peers:
                if float(np.linalg.norm(target_abs - peer_pos)) > self._sense_radius:
                    continue
                peer_time = self._intercept_time(
                    target_abs - peer_pos,
                    peer_vel,
                    np.asarray(row["velocity"], dtype=np.float64),
                    steps_left,
                )
                focus = self._peer_focus.get(peer_index)
                peer_age = focus[1] if focus is not None and focus[0] == target_index else 0
                peer_claim = -float(peer_time) + self.alpha * float(peer_age)
                key = (peer_claim, -peer_index)
                if best_key is None or key > best_key:
                    best_key = key
                    best_claim = peer_claim
                    best_peer = peer_index
            if best_key is None:
                continue
            if best_claim > own_claim + self.delta:
                row["yielded"] = True
            elif abs(best_claim - own_claim) <= 1e-9 and best_peer < my_index:
                row["yielded"] = True


def _rollout(case: ScenarioCase, alpha: float | None, delta: float | None) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    if alpha is None:
        policies: list[CoverFirstPolicy] = [CoverFirstPolicy() for _robot in range(n)]
    else:
        policies = [ClaimPolicy(alpha, float(delta)) for _robot in range(n)]
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
                if alpha is None:
                    current = None
                if previous[index] is not None and current is not None and current != previous[index]:
                    switches += 1
                if current is not None:
                    previous[index] = current
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
        "switches": float(switches),
    }


def _laws() -> list[tuple[str, float | None, float | None]]:
    found = [("e2", None, None)]
    for alpha in ALPHAS:
        for delta in DELTAS:
            found.append((f"a{alpha:g}_d{delta:g}", alpha, delta))
    return found


def _evaluate(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id, "group": case.group_id, "seed": int(case.scenario_seed)}
    for name, alpha, delta in _laws():
        result = _rollout(case, alpha, delta)
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
    base = _paired(rows, seeds, "e2")
    laws: dict[str, Any] = {}
    for name, _alpha, _delta in _laws():
        scores = _paired(rows, seeds, name)
        gaps = [new - old for old, new in zip(base, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        laws[name] = {
            "score": float(np.mean(scores)),
            "gap_vs_e2": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": float(np.mean([float(row[f"{name}_coverage"]) for row in rows])),
            "collision": float(np.mean([float(row[f"{name}_collision"]) for row in rows])),
            "switches": float(np.mean([float(row[f"{name}_switches"]) for row in rows])),
        }
    return {
        "note": "Q=-T+alpha*A。delta 是接管需要多出的报价。e2 是当前 entry.py。不是 evaluate_one。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
    }


def _public_score(law_name: str, alpha: float | None, delta: float | None) -> float:
    suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    grouped: dict[str, list[float]] = {}
    for group in suite.groups:
        for case in group.cases:
            result = _rollout(case, alpha, delta)
            grouped.setdefault(case.group_id, []).append(result["j"])
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
    summary["public_four"] = {
        name: _public_score(name, alpha, delta) for name, alpha, delta in _laws()
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
