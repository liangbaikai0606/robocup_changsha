"""把全局截击步数搬回局部观测。

开车始终是当前 entry.py 的远距满油门、近圈刹车和圈内跟速。
只改选哪个圈。五套都在主种子 20261003 的同一批种子上：

- baseline：现在的空圈优先，选最近。
- e1：空圈里选自己最早能罩住的。
- e2：可见队友至少快 1 步就让；截击步数相同则编号小的优先。
- e3：队友明显在追、随后离开视野时，这个圈再让 2 步。
- e4：新圈至少少 2 步才换。
- e5：裁判用全局位置速度做最小总 T，车仍用局部开车公式。只作上限，不是合法策略。

e1 到 e4 只读局部观测。
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
from oracle_search import _decode
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds
from time_to_intercept_oracle import _choose_assignment, _time_matrix


LAWS = ("baseline", "e1", "e2", "e3", "e4", "e5")
_YIELD_MARGIN = 1
_RESERVE_STEPS = 2
_SWITCH_SAVE = 2


class LocalAssignmentPolicy(CoverFirstPolicy):
    """在站圈优先上，用局部截击步数决定追哪个圈。"""

    def __init__(self, law: str, save_steps: int | None = None) -> None:
        super().__init__()
        self.law = law
        self.save_steps = _SWITCH_SAVE if law == "e4" and save_steps is None else save_steps
        self._held_target: int | None = None
        self._reserve_left: dict[int, int] = {}
        self.last_target: int | None = None
        self.last_reason = "none"
        self.last_alternative: int | None = None

    def reset(self, context: EpisodeContext) -> None:
        super().reset(context)
        self._held_target = None
        self._reserve_left = {}
        self.last_target = None
        self.last_reason = "none"
        self.last_alternative = None

    def _chase(self, observation: AgentObservation) -> np.ndarray:
        chosen = self._choose_target(observation)
        self.last_target = None if chosen is None else int(chosen[0])
        if chosen is None:
            self._held_target = None
            return np.zeros(2, dtype=np.float32)
        target_index, relative = chosen
        self_state = np.asarray(observation["self_state"], dtype=np.float64)
        self_velocity = self_state[2:4] * self._velocity_scale
        target_velocity = self._target_velocity_estimates.get(target_index, np.zeros(2, dtype=np.float64))
        return self._thrust_to_target(relative, self_velocity, target_velocity)

    def _choose_target(self, observation: AgentObservation) -> tuple[int, np.ndarray] | None:
        rows = self._visible_rows(observation)
        if not rows:
            self._held_target = None
            return None
        if self.law == "baseline":
            pool = [row for row in rows if not row["occupied"]] or rows
            pool.sort(key=lambda row: (row["distance"], row["index"]))
            chosen = pool[0]
            self._held_target = int(chosen["index"])
            self.last_reason = "nearest"
            self.last_alternative = None
            return int(chosen["index"]), chosen["relative"]

        self._mark_yields(observation, rows)
        free = [row for row in rows if not row["occupied"] and not row["yielded"]]
        occupied = [row for row in rows if row["occupied"]]
        pool = free or occupied
        if not pool:
            self._held_target = None
            self.last_reason = "none"
            self.last_alternative = None
            return None
        pool.sort(key=lambda row: (row["time"], row["distance"], row["index"]))
        raw = [row for row in rows if not row["occupied"]] or [row for row in rows if row["occupied"]]
        raw.sort(key=lambda row: (row["time"], row["distance"], row["index"]))
        chosen = pool[0]
        reason = "earliest"
        alternative: int | None = None
        if int(raw[0]["index"]) != int(chosen["index"]) and bool(raw[0]["yielded"]):
            reason = "yield_" + str(raw[0].get("yield_cause", "other"))
            alternative = int(raw[0]["index"])
        save_steps = self.save_steps
        if save_steps is not None and save_steps > 0 and self._held_target is not None:
            held = next((row for row in pool if row["index"] == self._held_target), None)
            if held is not None and not (int(chosen["time"]) + save_steps <= int(held["time"])):
                chosen = held
                reason = "save"
                alternative = None
        self._held_target = int(chosen["index"])
        self.last_reason = reason
        self.last_alternative = alternative
        return int(chosen["index"]), chosen["relative"]

    def _visible_rows(self, observation: AgentObservation) -> list[dict[str, Any]]:
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        peers = np.asarray(observation["peers"], dtype=np.float64)
        peer_visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
        _self_pos, self_vel = self._self_motion(observation)
        steps_left = self._horizon - int(observation["step_index"])
        rows: list[dict[str, Any]] = []
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            relative = targets[index, :2] * self._position_scale
            target_velocity = self._target_velocity_estimates.get(index, np.zeros(2, dtype=np.float64))
            rows.append(
                {
                    "index": index,
                    "relative": relative.astype(np.float64),
                    "distance": float(np.linalg.norm(relative)),
                    "occupied": self._occupied(relative, peers, peer_visible),
                    "velocity": target_velocity,
                    "time": self._intercept_time(relative, self_vel, target_velocity, steps_left),
                    "yielded": False,
                }
            )
        return rows

    def _mark_yields(self, observation: AgentObservation, rows: list[dict[str, Any]]) -> None:
        if self.law == "e1":
            return
        steps_left = self._horizon - int(observation["step_index"])
        self_pos, _self_vel = self._self_motion(observation)
        peers = self._visible_peers(observation, self_pos, _self_vel)
        my_index = int(observation["agent_index"])
        seen_strong: set[int] = set()
        for row in rows:
            if row["occupied"]:
                continue
            target_abs = self_pos + row["relative"]
            best_peer: tuple[int, int] | None = None
            for peer_index, peer_pos, peer_vel, _peer_dist in peers:
                if float(np.linalg.norm(target_abs - peer_pos)) > self._sense_radius:
                    continue
                peer_relative = target_abs - peer_pos
                peer_time = self._intercept_time(peer_relative, peer_vel, row["velocity"], steps_left)
                if best_peer is None or (peer_time, peer_index) < best_peer:
                    best_peer = (peer_time, peer_index)
                if self.law in ("e3", "e4") and self._closing_fast(peer_relative, peer_vel):
                    seen_strong.add(int(row["index"]))
            if best_peer is None:
                continue
            peer_time, peer_index = best_peer
            clearer = peer_time + _YIELD_MARGIN < int(row["time"])
            tied = peer_time == int(row["time"]) and peer_index < my_index
            if clearer or tied:
                row["yielded"] = True
                row["yield_cause"] = "faster" if clearer else "tie"
        if self.law not in ("e3", "e4"):
            return
        for row in rows:
            index = int(row["index"])
            if index in seen_strong:
                self._reserve_left[index] = _RESERVE_STEPS
                continue
            left = self._reserve_left.get(index, 0)
            if left <= 0 or row["occupied"]:
                continue
            still_seen = any(
                float(np.linalg.norm((self_pos + row["relative"]) - peer_pos)) <= self._sense_radius
                for _peer_index, peer_pos, _peer_vel, _peer_dist in peers
            )
            if still_seen:
                continue
            row["yielded"] = True
            row.setdefault("yield_cause", "reserve")
            self._reserve_left[index] = left - 1

    def _closing_fast(self, relative: np.ndarray, velocity: np.ndarray) -> bool:
        distance = float(np.linalg.norm(relative))
        if distance <= 1e-8:
            return False
        closing = float(np.dot(velocity, relative / distance))
        return closing > 0.05

    def _intercept_time(
        self,
        relative: np.ndarray,
        velocity: np.ndarray,
        target_velocity: np.ndarray,
        steps_left: int,
    ) -> int:
        """朝预测点满力，第一次进圈的步数。进不去记成剩余步数加 1。"""
        offset = np.asarray(relative, dtype=np.float64)
        if float(np.linalg.norm(offset)) <= self._cover_radius:
            return 0
        if steps_left <= 0:
            return 1
        pos = np.zeros(2, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64).copy()
        target = offset.copy()
        target_vel = np.asarray(target_velocity, dtype=np.float64)
        accel = self._dt / self._mass
        for step in range(1, steps_left + 1):
            action = np.clip(10.0 * (target - pos), -1.0, 1.0)
            pos = pos + vel * self._dt
            vel = (1.0 - self._damping) * vel + action * accel
            speed = float(np.linalg.norm(vel))
            if speed > self._max_speed:
                vel *= self._max_speed / speed
            target = target + target_vel * self._dt
            if float(np.linalg.norm(target - pos)) <= self._cover_radius:
                return step
        return steps_left + 1


def _rollout_local(case: ScenarioCase, law: str, save_steps: int | None = None) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    horizon = int(config.horizon)
    task = _public_task_params(case)
    policies = [LocalAssignmentPolicy("baseline" if law == "e5" else law, save_steps=save_steps) for _robot in range(n)]
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
    previous = [None for _robot in range(n)]
    try:
        for _step in range(horizon):
            if law == "e5":
                actions = _central_actions(env, policies, case)
            else:
                actions = {
                    agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                    for index, agent_id in enumerate(env.agents)
                }
            for index in range(n):
                current = policies[index].last_target
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


def _central_actions(env: Any, policies: list[LocalAssignmentPolicy], case: ScenarioCase) -> dict[str, np.ndarray]:
    """全局最小总 T 决定圈，力仍走局部开车公式。"""
    config = case.task_config
    n = int(config.num_agents)
    m = int(config.num_targets)
    robot_pos, robot_vel, target_pos, target_vel = _decode(env.state(), n, m)
    done = int(getattr(policies[0], "_central_steps", 0))
    times = _time_matrix(robot_pos, robot_vel, target_pos, target_vel, int(config.horizon) - done, config)
    chosen = _choose_assignment(times, None, 0)
    policies[0]._central_steps = done + 1
    actions: dict[str, np.ndarray] = {}
    for index, agent_id in enumerate(env.agents):
        relative = target_pos[chosen[index]] - robot_pos[index]
        policies[index].last_target = int(chosen[index])
        actions[agent_id] = policies[index]._thrust_to_target(relative, robot_vel[index], target_vel[chosen[index]])
    return actions


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id, "group": case.group_id, "seed": int(case.scenario_seed)}
    for law in LAWS:
        result = _rollout_local(case, law)
        for key, value in result.items():
            row[f"{law}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], law: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][f"{law}_j"]) + float(lookup[(seed, "cooperation")][f"{law}_j"]))
        for seed in seeds
    ]


def _law_summary(rows: list[dict[str, Any]], seeds: list[int], law: str, base: list[float]) -> dict[str, Any]:
    scores = _paired(rows, seeds, law)
    gaps = [new - old for old, new in zip(base, scores)]
    wins = sum(gap > 1e-12 for gap in gaps)
    losses = sum(gap < -1e-12 for gap in gaps)
    return {
        "score": float(np.mean(scores)),
        "gap_vs_baseline": float(np.mean(gaps)),
        "gap_ci95_normal": _mean_ci(gaps),
        "wins": wins,
        "ties": len(gaps) - wins - losses,
        "losses": losses,
        "coverage": float(np.mean([float(row[f"{law}_coverage"]) for row in rows])),
        "collision": float(np.mean([float(row[f"{law}_collision"]) for row in rows])),
        "switches": float(np.mean([float(row[f"{law}_switches"]) for row in rows])),
    }


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base = _paired(rows, seeds, "baseline")
    return {
        "oracle_warning": "e1 到 e4 只用局部观测。e5 分配用了全局状态，不是合法策略。都不是 evaluate_one。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": {law: _law_summary(rows, seeds, law, base) for law in LAWS},
    }


def _public_rows() -> list[dict[str, Any]]:
    suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    return [_evaluate_case(case) for group in suite.groups for case in group.cases]


def _public_score(rows: list[dict[str, Any]], law: str) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(float(row[f"{law}_j"]))
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
    public = _public_rows()
    summary["public_four"] = {law: _public_score(public, law) for law in LAWS}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
