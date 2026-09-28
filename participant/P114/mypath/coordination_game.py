"""E052：把最近的局部分配实验拆成差分功劳和让位反事实。

对照是 E046 的基线、E047 的让位（e2）、E048 的软预约（e3）。
开车不变。每一步用「拿掉这台车之后团队奖励变多少」算差分奖励。
让位的那一步另外锁住目标，向前滚 3 步：让位的车改去抢原来那个圈，看团队奖励升还是降。

读取全局状态只为了差分和反事实。e2/e3 选圈仍然只用局部观测。
不是 evaluate_one，不改 entry.py。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[3]
PARTICIPANT_ROOT = Path(__file__).resolve().parents[1]
MYPATH_ROOT = Path(__file__).resolve().parent
for folder in (REPO_ROOT, PARTICIPANT_ROOT, MYPATH_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.envs.motion import advance_targets
from coverage_bench.envs.physics import advance_robots
from coverage_bench.envs.scenario import snapshot
from coverage_bench.metrics import compute_step_metrics
from coverage_bench.protocol import EpisodeContext
from coverage_bench.runtime import _public_task_params
from coverage_bench.suites import ScenarioCase
from local_assignment_oracle import LocalAssignmentPolicy
from oracle_search import _decode
from reachability_debounce_oracle import MASTER_SEED, _layout_config, _mean_ci, _random_seeds
from time_to_intercept_oracle import _choose_assignment, _time_matrix


LAWS = ("baseline", "e1", "e2", "e3")
HORIZON_CREDIT = 3
TEAM_SIZE = 3


class _TracedPolicy(LocalAssignmentPolicy):
    """选圈规则与原来相同。act 若没走到选圈，原因记成对撞或承诺。"""

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        self.last_reason = "boost_or_commit"
        self.last_alternative = None
        action = super().act(observation)
        if self.last_reason == "boost_or_commit":
            self.last_target = None
        return action


def _team_parts(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    robot_radius: float,
    target_radius: float,
) -> tuple[float, float]:
    """覆盖率和碰撞参与率。碰撞分母固定是 3，拿掉一台车也不改。"""
    robot_count, target_count = len(robot_pos), len(target_pos)
    if robot_count == 0 or target_count == 0:
        return 0.0, 0.0
    distance = np.linalg.norm(target_pos[None, :, :] - robot_pos[:, None, :], axis=2)
    adjacency = (distance <= target_radius).astype(np.int32)
    matching = maximum_bipartite_matching(csr_matrix(adjacency), perm_type="column")
    matched = int(np.sum(np.asarray(matching) >= 0))
    hit = np.zeros(robot_count, dtype=bool)
    for left in range(robot_count):
        for right in range(left + 1, robot_count):
            if float(np.linalg.norm(robot_pos[left] - robot_pos[right])) < 2.0 * robot_radius:
                hit[left] = True
                hit[right] = True
    return matched / target_count, float(np.sum(hit)) / TEAM_SIZE


def _difference(
    robot_pos: np.ndarray,
    target_pos: np.ndarray,
    robot_radius: float,
    target_radius: float,
    weight: float,
) -> list[dict[str, float]]:
    coverage, collision = _team_parts(robot_pos, target_pos, robot_radius, target_radius)
    reward = coverage - weight * collision
    credits: list[dict[str, float]] = []
    for index in range(len(robot_pos)):
        kept = [i for i in range(len(robot_pos)) if i != index]
        cov_i, col_i = _team_parts(robot_pos[kept], target_pos, robot_radius, target_radius)
        credits.append(
            {
                "coverage": coverage - cov_i,
                "collision": (-weight * collision) - (-weight * col_i),
                "reward": reward - (cov_i - weight * col_i),
            }
        )
    return credits


def _open_loop(
    state: Any,
    policies: list[_TracedPolicy],
    targets: list[int | None],
    steps: int,
    weight: float,
) -> float:
    """每台车锁住一个圈，向前滚若干步，返回团队奖励之和。"""
    branch = copy.deepcopy(state)
    total = 0.0
    for _step in range(steps):
        actions: dict[str, np.ndarray] = {}
        for index in range(len(targets)):
            target = targets[index]
            if target is None:
                actions[f"agent_{index}"] = np.zeros(2, dtype=np.float32)
                continue
            relative = branch.target_positions[target] - branch.robot_positions[index]
            actions[f"agent_{index}"] = policies[index]._thrust_to_target(
                relative,
                branch.robot_velocities[index],
                branch.target_velocities[target],
            )
        advance_robots(branch, actions)
        advance_targets(branch)
        branch.step_index += 1
        metrics = compute_step_metrics(snapshot(branch))
        total += float(metrics.coverage_rate) - weight * float(metrics.collision_rate)
    return total


def _rollout(case: ScenarioCase, law: str) -> dict[str, float]:
    config = case.task_config
    env = make_training_env(config)
    observations, _info = env.reset(seed=int(case.scenario_seed))
    n = int(config.num_agents)
    m = int(config.num_targets)
    horizon = int(config.horizon)
    weight = float(config.collision_weight)
    radius = float(config.public.target_radius)
    robot_radius = float(config.public.robot_radius)
    task = _public_task_params(case)
    policies = [_TracedPolicy(law) for _robot in range(n)]
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=n,
                num_targets=m,
                horizon=horizon,
                task=task,
                policy_seed=0,
            )
        )
    total = 0.0
    coverage_sum = 0.0
    collision_sum = 0.0
    direct = 0.0
    redundant = 0
    inside = 0
    duplicate_steps = 0
    same_circle_steps = 0
    greedy_conflict = 0
    greedy_matches_team = 0
    reasons: dict[str, int] = {}
    credit_gap = 0.0
    cf_count = 0
    cf_helped = 0
    cf_hurt = 0
    cf_delta = 0.0
    cf1_delta = 0.0
    cf_by_reason: dict[str, list[float]] = {}
    try:
        for _step in range(horizon):
            robot_pos, robot_vel, target_pos, target_vel = _decode(env.state(), n, m)
            times = _time_matrix(
                robot_pos, robot_vel, target_pos, target_vel, horizon - _step, config
            )
            greedy: list[int] = []
            strict: list[bool] = []
            for index in range(n):
                order = np.argsort(times[index], kind="mergesort")
                greedy.append(int(order[0]))
                strict.append(float(times[index, order[0]]) + 1e-9 < float(times[index, order[1]]))
            fought: dict[int, int] = {}
            for index, target in enumerate(greedy):
                if strict[index]:
                    fought[target] = fought.get(target, 0) + 1
            if any(count >= 2 for count in fought.values()):
                greedy_conflict += 1
            team = _choose_assignment(times, None, 0)
            if tuple(greedy) == tuple(team):
                greedy_matches_team += 1

            actions = {
                agent_id: np.asarray(policies[index].act(observations[agent_id]), dtype=np.float32)
                for index, agent_id in enumerate(env.agents)
            }
            chosen = [policies[index].last_target for index in range(n)]
            named = [item for item in chosen if item is not None]
            if len(named) != len(set(named)):
                duplicate_steps += 1
            for index, reason in enumerate(reason for reason in (policies[i].last_reason for i in range(n))):
                reasons[reason] = reasons.get(reason, 0) + 1
            del index

            if law == "e2":
                state = env.unwrapped._scenario_state
                remain = min(HORIZON_CREDIT, horizon - int(state.step_index))
                if remain > 0:
                    for index, policy in enumerate(policies):
                        if not str(policy.last_reason).startswith("yield_") or policy.last_alternative is None:
                            continue
                        factual = list(chosen)
                        altered = list(chosen)
                        altered[index] = policy.last_alternative
                        keep = _open_loop(state, policies, factual, remain, weight)
                        grab = _open_loop(state, policies, altered, remain, weight)
                        delta = keep - grab
                        cf_count += 1
                        cf_delta += delta
                        cf_by_reason.setdefault(policy.last_reason, []).append(delta)
                        if delta > 1e-9:
                            cf_helped += 1
                        elif delta < -1e-9:
                            cf_hurt += 1
                        keep1 = _open_loop(state, policies, factual, 1, weight)
                        grab1 = _open_loop(state, policies, altered, 1, weight)
                        cf1_delta += keep1 - grab1

            observations, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage_sum += float(metrics.coverage_rate)
            collision_sum += float(metrics.collision_rate)
            snap = env.unwrapped._current_snapshot
            credits = _difference(
                snap.robot_positions, snap.target_positions, robot_radius, radius, weight
            )
            credit_gap += float(metrics.coverage_rate) - sum(item["coverage"] for item in credits)
            distance = np.linalg.norm(
                snap.target_positions[None, :, :] - snap.robot_positions[:, None, :], axis=2
            )
            if bool(np.any(np.sum(distance <= radius, axis=0) >= 2)):
                same_circle_steps += 1
            for index, item in enumerate(credits):
                if item["coverage"] > 0.2:
                    direct += item["coverage"]
                if bool(np.any(distance[index] <= radius)):
                    inside += 1
                    if item["coverage"] < 1e-9:
                        redundant += 1
    finally:
        env.close()

    row = {
        "j": total / horizon,
        "coverage": coverage_sum / horizon,
        "collision": collision_sum / horizon,
        "direct_coverage": direct / horizon,
        "redundant_steps": float(redundant),
        "inside_steps": float(inside),
        "duplicate_steps": float(duplicate_steps),
        "same_circle_steps": float(same_circle_steps),
        "greedy_conflict_steps": float(greedy_conflict),
        "greedy_matches_team_steps": float(greedy_matches_team),
        "credit_gap": credit_gap,
        "cf_count": float(cf_count),
        "cf_helped": float(cf_helped),
        "cf_hurt": float(cf_hurt),
        "cf_delta": cf_delta,
        "cf1_delta": cf1_delta,
    }
    for name in ("nearest", "earliest", "yield_faster", "yield_tie", "yield_reserve", "boost_or_commit", "save", "none"):
        row[f"reason_{name}"] = float(reasons.get(name, 0))
    for name, values in cf_by_reason.items():
        row[f"cf_delta_{name}"] = float(sum(values))
        row[f"cf_count_{name}"] = float(len(values))
    return row


def _evaluate_case(case: ScenarioCase) -> dict[str, Any]:
    row: dict[str, Any] = {"case_id": case.case_id, "group": case.group_id, "seed": int(case.scenario_seed)}
    for law in LAWS:
        result = _rollout(case, law)
        for key, value in result.items():
            row[f"{law}_{key}"] = value
    return row


def _paired(rows: list[dict[str, Any]], seeds: list[int], field: str) -> list[float]:
    lookup = {(int(row["seed"]), str(row["group"])): row for row in rows}
    return [
        500.0 * (float(lookup[(seed, "basic")][field]) + float(lookup[(seed, "cooperation")][field]))
        for seed in seeds
    ]


def _mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _summarize(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    base = _paired(rows, seeds, "baseline_j")
    laws: dict[str, Any] = {}
    for law in LAWS:
        scores = _paired(rows, seeds, f"{law}_j")
        gaps = [new - old for old, new in zip(base, scores)]
        wins = sum(gap > 1e-12 for gap in gaps)
        losses = sum(gap < -1e-12 for gap in gaps)
        coverage = _mean_field(rows, f"{law}_coverage")
        collision = _mean_field(rows, f"{law}_collision")
        laws[law] = {
            "score": float(np.mean(scores)),
            "gap_vs_baseline": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "wins": wins,
            "ties": len(gaps) - wins - losses,
            "losses": losses,
            "coverage": coverage,
            "collision": collision,
            "direct_coverage": _mean_field(rows, f"{law}_direct_coverage"),
            "redundant_steps_per_episode": _mean_field(rows, f"{law}_redundant_steps"),
            "inside_steps_per_episode": _mean_field(rows, f"{law}_inside_steps"),
            "duplicate_choice_steps_per_episode": _mean_field(rows, f"{law}_duplicate_steps"),
            "same_circle_steps_per_episode": _mean_field(rows, f"{law}_same_circle_steps"),
            "greedy_conflict_steps_per_episode": _mean_field(rows, f"{law}_greedy_conflict_steps"),
            "greedy_matches_team_steps_per_episode": _mean_field(rows, f"{law}_greedy_matches_team_steps"),
            "reasons_per_episode": {
                name: _mean_field(rows, f"{law}_reason_{name}")
                for name in ("nearest", "earliest", "yield_faster", "yield_tie", "yield_reserve", "boost_or_commit", "save", "none")
                if f"{law}_reason_{name}" in rows[0]
            },
        }
    base_cov = laws["baseline"]["coverage"]
    base_col = laws["baseline"]["collision"]
    e2_cov = laws["e2"]["coverage"]
    e2_col = laws["e2"]["collision"]
    cf_count = sum(float(row["e2_cf_count"]) for row in rows)
    cf_delta = sum(float(row["e2_cf_delta"]) for row in rows)
    cf1_delta = sum(float(row["e2_cf1_delta"]) for row in rows)
    by_reason: dict[str, Any] = {}
    for name in ("yield_faster", "yield_tie", "yield_reserve"):
        count_key = f"e2_cf_count_{name}"
        delta_key = f"e2_cf_delta_{name}"
        if count_key not in rows[0]:
            count = sum(float(row.get(count_key, 0.0)) for row in rows)
            delta = sum(float(row.get(delta_key, 0.0)) for row in rows)
        else:
            count = sum(float(row[count_key]) for row in rows)
            delta = sum(float(row[delta_key]) for row in rows)
        by_reason[name] = {
            "events": count,
            "mean_3step_delta": (delta / count) if count else 0.0,
        }
    return {
        "note": "脚本自算，不是 evaluate_one。差分奖励看拿掉这台车后这一步团队奖励掉多少。让位反事实把车锁在圈上滚 3 步，用的是真实目标速度，不是策略自己的估计。",
        "master_seed": MASTER_SEED,
        "seeds": len(seeds),
        "laws": laws,
        "e2_gap_from_coverage": 1000.0 * (e2_cov - base_cov),
        "e2_gap_from_collision": 200.0 * (base_col - e2_col),
        "credit_abs_gap_mean": _mean_field(rows, "e2_credit_gap"),
        "yield_counterfactual": {
            "events": cf_count,
            "helped": sum(float(row["e2_cf_helped"]) for row in rows),
            "hurt": sum(float(row["e2_cf_hurt"]) for row in rows),
            "mean_3step_delta": (cf_delta / cf_count) if cf_count else 0.0,
            "mean_1step_delta": (cf1_delta / cf_count) if cf_count else 0.0,
            "by_reason": by_reason,
        },
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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for row in rows:
        for key in fieldnames:
            row.setdefault(key, 0.0)
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, seeds)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
