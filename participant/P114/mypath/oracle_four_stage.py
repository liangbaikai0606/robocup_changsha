"""四阶段全局 Oracle：满油门、提前收油、反向制动、圈内跟速。

这是读取 ``env.state()`` 的诊断脚本，不是合法提交策略。它在每个场景中枚举
固定分配、目标提前量和少量控制参数，并与 E025 的旧固定分配 Oracle 对照。
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
from dataclasses import dataclass
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
from oracle_search import _decode, _rollout as old_rollout


MASTER_SEED = 20260927
PUBLIC_SEEDS = {1001, 1002, 2001, 2002}
LEADS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


@dataclass(frozen=True)
class ControlPreset:
    name: str
    hold_radius: float
    approach_radius: float
    stable_margin: float


PRESETS = (
    ControlPreset("center-early", 0.00, 0.40, 0.025),
    ControlPreset("center-late", 0.00, 0.28, 0.020),
    ControlPreset("inner-early", 0.05, 0.40, 0.020),
    ControlPreset("inner-mid", 0.05, 0.32, 0.015),
    ControlPreset("inner-late", 0.05, 0.25, 0.010),
    ControlPreset("wide-mid", 0.10, 0.32, 0.010),
)


def _stopping_distance(closing_speed: float, dt: float, damping: float) -> float:
    """满反向力时，沿目标方向的离散制动距离。"""
    speed = max(float(closing_speed), 0.0)
    distance = 0.0
    for _ in range(32):
        if speed <= 1e-6:
            break
        distance += speed * dt
        speed = max((1.0 - damping) * speed - dt, 0.0)
    return distance


def _inverse_velocity(
    velocity: np.ndarray,
    desired_velocity: np.ndarray,
    dt: float,
    damping: float,
) -> np.ndarray:
    action = (desired_velocity - (1.0 - damping) * velocity) / dt
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _four_stage_actions(
    state: np.ndarray,
    assignment: tuple[int, ...],
    config: TaskConfig,
    lead_steps: float,
    preset: ControlPreset,
) -> dict[str, np.ndarray]:
    """按四段状态生成联合动作；每台车仍保持固定目标分配。"""
    n = int(config.num_agents)
    m = int(config.num_targets)
    dt = float(config.public.dt)
    damping = float(config.public.damping)
    cover_radius = float(config.public.target_radius)
    robot_pos, robot_vel, target_pos, target_vel = _decode(state, n, m)
    actions: dict[str, np.ndarray] = {}

    for robot_index, target_index in enumerate(assignment):
        pos = robot_pos[robot_index]
        vel = robot_vel[robot_index]
        target = target_pos[target_index]
        target_v = target_vel[target_index]
        goal = target + target_v * (lead_steps * dt)

        # 本步动作不会改变本步位移，因此用双方“已经承诺”的下一拍位置判断。
        committed_robot = pos + vel * dt
        committed_target = target + target_v * dt
        committed_delta = committed_target - committed_robot
        committed_dist = float(np.linalg.norm(committed_delta))
        delta = goal - pos
        dist = float(np.linalg.norm(delta))
        direction = delta / dist if dist > 1e-9 else np.zeros(2, dtype=np.float64)
        relative_velocity = vel - target_v
        closing_speed = float(np.dot(relative_velocity, direction))
        stop_distance = _stopping_distance(closing_speed, dt, damping)
        room = max(committed_dist - preset.hold_radius, 0.0)

        stable_limit = max(cover_radius - preset.stable_margin, 0.0)
        if committed_dist <= stable_limit:
            # 已稳定进圈：消掉相对速度，跟随目标。
            desired_velocity = target_v
            action = _inverse_velocity(vel, desired_velocity, dt, damping)
        elif closing_speed > 0.0 and stop_distance >= room:
            # 现有速度会冲过目标：反向制动到目标速度。
            desired_velocity = target_v
            action = _inverse_velocity(vel, desired_velocity, dt, damping)
        elif committed_dist > preset.approach_radius:
            # 还远：朝提前点满油门。
            action = np.clip(delta * 10.0, -1.0, 1.0).astype(np.float32)
        else:
            # 接近：速度上限随剩余距离下降，提前收油。
            safe_speed = math.sqrt(max(2.0 * room, 0.0))
            proportional_speed = room / max(2.0 * dt, 1e-9)
            desired_closing = min(0.40, safe_speed, proportional_speed)
            desired_velocity = target_v + direction * desired_closing
            action = _inverse_velocity(vel, desired_velocity, dt, damping)

        actions[f"agent_{robot_index}"] = action
    return actions


def _rollout_four_stage(
    case: ScenarioCase,
    assignment: tuple[int, ...],
    lead_steps: float,
    preset: ControlPreset,
) -> tuple[float, float, float, float]:
    config = case.task_config
    env = make_training_env(config)
    env.reset(seed=case.scenario_seed)
    total = 0.0
    coverage: list[float] = []
    collision: list[float] = []
    try:
        for _ in range(int(config.horizon)):
            actions = _four_stage_actions(env.state(), assignment, config, lead_steps, preset)
            _obs, rewards, _terms, _truncs, infos = env.step(actions)
            agent = next(iter(rewards))
            total += float(rewards[agent])
            metrics = infos[agent]["metrics"]
            coverage.append(float(metrics.coverage_rate))
            collision.append(float(metrics.collision_rate))
    finally:
        env.close()
    horizon = int(config.horizon)
    return total, total / horizon, float(np.mean(coverage)), float(np.mean(collision))


def _best_for_case(case: ScenarioCase) -> dict[str, Any]:
    assignments = list(permutations(range(case.task_config.num_targets)))
    old_best: tuple[float, float, float, float, str] | None = None
    new_best: tuple[float, float, float, float, str] | None = None
    for assignment in assignments:
        for lead in (1.0, 2.0, 3.0, 4.0, 5.0):
            rec = old_rollout(case, assignment, lead)
            label = f"assign={assignment};lead={lead}"
            candidate = (*rec, label)
            if old_best is None or candidate[0] > old_best[0]:
                old_best = candidate
        for lead in LEADS:
            for preset in PRESETS:
                rec = _rollout_four_stage(case, assignment, lead, preset)
                label = f"assign={assignment};lead={lead};preset={preset.name}"
                candidate = (*rec, label)
                if new_best is None or candidate[0] > new_best[0]:
                    new_best = candidate
    assert old_best is not None and new_best is not None
    return {
        "case_id": case.case_id,
        "group": case.group_id,
        "seed": int(case.scenario_seed),
        "old_return": old_best[0],
        "old_j": old_best[1],
        "old_coverage": old_best[2],
        "old_collision": old_best[3],
        "old_plan": old_best[4],
        "four_return": new_best[0],
        "four_j": new_best[1],
        "four_coverage": new_best[2],
        "four_collision": new_best[3],
        "four_plan": new_best[4],
        "gap_j": new_best[1] - old_best[1],
    }


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


def _mean_ci(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return [mean, mean]
    half = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return [mean - half, mean + half]


def _summary(rows: list[dict[str, Any]], random_seeds: list[int]) -> dict[str, Any]:
    public_rows = [row for row in rows if str(row["case_id"]).startswith(("basic-", "coop-"))]
    random_rows = [row for row in rows if str(row["case_id"]).startswith("random-")]

    def score_for(items: list[dict[str, Any]], key: str) -> float:
        groups = {
            group: [float(row[key]) for row in items if row["group"] == group]
            for group in ("basic", "cooperation")
        }
        return 500.0 * (float(np.mean(groups["basic"])) + float(np.mean(groups["cooperation"])))

    paired_old: list[float] = []
    paired_four: list[float] = []
    lookup = {(int(row["seed"]), str(row["group"])): row for row in random_rows}
    for seed in random_seeds:
        basic = lookup[(seed, "basic")]
        coop = lookup[(seed, "cooperation")]
        paired_old.append(500.0 * (float(basic["old_j"]) + float(coop["old_j"])))
        paired_four.append(500.0 * (float(basic["four_j"]) + float(coop["four_j"])))
    gaps = [new - old for old, new in zip(paired_old, paired_four)]
    return {
        "oracle_warning": "Privileged diagnostic with hindsight parameter selection; not legal and not a mathematical upper bound.",
        "public": {
            "cases": public_rows,
            "old_score": score_for(public_rows, "old_j"),
            "four_stage_score": score_for(public_rows, "four_j"),
        },
        "random": {
            "master_seed": MASTER_SEED,
            "unique_seeds": len(random_seeds),
            "episodes": len(random_rows),
            "old_score": float(np.mean(paired_old)),
            "four_stage_score": float(np.mean(paired_four)),
            "four_minus_old": float(np.mean(gaps)),
            "gap_ci95_normal": _mean_ci(gaps),
            "four_better_fraction": float(np.mean(np.asarray(gaps) > 1e-12)),
            "ties_fraction": float(np.mean(np.abs(np.asarray(gaps)) <= 1e-12)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        parser.error("--count and --workers must be positive")

    public_suite = load_suite(REPO_ROOT / "configs" / "public-suite-v1.yaml")
    cases = [case for group in public_suite.groups for case in group.cases]
    random_seeds = _random_seeds(args.count, args.master_seed)
    basic = _layout_config("uniform")
    cooperation = _layout_config("crossing")
    for seed in random_seeds:
        cases.append(ScenarioCase(f"random-basic-{seed}", "basic", basic, seed))
        cases.append(ScenarioCase(f"random-cooperation-{seed}", "cooperation", cooperation, seed))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_best_for_case, case): case.case_id for case in cases}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 10 == 0 or index == len(cases):
                print(f"completed {index}/{len(cases)}", flush=True)

    rows.sort(key=lambda row: (str(row["group"]), int(row["seed"])))
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, random_seeds)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
