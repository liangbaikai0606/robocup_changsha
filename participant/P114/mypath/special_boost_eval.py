"""E026：碰撞助推规则的人工特例诊断。

本脚本不修改官方环境，也不属于正式提交入口。它通过官方环境真实推进，
只把 reset 后的初始状态替换成几个可复现的人工布置，用来回答：

* 满足“近车、远圈 6 步单车够不到、碰撞后两车都能到圈”时，E024 是否触发；
* 不满足时，E024 是否保持不碰撞；
* 与关闭助推的同场景基线相比，覆盖、回报和碰撞怎样变化。

输出的 diagnostic_score = 1000 * 平均团队回报，仅用于比较，不能当正式成绩。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
P114_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(P114_ROOT) not in sys.path:
    sys.path.insert(0, str(P114_ROOT))

from coverage_bench.config import load_task_config
from coverage_bench.envs.factory import make_training_env
from coverage_bench.envs.scenario import snapshot
from coverage_bench.observations import observe_agent
from coverage_bench.protocol import EpisodeContext, PublicTaskParams
from entry import CoverFirstPolicy


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    robot_positions: tuple[tuple[float, float], ...]
    target_positions: tuple[tuple[float, float], ...]


CASES = (
    Case(
        name="helpful_collision",
        description="右车单车 6 步到不了右侧远圈；碰撞弹开后，左车仍能到左侧近圈",
        robot_positions=((-0.05, 0.0), (0.05, 0.0), (0.85, -0.85)),
        target_positions=((-0.35, 0.0), (0.45, 0.0), (0.0, 0.85)),
    ),
    Case(
        name="far_circle_not_shared",
        description="远圈离另一车超过感知半径，不应单方面发起碰撞",
        robot_positions=((-0.03, 0.0), (0.03, 0.0), (0.80, 0.80)),
        target_positions=((-0.23, 0.0), (0.72, 0.0), (0.0, -0.80)),
    ),
    Case(
        name="near_circle_would_be_lost",
        description="碰撞后留下的车无法保住近圈，不应触发",
        robot_positions=((-0.03, 0.0), (0.03, 0.0), (0.80, 0.80)),
        target_positions=((-0.70, 0.0), (0.57, 0.0), (0.0, -0.80)),
    ),
)


class NoBoostPolicy(CoverFirstPolicy):
    """E024 的同款追圈逻辑，但禁用碰撞助推，作为诊断对照。"""

    def _start_boost(self, observation):
        del observation
        return None


def _public_params(config):
    task = config.public
    return PublicTaskParams(
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
    )


def _prepare_env(config, case: Case, seed: int):
    env = make_training_env(config)
    env.reset(seed=seed)
    state = env._scenario_state
    state.robot_positions[:] = np.asarray(case.robot_positions, dtype=np.float64)
    state.robot_velocities[:] = 0.0
    state.target_positions[:] = np.asarray(case.target_positions, dtype=np.float64)
    state.target_velocities[:] = 0.0
    state.target_next_turn[:] = config.horizon + 100
    state.step_index = 0
    env._current_snapshot = snapshot(state)
    observations = {
        agent: observe_agent(env._current_snapshot, index, config, env.spec)
        for index, agent in enumerate(env.agents)
    }
    return env, observations


def _run_case(config, case: Case, policy_type, seed: int):
    env, observations = _prepare_env(config, case, seed)
    policies = [policy_type() for _ in range(config.num_agents)]
    task = _public_params(config)
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=config.num_agents,
                num_targets=config.num_targets,
                horizon=config.horizon,
                task=task,
                policy_seed=seed + index,
            )
        )

    total = 0.0
    coverages = []
    collisions = []
    boost_starts = 0
    try:
        for _step in range(config.horizon):
            actions = {}
            for index, agent in enumerate(env.agents):
                before = policies[index]._commit is not None
                actions[agent] = policies[index].act(observations[agent])
                after = policies[index]._commit is not None
                boost_starts += int(not before and after)
            observations, rewards, _terms, _truncs, infos = env.step(actions)
            first_agent = next(iter(rewards))
            total += float(rewards[first_agent])
            metrics = infos[first_agent]["metrics"]
            coverages.append(float(metrics.coverage_rate))
            collisions.append(float(metrics.collision_rate))
    finally:
        for policy in policies:
            policy.close()
        env.close()

    mean_return = total / config.horizon
    return {
        "return": total,
        "mean_j": mean_return,
        "diagnostic_score": 1000.0 * mean_return,
        "mean_coverage": float(np.mean(coverages)),
        "mean_collision": float(np.mean(collisions)),
        "boost_starts": boost_starts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026092701)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "P114" / "special-boost-e026",
    )
    args = parser.parse_args()
    config = load_task_config(REPO_ROOT / "configs" / "task-v1.yaml")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for case in CASES:
        boosted = _run_case(config, case, CoverFirstPolicy, args.seed)
        baseline = _run_case(config, case, NoBoostPolicy, args.seed)
        row = {
            "case": case.name,
            "description": case.description,
            "e024": boosted,
            "no_boost": baseline,
        }
        rows.append(row)

    (output / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "E026",
                "seed": args.seed,
                "cases": rows,
                "warning": "diagnostic_score is 1000 * mean team reward, not an official public-suite score",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "policy",
                "diagnostic_score",
                "mean_j",
                "mean_coverage",
                "mean_collision",
                "boost_starts",
            ]
        )
        for row in rows:
            for label in ("e024", "no_boost"):
                result = row[label]
                writer.writerow(
                    [
                        row["case"],
                        label,
                        f"{result['diagnostic_score']:.6f}",
                        f"{result['mean_j']:.6f}",
                        f"{result['mean_coverage']:.6f}",
                        f"{result['mean_collision']:.6f}",
                        result["boost_starts"],
                    ]
                )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
