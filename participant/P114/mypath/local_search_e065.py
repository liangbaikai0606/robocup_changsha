"""E065：合法的局部在线动作搜索。诊断脚本，不替换 entry.py。

每台车只看 AgentObservation 和自己记下的两帧目标速度。
想象的未来用公开任务参数积分，目标碰到边界就反射。
看不见的队友、真实未来轨迹、env.state() 都不进动作。
baseline 是当前 CoverFirstPolicy。搜索只在它不准备对撞时替换追圈的那一步。
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MYPATH = Path(__file__).resolve().parent
ENTRY = ROOT / "participant" / "P114"
for _path in (ROOT, MYPATH, ENTRY):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import AgentObservation, EpisodeContext, PublicTaskParams
from entry import CoverFirstPolicy
from reachability_debounce_oracle import _layout_config, _mean_ci, _random_seeds

OUT = ROOT / "outputs" / "P114" / "local-search-e065"
DEV_MASTER = 20261003
HOLD_MASTER = 20261114
N_SEEDS = 12
HORIZONS = (3, 5, 10)
SHOOT_N = 32
CEM_POP = 64
CEM_ITERS = 2
ELITE = 16
SOFT_WEIGHT = 0.05


@dataclass
class _WorldGuess:
    """从一台车的局部观测还原出来的这一步场面。"""

    self_pos: np.ndarray
    self_vel: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray
    target_ok: np.ndarray
    peer_pos: np.ndarray
    peer_vel: np.ndarray
    peer_ok: np.ndarray


class LocalSearchPolicy:
    """滚动搜索自己接下来几步的力。只执行选出的第一步。"""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.base = CoverFirstPolicy()
        self._position_scale = 1.0
        self._velocity_scale = 1.0
        self._dt = 0.1
        self._damping = 0.25
        self._accel = 1.0
        self._max_speed = 1.0
        self._half = 1.0
        self._robot_r = 0.05
        self._target_r = 0.15
        self._cover_r = 0.15
        self.act_seconds = 0.0
        self.act_calls = 0

    def reset(self, context: EpisodeContext) -> None:
        task = context.task
        self.base.reset(context)
        self._position_scale = float(self.base._position_scale)
        self._velocity_scale = float(self.base._velocity_scale)
        self._dt = float(task.dt)
        self._damping = float(task.damping)
        self._accel = float(task.drive_force) / float(task.robot_mass) * self._dt
        self._max_speed = float(task.robot_max_speed)
        self._half = float(task.map_half_extent)
        self._robot_r = float(task.robot_radius)
        self._target_r = float(task.target_radius)
        self._cover_r = float(task.target_radius)

    def close(self) -> None:
        self.base.close()

    def act(self, observation: AgentObservation) -> np.ndarray:
        started = time.perf_counter()
        action = self._choose(observation)
        self.act_seconds += time.perf_counter() - started
        self.act_calls += 1
        return action

    def _choose(self, observation: AgentObservation) -> np.ndarray:
        if self.mode == "baseline":
            return self.base.act(observation)
        self.base._update_target_velocity(observation)
        if self.base._commit is not None:
            committed = self.base._follow_commit(observation)
            if committed is not None:
                return committed
        boost = self.base._start_boost(observation)
        if boost is not None:
            return boost
        baseline = np.asarray(self.base._chase(observation), dtype=np.float64)
        if self.mode.startswith("shoot"):
            horizon = int(self.mode.split("-")[1])
            return self._search(observation, baseline, horizon, SHOOT_N, 1, joint=False)
        if self.mode == "cem-self":
            return self._search(observation, baseline, 5, CEM_POP, CEM_ITERS, joint=False)
        if self.mode == "cem-joint":
            return self._search(observation, baseline, 5, CEM_POP, CEM_ITERS, joint=True)
        raise ValueError(self.mode)

    def _search(
        self,
        observation: AgentObservation,
        baseline: np.ndarray,
        horizon: int,
        population: int,
        iterations: int,
        joint: bool,
    ) -> np.ndarray:
        guess = self._guess(observation)
        if not np.any(guess.target_ok):
            return baseline.astype(np.float32)
        step = int(observation["step_index"])
        agent = int(observation["agent_index"])
        rng = np.random.default_rng(agent * 10007 + step * 17 + horizon)
        sequences = self._seed_sequences(baseline, horizon, population, rng)
        mean = np.repeat(baseline.reshape(1, 2), horizon, axis=0)
        std = np.full((horizon, 2), 0.6, dtype=np.float64)
        best_seq = sequences[0]
        best_score = -1e9
        for _ in range(iterations):
            scores = self._score_batch(guess, sequences, joint)
            order = np.argsort(scores)[::-1]
            if float(scores[order[0]]) > best_score:
                best_score = float(scores[order[0]])
                best_seq = sequences[order[0]]
            elite = sequences[order[:ELITE]]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), 0.05)
            sequences = mean + rng.normal(0.0, 1.0, size=(population, horizon, 2)) * std
            sequences = np.clip(sequences, -1.0, 1.0)
            sequences[0] = baseline.reshape(1, 2).repeat(horizon, axis=0)
            sequences[1] = best_seq
        return np.clip(best_seq[0], -1.0, 1.0).astype(np.float32)

    def _seed_sequences(
        self,
        baseline: np.ndarray,
        horizon: int,
        population: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        sequences = rng.uniform(-1.0, 1.0, size=(population, horizon, 2))
        hold = np.repeat(baseline.reshape(1, 2), horizon, axis=0)
        sequences[0] = hold
        sequences[1] = np.zeros((horizon, 2))
        sequences[2] = np.clip(hold * 1.5, -1.0, 1.0)
        brake = hold.copy()
        brake[horizon // 2 :] = -np.clip(hold[horizon // 2 :], -1.0, 1.0)
        sequences[3] = brake
        for axis in range(2):
            ones = np.zeros((horizon, 2))
            ones[:, axis] = 1.0
            sequences[4 + axis] = ones
            sequences[6 + axis] = -ones
        return np.clip(sequences, -1.0, 1.0)

    def _guess(self, observation: AgentObservation) -> _WorldGuess:
        state = np.asarray(observation["self_state"], dtype=np.float64)
        self_pos = state[:2] * self._position_scale
        self_vel = state[2:4] * self._velocity_scale
        targets = np.asarray(observation["targets"], dtype=np.float64)
        target_ok = np.asarray(observation["target_visible"], dtype=np.bool_)
        target_pos = np.zeros((targets.shape[0], 2), dtype=np.float64)
        target_vel = np.zeros_like(target_pos)
        for index, ok in enumerate(target_ok):
            if not ok:
                continue
            target_pos[index] = self_pos + targets[index, :2] * self._position_scale
            target_vel[index] = self.base._target_velocity_estimates.get(
                index, np.zeros(2, dtype=np.float64)
            )
        peers = np.asarray(observation["peers"], dtype=np.float64)
        peer_ok = np.asarray(observation["peer_visible"], dtype=np.bool_)
        peer_pos = np.zeros((peers.shape[0], 2), dtype=np.float64)
        peer_vel = np.zeros_like(peer_pos)
        for index, ok in enumerate(peer_ok):
            if not ok:
                continue
            peer_pos[index] = self_pos + peers[index, :2] * self._position_scale
            peer_vel[index] = self_vel + peers[index, 2:4] * self._velocity_scale
        return _WorldGuess(self_pos, self_vel, target_pos, target_vel, target_ok, peer_pos, peer_vel, peer_ok)

    def _score_batch(self, guess: _WorldGuess, sequences: np.ndarray, joint: bool) -> np.ndarray:
        """sequences: (P, H, 2)。返回每条序列的想象分。"""
        population, horizon, _ = sequences.shape
        pos = np.repeat(guess.self_pos.reshape(1, 2), population, axis=0)
        vel = np.repeat(guess.self_vel.reshape(1, 2), population, axis=0)
        target_pos = guess.target_pos.copy()
        target_vel = guess.target_vel.copy()
        peer_pos = np.repeat(guess.peer_pos.reshape(1, -1, 2), population, axis=0)
        peer_vel = np.repeat(guess.peer_vel.reshape(1, -1, 2), population, axis=0)
        scores = np.zeros(population, dtype=np.float64)
        robot_limit = self._half - self._robot_r
        target_limit = self._half - self._target_r
        for step in range(horizon):
            pos, vel = _integrate_robots(
                pos, vel, sequences[:, step, :], self._dt, self._damping, self._accel, self._max_speed, robot_limit
            )
            target_pos, target_vel = _reflect_targets(target_pos, target_vel, self._dt, target_limit)
            if joint:
                peer_action = _peer_pd(peer_pos, peer_vel, target_pos, target_vel, guess.target_ok)
                peer_pos, peer_vel = _integrate_robots(
                    peer_pos.reshape(-1, 2),
                    peer_vel.reshape(-1, 2),
                    peer_action.reshape(-1, 2),
                    self._dt,
                    self._damping,
                    self._accel,
                    self._max_speed,
                    robot_limit,
                )
                peer_pos = peer_pos.reshape(population, -1, 2)
                peer_vel = peer_vel.reshape(population, -1, 2)
            else:
                peer_pos = peer_pos + peer_vel * self._dt
                peer_pos = np.clip(peer_pos, -robot_limit, robot_limit)
            scores += _step_value(
                pos, target_pos, guess.target_ok, peer_pos, guess.peer_ok, self._cover_r, self._robot_r, joint
            )
        return scores


def _integrate_robots(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    dt: float,
    damping: float,
    accel: float,
    max_speed: float,
    limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = pos + vel * dt
    vel = (1.0 - damping) * vel + action * accel
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    too_fast = speed > max_speed
    vel = np.where(too_fast, vel * (max_speed / np.maximum(speed, 1e-9)), vel)
    pos = np.clip(pos, -limit, limit)
    outward_hi = (pos >= limit) & (vel > 0.0)
    outward_lo = (pos <= -limit) & (vel < 0.0)
    vel = np.where(outward_hi | outward_lo, 0.0, vel)
    return pos, vel


def _reflect_targets(
    pos: np.ndarray, vel: np.ndarray, dt: float, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    pos = pos + vel * dt
    over = pos > limit
    under = pos < -limit
    pos = np.where(over, 2.0 * limit - pos, pos)
    pos = np.where(under, -2.0 * limit - pos, pos)
    vel = np.where(over | under, -vel, vel)
    return pos, vel


def _peer_pd(
    peer_pos: np.ndarray,
    peer_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_ok: np.ndarray,
) -> np.ndarray:
    """可见队友按同一套位置加相对速度，朝自己最近的可见圈开。"""
    population, peer_count, _ = peer_pos.shape
    action = np.zeros((population, peer_count, 2), dtype=np.float64)
    if not np.any(target_ok):
        return action
    usable = target_pos[target_ok]
    usable_vel = target_vel[target_ok]
    delta = usable.reshape(1, 1, -1, 2) - peer_pos[:, :, None, :]
    dist = np.linalg.norm(delta, axis=-1)
    nearest = np.argmin(dist, axis=-1)
    offset = np.take_along_axis(delta, nearest[:, :, None, None], axis=2)[:, :, 0, :]
    speed = np.take_along_axis(usable_vel.reshape(1, 1, -1, 2), nearest[:, :, None, None], axis=2)[:, :, 0, :]
    action = np.clip(20.0 * offset + 4.0 * (speed - peer_vel), -1.0, 1.0)
    return action


def _step_value(
    self_pos: np.ndarray,
    target_pos: np.ndarray,
    target_ok: np.ndarray,
    peer_pos: np.ndarray,
    peer_ok: np.ndarray,
    cover_r: float,
    robot_r: float,
    joint: bool,
) -> np.ndarray:
    population = self_pos.shape[0]
    value = np.zeros(population, dtype=np.float64)
    if np.any(target_ok):
        usable = target_pos[target_ok]
        self_dist = np.linalg.norm(self_pos[:, None, :] - usable.reshape(1, -1, 2), axis=-1)
        nearest = self_dist.min(axis=1)
        value += SOFT_WEIGHT * np.exp(-nearest / 0.3)
        self_cover = self_dist <= cover_r
        if joint and np.any(peer_ok):
            peers = peer_pos[:, peer_ok, :]
            peer_dist = np.linalg.norm(peers[:, :, None, :] - usable.reshape(1, 1, -1, 2), axis=-1)
            peer_cover = peer_dist <= cover_r
            claimed = np.any(peer_cover, axis=1)
            self_cover = self_cover & ~claimed
        value += np.any(self_cover, axis=1).astype(np.float64) / 3.0
    if np.any(peer_ok):
        peers = peer_pos[:, peer_ok, :]
        gap = np.linalg.norm(self_pos[:, None, :] - peers, axis=-1)
        hit = np.any(gap < 2.0 * robot_r, axis=1)
        value -= 0.2 * hit.astype(np.float64) / 3.0
    return value


def _episode(mode: str, seed: int, layout: str) -> dict[str, float]:
    """用训练环境跑一局。分数和 E063 一样，是回报除以 10 步。"""
    case_task = _layout_config(layout)
    env = make_training_env(case_task)
    observations, _infos = env.reset(seed=seed)
    policies = [LocalSearchPolicy(mode) for _agent in env.agents]
    task = case_task.public
    for index, policy in enumerate(policies):
        policy.reset(
            EpisodeContext(
                agent_index=index,
                num_agents=int(case_task.num_agents),
                num_targets=int(case_task.num_targets),
                horizon=int(case_task.horizon),
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
        )
    horizon = int(case_task.horizon)
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
    act_ms = [1000.0 * policy.act_seconds / max(policy.act_calls, 1) for policy in policies]
    return {
        "j": total / horizon,
        "coverage": float(np.mean(coverage)),
        "collision": float(np.mean(collision)),
        "act_ms": float(np.mean(act_ms)),
    }


def _run_mode(mode: str, master: int) -> dict[str, object]:
    seeds = _random_seeds(N_SEEDS, master)
    jobs = [(mode, int(seed), layout) for seed in seeds for layout in ("uniform", "crossing")]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_episode, *zip(*jobs)))
    by_seed: dict[int, dict[str, dict[str, float]]] = {}
    act_ms = []
    for (mode_name, seed, layout), row in zip(jobs, rows):
        by_seed.setdefault(seed, {})[layout] = row
        act_ms.append(row["act_ms"])
    paired = [
        500.0 * (layouts["uniform"]["j"] + layouts["crossing"]["j"])
        for layouts in by_seed.values()
    ]
    coverages = [row["coverage"] for row in rows]
    collisions = [row["collision"] for row in rows]
    ordered = np.sort(np.asarray(paired, dtype=np.float64))
    return {
        "mode": mode,
        "master": master,
        "seed_count": N_SEEDS,
        "legal": "local observation only; target velocity from two frames; boundary reflection; no env.state",
        "elapsed_sec": time.perf_counter() - started,
        "paired_score": float(np.mean(paired)),
        "ci95": _mean_ci(paired),
        "p10": float(np.quantile(ordered, 0.10)),
        "p50": float(np.quantile(ordered, 0.50)),
        "p90": float(np.quantile(ordered, 0.90)),
        "coverage_mean": float(np.mean(coverages)),
        "collision_mean": float(np.mean(collisions)),
        "act_ms_mean": float(np.mean(act_ms)),
        "act_ms_p90": float(np.quantile(act_ms, 0.9)),
        "per_seed": {
            str(seed): {layout: row["j"] for layout, row in layouts.items()}
            for seed, layouts in by_seed.items()
        },
    }


def _compare(base: dict[str, object], other: dict[str, object]) -> dict[str, object]:
    base_scores = []
    other_scores = []
    for seed, layouts in base["per_seed"].items():
        base_scores.append(500.0 * (layouts["uniform"] + layouts["crossing"]) / 10.0)
        other_layouts = other["per_seed"][seed]
        other_scores.append(500.0 * (other_layouts["uniform"] + other_layouts["crossing"]) / 10.0)
    delta = np.asarray(other_scores) - np.asarray(base_scores)
    wins = int(np.sum(delta > 1e-9))
    losses = int(np.sum(delta < -1e-9))
    ties = int(delta.size - wins - losses)
    mean = float(delta.mean())
    half = 1.96 * float(delta.std(ddof=1) / np.sqrt(delta.size))
    return {
        "versus": other["mode"],
        "delta": mean,
        "ci95": [mean - half, mean + half],
        "win_tie_loss": [wins, ties, losses],
    }


def main() -> None:
    modes = ["baseline", "shoot-3", "shoot-5", "shoot-10", "cem-self", "cem-joint"]
    dev = [_run_mode(mode, DEV_MASTER) for mode in modes]
    best_name = max(
        (row for row in dev if row["mode"] != "baseline"),
        key=lambda row: float(row["paired_score"]),
    )["mode"]
    hold_modes = ["baseline", best_name]
    hold = [_run_mode(mode, HOLD_MASTER) for mode in hold_modes]
    dev_base = dev[0]
    hold_base = hold[0]
    summary = {
        "label": "diagnostic local search, not evaluate_one, entry.py unchanged",
        "changed": "action search on top of CoverFirstPolicy; physics, assignment fallback and seeds splits fixed inside each arm",
        "dev_master": DEV_MASTER,
        "hold_master": HOLD_MASTER,
        "seed_count_each": N_SEEDS,
        "note": "12 seeds per split, not the full 300. Holdout 20261114 was not used to choose Kp or assignment.",
        "dev": [{key: value for key, value in row.items() if key != "per_seed"} | {"comparisons": _compare(dev_base, row)} for row in dev],
        "holdout": [{key: value for key, value in row.items() if key != "per_seed"} | {"comparisons": _compare(hold_base, row)} for row in hold],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"dev": summary["dev"], "holdout": summary["holdout"]}, indent=2))


if __name__ == "__main__":
    main()
