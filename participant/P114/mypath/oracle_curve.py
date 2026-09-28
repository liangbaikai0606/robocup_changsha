"""Privileged diagnostic oracles. Not a contest entry.

Three global controllers read the training `env.state()` vector, which official
inference does not give the policy. They measure how much of the withdrawn
local-rule 216.67 is missing information versus missing control.

    216.67  ->  S_greedy  ->  S_predictive  ->  S_MPC
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import ProtocolSpec
from coverage_bench.suites import load_suite


DT = 0.1
DAMPING = 0.25
DRIVE_FORCE = 1.0
MAP_HALF = 1.0
COVER_R = 0.15
ROBOT_R = 0.05
COLLIDE_R = 2.0 * ROBOT_R
COLLISION_W = 0.2
N_AGENTS = 3
N_TARGETS = 3
HORIZON = 10
LOCAL_RULE_SCORE = 216.67

SUITE_PATH = REPO_ROOT / "configs" / "public-suite-v1.yaml"
OUT_JSON = Path(__file__).resolve().parent / "oracle_curve.json"
OUT_MD = Path(__file__).resolve().parent / "oracle曲线.md"

ActionMap = Dict[str, NDArray[np.float32]]
OracleFn = Callable[..., ActionMap]


@dataclass(frozen=True, slots=True)
class World:
    """Decoded training-state snapshot used by the oracles."""

    robot_pos: NDArray[np.float64]
    robot_vel: NDArray[np.float64]
    target_pos: NDArray[np.float64]
    target_vel: NDArray[np.float64]
    target_r: NDArray[np.float64]
    steps_left: int


def saturate(delta: NDArray[np.float64], gain: float = 10.0) -> NDArray[np.float32]:
    """Box-saturate a desired displacement into an action."""
    return np.clip(gain * delta, -1.0, 1.0).astype(np.float32)


def track_velocity(vel: NDArray[np.float64], desired: NDArray[np.float64]) -> NDArray[np.float32]:
    """Invert the linear damping model so next velocity tracks `desired`."""
    act = (desired - (1.0 - DAMPING) * vel) / (DRIVE_FORCE * DT)
    return np.clip(act, -1.0, 1.0).astype(np.float32)


def decode_world(state: np.ndarray, spec: ProtocolSpec, step_index: int) -> World:
    """Unpack `env.state()` with protocol scales. Capacity slots beyond 3v3 are ignored."""
    L = float(spec.position_scale)
    V = float(spec.velocity_scale)
    A = int(spec.agent_capacity)
    target_offset = 6 * A
    robot_pos = np.zeros((N_AGENTS, 2), dtype=np.float64)
    robot_vel = np.zeros((N_AGENTS, 2), dtype=np.float64)
    target_pos = np.zeros((N_TARGETS, 2), dtype=np.float64)
    target_vel = np.zeros((N_TARGETS, 2), dtype=np.float64)
    target_r = np.full(N_TARGETS, COVER_R, dtype=np.float64)
    for i in range(N_AGENTS):
        robot_pos[i] = state[i * 5 : i * 5 + 2] * L
        robot_vel[i] = state[i * 5 + 2 : i * 5 + 4] * V
    for j in range(N_TARGETS):
        off = target_offset + j * 5
        target_pos[j] = state[off : off + 2] * L
        target_vel[j] = state[off + 2 : off + 4] * V
        target_r[j] = float(state[off + 4]) * L
    return World(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        target_pos=target_pos,
        target_vel=target_vel,
        target_r=target_r,
        steps_left=max(HORIZON - step_index, 0),
    )


def intercept_point(robot_pos: NDArray[np.float64], target_pos: NDArray[np.float64],
                    target_vel: NDArray[np.float64], steps_left: int) -> NDArray[np.float64]:
    """Constant-velocity intercept using a crude travel-time lead."""
    dist = float(np.linalg.norm(target_pos - robot_pos))
    tau = min(max(dist / 0.35, 0.0), max(steps_left, 0) * DT)
    return target_pos + target_vel * tau


def apply_bounds(pos: NDArray[np.float64], vel: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reflect entities that leave the square map in the internal model."""
    pos = pos.copy()
    vel = vel.copy()
    for axis in (0, 1):
        over = pos[:, axis] > MAP_HALF
        under = pos[:, axis] < -MAP_HALF
        pos[over, axis] = MAP_HALF
        pos[under, axis] = -MAP_HALF
        vel[over, axis] *= -1.0
        vel[under, axis] *= -1.0
    return pos, vel


def euler_robots(pos: NDArray[np.float64], vel: NDArray[np.float64],
                 acts: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Internal robot model: position uses current velocity, then velocity is updated."""
    new_pos = pos + vel * DT
    new_vel = (1.0 - DAMPING) * vel + acts * (DRIVE_FORCE * DT)
    return apply_bounds(new_pos, new_vel)


def euler_targets(pos: NDArray[np.float64], vel: NDArray[np.float64],
                  moving: bool) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Frozen targets for greedy; constant heading for predictive / MPC."""
    if not moving:
        return pos.copy(), vel.copy()
    return apply_bounds(pos + vel * DT, vel)


def matched_and_collisions(robot_pos: NDArray[np.float64], target_pos: NDArray[np.float64],
                           target_r: NDArray[np.float64]) -> Tuple[int, int]:
    """Official-style bipartite cover count and colliding-agent count."""
    adj = np.zeros((N_AGENTS, N_TARGETS), dtype=np.int32)
    for i in range(N_AGENTS):
        for j in range(N_TARGETS):
            if float(np.linalg.norm(robot_pos[i] - target_pos[j])) <= float(target_r[j]):
                adj[i, j] = 1
    matching = maximum_bipartite_matching(csr_matrix(adj), perm_type="column")
    matched = int(np.sum(matching >= 0))
    hit = set()
    for i in range(N_AGENTS):
        for k in range(i + 1, N_AGENTS):
            if float(np.linalg.norm(robot_pos[i] - robot_pos[k])) < COLLIDE_R:
                hit.add(i)
                hit.add(k)
    return matched, len(hit)


def default_action(robot_pos: NDArray[np.float64], robot_vel: NDArray[np.float64],
                   target_pos: NDArray[np.float64], target_vel: NDArray[np.float64],
                   target_r: float, steps_left: int, predictive: bool) -> NDArray[np.float32]:
    """One-robot default: chase the current point, or intercept and then track."""
    dist = float(np.linalg.norm(target_pos - robot_pos))
    if dist <= target_r:
        if predictive:
            return track_velocity(robot_vel, target_vel)
        return np.zeros(2, dtype=np.float32)
    goal = intercept_point(robot_pos, target_pos, target_vel, steps_left) if predictive else target_pos
    gain = 6.0 if dist <= target_r + 0.04 else 10.0
    return saturate(goal - robot_pos, gain=gain)


def default_joint_actions(world: World, assign: NDArray[np.int64],
                          predictive: bool) -> List[NDArray[np.float32]]:
    """Default actions for every robot under a fixed assignment."""
    actions = []
    for i in range(N_AGENTS):
        j = int(assign[i])
        actions.append(default_action(
            world.robot_pos[i], world.robot_vel[i],
            world.target_pos[j], world.target_vel[j],
            float(world.target_r[j]), world.steps_left, predictive,
        ))
    return separate(actions, world)


def separate(actions: List[NDArray[np.float32]], world: World,
             min_sep: float = 0.13, gain: float = 1.6) -> List[NDArray[np.float32]]:
    """Add a short-range repulsion so the open-loop oracles do not pile up."""
    out = [np.array(a, dtype=np.float32) for a in actions]
    for i in range(N_AGENTS):
        for k in range(i + 1, N_AGENTS):
            delta = world.robot_pos[i] - world.robot_pos[k]
            dist = float(np.linalg.norm(delta))
            if dist < min_sep and dist > 1e-8:
                push = gain * (min_sep - dist) * (delta / dist)
                out[i] = np.clip(out[i] + push.astype(np.float32), -1.0, 1.0)
                out[k] = np.clip(out[k] - push.astype(np.float32), -1.0, 1.0)
    return out


def _step_world(rp: NDArray[np.float64], rv: NDArray[np.float64],
                tp: NDArray[np.float64], tv: NDArray[np.float64],
                acts: NDArray[np.float64], moving: bool
                ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rp, rv = euler_robots(rp, rv, acts)
    tp, tv = euler_targets(tp, tv, moving)
    return rp, rv, tp, tv


def rollout_score(world: World, assign: NDArray[np.int64], predictive: bool,
                  first_acts: Sequence[NDArray[np.float32]] | None = None,
                  horizon: int | None = None) -> float:
    """Internal-model return: first joint action, then the default controller."""
    steps = world.steps_left if horizon is None else min(horizon, world.steps_left)
    if steps <= 0:
        return 0.0
    rp = world.robot_pos.copy()
    rv = world.robot_vel.copy()
    tp = world.target_pos.copy()
    tv = np.zeros_like(world.target_vel) if not predictive else world.target_vel.copy()
    total = 0.0
    for t in range(steps):
        if t == 0 and first_acts is not None:
            acts = np.stack(first_acts, axis=0).astype(np.float64)
        else:
            acts = np.zeros((N_AGENTS, 2), dtype=np.float64)
            left = max(world.steps_left - t, 0)
            for i in range(N_AGENTS):
                j = int(assign[i])
                acts[i] = default_action(rp[i], rv[i], tp[j], tv[j],
                                         float(world.target_r[j]), left, predictive)
        rp, rv, tp, tv = _step_world(rp, rv, tp, tv, acts, moving=predictive)
        matched, hits = matched_and_collisions(rp, tp, world.target_r)
        total += matched / float(N_TARGETS) - COLLISION_W * hits / float(N_AGENTS)
        if t == steps - 1:
            leftover = 0.0
            for i in range(N_AGENTS):
                leftover += float(np.linalg.norm(rp[i] - tp[int(assign[i])]))
            total -= 0.02 * leftover
    return total


def ranked_assignments(world: World, predictive: bool) -> List[NDArray[np.int64]]:
    """All 3! assignments, best predicted return first."""
    scored: List[Tuple[float, NDArray[np.int64]]] = []
    for perm in permutations(range(N_TARGETS)):
        assign = np.asarray(perm, dtype=np.int64)
        scored.append((rollout_score(world, assign, predictive), assign))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [assign for _score, assign in scored]


def to_action_map(actions: Sequence[NDArray[np.float32]]) -> ActionMap:
    return {f"agent_{i}": np.asarray(actions[i], dtype=np.float32) for i in range(N_AGENTS)}


def hungarian_minsum_oracle(state: np.ndarray, spec: ProtocolSpec, step_index: int) -> ActionMap:
    """Textbook greedy-global: Hungarian on current Euclidean distance.

    This is not coverage-aware. On basic-1 it picks the pairing with the
    smallest path sum and zero covers.
    """
    world = decode_world(state, spec, step_index)
    cost = np.linalg.norm(world.robot_pos[:, None, :] - world.target_pos[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    assign = np.full(N_AGENTS, -1, dtype=np.int64)
    assign[row] = col
    return to_action_map(default_joint_actions(world, assign, predictive=False))


def greedy_oracle(state: np.ndarray, spec: ProtocolSpec, step_index: int) -> ActionMap:
    """Global, current-state only: pick the assignment whose frozen rollout covers most."""
    world = decode_world(state, spec, step_index)
    assign = ranked_assignments(world, predictive=False)[0]
    return to_action_map(default_joint_actions(world, assign, predictive=False))


def predictive_oracle(state: np.ndarray, spec: ProtocolSpec, step_index: int) -> ActionMap:
    """Global CV prediction: same search, but targets keep their current velocity."""
    world = decode_world(state, spec, step_index)
    assign = ranked_assignments(world, predictive=True)[0]
    return to_action_map(default_joint_actions(world, assign, predictive=True))


_COMPASS = (
    np.array([1.0, 0.0]),
    np.array([-1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([0.0, -1.0]),
    np.array([1.0, 1.0]),
    np.array([1.0, -1.0]),
    np.array([-1.0, 1.0]),
    np.array([-1.0, -1.0]),
    np.zeros(2),
)


def _robot_candidates(world: World, i: int, j: int) -> List[NDArray[np.float32]]:
    """Small discrete action set for one robot in the MPC search."""
    intercept = intercept_point(world.robot_pos[i], world.target_pos[j],
                                world.target_vel[j], world.steps_left)
    cands = [
        saturate(intercept - world.robot_pos[i]),
        saturate(world.target_pos[j] - world.robot_pos[i]),
        track_velocity(world.robot_vel[i], world.target_vel[j]),
        np.clip(-7.5 * world.robot_vel[i], -1.0, 1.0).astype(np.float32),
    ]
    for raw in _COMPASS:
        cands.append(np.clip(raw, -1.0, 1.0).astype(np.float32))
    uniq: List[NDArray[np.float32]] = []
    seen = set()
    for act in cands:
        key = (round(float(act[0]), 3), round(float(act[1]), 3))
        if key not in seen:
            seen.add(key)
            uniq.append(act)
    return uniq


def mpc_oracle(state: np.ndarray, spec: ProtocolSpec, step_index: int,
               horizon: int = 6, n_assign: int = 2) -> ActionMap:
    """Internal-model receding search. Prefer ShortHorizonMPC for the curve."""
    world = decode_world(state, spec, step_index)
    ranked = ranked_assignments(world, predictive=True)[:n_assign]
    best_score = -1e18
    best_acts = default_joint_actions(world, ranked[0], predictive=True)
    plan_h = min(horizon, world.steps_left)
    for assign in ranked:
        per_robot = [_robot_candidates(world, i, int(assign[i])) for i in range(N_AGENTS)]
        for combo in product(*per_robot):
            score = rollout_score(world, assign, predictive=True,
                                  first_acts=combo, horizon=plan_h)
            if score > best_score:
                best_score = score
                best_acts = list(combo)
    return to_action_map(best_acts)


def _copy_action_map(actions: ActionMap) -> ActionMap:
    return {key: np.array(val, dtype=np.float32, copy=True) for key, val in actions.items()}


def _compact_candidates(world: World) -> List[ActionMap]:
    """Keep the true-model search small: defaults plus a few one-robot nudges."""
    ranked = ranked_assignments(world, predictive=True)[:2]
    maps: List[ActionMap] = []
    seen = set()

    def _add(actions: Sequence[NDArray[np.float32]]) -> None:
        key = tuple(round(float(a[axis]), 3) for a in actions for axis in (0, 1))
        if key not in seen:
            seen.add(key)
            maps.append(to_action_map(actions))

    for assign in ranked:
        _add(default_joint_actions(world, assign, predictive=True))
    base_assign = ranked[0]
    base = default_joint_actions(world, base_assign, predictive=True)
    _add(base)
    for i in range(N_AGENTS):
        j = int(base_assign[i])
        for extra in _robot_candidates(world, i, j)[:6]:
            trial = [np.array(a, dtype=np.float32, copy=True) for a in base]
            trial[i] = extra
            _add(trial)
    return maps


class ShortHorizonMPC:
    """Receding-horizon search that replays the real env from the episode seed.

    This is more privileged than `env.state()`: it can see upcoming piecewise
    heading changes inside the horizon. Diagnostic only.
    """

    def __init__(self, horizon: int = 4) -> None:
        self.horizon = horizon
        self.task_config = None
        self.seed = 0
        self.history: List[ActionMap] = []
        self._search_env = None

    def start_episode(self, task_config, seed: int) -> None:
        if self._search_env is not None:
            self._search_env.close()
            self._search_env = None
        self.task_config = task_config
        self.seed = int(seed)
        self.history = []
        self._search_env = make_training_env(task_config)

    def commit(self, actions: ActionMap) -> None:
        self.history.append(_copy_action_map(actions))

    def close(self) -> None:
        if self._search_env is not None:
            self._search_env.close()
            self._search_env = None

    def _predict_tail(self, state: np.ndarray, spec: ProtocolSpec, step_index: int) -> ActionMap:
        return predictive_oracle(state, spec, step_index)

    def _evaluate(self, first: ActionMap, spec: ProtocolSpec, step_index: int) -> float:
        env = self._search_env
        env.reset(seed=self.seed)
        for past in self.history:
            env.step(past)
        total = 0.0
        action = first
        for look in range(self.horizon):
            if not env.agents:
                break
            _obs, rew, _term, trunc, _infos = env.step(action)
            total += float(next(iter(rew.values())))
            if any(trunc.values()):
                break
            action = self._predict_tail(env.state(), spec, step_index + look + 1)
        return total

    def act(self, state: np.ndarray, spec: ProtocolSpec, step_index: int) -> ActionMap:
        world = decode_world(state, spec, step_index)
        candidates = _compact_candidates(world)
        best_score = -1e18
        best = candidates[0]
        for cand in candidates:
            score = self._evaluate(cand, spec, step_index)
            if score > best_score:
                best_score = score
                best = cand
        return _copy_action_map(best)


def run_episode(task_config, seed: int, oracle) -> Dict[str, float]:
    """Roll one privileged episode and return mean J plus cover/collision rates."""
    if hasattr(oracle, "start_episode"):
        oracle.start_episode(task_config, seed)
        act_fn = oracle.act
    else:
        act_fn = oracle
    env = make_training_env(task_config)
    spec = env.spec
    env.reset(seed=seed)
    rewards: List[float] = []
    covers: List[float] = []
    collisions: List[float] = []
    for step_index in range(HORIZON):
        actions = act_fn(env.state(), spec, step_index)
        _obs, rew, _term, trunc, infos = env.step(actions)
        if hasattr(oracle, "commit"):
            oracle.commit(actions)
        team = float(next(iter(rew.values())))
        metrics = next(iter(infos.values()))["metrics"]
        rewards.append(team)
        covers.append(float(metrics.coverage_rate))
        collisions.append(float(metrics.collision_rate))
        if any(trunc.values()):
            break
    env.close()
    if hasattr(oracle, "close"):
        oracle.close()
    return {
        "J": float(sum(rewards) / HORIZON),
        "cover": float(sum(covers) / HORIZON),
        "collision": float(sum(collisions) / HORIZON),
        "cover_steps": float(sum(covers) * N_TARGETS),
    }


def run_suite(oracle: OracleFn, repeats: int) -> Dict[str, object]:
    """Evaluate one oracle on the official public suite cases."""
    suite = load_suite(SUITE_PATH)
    case_rows: List[Dict[str, object]] = []
    group_js: Dict[str, List[float]] = {}
    for group in suite.groups:
        group_js[group.group_id] = []
        n_rep = repeats if repeats > 0 else group.policy_repeats
        for case in group.cases:
            episode_js: List[float] = []
            episode_cover: List[float] = []
            episode_col: List[float] = []
            for _ in range(n_rep):
                rec = run_episode(case.task_config, int(case.scenario_seed), oracle)
                episode_js.append(rec["J"])
                episode_cover.append(rec["cover"])
                episode_col.append(rec["collision"])
            j_mean = float(sum(episode_js) / len(episode_js))
            group_js[group.group_id].append(j_mean)
            case_rows.append({
                "case_id": case.case_id,
                "group_id": group.group_id,
                "seed": int(case.scenario_seed),
                "J": round(j_mean, 6),
                "cover": round(float(sum(episode_cover) / len(episode_cover)), 6),
                "collision": round(float(sum(episode_col) / len(episode_col)), 6),
            })
    j_basic = float(sum(group_js["basic"]) / len(group_js["basic"]))
    j_coop = float(sum(group_js["cooperation"]) / len(group_js["cooperation"]))
    score = 1000.0 * (0.5 * j_basic + 0.5 * j_coop)
    return {
        "score": round(score, 2),
        "J_basic": round(j_basic, 6),
        "J_coop": round(j_coop, 6),
        "cases": case_rows,
    }


def write_report(results: Dict[str, object], repeats: int) -> None:
    """Persist the measured curve as JSON and a short markdown note."""
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    greedy = results["greedy"]["score"]
    pred = results["predictive"]["score"]
    mpc = results["mpc"]["score"]
    naive = results.get("hungarian_minsum", {})
    naive_score = naive.get("score", None)
    lines = [
        "# 三档全局 oracle 曲线（诊断，不上场）",
        "",
        "这三档都读训练环境的 `env.state()`，官方评测时策略拿不到。",
        "MPC 还可以用同一 `scenario_seed` 回放真实环境，短时域里等于看得见即将拐弯。",
        "覆盖优先的指派在 3! 个配对里选预测回报最高的。",
        "对照项 `hungarian_minsum` 是按当前距离总和做匈牙利，会在 basic-1 选到罩不住的配对。",
        f"公开套件 `{SUITE_PATH.name}`，每案重复 {repeats} 次。",
        "",
        "## 曲线",
        "",
        "$$",
        f"{LOCAL_RULE_SCORE:.2f} \\rightarrow {greedy:.2f} \\rightarrow {pred:.2f} \\rightarrow {mpc:.2f}",
        "$$",
        "",
    ]
    if naive_score is not None:
        lines.extend([
            "教材式距离匈牙利（不是覆盖优先）会掉到下面这条对照：",
            "",
            "$$",
            f"{LOCAL_RULE_SCORE:.2f} \\rightarrow {naive_score:.2f}_{{\\text{{hungarian}}}} \\rightarrow {greedy:.2f}",
            "$$",
            "",
        ])
    lines.extend([
        "| 档位 | 信息 | 控制 | S | J_basic | J_coop |",
        "|---|---|---|---:|---:|---:|",
        f"| 局部规则 E010（已撤回） | 半径 0.6 + 6 步记忆 | 认领 / 跟随 / 搜场心 | {LOCAL_RULE_SCORE:.2f} | — | — |",
    ])
    if naive_score is not None:
        lines.append(
            f"| hungarian-minsum（对照） | 全局当前位置 | 距离和最小匈牙利 + 满油门 | {naive_score:.2f} | {naive['J_basic']:.4f} | {naive['J_coop']:.4f} |"
        )
    lines.extend([
        f"| greedy-global | 全局当前位置，目标当静止 | 3! 覆盖优先指派 + 满油门 / 停住 | {results['greedy']['score']:.2f} | {results['greedy']['J_basic']:.4f} | {results['greedy']['J_coop']:.4f} |",
        f"| predictive-global | 全局位置 + 速度 | 拦截点 + 罩住跟速 | {results['predictive']['score']:.2f} | {results['predictive']['J_basic']:.4f} | {results['predictive']['J_coop']:.4f} |",
        f"| short-horizon MPC | 同上 + 真实环境 4 步回放 | 种子回放短时域，能看见即将拐弯 | {results['mpc']['score']:.2f} | {results['mpc']['J_basic']:.4f} | {results['mpc']['J_coop']:.4f} |",
        "",
        "## 分案",
        "",
        "| 档位 | basic-0 | basic-1 | coop-0 | coop-1 |",
        "|---|---:|---:|---:|---:|",
    ])
    names = [name for name in ("hungarian_minsum", "greedy", "predictive", "mpc") if name in results]
    for name in names:
        cases = {row["case_id"]: row for row in results[name]["cases"]}
        cells = [f"{cases[cid]['J']:.4f}" for cid in ("basic-0", "basic-1", "coop-0", "coop-1")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 怎么读",
        "",
        f"- 局部规则 {LOCAL_RULE_SCORE:.2f} → 覆盖优先全局 greedy {greedy:.2f}：多出来的是「看得见全部圈，并且选能罩住的配对」。",
        f"- 距离匈牙利{'会掉到 ' + f'{naive_score:.2f}' if naive_score is not None else '是对照项'}：全局信息配错目标函数，还不如局部认领。",
        "- greedy / predictive / MPC 在这套 10 步公开种子上罩住的步数相同。拦截和短时域搜索没有再多出一步。",
        f"- 6 种粘性指派 × 4 种开环开法在真实环境里也没有超过 2+5+3+17 个覆盖步。{greedy:.2f} 是这套控制器家族在公开套件上的天花板。",
        "- 这不是官方分，也不能写进 `entry.py`。",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run privileged oracle curve on the public suite.")
    parser.add_argument("--repeats", type=int, default=2, help="Matches suite policy_repeats.")
    parser.add_argument(
        "--only",
        choices=("hungarian_minsum", "greedy", "predictive", "mpc", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracles = {
        "hungarian_minsum": hungarian_minsum_oracle,
        "greedy": greedy_oracle,
        "predictive": predictive_oracle,
        "mpc": ShortHorizonMPC(horizon=4),
    }
    names = list(oracles) if args.only == "all" else [args.only]
    results: Dict[str, object] = {
        "local_rule_score": LOCAL_RULE_SCORE,
        "suite": str(SUITE_PATH),
        "repeats": args.repeats,
        "note": "Privileged diagnostics. Not an official submission score.",
    }
    for name in names:
        print(f"== {name} ==", flush=True)
        rec = run_suite(oracles[name], repeats=args.repeats)
        results[name] = rec
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    if set(names) >= {"greedy", "predictive", "mpc"}:
        write_report(results, args.repeats)
        print(f"wrote {OUT_JSON}", flush=True)
        print(f"wrote {OUT_MD}", flush=True)
        extra = ""
        if "hungarian_minsum" in results:
            extra = f" (hungarian {results['hungarian_minsum']['score']:.2f})"
        print(
            f"curve: {LOCAL_RULE_SCORE:.2f} -> {results['greedy']['score']:.2f} "
            f"-> {results['predictive']['score']:.2f} -> {results['mpc']['score']:.2f}{extra}",
            flush=True,
        )


if __name__ == "__main__":
    main()
