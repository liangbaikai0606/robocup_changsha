"""E057：选圈仍是 E052。开车改成位置加相对速度。

a = clip(20*Δp + 4*(v_T - v_R), -1, 1)。v_T 仍是连续两步估出来的目标速度。
把估计速率裁到目标最大速度，或把新旧估计各取一半，在这批种子上没有再加分。
"""

from dataclasses import dataclass

import numpy as np

from coverage_bench.protocol import AgentObservation, EpisodeContext, get_protocol_spec

_PROTOCOL_SPEC = get_protocol_spec()
_CLOSE_DISTANCE = 0.16
_HARD_STEPS = 6
_APPROACH_LIMIT = 5
_COLLISION_WEIGHT = 0.2
_TEAM_SIZE = 3
_BOOST_MIN_NET_GAIN = 0.05
_YIELD_MARGIN = 1


@dataclass
class _Commit:
    """一次已经答应的对撞：先朝队友走，分开之后再去自己的那个圈。"""

    partner_index: int
    circle_index: int
    circle_abs: np.ndarray
    phase: str
    touched: bool
    last_dist: float
    approach_steps: int


class CoverFirstPolicy:
    """每台车只看自己的局部观测。对撞要两边各自算出同一次，才一起朝对方加力。"""

    def __init__(self) -> None:
        self._position_scale = float(_PROTOCOL_SPEC.position_scale)
        self._velocity_scale = float(_PROTOCOL_SPEC.velocity_scale)
        self._cover_radius = 0.15
        self._dt = 0.1
        self._damping = 0.25
        self._mass = 1.0
        self._contact_force = 100.0
        self._contact_margin = 0.001
        self._max_speed = 1.0
        self._half_extent = 1.0
        self._robot_radius = 0.05
        self._sense_radius = 0.6
        self._horizon = int(_PROTOCOL_SPEC.max_episode_steps)
        self._commit: _Commit | None = None
        self._last_target_abs: dict[int, np.ndarray] = {}
        self._target_velocity_estimates: dict[int, np.ndarray] = {}

    def reset(self, context: EpisodeContext) -> None:
        """新回合清掉上一局的对撞约定，并记下公开物理参数。"""
        task = context.task
        self._cover_radius = float(task.target_radius)
        self._dt = float(task.dt)
        self._damping = float(task.damping)
        self._mass = float(task.robot_mass)
        self._contact_force = float(task.contact_force)
        self._contact_margin = float(task.contact_margin)
        self._max_speed = float(task.robot_max_speed)
        self._half_extent = float(task.map_half_extent)
        self._robot_radius = float(task.robot_radius)
        self._sense_radius = float(task.sense_radius)
        self._horizon = int(context.horizon)
        self._commit = None
        self._last_target_abs = {}
        self._target_velocity_estimates = {}

    def act(self, observation: AgentObservation) -> np.ndarray:
        """先完成已经开始的对撞；否则看这一步要不要新开；都不是就追圈。"""
        self._update_target_velocity(observation)
        if self._commit is not None:
            action = self._follow_commit(observation)
            if action is not None:
                return action
        boost = self._start_boost(observation)
        if boost is not None:
            return boost
        return self._chase(observation)

    def close(self) -> None:
        """规则策略没有要释放的模型。"""

    def _chase(self, observation: AgentObservation) -> np.ndarray:
        """空圈里选自己最早能罩住的。可见队友更快，或步数相同但编号更小，就让出这个圈。"""
        chosen = self._choose_intercept_target(observation)
        if chosen is None:
            return np.zeros(2, dtype=np.float32)
        target_index, relative = chosen
        self_state = np.asarray(observation["self_state"], dtype=np.float64)
        self_velocity = self_state[2:4] * self._velocity_scale
        target_velocity = self._target_velocity_estimates.get(target_index, np.zeros(2, dtype=np.float64))
        return self._thrust_to_target(relative, self_velocity, target_velocity)

    def _choose_intercept_target(self, observation: AgentObservation) -> tuple[int, np.ndarray] | None:
        """只在看得见的圈里选。已被队友站上的圈不和空圈抢，除非没有空圈。"""
        rows = self._visible_intercept_rows(observation)
        if not rows:
            return None
        self._mark_peer_yields(observation, rows)
        free = [row for row in rows if not row["occupied"] and not row["yielded"]]
        occupied = [row for row in rows if row["occupied"]]
        pool = free or occupied
        if not pool:
            return None
        pool.sort(key=lambda row: (row["time"], row["distance"], row["index"]))
        chosen = pool[0]
        return int(chosen["index"]), chosen["relative"]

    def _visible_intercept_rows(self, observation: AgentObservation) -> list[dict[str, object]]:
        """每个可见圈记下距离、是否已被站上，以及自己还要几步能罩住。"""
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        peers = np.asarray(observation["peers"], dtype=np.float64)
        peer_visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
        _self_pos, self_vel = self._self_motion(observation)
        steps_left = self._horizon - int(observation["step_index"])
        rows: list[dict[str, object]] = []
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

    def _mark_peer_yields(self, observation: AgentObservation, rows: list[dict[str, object]]) -> None:
        """队友截击至少快 1 步，或同样快但编号更小，这个圈记成让出。"""
        steps_left = self._horizon - int(observation["step_index"])
        self_pos, self_vel = self._self_motion(observation)
        peers = self._visible_peers(observation, self_pos, self_vel)
        my_index = int(observation["agent_index"])
        for row in rows:
            if row["occupied"]:
                continue
            relative = np.asarray(row["relative"], dtype=np.float64)
            target_abs = self_pos + relative
            best_peer: tuple[int, int] | None = None
            for peer_index, peer_pos, peer_vel, _peer_dist in peers:
                if float(np.linalg.norm(target_abs - peer_pos)) > self._sense_radius:
                    continue
                peer_relative = target_abs - peer_pos
                peer_time = self._intercept_time(
                    peer_relative,
                    peer_vel,
                    np.asarray(row["velocity"], dtype=np.float64),
                    steps_left,
                )
                if best_peer is None or (peer_time, peer_index) < best_peer:
                    best_peer = (peer_time, peer_index)
            if best_peer is None:
                continue
            peer_time, peer_index = best_peer
            clearer = peer_time + _YIELD_MARGIN < int(row["time"])
            tied = peer_time == int(row["time"]) and peer_index < my_index
            if clearer or tied:
                row["yielded"] = True

    def _intercept_time(
        self,
        relative: np.ndarray,
        velocity: np.ndarray,
        target_velocity: np.ndarray,
        steps_left: int,
    ) -> int:
        """朝当前估计的目标速度预测点满力，第一次进圈的步数。进不去就记成剩余步数加 1。"""
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

    def _visible_targets(
        self, observation: AgentObservation
    ) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
        """看得见的圈按离自己从近到远，分成空的和已被可见队友占住的。"""
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        peers = np.asarray(observation["peers"], dtype=np.float64)
        peer_visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
        empty: list[tuple[float, int, np.ndarray]] = []
        occupied: list[tuple[float, int, np.ndarray]] = []
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            relative = targets[index, :2] * self._position_scale
            distance = float(np.linalg.norm(relative))
            bucket = occupied if self._occupied(relative, peers, peer_visible) else empty
            bucket.append((distance, index, relative.astype(np.float32)))
        empty.sort(key=lambda item: item[0])
        occupied.sort(key=lambda item: item[0])
        return (
            [(item[1], item[2]) for item in empty],
            [(item[1], item[2]) for item in occupied],
        )

    def _update_target_velocity(self, observation: AgentObservation) -> None:
        """用连续两步的相对位置和自身位移估计可见目标速度。"""
        self_state = np.asarray(observation["self_state"], dtype=np.float64)
        self_pos = self_state[:2] * self._position_scale
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        step = int(observation["step_index"])
        if step <= 0:
            self._last_target_abs = {}
            self._target_velocity_estimates = {}
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            target_abs = self_pos + targets[index, :2] * self._position_scale
            previous = self._last_target_abs.get(index)
            if previous is not None and self._dt > 0.0:
                estimate = (target_abs - previous) / self._dt
                self._target_velocity_estimates[index] = np.clip(
                    estimate, -self._max_speed, self._max_speed
                )
            self._last_target_abs[index] = target_abs.copy()

    def _occupied(
        self,
        target_relative: np.ndarray,
        peers: np.ndarray,
        peer_visible: np.ndarray,
    ) -> bool:
        """可见队友的中心进入圈半径，这个圈就算被占了。"""
        for index, is_visible in enumerate(peer_visible):
            if not is_visible:
                continue
            peer_relative = peers[index, :2] * self._position_scale
            if float(np.linalg.norm(peer_relative - target_relative)) <= self._cover_radius:
                return True
        return False

    def _start_boost(self, observation: AgentObservation) -> np.ndarray | None:
        """两边都看得见的近圈和远圈满足条件时，约定对撞并立刻朝队友加力。"""
        steps_left = self._horizon - int(observation["step_index"])
        if steps_left < _HARD_STEPS:
            return None
        self_pos, self_vel = self._self_motion(observation)
        peers = self._visible_peers(observation, self_pos, self_vel)
        targets = self._visible_target_points(observation, self_pos)
        partner = self._mutual_partner(self_pos, peers)
        if partner is None:
            return None
        shared = [
            item
            for item in targets
            if float(np.linalg.norm(item[1] - partner[1])) <= self._sense_radius
        ]
        if len(shared) < 2:
            return None
        if self._private_chase(self_pos, targets, peers, shared) is not None:
            return None
        plan = self._best_plan(
            int(observation["agent_index"]), self_pos, self_vel, partner, shared, steps_left
        )
        if plan is None:
            return None
        _role, circle_index, circle_abs, _improve = plan
        self._commit = _Commit(
            partner_index=int(partner[0]),
            circle_index=int(circle_index),
            circle_abs=np.array(circle_abs, dtype=np.float64),
            phase="approach",
            touched=False,
            last_dist=float(np.linalg.norm(partner[1] - self_pos)),
            approach_steps=0,
        )
        return self._follow_commit(observation)

    def _best_plan(
        self,
        my_index: int,
        self_pos: np.ndarray,
        self_vel: np.ndarray,
        partner: tuple[int, np.ndarray, np.ndarray, float],
        shared: list[tuple[int, np.ndarray]],
        steps_left: int,
    ) -> tuple[str, int, np.ndarray, float] | None:
        """在两种角色里挑撞完两边都能进圈、而且远圈单独 6 步不够的那一次。"""
        partner_index, partner_pos, partner_vel, _partner_dist = partner
        cars = {
            "self": (self_pos, self_vel),
            "peer": (partner_pos, partner_vel),
        }
        found: list[tuple[float, float, int, int, str, int, np.ndarray, np.ndarray]] = []
        for flyer_name in ("self", "peer"):
            keeper_name = "peer" if flyer_name == "self" else "self"
            keeper_pos, keeper_vel = cars[keeper_name]
            flyer_pos, flyer_vel = cars[flyer_name]
            occupiers = [flyer_pos] if keeper_name == "self" else [self_pos]
            near = self._chase_point(keeper_pos, shared, occupiers)
            if near is None:
                continue
            near_index, near_xy = near
            keeper_dist = float(np.linalg.norm(keeper_pos - near_xy))
            push = flyer_pos - keeper_pos
            push_norm = float(np.linalg.norm(push))
            if push_norm < 1e-8:
                continue
            push = push / push_norm
            for far_index, far_xy in shared:
                if far_index == near_index:
                    continue
                toward_far = far_xy - flyer_pos
                far_norm = float(np.linalg.norm(toward_far))
                if far_norm < 1e-8 or float(np.dot(push, toward_far / far_norm)) < 0.75:
                    continue
                if self._min_distance(flyer_pos, flyer_vel, far_xy, _HARD_STEPS) <= self._cover_radius:
                    continue
                reached, _keeper_best, flyer_best, boost_coverage, collision_steps = self._boost_reaches(
                    keeper_pos, keeper_vel, flyer_pos, flyer_vel, near_xy, far_xy, steps_left
                )
                if not reached:
                    continue
                solo = self._min_distance(flyer_pos, flyer_vel, far_xy, steps_left)
                solo_coverage = self._solo_pair_coverage(
                    keeper_pos, keeper_vel, flyer_pos, flyer_vel, near_xy, far_xy, steps_left
                )
                coverage_gain = (boost_coverage - solo_coverage) / float(_TEAM_SIZE)
                collision_cost = (
                    _COLLISION_WEIGHT * (2.0 / float(_TEAM_SIZE)) * collision_steps
                )
                net_gain = coverage_gain - collision_cost
                if net_gain < _BOOST_MIN_NET_GAIN:
                    continue
                improve = net_gain
                keeper_index = my_index if keeper_name == "self" else partner_index
                found.append(
                    (improve, keeper_dist, far_index, keeper_index, flyer_name, near_index, near_xy, far_xy)
                )
        if not found:
            return None
        found.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        improve, _keeper_dist, far_index, _keeper_index, flyer_name, near_index, near_xy, far_xy = found[0]
        if flyer_name == "self":
            return "fly", far_index, far_xy, improve
        return "keep", near_index, near_xy, improve

    def _mutual_partner(
        self,
        self_pos: np.ndarray,
        peers: list[tuple[int, np.ndarray, np.ndarray, float]],
    ) -> tuple[int, np.ndarray, np.ndarray, float] | None:
        """只和离自己最近、而且自己也是对方最近的那台车商量。第三台太近就先不撞。"""
        if not peers:
            return None
        partner = min(peers, key=lambda item: (item[3], item[0]))
        if partner[3] > _CLOSE_DISTANCE:
            return None
        for other in peers:
            if other[0] == partner[0]:
                continue
            if float(np.linalg.norm(other[1] - partner[1])) < partner[3]:
                return None
            if min(
                float(np.linalg.norm(other[1] - self_pos)),
                float(np.linalg.norm(other[1] - partner[1])),
            ) < 0.25:
                return None
        return partner

    def _private_chase(
        self,
        self_pos: np.ndarray,
        targets: list[tuple[int, np.ndarray]],
        peers: list[tuple[int, np.ndarray, np.ndarray, float]],
        shared: list[tuple[int, np.ndarray]],
    ) -> tuple[int, np.ndarray] | None:
        """自己真正要追的圈队友看不见时，返回那个圈。这样就不会单方面去撞。"""
        shared_ids = {index for index, _xy in shared}
        occupiers = [peer[1] for peer in peers]
        mine = self._chase_point(self_pos, targets, occupiers)
        if mine is None or mine[0] in shared_ids:
            return None
        return mine

    def _chase_point(
        self,
        origin: np.ndarray,
        pool: list[tuple[int, np.ndarray]],
        occupiers: list[np.ndarray],
    ) -> tuple[int, np.ndarray] | None:
        """从 origin 看，空圈优先，否则最近的圈。占用只按给定的那几台车算。"""
        empty: list[tuple[float, int, np.ndarray]] = []
        taken: list[tuple[float, int, np.ndarray]] = []
        for index, xy in pool:
            distance = float(np.linalg.norm(xy - origin))
            occupied = any(
                float(np.linalg.norm(other - xy)) <= self._cover_radius for other in occupiers
            )
            (taken if occupied else empty).append((distance, index, xy))
        bucket = empty or taken
        if not bucket:
            return None
        bucket.sort(key=lambda item: (item[0], item[1]))
        _distance, index, xy = bucket[0]
        return index, xy

    def _follow_commit(self, observation: AgentObservation) -> np.ndarray | None:
        """约定还在就继续：靠近阶段朝队友，弹开之后朝自己的圈。"""
        commit = self._commit
        if commit is None:
            return None
        self_pos, _self_vel = self._self_motion(observation)
        circle = self._refresh_circle(observation, self_pos, commit)
        if commit.phase == "approach":
            partner = self._partner_position(observation, self_pos, commit.partner_index)
            if partner is None:
                self._commit = None
                return None
            dist = float(np.linalg.norm(partner - self_pos))
            if dist < self._robot_radius * 2.0:
                commit.touched = True
            if (commit.touched and dist > commit.last_dist) or commit.approach_steps >= _APPROACH_LIMIT:
                commit.phase = "ride"
            commit.last_dist = dist
            if commit.phase == "approach":
                commit.approach_steps += 1
                return self._thrust_relative(partner - self_pos)
        return self._thrust_relative(circle - self_pos)

    def _refresh_circle(
        self, observation: AgentObservation, self_pos: np.ndarray, commit: _Commit
    ) -> np.ndarray:
        """圈还看得见就用新位置，看不见就用上次记下的位置。"""
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        targets = np.asarray(observation["targets"], dtype=np.float64)
        if 0 <= commit.circle_index < len(visible) and bool(visible[commit.circle_index]):
            commit.circle_abs = self_pos + targets[commit.circle_index, :2] * self._position_scale
        return commit.circle_abs

    def _boost_reaches(
        self,
        keeper_pos: np.ndarray,
        keeper_vel: np.ndarray,
        flyer_pos: np.ndarray,
        flyer_vel: np.ndarray,
        near: np.ndarray,
        far: np.ndarray,
        steps: int,
    ) -> tuple[bool, float, float, int, int]:
        """先对撞再分头追，并估计覆盖收益与碰撞成本。圈先当成不动。"""
        pos = np.stack([np.array(keeper_pos, dtype=np.float64), np.array(flyer_pos, dtype=np.float64)])
        vel = np.stack([np.array(keeper_vel, dtype=np.float64), np.array(flyer_vel, dtype=np.float64)])
        touched = False
        last = float(np.linalg.norm(pos[0] - pos[1]))
        phase = "approach"
        keeper_best = float(np.linalg.norm(pos[0] - near))
        flyer_best = float(np.linalg.norm(pos[1] - far))
        coverage_steps = 0
        collision_steps = 0
        for tick in range(steps):
            dist = float(np.linalg.norm(pos[0] - pos[1]))
            if dist < self._robot_radius * 2.0:
                touched = True
            if phase == "approach" and ((touched and dist > last) or tick >= _APPROACH_LIMIT):
                phase = "ride"
            if phase == "approach":
                keeper_force = self._thrust_relative(pos[1] - pos[0])
                flyer_force = self._thrust_relative(pos[0] - pos[1])
            else:
                keeper_force = self._thrust_relative(near - pos[0])
                flyer_force = self._thrust_relative(far - pos[1])
            pos, vel = self._integrate(pos, vel, np.stack([keeper_force, flyer_force]), contact=True)
            keeper_best = min(keeper_best, float(np.linalg.norm(pos[0] - near)))
            flyer_best = min(flyer_best, float(np.linalg.norm(pos[1] - far)))
            if float(np.linalg.norm(pos[0] - pos[1])) < self._robot_radius * 2.0:
                collision_steps += 1
            coverage_steps += int(float(np.linalg.norm(pos[0] - near)) <= self._cover_radius)
            coverage_steps += int(float(np.linalg.norm(pos[1] - far)) <= self._cover_radius)
            last = dist
        covered = keeper_best <= self._cover_radius and flyer_best <= self._cover_radius
        return covered, keeper_best, flyer_best, coverage_steps, collision_steps

    def _solo_pair_coverage(
        self,
        keeper_pos: np.ndarray,
        keeper_vel: np.ndarray,
        flyer_pos: np.ndarray,
        flyer_vel: np.ndarray,
        near: np.ndarray,
        far: np.ndarray,
        steps: int,
    ) -> int:
        """估计两车不碰撞、各自追一个圈时的累计覆盖次数。"""
        pos = np.stack([np.array(keeper_pos, dtype=np.float64), np.array(flyer_pos, dtype=np.float64)])
        vel = np.stack([np.array(keeper_vel, dtype=np.float64), np.array(flyer_vel, dtype=np.float64)])
        coverage_steps = 0
        for _tick in range(steps):
            forces = np.stack(
                [self._thrust_relative(near - pos[0]), self._thrust_relative(far - pos[1])]
            )
            pos, vel = self._integrate(pos, vel, forces, contact=False)
            coverage_steps += int(float(np.linalg.norm(pos[0] - near)) <= self._cover_radius)
            coverage_steps += int(float(np.linalg.norm(pos[1] - far)) <= self._cover_radius)
        return coverage_steps

    def _min_distance(self, pos: np.ndarray, vel: np.ndarray, target: np.ndarray, steps: int) -> float:
        """一台车单独朝圈满力，这几步里离圈心最近是多少。不算碰撞。"""
        point = np.array(pos, dtype=np.float64).reshape(1, 2)
        speed = np.array(vel, dtype=np.float64).reshape(1, 2)
        best = float(np.linalg.norm(point[0] - target))
        for _tick in range(steps):
            if best <= self._cover_radius:
                return best
            force = self._thrust_relative(target - point[0]).reshape(1, 2)
            point, speed = self._integrate(point, speed, force, contact=False)
            best = min(best, float(np.linalg.norm(point[0] - target)))
        return best

    def _integrate(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        force: np.ndarray,
        contact: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """和官方相同的一步：先用旧速度改位置，再加阻尼、油门和接触力。"""
        point = np.array(pos, dtype=np.float64, copy=True)
        speed = np.array(vel, dtype=np.float64, copy=True)
        applied = np.array(force, dtype=np.float64, copy=True)
        if contact and len(point) >= 2:
            delta = point[0] - point[1]
            dist = float(np.linalg.norm(delta))
            dist_min = self._robot_radius * 2.0
            margin = self._contact_margin
            if dist == 0.0:
                normal = np.array([1.0, 0.0], dtype=np.float64)
                penetration = float(np.logaddexp(0.0, dist_min / margin) * margin)
                push = self._contact_force * normal * penetration
            else:
                penetration = float(np.logaddexp(0.0, -(dist - dist_min) / margin) * margin)
                push = self._contact_force * (delta / dist) * penetration
            applied[0] = applied[0] + push
            applied[1] = applied[1] - push
        point = point + speed * self._dt
        speed = (1.0 - self._damping) * speed + (applied / self._mass) * self._dt
        for index in range(len(speed)):
            magnitude = float(np.linalg.norm(speed[index]))
            if magnitude > self._max_speed:
                speed[index] *= self._max_speed / magnitude
        limit = self._half_extent - self._robot_radius
        for index in range(len(point)):
            for axis in (0, 1):
                if point[index, axis] > limit:
                    point[index, axis] = limit
                    if speed[index, axis] > 0.0:
                        speed[index, axis] = 0.0
                elif point[index, axis] < -limit:
                    point[index, axis] = -limit
                    if speed[index, axis] < 0.0:
                        speed[index, axis] = 0.0
        return point, speed

    def _self_motion(self, observation: AgentObservation) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(observation["self_state"], dtype=np.float64)
        pos = state[:2] * self._position_scale
        vel = state[2:4] * self._velocity_scale
        return pos, vel

    def _visible_peers(
        self,
        observation: AgentObservation,
        self_pos: np.ndarray,
        self_vel: np.ndarray,
    ) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        peers = np.asarray(observation["peers"], dtype=np.float64)
        visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
        found: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            relative = peers[index, :2] * self._position_scale
            pos = self_pos + relative
            vel = self_vel + peers[index, 2:4] * self._velocity_scale
            found.append((index, pos, vel, float(np.linalg.norm(relative))))
        return found

    def _visible_target_points(
        self, observation: AgentObservation, self_pos: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        targets = np.asarray(observation["targets"], dtype=np.float64)
        visible = np.asarray(observation["target_visible"], dtype=np.bool_)
        found: list[tuple[int, np.ndarray]] = []
        for index, is_visible in enumerate(visible):
            if not is_visible:
                continue
            found.append((index, self_pos + targets[index, :2] * self._position_scale))
        return found

    def _partner_position(
        self, observation: AgentObservation, self_pos: np.ndarray, partner_index: int
    ) -> np.ndarray | None:
        visible = np.asarray(observation["peer_visible"], dtype=np.bool_)
        peers = np.asarray(observation["peers"], dtype=np.float64)
        if partner_index < 0 or partner_index >= len(visible) or not bool(visible[partner_index]):
            return None
        return self_pos + peers[partner_index, :2] * self._position_scale

    def _thrust_relative(self, relative: np.ndarray) -> np.ndarray:
        """两个分量都可以打满。近处不会缩成单位向量。"""
        return np.clip(np.asarray(relative, dtype=np.float64) * 10.0, -1.0, 1.0).astype(np.float32)

    def _thrust_to_target(
        self,
        relative: np.ndarray,
        self_velocity: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        """全程用位置差和相对速度。近处不再单独收油门。"""
        offset = np.asarray(relative, dtype=np.float64)
        velocity = np.asarray(self_velocity, dtype=np.float64)
        target_speed = np.asarray(target_velocity, dtype=np.float64)
        return np.clip(20.0 * offset + 4.0 * (target_speed - velocity), -1.0, 1.0).astype(np.float32)


def build_policy(context: object) -> CoverFirstPolicy:
    """评测入口。规则不读取模型文件，也不使用开局种子。"""
    del context
    return CoverFirstPolicy()
