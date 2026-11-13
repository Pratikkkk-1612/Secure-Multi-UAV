"""Physical network state and per-step link evaluation for Secure Multi-UAV SAGIN.

The PettingZoo environment owns rewards/episodes; MASAC owns decisions.
All node positions and velocities have shape (N, 3), including in 2D mode.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import DEFAULT_CONFIG, ProjectConfig
from channels.a2g_channel import sample_a2g
from channels.a2a_channel import a2a_sinr, sample_a2a
from security.secrecy_rate import SecurityMetrics, evaluate_security


@dataclass(frozen=True)
class NetworkMetrics:
    """Metrics for one channel realization; rates are in bits/second."""

    association: np.ndarray                 # (N_GU,), zero-based UAV IDs
    schedule: np.ndarray                    # (N_UAV, N_RB), 0 idle; 1-based GU IDs
    power_w: np.ndarray                     # (N_UAV,)
    gu_sinr: np.ndarray                     # (N_GU,), linear scale
    gu_rate_bps: np.ndarray                 # (N_GU,)
    eav_rate_bps: np.ndarray                # (N_GU,), stream leakage to its UAV's selected EAV
    secrecy_rate_bps: np.ndarray            # (N_GU,), stream diagnostic only
    per_uav_secrecy_bps: np.ndarray         # (N_UAV,)
    total_rate_bps: float
    total_secrecy_bps: float
    a2g_state: Any
    a2a_state: Any | None
    a2a_sinr: Any | None
    security: SecurityMetrics


class NetworkModel:
    """Stateful network simulation, without PettingZoo or RL dependencies."""

    def __init__(
        self, config: ProjectConfig = DEFAULT_CONFIG, *, seed: int | None = None,
        spawn_layout: str = "random", eav_policy: str = "random",
        enforce_average_power_budget: bool = True,
    ):
        if spawn_layout not in ("random", "spread") or eav_policy not in ("random", "pursuit"):
            raise ValueError("Unknown spawn layout or EAV mobility policy")
        self.spawn_layout = spawn_layout
        self.eav_policy = eav_policy
        self.enforce_average_power_budget = enforce_average_power_budget
        self.config = config
        self.net = config.network
        self.sec = config.security
        self.a2g_config = config.a2g
        self.rng = np.random.default_rng()
        self.step_count = 0
        self.last_metrics: NetworkMetrics | None = None
        self.boundary_violations = np.zeros(self.net.n_uav, dtype=bool)
        self.reset(seed)

    def _altitude_limits(self, role: str) -> tuple[float, float]:
        if self.net.mobility_mode == "2d":
            if role == "GU":
                return (self.net.gu_altitude_m,) * 2
            return (self.net.fixed_2d_altitude_m,) * 2
        if role == "GU":
            return (self.net.gu_altitude_m,) * 2
        limits = (
            self.net.uav_3d_altitude_limits_m
            if role == "UAV"
            else self.net.eav_3d_altitude_limits_m
        )
        if limits is None:
            raise ValueError(f"3D {role} altitude limits must be configured")
        return limits

    def _positions(self, count: int, role: str) -> np.ndarray:
        if role == "UAV" and self.spawn_layout == "spread":
            cols = int(np.ceil(np.sqrt(count)))
            rows = int(np.ceil(count / cols))
            xs = (np.arange(cols) + 0.5) * self.net.area_x_m / cols
            ys = (np.arange(rows) + 0.5) * self.net.area_y_m / rows
            xy = np.array([(x, y) for y in ys for x in xs])[:count]
        else:
            xy = self.rng.uniform(
                [0.0, 0.0], [self.net.area_x_m, self.net.area_y_m], size=(count, 2)
            )
        if role == "GU":
            z = np.full(count, self.net.gu_altitude_m)
        elif self.net.mobility_mode == "2d":
            z = np.full(count, self.net.fixed_2d_altitude_m)
        else:
            initial = (
                self.net.initial_uav_3d_altitude_m
                if role == "UAV"
                else self.net.initial_eav_3d_altitude_m
            )
            z = np.full(count, initial)
        return np.column_stack((xy, z))

    def _velocities(self, count: int, speed: float, *, three_d: bool) -> np.ndarray:
        if count == 0 or speed == 0:
            return np.zeros((count, 3), dtype=float)
        v = self.rng.normal(size=(count, 3 if three_d else 2))
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        if not three_d:
            v = np.column_stack((v, np.zeros(count)))
        return speed * v

    def reset(self, seed: int | None = None) -> NetworkMetrics:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.uav_pos = self._positions(self.net.n_uav, "UAV")
        self.gu_pos = self._positions(self.net.n_gu, "GU")
        self.uav_vel = np.zeros((self.net.n_uav, 3), dtype=float)
        self.gu_vel = self._velocities(
            self.net.n_gu, self.net.gu_speed_mps, three_d=False
        )
        self.eav_pos = self._positions(self.net.n_eav, "EAV")
        self.eav_vel = self._velocities(
            self.net.n_eav,
            self.net.eav_speed_mps,
            three_d=(self.net.mobility_mode == "3d"),
        )
        self.power_w = np.full(
            self.net.n_uav, self.sec.initial_uav_power_w, dtype=float
        )
        self.power_sum_w_steps = np.zeros(self.net.n_uav)
        self.boundary_violations = np.zeros(self.net.n_uav, dtype=bool)
        self.last_metrics = self.evaluate()
        return self.last_metrics

    def _reflect(self, pos: np.ndarray, vel: np.ndarray, role: str) -> None:
        bounds = (
            (0.0, self.net.area_x_m),
            (0.0, self.net.area_y_m),
            self._altitude_limits(role),
        )
        for axis, (low, high) in enumerate(bounds):
            if high == low:
                pos[:, axis] = low
                vel[:, axis] = 0.0
                continue
            span = high - low
            raw = pos[:, axis] + vel[:, axis] * self.net.dt_s
            wrapped = (raw - low) % (2 * span)
            pos[:, axis] = low + np.where(
                wrapped <= span, wrapped, 2 * span - wrapped
            )
            vel[:, axis] *= np.where(np.floor((raw - low) / span) % 2 == 0, 1.0, -1.0)

    def _move_uavs(self, directions: np.ndarray) -> None:
        velocities = np.zeros((self.net.n_uav, 3), dtype=float)
        dimensions = self.net.movement_dimensions
        velocities[:, :dimensions] = directions
        lengths = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocities /= np.maximum(lengths, 1.0)
        velocities *= self.net.uav_max_speed_mps

        previous = self.uav_pos.copy()
        candidate = previous + velocities * self.net.dt_s
        lower = np.array([0.0, 0.0, self._altitude_limits("UAV")[0]])
        upper = np.array([
            self.net.area_x_m,
            self.net.area_y_m,
            self._altitude_limits("UAV")[1],
        ])
        self.boundary_violations = np.any(
            (candidate < lower) | (candidate > upper), axis=1
        )
        self.uav_pos = np.clip(candidate, lower, upper)
        self.uav_vel = (self.uav_pos - previous) / self.net.dt_s

    def step(self, actions: np.ndarray) -> NetworkMetrics:
        """Apply normalized per-UAV [movement..., absolute_power] commands."""
        if self.step_count >= self.net.n_steps:
            raise RuntimeError("Episode finished; call reset()")
        action = np.asarray(actions, dtype=float)
        expected = (self.net.n_uav, self.config.action_dimensions_per_uav)
        if action.shape != expected or not np.all(np.isfinite(action)):
            raise ValueError(f"actions must be a finite array of shape {expected}")
        action = np.clip(action, -1.0, 1.0)

        dimensions = self.net.movement_dimensions
        self._move_uavs(action[:, :dimensions])
        power_action = (action[:, dimensions] + 1.0) / 2.0

        power_range = self.sec.max_uav_power_w - self.sec.min_uav_power_w
        self.power_w = self.sec.min_uav_power_w + power_action * power_range

        if self.enforce_average_power_budget:
            future_steps = self.net.n_steps - self.step_count - 1
            available = (self.net.n_steps * self.sec.average_uav_power_budget_w
                         - self.power_sum_w_steps - future_steps * self.sec.min_uav_power_w)
            self.power_w = np.maximum(self.sec.min_uav_power_w, np.minimum(self.power_w, available))
        self.power_sum_w_steps += self.power_w

        self._reflect(self.gu_pos, self.gu_vel, "GU")
        if self.net.n_eav and self.eav_policy == "pursuit":
            diff = self.uav_pos[None, :, :] - self.eav_pos[:, None, :]
            nearest = np.linalg.norm(diff, axis=-1).argmin(axis=1)
            direction = diff[np.arange(self.net.n_eav), nearest]
            lengths = np.linalg.norm(direction, axis=1, keepdims=True)
            self.eav_vel = direction * self.net.eav_speed_mps / np.maximum(lengths, 1e-12)
        self._reflect(self.eav_pos, self.eav_vel, "EAV")
        self.step_count += 1
        self.last_metrics = self.evaluate()
        return self.last_metrics

    def associate_users(self) -> np.ndarray:
        """Greedy capacity-feasible association; every GU is assigned once."""
        distances = np.linalg.norm(self.uav_pos[:, None, :] - self.gu_pos[None, :, :], axis=-1)
        association = np.argmin(distances, axis=0)
        counts = np.bincount(association, minlength=self.net.n_uav)
        low, high = max(1, self.net.min_gu_per_uav), self.net.max_gu_per_uav

        while np.any(counts > high):
            src = int(np.argmax(counts - high))
            movable = np.flatnonzero(association == src)
            destinations = np.flatnonzero(counts < high)
            penalties = distances[np.ix_(destinations, movable)] - distances[src, movable]
            d, u = np.unravel_index(np.argmin(penalties), penalties.shape)
            dst, gu = int(destinations[d]), int(movable[u])
            association[gu] = dst
            counts[src] -= 1
            counts[dst] += 1

        while np.any(counts < low):
            dst = int(np.flatnonzero(counts < low)[0])
            movable = np.flatnonzero(counts[association] > low)
            if len(movable) == 0:
                raise RuntimeError("User association is infeasible under current constraints")
            penalties = distances[dst, movable] - distances[association[movable], movable]
            gu = int(movable[np.argmin(penalties)])
            src = int(association[gu])
            association[gu] = dst
            counts[src] -= 1
            counts[dst] += 1

        return association

    def schedule_users(self, association: np.ndarray) -> np.ndarray:
        """MATLAB-compatible: 0 idle RB, otherwise 1-based GU index."""
        result = np.zeros(
            (self.net.n_uav, self.net.n_resource_blocks_per_uav), dtype=int
        )
        for u in range(self.net.n_uav):
            gu_ids = np.flatnonzero(association == u)
            if len(gu_ids) > result.shape[1]:
                raise RuntimeError("More assigned users than resource blocks")
            result[u, : len(gu_ids)] = gu_ids + 1
        return result

    def evaluate(self) -> NetworkMetrics:
        """Sample channels once and compute rates for the current topology."""
        association = self.associate_users()
        schedule = self.schedule_users(association)
        ground = sample_a2g(
            self.uav_pos, self.gu_pos, config=self.a2g_config, rng=self.rng
        )

        air = None
        air_sinr = None
        if self.net.n_eav:
            air = sample_a2a(
                self.uav_pos, self.eav_pos, self.uav_vel, self.eav_vel,
                self.step_count * self.net.dt_s,
                config=self.config.a2a, rng=self.rng,
            )
            air_sinr = a2a_sinr(
                air, self.power_w, config=self.config.a2a
            )
        security = evaluate_security(ground, air, schedule, self.power_w, config=self.config)
        return NetworkMetrics(
            association=association, schedule=schedule, power_w=self.power_w.copy(),
            gu_sinr=security.legitimate_sinr, gu_rate_bps=security.gu_rate_bps,
            eav_rate_bps=security.worst_eav_rate_bps, secrecy_rate_bps=security.secrecy_rate_bps,
            per_uav_secrecy_bps=security.per_uav_secrecy_rate_bps,
            total_rate_bps=float(security.gu_rate_bps.sum()),
            total_secrecy_bps=security.total_secrecy_rate_bps,
            a2g_state=ground, a2a_state=air, a2a_sinr=air_sinr, security=security,
        )

    def satellite_diagnostics(
        self, satellite_pos: np.ndarray, satellite_vel: np.ndarray
    ) -> tuple[Any, np.ndarray]:
        """Optional A2S uplink diagnostic; never used in the RL reward.

        Satellite positions and velocities must use the same 3D Cartesian
        frame and SI units as the UAVs. Orbit propagation is not included.
        """
        from channels.a2s_channel import a2s_sinr, sample_a2s

        satellite_config = getattr(self.config, "a2s", None)
        channel = sample_a2s(
            self.uav_pos, satellite_pos, self.uav_vel, satellite_vel,
            self.step_count * self.net.dt_s,
            config=satellite_config, rng=self.rng,
        )
        sinr = a2s_sinr(channel, self.power_w, config=satellite_config)
        return channel, sinr
