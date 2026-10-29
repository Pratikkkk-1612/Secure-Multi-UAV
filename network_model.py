"""Physical network state and per-step link evaluation for Secure Multi-UAV SAGIN.

The PettingZoo environment owns rewards/episodes; MASAC owns decisions.
All node positions and velocities have shape (N, 3), including in 2D mode.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import DEFAULT_CONFIG, ProjectConfig
from channels.a2g_channel import (
    a2g_sinr,
    sample_a2g,
)
from channels.a2a_channel import a2a_sinr, sample_a2a


class _A2GConfigView:
    """Bridge both the earlier A2G module and the shared project config."""

    _aliases = {
        "urban_a": "los_a",
        "urban_b": "los_b",
        "gas_attenuation_db_per_km": "oxygen_attenuation_db_per_km",
        "noise_density_dbm_hz": "thermal_noise_density_dbm_per_hz",
    }

    def __init__(self, source: Any, n_rb: int):
        self.source = source
        self.n_resource_blocks = n_rb

    @property
    def resource_block_bandwidth_hz(self) -> float:
        return self.source.total_bandwidth_hz / self.n_resource_blocks

    @property
    def noise_power_w(self) -> float:
        if hasattr(self.source, "noise_power_w_per_rb"):
            return float(self.source.noise_power_w_per_rb(self.n_resource_blocks))
        density = getattr(
            self.source, "noise_density_dbm_hz",
            getattr(self.source, "thermal_noise_density_dbm_per_hz", None),
        )
        if density is None:
            raise AttributeError("A2G noise density is missing from config")
        noise_dbm = (
            density
            + 10.0 * np.log10(self.resource_block_bandwidth_hz)
            + self.source.noise_figure_db
            + self.source.implementation_margin_db
        )
        return float(10.0 ** ((noise_dbm - 30.0) / 10.0))

    def rb_bandwidth_hz(self, n_resource_blocks: int) -> float:
        return self.source.total_bandwidth_hz / n_resource_blocks

    def noise_power_w_per_rb(self, n_resource_blocks: int) -> float:
        if n_resource_blocks == self.n_resource_blocks:
            return self.noise_power_w
        if hasattr(self.source, "noise_power_w_per_rb"):
            return self.source.noise_power_w_per_rb(n_resource_blocks)
        bandwidth = self.source.total_bandwidth_hz / n_resource_blocks
        density = getattr(self.source, "noise_density_dbm_hz", None)
        if density is None:
            density = self.source.thermal_noise_density_dbm_per_hz
        dbm = density + 10 * np.log10(bandwidth)
        dbm += self.source.noise_figure_db + self.source.implementation_margin_db
        return float(10 ** ((dbm - 30) / 10))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, self._aliases.get(name, name))


@dataclass(frozen=True)
class NetworkMetrics:
    """Metrics for one channel realization; rates are in bits/second."""

    association: np.ndarray                 # (N_GU,), zero-based UAV IDs
    schedule: np.ndarray                    # (N_UAV, N_RB), 0 idle; 1-based GU IDs
    power_w: np.ndarray                     # (N_UAV,)
    gu_sinr: np.ndarray                     # (N_GU,), linear scale
    gu_rate_bps: np.ndarray                 # (N_GU,)
    eav_rate_bps: np.ndarray                # (N_GU,), worst EAV on each user's RB
    secrecy_rate_bps: np.ndarray            # (N_GU,), clipped at zero
    per_uav_secrecy_bps: np.ndarray         # (N_UAV,)
    total_rate_bps: float
    total_secrecy_bps: float
    a2g_state: Any
    a2a_state: Any | None
    a2a_sinr: Any | None


class NetworkModel:
    """Stateful network simulation, without PettingZoo or RL dependencies."""

    def __init__(self, config: ProjectConfig = DEFAULT_CONFIG):
        self.config = config
        self.net = config.network
        self.sec = config.security
        self.a2g_config = _A2GConfigView(
            config.a2g, self.net.n_resource_blocks_per_uav
        )
        self.rng = np.random.default_rng()
        self.step_count = 0
        self.last_metrics: NetworkMetrics | None = None
        self.boundary_violations = np.zeros(self.net.n_uav, dtype=bool)
        self.reset()

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
        self.eav_pos = self._positions(self.net.n_eav, "EAV")
        self.uav_vel = np.zeros((self.net.n_uav, 3), dtype=float)
        self.gu_vel = self._velocities(
            self.net.n_gu, self.net.gu_speed_mps, three_d=False
        )
        self.eav_vel = self._velocities(
            self.net.n_eav,
            self.net.eav_speed_mps,
            three_d=(self.net.mobility_mode == "3d"),
        )
        self.power_w = np.full(
            self.net.n_uav, self.sec.initial_uav_power_w, dtype=float
        )
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
            vel[:, axis] *= np.where(wrapped <= span, 1.0, -1.0)

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

        self.power_w = (
            self.sec.min_uav_power_w
                + power_action * (
                self.sec.max_uav_power_w - self.sec.min_uav_power_w
            )
        )

        self._reflect(self.gu_pos, self.gu_vel, "GU")
        self._reflect(self.eav_pos, self.eav_vel, "EAV")
        self.step_count += 1
        self.last_metrics = self.evaluate()
        return self.last_metrics

    def associate_users(self) -> np.ndarray:
        """Greedy capacity-feasible association; every GU is assigned once."""
        distances = np.linalg.norm(
            self.uav_pos[:, None, :] - self.gu_pos[None, :, :], axis=-1
        )
        n_uav, n_gu = distances.shape
        assigned = np.full(n_gu, -1, dtype=int)
        counts = np.zeros(n_uav, dtype=int)
        # The existing A2G scheduler requires every UAV to serve >=1 GU.
        if n_gu < n_uav:
            raise ValueError("The A2G model requires at least one GU per UAV")
        min_users = max(1, self.net.min_gu_per_uav)
        max_users = min(
            self.net.max_gu_per_uav, self.net.n_resource_blocks_per_uav
        )
        pairs = np.argsort(distances, axis=None, kind="stable")

        # First ensure minimum loads without assigning any GU twice.
        for flat in pairs:
            u, g = divmod(int(flat), n_gu)
            if assigned[g] == -1 and counts[u] < min_users:
                assigned[g] = u
                counts[u] += 1

        # Then serve the remaining GUs under each UAV's RB capacity.
        for flat in pairs:
            u, g = divmod(int(flat), n_gu)
            if assigned[g] == -1 and counts[u] < max_users:
                assigned[g] = u
                counts[u] += 1

        if np.any(assigned < 0):
            raise RuntimeError("GU association is infeasible for this network")
        return assigned

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

    def _mrt_weights(self, state: Any, schedule: np.ndarray) -> np.ndarray:
        n_uav, n_gu, n_ant = state.h.shape
        weights = np.zeros(
            (n_uav, self.net.n_resource_blocks_per_uav, n_ant), dtype=complex
        )
        for u in range(n_uav):
            active = np.flatnonzero(schedule[u])
            h = state.h[u, schedule[u, active] - 1]
            weights[u, active] = h / np.linalg.norm(
                h, axis=1, keepdims=True
            ).clip(min=1e-30)
        return weights

    def _eavesdropper_rates(
        self, eav_channels: np.ndarray | None, schedule: np.ndarray,
        weights: np.ndarray, association: np.ndarray
    ) -> np.ndarray:
        result = np.zeros(self.net.n_gu, dtype=float)
        if eav_channels is None:
            return result

        counts = np.bincount(association, minlength=self.net.n_uav)
        power_per_rb = self.power_w / counts
        bandwidth = self.a2g_config.resource_block_bandwidth_hz
        # EAV receiver noise must use per-RB, not total, bandwidth.
        a2a = self.config.a2a
        noise_dbm = (
            getattr(
                a2a, "thermal_noise_density_dbm_per_hz",
                getattr(a2a, "noise_density_dbm_hz", -174.0),
            )
            + 10.0 * np.log10(bandwidth)
            + a2a.noise_figure_db
            + a2a.implementation_margin_db
        )
        noise = 10.0 ** ((noise_dbm - 30.0) / 10.0)

        for rb in range(schedule.shape[1]):
            active = np.flatnonzero(schedule[:, rb])
            if len(active) == 0:
                continue
            # Each active UAV uses its own beam, on this specific RB.
            h = eav_channels[active]          # (active UAVs, EAVs, antennas)
            w = weights[active, rb]           # (active UAVs, antennas)
            gain = np.abs(np.einsum("uen,un->ue", h.conj(), w)) ** 2
            received = power_per_rb[active, None] * gain
            interference = np.maximum(
                received.sum(axis=0, keepdims=True) - received, 0.0
            )
            eve_sinr = received / (interference + noise)
            result[schedule[active, rb] - 1] = (
                bandwidth * np.log2(1.0 + eve_sinr).max(axis=1)
            )
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
        eve_h = None
        if self.net.n_eav:
            air = sample_a2a(
                self.uav_pos, self.eav_pos, self.uav_vel, self.eav_vel,
                self.step_count * self.net.dt_s,
                config=self.config.a2a, rng=self.rng,
            )
            air_sinr = a2a_sinr(
                air, self.power_w, config=self.config.a2a
            )
            eve_h = air.uav_eav.h

        nt = ground.h.shape[-1]
        
        weights = np.zeros(
            (self.net.n_uav, schedule.shape[1], nt),
            dtype=np.complex128,
        )

        weights[schedule > 0] = np.ones(nt) / np.sqrt(nt)  

        sinr = a2g_sinr(
            ground, schedule, self.power_w, config=self.a2g_config
        )
        rates = self.a2g_config.resource_block_bandwidth_hz * np.log2(1 + sinr)
        eve_rates = self._eavesdropper_rates(
            eve_h, schedule, weights, association
        )
        secrecy = np.maximum(rates - eve_rates, 0.0)
        by_uav = np.bincount(
            association, weights=secrecy, minlength=self.net.n_uav
        )
        return NetworkMetrics(
            association=association, schedule=schedule,
            power_w=self.power_w.copy(), gu_sinr=sinr,
            gu_rate_bps=rates, eav_rate_bps=eve_rates,
            secrecy_rate_bps=secrecy, per_uav_secrecy_bps=by_uav,
            total_rate_bps=float(rates.sum()),
            total_secrecy_bps=float(secrecy.sum()),
            a2g_state=ground, a2a_state=air, a2a_sinr=air_sinr,
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
