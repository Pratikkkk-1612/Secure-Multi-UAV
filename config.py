"""Configuration for the Secure Multi-UAV SAGIN simulator."""

from dataclasses import dataclass, field
from math import isclose, log10


def _noise_power_w(
    bandwidth_hz: float,
    density_dbm_hz: float,
    noise_figure_db: float,
    margin_db: float,
) -> float:
    if bandwidth_hz <= 0:
        raise ValueError("Noise bandwidth must be positive")
    noise_dbm = (
        density_dbm_hz
        + 10 * log10(bandwidth_hz)
        + noise_figure_db
        + margin_db
    )
    return 10 ** ((noise_dbm - 30) / 10)


@dataclass(frozen=True)
class NetworkConfig:
    n_uav: int = 4
    n_gu: int = 200
    n_eav: int = 4
    area_x_m: float = 1000.0
    area_y_m: float = 1000.0

    mobility_mode: str = "2d"
    fixed_2d_altitude_m: float = 400.0
    gu_altitude_m: float = 1.5
    initial_uav_3d_altitude_m: float = 400.0
    initial_eav_3d_altitude_m: float = 400.0
    uav_3d_altitude_limits_m: tuple[float, float] = (300.0, 600.0)
    eav_3d_altitude_limits_m: tuple[float, float] = (300.0, 600.0)

    uav_max_speed_mps: float = 20.0
    gu_speed_mps: float = 1.0
    eav_speed_mps: float = 20.0
    dt_s: float = 0.1
    episode_duration_s: float = 10.0

    n_resource_blocks_per_uav: int = 100
    min_gu_per_uav: int = 15
    max_gu_per_uav: int = 100

    @property
    def movement_dimensions(self) -> int:
        return 2 if self.mobility_mode == "2d" else 3

    @property
    def uav_altitude_m(self) -> float:
        return (
            self.fixed_2d_altitude_m
            if self.mobility_mode == "2d"
            else self.initial_uav_3d_altitude_m
        )

    @property
    def eav_altitude_m(self) -> float:
        return (
            self.fixed_2d_altitude_m
            if self.mobility_mode == "2d"
            else self.initial_eav_3d_altitude_m
        )

    @property
    def n_steps(self) -> int:
        return round(self.episode_duration_s / self.dt_s)

    @property
    def max_uav_step_m(self) -> float:
        return self.uav_max_speed_mps * self.dt_s

    @property
    def max_eav_step_m(self) -> float:
        return self.eav_speed_mps * self.dt_s

    @property
    def n_resource_blocks(self) -> int:
        return self.n_resource_blocks_per_uav


@dataclass(frozen=True)
class ChannelConfig:
    n_tx_antennas: int = 2
    carrier_frequency_hz: float = 28e9
    speed_of_light_mps: float = 3e8
    total_bandwidth_hz: float = 200e6
    outage_sinr_threshold_db: float = 5.0
    tx_antenna_gain_dbi: float = 15.0
    thermal_noise_density_dbm_per_hz: float = -174.0
    noise_figure_db: float = 8.0
    implementation_margin_db: float = 10.0
    monte_carlo_realizations: int = 5000

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light_mps / self.carrier_frequency_hz

    @property
    def noise_density_dbm_hz(self) -> float:
        """Backward-compatible name used by the A2G channel."""
        return self.thermal_noise_density_dbm_per_hz

    @property
    def noise_power_w(self) -> float:
        """Receiver noise across the entire channel bandwidth."""
        return _noise_power_w(
            self.total_bandwidth_hz,
            self.thermal_noise_density_dbm_per_hz,
            self.noise_figure_db,
            self.implementation_margin_db,
        )


@dataclass(frozen=True)
class A2GConfig(ChannelConfig):
    n_resource_blocks: int = 100

    tx_antenna_gain_dbi: float = 18.0
    gu_rx_antenna_gain_dbi: float = 9.0

    nakagami_m: float = 2.0
    los_a: float = 9.61
    los_b: float = 0.16
    los_excess_loss_db: float = 1.0
    nlos_excess_loss_db: float = 20.0

    oxygen_attenuation_db_per_km: float = 0.11
    rain_attenuation_db_per_km: float = 0.25
    blockage_beta_per_m: float = 0.001

    # For standalone Monte Carlo validation, not every RL environment step.
    monte_carlo_realizations: int = 1_000_000
    power_sweep_dbm: tuple[float, ...] = tuple(
        sorted((*range(0, 37, 3), 19.03))
    )

    @property
    def urban_a(self) -> float:
        return self.los_a

    @property
    def urban_b(self) -> float:
        return self.los_b

    @property
    def gas_attenuation_db_per_km(self) -> float:
        return self.oxygen_attenuation_db_per_km

    @property
    def resource_block_bandwidth_hz(self) -> float:
        return self.rb_bandwidth_hz(self.n_resource_blocks)

    @property
    def noise_power_w(self) -> float:
        """A2G noise per resource block, as used by a2g_sinr()."""
        return self.noise_power_w_per_rb(self.n_resource_blocks)

    def rb_bandwidth_hz(self, n_resource_blocks: int) -> float:
        if n_resource_blocks <= 0:
            raise ValueError("Resource-block count must be positive")
        return self.total_bandwidth_hz / n_resource_blocks

    def noise_power_w_per_rb(self, n_resource_blocks: int) -> float:
        return _noise_power_w(
            self.rb_bandwidth_hz(n_resource_blocks),
            self.thermal_noise_density_dbm_per_hz,
            self.noise_figure_db,
            self.implementation_margin_db,
        )


@dataclass(frozen=True)
class A2AConfig(ChannelConfig):
    n_resource_blocks: int = 100

    tx_antenna_gain_dbi: float = 18.0
    uav_rx_antenna_gain_dbi: float = 15.0
    eav_rx_antenna_gain_dbi: float = 5.0

    nakagami_m: float = 2.0
    obstacle_height_scale_m: float = 20.0
    obstacle_density_beta: float = 2e-3
    gas_attenuation_db_per_km: float = 0.07
    rain_attenuation_db_per_km: float = 2.5

    @property
    def resource_block_bandwidth_hz(self) -> float:
        if self.n_resource_blocks <= 0:
            raise ValueError("Resource-block count must be positive")
        return self.total_bandwidth_hz / self.n_resource_blocks

    @property
    def rb_noise_power_w(self) -> float:
        """A2A receiver noise per RB for scheduled EAV interception."""
        return _noise_power_w(
            self.resource_block_bandwidth_hz,
            self.thermal_noise_density_dbm_per_hz,
            self.noise_figure_db,
            self.implementation_margin_db,
        )


@dataclass(frozen=True)
class A2SConfig(ChannelConfig):
    # Retained from the existing configuration; validate against the final FSO model.
    gu_rx_antenna_gain_dbi: float = 5.0
    nakagami_m_los: float = 2.0
    nakagami_m_nlos: float = 1.0
    los_a: float = 9.61
    los_b: float = 0.16
    nlos_excess_loss_db: float = 20.0
    gas_attenuation_db_per_km: float = 0.07
    rain_attenuation_db_per_km: float = 2.5
    full_frequency_reuse: bool = True
    random_seed: int = 1


@dataclass(frozen=True)
class SecurityConfig:
    min_uav_power_w: float = 0.0
    average_uav_power_budget_w: float = 0.1
    max_uav_power_w: float = 0.4
    n_tx_antennas: int = 2

    @property
    def initial_uav_power_w(self) -> float:
        return self.average_uav_power_budget_w


@dataclass(frozen=True)
class ProjectConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    a2g: A2GConfig = field(default_factory=A2GConfig)
    a2a: A2AConfig = field(default_factory=A2AConfig)
    a2s: A2SConfig = field(default_factory=A2SConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    @property
    def action_dimensions_per_uav(self) -> int:
        return self.network.movement_dimensions + 1

    def __post_init__(self) -> None:
        n, s = self.network, self.security
        if n.mobility_mode not in {"2d", "3d"}:
            raise ValueError("mobility_mode must be '2d' or '3d'")
        if min(n.n_uav, n.n_gu, n.n_resource_blocks_per_uav) <= 0 or n.n_eav < 0:
            raise ValueError("Invalid network entity or resource-block count")
        if min(n.area_x_m, n.area_y_m, n.fixed_2d_altitude_m) <= 0 or n.gu_altitude_m < 0:
            raise ValueError("Invalid area or altitude")
        if n.uav_max_speed_mps <= 0 or min(n.gu_speed_mps, n.eav_speed_mps) < 0:
            raise ValueError("Invalid mobility speed")
        if n.dt_s <= 0 or n.episode_duration_s <= 0 or not isclose(
            n.n_steps * n.dt_s, n.episode_duration_s, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("Episode duration must be a positive whole number of steps")
        if not (0 <= n.min_gu_per_uav <= n.max_gu_per_uav <= n.n_resource_blocks_per_uav):
            raise ValueError("Invalid user scheduling capacity")
        if not (n.min_gu_per_uav * n.n_uav <= n.n_gu <= n.max_gu_per_uav * n.n_uav):
            raise ValueError("User association is infeasible")
        if n.mobility_mode == "3d":
            for limits, initial in (
                (n.uav_3d_altitude_limits_m, n.initial_uav_3d_altitude_m),
                (n.eav_3d_altitude_limits_m, n.initial_eav_3d_altitude_m),
            ):
                if not (0 < limits[0] < limits[1] and limits[0] <= initial <= limits[1]):
                    raise ValueError("Invalid 3D altitude bounds")
        if not (0 <= s.min_uav_power_w <= s.average_uav_power_budget_w <= s.max_uav_power_w):
            raise ValueError("Invalid power budget")
        if any(cfg.n_tx_antennas != s.n_tx_antennas for cfg in (self.a2g, self.a2a, self.a2s)):
            raise ValueError("Transmit antenna counts must agree")
        if not (
            n.n_resource_blocks == self.a2g.n_resource_blocks == self.a2a.n_resource_blocks
        ):
            raise ValueError("A2G/A2A resource-block counts must agree")
        if self.a2g.carrier_frequency_hz != self.a2a.carrier_frequency_hz:
            raise ValueError("A2G/A2A carrier frequencies must agree")
        if self.a2g.total_bandwidth_hz != self.a2a.total_bandwidth_hz:
            raise ValueError("A2G/A2A bandwidths must agree")
        if self.a2g.tx_antenna_gain_dbi != self.a2a.tx_antenna_gain_dbi:
            raise ValueError("UAV transmit antenna gains must agree")
        for cfg in (self.a2g, self.a2a, self.a2s):
            if cfg.carrier_frequency_hz <= 0 or cfg.total_bandwidth_hz <= 0:
                raise ValueError("Channel frequency and bandwidth must be positive")
            if cfg.monte_carlo_realizations <= 0:
                raise ValueError("Monte Carlo realizations must be positive")


DEFAULT_CONFIG = ProjectConfig()
