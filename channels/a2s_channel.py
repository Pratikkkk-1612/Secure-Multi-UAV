"""Provisional UAV-to-SATELLITE (A2S) uplink channel.

CAUTION: the supplied 'A2S_Channel(1).m' actually models UAV -> GU,
not UAV -> satellite. Only generic FSPL, Nakagami fading, Doppler,
frequency/noise/power-sweep parameters can be reused. This is a physically
identified baseline, NOT a validated 3GPP NTN, orbital or rain-fade model.

UAV and satellite 3-D positions/velocities must be in the SAME Cartesian
frame in metres / m/s. No orbit propagation or Earth-horizon checking.
Atmospheric attenuation uses a specified *effective atmospheric path*
(not the entire satellite slant range). Default 0 km avoids inventing one.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class A2SConfig:
    carrier_frequency_hz: float = 28e9
    speed_of_light_mps: float = 3e8
    total_bandwidth_hz: float = 200e6
    noise_density_dbm_hz: float = -174.0
    noise_figure_db: float = 8.0
    implementation_margin_db: float = 10.0
    n_tx_antennas: int = 2
    tx_antenna_gain_dbi: float = 15.0
    satellite_rx_antenna_gain_dbi: float = 5.0  # Placeholder: copied GU gain, NOT calibrated satellite gain
    nakagami_m_los: float = 2.0
    nakagami_m_nlos: float = 1.0
    los_probability: float = 1.0             # Clear unobstructed link assumption
    nlos_excess_loss_db: float = 20.0        # From MATLAB, only relevant if p_LOS < 1
    gas_attenuation_db_per_km: float = 0.07
    rain_attenuation_db_per_km: float = 2.5
    effective_atmospheric_path_km: float = 0.0  # Must be set from satellite geometry/weather
    outage_sinr_threshold_db: float = 5.0
    monte_carlo_realizations: int = 5000
    power_sweep_dbm: tuple[float, ...] = tuple(sorted((*range(0, 37, 3), 19.03)))

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light_mps / self.carrier_frequency_hz

    @property
    def noise_power_w(self) -> float:
        dbm = (self.noise_density_dbm_hz + 10 * np.log10(self.total_bandwidth_hz)
               + self.noise_figure_db + self.implementation_margin_db)
        return float(10 ** ((dbm - 30) / 10))


@dataclass(frozen=True)
class A2SLinkState:
    distance_m: np.ndarray              # (N_UAV, N_SAT)
    radial_velocity_mps: np.ndarray
    doppler_hz: np.ndarray
    los_probability: np.ndarray
    is_los: np.ndarray
    path_loss_db: np.ndarray
    large_scale_gain: np.ndarray
    h: np.ndarray                       # (N_UAV, N_SAT, Nt)
    gain: np.ndarray                    # sum(|h|^2), MATLAB aggregation convention
    interference_gain: np.ndarray       # sum of other UAV gains, equal-power sweep


@dataclass(frozen=True)
class A2SPerformance:
    power_dbm: np.ndarray
    outage: np.ndarray
    mean_sinr: np.ndarray
    mean_sinr_db: np.ndarray
    mean_rate_mbps: np.ndarray
    last_state: A2SLinkState


def _nodes(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be a finite, nonempty (N,3) array")
    return arr


def sample_a2s(uav_pos: np.ndarray, sat_pos: np.ndarray,
               uav_vel: np.ndarray, sat_vel: np.ndarray, t_s: float, *,
               config: A2SConfig | None = None,
               rng: np.random.Generator | None = None) -> A2SLinkState:
    """Sample one UAV-to-satellite uplink fading realization."""
    cfg = config or A2SConfig()
    rng = rng if rng is not None else np.random.default_rng()
    uav_pos, sat_pos = _nodes(uav_pos, "uav_pos"), _nodes(sat_pos, "sat_pos")
    uav_vel, sat_vel = _nodes(uav_vel, "uav_vel"), _nodes(sat_vel, "sat_vel")
    if uav_pos.shape != uav_vel.shape or sat_pos.shape != sat_vel.shape:
        raise ValueError("Velocities must have the same shape as their positions")
    if (not np.isfinite(t_s) or cfg.carrier_frequency_hz <= 0
            or cfg.total_bandwidth_hz <= 0 or cfg.n_tx_antennas < 1
            or cfg.nakagami_m_los < 0.5 or cfg.nakagami_m_nlos < 0.5
            or not 0 <= cfg.los_probability <= 1
            or cfg.effective_atmospheric_path_km < 0):
        raise ValueError("Invalid A2S configuration")
    displacement = sat_pos[None, :, :] - uav_pos[:, None, :]
    distance = np.linalg.norm(displacement, axis=-1)
    if np.any(distance <= 0):
        raise ValueError("A UAV and satellite cannot be colocated")
    vrel = sat_vel[None, :, :] - uav_vel[:, None, :]
    radial = np.sum(vrel * displacement, axis=-1) / distance
    doppler = radial / cfg.wavelength_m
    p_los = np.full(distance.shape, cfg.los_probability)
    is_los = rng.random(distance.shape) < p_los

    free_space = 20 * np.log10(4 * np.pi * distance / cfg.wavelength_m)
    atmospheric = (cfg.gas_attenuation_db_per_km + cfg.rain_attenuation_db_per_km
                   ) * cfg.effective_atmospheric_path_km
    path_loss = free_space + atmospheric + np.where(is_los, 0, cfg.nlos_excess_loss_db)
    gain_large = 10 ** ((cfg.tx_antenna_gain_dbi + cfg.satellite_rx_antenna_gain_dbi
                         - path_loss) / 10)
    m = np.where(is_los, cfg.nakagami_m_los, cfg.nakagami_m_nlos)
    shape = (*distance.shape, cfg.n_tx_antennas)
    fading = rng.gamma(shape=m[..., None], scale=1 / m[..., None], size=shape)
    phases = 2 * np.pi * rng.random(shape) + 2 * np.pi * doppler[..., None] * t_s
    h = np.sqrt(gain_large[..., None] * fading) * np.exp(1j * phases)
    channel_gain = np.sum(np.abs(h) ** 2, axis=-1)
    interference_gain = np.maximum(channel_gain.sum(axis=0, keepdims=True) - channel_gain, 0)
    return A2SLinkState(distance, radial, doppler, p_los, is_los, path_loss,
                        gain_large, h, channel_gain, interference_gain)


def a2s_sinr(state: A2SLinkState, transmit_power_w: float | np.ndarray, *,
             config: A2SConfig | None = None) -> np.ndarray:
    """Full-frequency-reuse uplink SINR for each candidate UAV-satellite link.

    Every UAV is assumed to transmit simultaneously toward each satellite.
    Scheduling/beamforming are not yet represented (same simplification as A2A).
    """
    cfg = config or A2SConfig()
    n_uav = state.gain.shape[0]
    power = np.asarray(transmit_power_w, dtype=float)
    if power.ndim == 0:
        power = np.full(n_uav, float(power))
    if power.shape != (n_uav,) or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("Power must be nonnegative, scalar or (N_UAV,)")
    received = power[:, None] * state.gain
    interference = np.maximum(received.sum(axis=0, keepdims=True) - received, 0)
    return received / (interference + cfg.noise_power_w)


def evaluate_a2s(uav_pos: np.ndarray, sat_pos: np.ndarray,
                 uav_vel: np.ndarray, sat_vel: np.ndarray, t_s: float, *,
                 config: A2SConfig | None = None,
                 n_realizations: int | None = None,
                 rng: np.random.Generator | None = None) -> A2SPerformance:
    """Independent-channel Monte Carlo power sweep at fixed geometry."""
    cfg = config or A2SConfig()
    n_mc = cfg.monte_carlo_realizations if n_realizations is None else n_realizations
    if not isinstance(n_mc, (int, np.integer)) or n_mc < 1:
        raise ValueError("n_realizations must be a positive integer")
    rng = rng if rng is not None else np.random.default_rng()
    power_dbm = np.asarray(cfg.power_sweep_dbm, dtype=float)
    power_w = 10 ** ((power_dbm - 30) / 10)
    outage = np.zeros(len(power_w))
    sum_sinr = np.zeros(len(power_w))
    sum_rate = np.zeros(len(power_w))
    for _ in range(n_mc):
        state = sample_a2s(uav_pos, sat_pos, uav_vel, sat_vel, t_s, config=cfg, rng=rng)
        gamma = power_w[:, None, None] * state.gain[None, :, :] / (
            power_w[:, None, None] * state.interference_gain[None, :, :] + cfg.noise_power_w
        )
        outage += np.sum(gamma < 10 ** (cfg.outage_sinr_threshold_db / 10), axis=(1, 2))
        sum_sinr += gamma.sum(axis=(1, 2))
        sum_rate += (cfg.total_bandwidth_hz * np.log2(1 + gamma)).sum(axis=(1, 2))
    n_obs = n_mc * state.gain.size
    mean_sinr = sum_sinr / n_obs
    return A2SPerformance(power_dbm, outage / n_obs, mean_sinr,
                          10 * np.log10(mean_sinr), sum_rate / n_obs / 1e6, state)
