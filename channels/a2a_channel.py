"""Air-to-air channel"""

from dataclasses import dataclass

import numpy as np

from config import A2AConfig


# MATLAB reference sweep, used when the shared config does not specify one.
MATLAB_POWER_SWEEP_DBM = tuple(sorted((*range(0, 37, 3), 19.03)))


@dataclass(frozen=True)
class A2ALinkState:
    distance_m: np.ndarray
    radial_velocity_mps: np.ndarray
    doppler_hz: np.ndarray
    los_probability: np.ndarray
    is_los: np.ndarray
    path_loss_db: np.ndarray
    large_scale_gain: np.ndarray
    h: np.ndarray  
    gain: np.ndarray
    interference_gain: np.ndarray


@dataclass(frozen=True)
class A2AChannelState:
    uav_uav: A2ALinkState
    uav_eav: A2ALinkState


@dataclass(frozen=True)
class A2ASINR:
    uav_uav: np.ndarray
    uav_eav: np.ndarray


@dataclass(frozen=True)
class A2APerformance:
    power_dbm: np.ndarray
    outage_uav_uav: np.ndarray
    outage_uav_eav: np.ndarray
    mean_sinr_uav_uav: np.ndarray
    mean_sinr_uav_eav: np.ndarray
    mean_sinr_uav_uav_db: np.ndarray
    mean_sinr_uav_eav_db: np.ndarray
    mean_rate_uav_uav_mbps: np.ndarray
    mean_rate_uav_eav_mbps: np.ndarray
    last_state: A2AChannelState


def _nodes(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be a finite (N, 3) array")
    return values


def _geometry(tx: np.ndarray, rx: np.ndarray, tx_vel: np.ndarray,
              rx_vel: np.ndarray, wavelength_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacement = rx[None, :, :] - tx[:, None, :]
    distance = np.linalg.norm(displacement, axis=-1)
    relative_velocity = rx_vel[None, :, :] - tx_vel[:, None, :]
    radial = np.sum(relative_velocity * displacement, axis=-1) / np.maximum(distance, 1e-9)
    return distance, radial, radial / wavelength_m


def _los_probability(tx: np.ndarray, rx: np.ndarray, distance: np.ndarray,
                     cfg: A2AConfig) -> np.ndarray:
    mean_altitude = (tx[:, None, 2] + rx[None, :, 2]) / 2
    fresnel_radius = np.sqrt(cfg.wavelength_m * np.maximum(distance, 1e-9)) / 2
    area = (np.pi * distance / 2) * fresnel_radius
    p_clear = 1 - np.exp(-(mean_altitude ** 2) / (2 * cfg.obstacle_height_scale_m ** 2))
    return np.clip(p_clear ** (area * cfg.obstacle_density_beta), 0.0, 1.0)


def _knife_edge_loss_db(tx: np.ndarray, rx: np.ndarray, distance_m: float,
                        cfg: A2AConfig, rng: np.random.Generator) -> float:
    # Exact sampleKED() assumptions from the MATLAB file: midpoint obstruction.
    d1 = d2 = distance_m / 2
    obstacle_height = cfg.obstacle_height_scale_m * np.sqrt(
        -2 * np.log(max(float(rng.random()), np.finfo(float).tiny))
    )
    line_height = tx[2] + (rx[2] - tx[2]) * (d1 / distance_m)
    clearance = obstacle_height - line_height
    v = clearance * np.sqrt(
        2 * (d1 + d2) /
        (cfg.wavelength_m * d1 * d2)
    )

    if v <= -0.78:
        return 0.0

    loss_db = 6.9 + 20 * np.log10(
        np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1
    )

    return float(max(loss_db, 0.0))     


def _sample_links(tx: np.ndarray, rx: np.ndarray, tx_vel: np.ndarray,
                  rx_vel: np.ndarray, t_s: float, receiver_gain_dbi: float,
                  cfg: A2AConfig, rng: np.random.Generator,
                  self_links: bool) -> A2ALinkState:
    d, vrad, fd = _geometry(tx, rx, tx_vel, rx_vel, cfg.wavelength_m)
    if self_links:
        invalid = np.eye(len(tx), dtype=bool)
        if np.any(d[~invalid] <= 0):
            raise ValueError("Distinct UAVs must not occupy identical positions")
    else:
        invalid = np.zeros(d.shape, dtype=bool)
        if np.any(d <= 0):
            raise ValueError("A UAV and an EAV must not occupy identical positions")

    p_los = _los_probability(tx, rx, d, cfg)
    is_los = rng.random(d.shape) < p_los
    is_los[invalid] = False

    fspl_db = 20 * np.log10(4 * np.pi * np.maximum(d, 1e-9) / cfg.wavelength_m)
    atmospheric_db = (
        cfg.gas_attenuation_db_per_km + cfg.rain_attenuation_db_per_km
    ) * (d / 1000)
    path_loss_db = fspl_db + atmospheric_db

    for i, j in np.argwhere(~is_los & ~invalid):
        path_loss_db[i, j] += _knife_edge_loss_db(tx[i], rx[j], d[i, j], cfg, rng)

    large_scale_gain = 10 ** ((cfg.tx_antenna_gain_dbi + receiver_gain_dbi - path_loss_db) / 10)
    large_scale_gain[invalid] = np.nan

    fading_power = rng.gamma(cfg.nakagami_m, 1 / cfg.nakagami_m,
                             size=(*d.shape, cfg.n_tx_antennas))
    phase = 2 * np.pi * rng.random((*d.shape, cfg.n_tx_antennas)) + 2 * np.pi * fd[..., None] * t_s
    h = np.sqrt(large_scale_gain[..., None] * fading_power) * np.exp(1j * phase)
    gain = np.abs(np.sum(h, axis=-1)) ** 2 / cfg.n_tx_antennas

    # Full-frequency reuse; exclude desired transmitter, and exclude the UAV
    # receiver's nonexistent self-transmission for UAV-UAV links.
    total_per_receiver = np.nansum(gain, axis=0)
    interference_gain = np.maximum(total_per_receiver[None, :] - np.nan_to_num(gain, nan=0), 0)
    interference_gain[invalid] = 0
    vrad[invalid] = np.nan
    fd[invalid] = np.nan
    p_los[invalid] = np.nan

    return A2ALinkState(d, vrad, fd, p_los, is_los, path_loss_db,
                        large_scale_gain, h, gain, interference_gain)


def sample_a2a(uav_pos: np.ndarray, eav_pos: np.ndarray,
               uav_vel: np.ndarray, eav_vel: np.ndarray, t_s: float, *,
               config: A2AConfig | None = None,
               rng: np.random.Generator | None = None) -> A2AChannelState:
    """Sample one channel realization at simulation time ``t_s`` (seconds).

    Returned gains include both transmit antennas. UAV-UAV arrays have shape
    (N_UAV, N_UAV), UAV-EAV arrays (N_UAV, N_EAV); ``h`` adds an antenna axis.
    UAV-UAV diagonal entries are invalid self-links.
    """
    cfg = config if config is not None else A2AConfig()
    rng = rng if rng is not None else np.random.default_rng()
    uav_pos, eav_pos = _nodes(uav_pos, "uav_pos"), _nodes(eav_pos, "eav_pos")
    uav_vel, eav_vel = _nodes(uav_vel, "uav_vel"), _nodes(eav_vel, "eav_vel")

    if not len(uav_pos) or not len(eav_pos):
        raise ValueError("At least one UAV and one EAV are required")
    if uav_pos.shape != uav_vel.shape or eav_pos.shape != eav_vel.shape:
        raise ValueError("Each velocity array must match its position array")
    if not np.isfinite(t_s):
        raise ValueError("t_s must be finite")
    if (cfg.n_tx_antennas < 1 or cfg.nakagami_m <= 0 or
            cfg.obstacle_height_scale_m <= 0 or cfg.wavelength_m <= 0):
        raise ValueError("Invalid A2A physical configuration")

    uu = _sample_links(uav_pos, uav_pos, uav_vel, uav_vel, t_s,
                       cfg.uav_rx_antenna_gain_dbi, cfg, rng, True)
    ue = _sample_links(uav_pos, eav_pos, uav_vel, eav_vel, t_s,
                       cfg.eav_rx_antenna_gain_dbi, cfg, rng, False)
    return A2AChannelState(uu, ue)


def a2a_sinr(state: A2AChannelState, transmit_power_w: float | np.ndarray, *,
             config: A2AConfig | None = None) -> A2ASINR:
    """Compute SINR with a common power or one RL-selected power per UAV."""
    cfg = config if config is not None else A2AConfig()
    n_uav = state.uav_uav.gain.shape[0]
    power = np.asarray(transmit_power_w, dtype=float)
    if power.ndim == 0:
        power = np.full(n_uav, float(power))
    if power.shape != (n_uav,) or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("transmit_power_w must be nonnegative and scalar or (N_UAV,)")

    def sinr(gain: np.ndarray) -> np.ndarray:
        weighted_gain = power[:, None] * gain
        received_total = np.nansum(weighted_gain, axis=0, keepdims=True)
        interference = np.maximum(received_total - np.nan_to_num(weighted_gain, nan=0), 0)
        return weighted_gain / (interference + cfg.noise_power_w)

    return A2ASINR(sinr(state.uav_uav.gain), sinr(state.uav_eav.gain))



def a2a_eav_sinr_rb(
    state: A2AChannelState,
    scheduled_gu: np.ndarray,
    transmit_power_w: float | np.ndarray,
    *,
    config: A2AConfig | None = None,
) -> np.ndarray:
    """Eavesdropper SINR per UAV, EAV and resource block."""

    cfg = config or A2AConfig()

    gain = state.uav_eav.gain
    n_uav, n_eav = gain.shape

    schedule = np.asarray(scheduled_gu)

    if (schedule.shape != (n_uav, cfg.n_resource_blocks)
            or not np.issubdtype(schedule.dtype, np.integer)
            or np.any(schedule < 0)):
        raise ValueError("Invalid resource-block schedule")
    
    assigned = schedule[schedule > 0]
    if (
        len(np.unique(assigned)) != len(assigned)
        or not np.array_equal(
            np.sort(assigned),
            np.arange(1, len(assigned) + 1)
        )
    ):
        raise ValueError("Invalid resource-block schedule") 

    power = np.asarray(transmit_power_w, dtype=float)

    if power.ndim == 0:
        power = np.full(n_uav, float(power))

    if (power.shape != (n_uav,)
            or not np.all(np.isfinite(power))
            or np.any(power < 0)):
        raise ValueError("Invalid transmit power")

    counts = np.count_nonzero(schedule, axis=1)

    if np.any(counts == 0):
        raise ValueError("Each UAV must serve at least one GU")

    stream_power = power / counts

    sinr = np.zeros(
        (n_uav, n_eav, cfg.n_resource_blocks),
        dtype=float
    )

    for rb in range(cfg.n_resource_blocks):

        active = schedule[:, rb] > 0

        if not np.any(active):
            continue

        received = (
            stream_power[:, None]
            * gain
            * active[:, None]
        )

        interference = (
            received.sum(axis=0, keepdims=True)
            - received
        )

        sinr[:, :, rb] = received / (
            np.maximum(interference, 0.0)
            + cfg.rb_noise_power_w
        )

    return sinr


def evaluate_a2a(uav_pos: np.ndarray, eav_pos: np.ndarray,
                 uav_vel: np.ndarray, eav_vel: np.ndarray, t_s: float, *,
                 config: A2AConfig | None = None,
                 n_realizations: int | None = None,
                 rng: np.random.Generator | None = None,
                 power_sweep_dbm: tuple[float, ...] | np.ndarray | None = None
                 ) -> A2APerformance:
    """Replicate MATLAB power-sweep Monte Carlo metrics at fixed geometry.

    The sweep uses the same power at every UAV, unlike RL operation.
    With no explicit sweep, use the config's sweep if present; otherwise use
    the original MATLAB sweep (0:3:36 dBm plus 19.03 dBm).
    Mean SINR is averaged in linear scale before conversion to dB.
    """
    cfg = config if config is not None else A2AConfig()
    n_mc = cfg.monte_carlo_realizations if n_realizations is None else n_realizations
    if not isinstance(n_mc, (int, np.integer)) or n_mc < 1:
        raise ValueError("n_realizations must be a positive integer")
    rng = rng if rng is not None else np.random.default_rng()
    sweep = (power_sweep_dbm if power_sweep_dbm is not None else
             getattr(cfg, "power_sweep_dbm", MATLAB_POWER_SWEEP_DBM))
    power_dbm = np.asarray(sweep, dtype=float)
    if (power_dbm.ndim != 1 or power_dbm.size == 0 or
            not np.all(np.isfinite(power_dbm))):
        raise ValueError("power_sweep_dbm must contain finite dBm values")
    power_w = 10 ** ((power_dbm - 30) / 10)
    outage = np.zeros((2, len(power_w)))
    sinr_sum = np.zeros_like(outage)
    rate_sum = np.zeros_like(outage)
    threshold = 10 ** (cfg.outage_sinr_threshold_db / 10)

    for _ in range(n_mc):
        state = sample_a2a(uav_pos, eav_pos, uav_vel, eav_vel,
                           t_s, config=cfg, rng=rng)
        for link_id, link in enumerate((state.uav_uav, state.uav_eav)):
            gain = link.gain[np.isfinite(link.gain)]
            interference = link.interference_gain[np.isfinite(link.gain)]
            sinr = power_w[:, None] * gain[None, :] / (
                power_w[:, None] * interference[None, :] + cfg.noise_power_w
            )
            outage[link_id] += np.sum(sinr < threshold, axis=1)
            sinr_sum[link_id] += np.sum(sinr, axis=1)
            rate_sum[link_id] += np.sum(cfg.total_bandwidth_hz * np.log2(1 + sinr), axis=1)

    n_uav, n_eav = state.uav_eav.gain.shape
    observations = n_mc * np.array([n_uav * (n_uav - 1), n_uav * n_eav])
    if np.any(observations == 0):
        raise ValueError("Monte Carlo metrics require at least two UAVs")
    mean_sinr = sinr_sum / observations[:, None]
    mean_rate_mbps = rate_sum / observations[:, None] / 1e6
    return A2APerformance(
        power_dbm, *(outage / observations[:, None]),
        *mean_sinr, *(10 * np.log10(mean_sinr)), *mean_rate_mbps, state,
    )
