"""UAV-to-ground channel, following A2G_Channel(1).m.

Fixed equal-power transmit weights across antennas (no beamforming).
The MATLAB schedule format is used: 0 = idle RB, 1..N_GU = GU indices.
Inputs must use the same 3-D Cartesian coordinate frame (metres).
"""

from dataclasses import dataclass
import numpy as np
from config import A2GConfig



@dataclass(frozen=True)
class A2GLinkState:
    distance_m: np.ndarray               # (N_UAV, N_GU)
    horizontal_distance_m: np.ndarray
    elevation_deg: np.ndarray
    los_probability: np.ndarray
    is_los: np.ndarray
    path_loss_db: np.ndarray              # includes excess/atmospheric loss; not blockage
    large_scale_gain: np.ndarray          # includes MATLAB exp(-beta*d) power factor
    h: np.ndarray                         # (N_UAV, N_GU, Nt)
    gain: np.ndarray                      # mean |h|^2 over Tx antennas


@dataclass(frozen=True)
class A2GPerformance:
    power_dbm: np.ndarray
    outage: np.ndarray
    mean_sinr: np.ndarray
    mean_sinr_db: np.ndarray
    mean_rate_mbps: np.ndarray
    last_state: A2GLinkState


def _nodes(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a nonempty, finite (N, 3) array")
    return arr


def _power_vector(power_w: float | np.ndarray, n_uav: int) -> np.ndarray:
    power = np.asarray(power_w, dtype=float)
    if power.ndim == 0:
        power = np.full(n_uav, float(power))
    if power.shape != (n_uav,) or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("Power must be nonnegative, scalar or (N_UAV,)")
    return power


def _schedule(schedule: np.ndarray, n_uav: int, n_gu: int, cfg: A2GConfig) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(schedule)
    if s.shape != (n_uav, cfg.n_resource_blocks) or not np.issubdtype(s.dtype, np.integer):
        raise ValueError("scheduled_gu must be an integer (N_UAV, N_RB) array")
    if np.any(s < 0) or np.any(s > n_gu):
        raise ValueError("GU indices must be 0 (idle) or 1..N_GU")
    assigned = s[s > 0]
    if len(assigned) != n_gu or not np.array_equal(np.sort(assigned), np.arange(1, n_gu + 1)):
        raise ValueError("Each GU must appear exactly once")
    counts = np.count_nonzero(s, axis=1)
    if np.any(counts == 0):
        raise ValueError("Every UAV must serve at least one GU")
    return s, counts


def sample_a2g(uav_pos: np.ndarray, gu_pos: np.ndarray, *,
               config: A2GConfig | None = None,
               rng: np.random.Generator | None = None) -> A2GLinkState:
    cfg = config or A2GConfig()
    rng = rng if rng is not None else np.random.default_rng()
    uav_pos = _nodes(uav_pos, "uav_pos")
    gu_pos = _nodes(gu_pos, "gu_pos")
    if (cfg.carrier_frequency_hz <= 0 or cfg.total_bandwidth_hz <= 0
            or cfg.n_resource_blocks < 1 or cfg.n_tx_antennas < 1
            or cfg.nakagami_m < 0.5 or cfg.blockage_beta_per_m < 0):
        raise ValueError("Invalid A2G physical configuration")

    delta = uav_pos[:, None, :] - gu_pos[None, :, :]
    horizontal = np.linalg.norm(delta[..., :2], axis=-1)
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(delta[..., 2] <= 0):
        raise ValueError("All UAVs must be above all GUs for this reference model")
    elevation = np.degrees(np.arctan2(delta[..., 2], horizontal))
    p_los = 1 / (1 + cfg.urban_a * np.exp(-cfg.urban_b * (elevation - cfg.urban_a)))
    los = rng.random(distance.shape) < p_los

    free_space_db = 20 * np.log10(4 * np.pi * distance / cfg.wavelength_m)
    atmospheric_db = (cfg.gas_attenuation_db_per_km
                      + cfg.rain_attenuation_db_per_km) * distance / 1000
    path_loss_db = (free_space_db + atmospheric_db
                    + np.where(los, cfg.los_excess_loss_db, cfg.nlos_excess_loss_db))
    blockage_power = np.exp(-cfg.blockage_beta_per_m * distance)
    large_scale_gain = (10 ** ((cfg.tx_antenna_gain_dbi + cfg.gu_rx_antenna_gain_dbi
                                - path_loss_db) / 10) * blockage_power)

    shape = (*distance.shape, cfg.n_tx_antennas)
    fading_power = rng.gamma(cfg.nakagami_m, 1 / cfg.nakagami_m, size=shape)
    phase = 2 * np.pi * rng.random(shape)
    h = np.sqrt(large_scale_gain[..., None] * fading_power) * np.exp(1j * phase)
    gain = np.abs(np.sum(h, axis=-1)) ** 2 / cfg.n_tx_antennas
    return A2GLinkState(distance, horizontal, elevation, p_los, los,path_loss_db, large_scale_gain, h, gain)


def expected_a2g_gain(uav_pos, gu_pos, *, config=None):
    """Expected large-scale channel power gain for every UAV-GU link."""
    cfg = config or A2GConfig()

    uav_pos = _nodes(uav_pos, "uav_pos")
    gu_pos = _nodes(gu_pos, "gu_pos")

    delta = uav_pos[:, None, :] - gu_pos[None, :, :]

    horizontal = np.linalg.norm(delta[..., :2], axis=-1)
    distance = np.linalg.norm(delta, axis=-1)

    if np.any(delta[..., 2] <= 0):
        raise ValueError("UAVs must be above ground users")

    elevation = np.degrees(
        np.arctan2(delta[..., 2], horizontal)
    )

    p_los = 1.0 / (
        1.0 + cfg.urban_a * np.exp(
            -cfg.urban_b * (elevation - cfg.urban_a)
        )
    )

    fspl_db = 20.0 * np.log10(
        4.0 * np.pi * distance / cfg.wavelength_m
    )

    atmospheric_db = (
        cfg.gas_attenuation_db_per_km
        + cfg.rain_attenuation_db_per_km
    ) * distance / 1000.0

    common_gain = 10.0 ** (
        (
            cfg.tx_antenna_gain_dbi
            + cfg.gu_rx_antenna_gain_dbi
            - fspl_db
            - atmospheric_db
        ) / 10.0
    )

    blockage = np.exp(
        -cfg.blockage_beta_per_m * distance
    )

    g_los = (
        common_gain
        * blockage
        * 10.0 ** (-cfg.los_excess_loss_db / 10.0)
    )

    g_nlos = (
        common_gain
        * blockage
        * 10.0 ** (-cfg.nlos_excess_loss_db / 10.0)
    )

    return p_los * g_los + (1.0 - p_los) * g_nlos


def _scheduled_coefficients(
    state: A2GLinkState,
    scheduled_gu: np.ndarray,
    cfg: A2GConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """coefficients[u, g]: power gain of UAV u at GU g, split over UAV u's GUs.
    serving[g]: index of the UAV serving GU g."""
    n_uav, n_gu = state.gain.shape
    schedule, counts = _schedule(scheduled_gu, n_uav, n_gu, cfg)

    uav_idx, rb_idx = np.nonzero(schedule)
    gu_idx = schedule[uav_idx, rb_idx] - 1
    serving = np.empty(n_gu, dtype=int)
    rb_of_gu = np.empty(n_gu, dtype=int)
    serving[gu_idx] = uav_idx
    rb_of_gu[gu_idx] = rb_idx

    # UAV u interferes with GU g if u is also transmitting on g's RB.
    active = schedule[:, rb_of_gu] > 0                     # (N_UAV, N_GU)
    coefficients = np.where(active, state.gain / counts[:, None], 0.0)
    return coefficients, serving




def a2g_sinr(
    state: A2GLinkState,
    scheduled_gu: np.ndarray,
    transmit_power_w: float | np.ndarray,
    *,
    config: A2GConfig | None = None,
) -> np.ndarray:

    cfg = config or A2GConfig()

    n_uav, n_gu = state.gain.shape
    power = _power_vector(transmit_power_w, n_uav)

    coefficients, serving = _scheduled_coefficients(
        state, scheduled_gu, cfg
    )

    received = power[:, None] * coefficients
    desired = received[serving, np.arange(n_gu)]
    interference = received.sum(axis=0) - desired

    return desired / (
        np.maximum(interference, 0.0) + cfg.noise_power_w
    )



def evaluate_a2g(uav_pos: np.ndarray, gu_pos: np.ndarray,
                 scheduled_gu: np.ndarray, *,
                 config: A2GConfig | None = None,
                 n_realizations: int | None = None,
                 rng: np.random.Generator | None = None) -> A2GPerformance:
    """Monte Carlo power sweep at fixed geometry. Offline validation only;
    never call this inside the RL loop."""
    cfg = config or A2GConfig()
    n_mc = cfg.monte_carlo_realizations if n_realizations is None else n_realizations
    if not isinstance(n_mc, (int, np.integer)) or n_mc < 1:
        raise ValueError("n_realizations must be a positive integer")
    rng = rng if rng is not None else np.random.default_rng()
    powers_dbm = np.asarray(cfg.power_sweep_dbm, dtype=float)
    powers = 10 ** ((powers_dbm - 30) / 10)
    threshold = 10 ** (cfg.outage_sinr_threshold_db / 10)
    sum_sinr = np.zeros(len(powers))
    sum_rate = np.zeros(len(powers))
    outage = np.zeros(len(powers))
    for _ in range(n_mc):
        state = sample_a2g(uav_pos, gu_pos, config=cfg, rng=rng)
        coefficients, serving = _scheduled_coefficients(state, scheduled_gu, cfg)
        n_gu = state.gain.shape[1]
        desired = coefficients[serving, np.arange(n_gu)]
        interf = np.maximum(coefficients.sum(axis=0) - desired, 0)
        gamma = powers[:, None] * desired[None, :] / (
            powers[:, None] * interf[None, :] + cfg.noise_power_w)
        sum_sinr += gamma.sum(axis=1)
        outage += (gamma < threshold).sum(axis=1)
        sum_rate += (cfg.resource_block_bandwidth_hz * np.log2(1 + gamma)).sum(axis=1)
    total = n_mc * n_gu
    mean_sinr = sum_sinr / total
    return A2GPerformance(powers_dbm, outage / total, mean_sinr,
                          10 * np.log10(mean_sinr), sum_rate / total / 1e6, state)