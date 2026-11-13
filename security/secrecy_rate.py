from dataclasses import dataclass

import numpy as np

from config import DEFAULT_CONFIG, ProjectConfig


@dataclass(frozen=True)
class SecurityMetrics:
    """One slot. GU-indexed rates are stream diagnostics for the selected listener.

    secrecy_rate_bps clips each stream separately for diagnostics only. The
    training objective clips once per UAV, after summing all its streams.
    """
    legitimate_sinr: np.ndarray
    eav_sinr: np.ndarray
    gu_rate_bps: np.ndarray
    worst_eav_rate_bps: np.ndarray
    secrecy_rate_bps: np.ndarray
    per_uav_secrecy_rate_bps: np.ndarray
    transmit_weights: np.ndarray

    per_uav_gu_rate_bps: np.ndarray
    per_uav_eav_rate_bps: np.ndarray       # (UAV, EAV), summed over active RBs
    worst_eav_index: np.ndarray            # (UAV,), -1 when no EAV is present
    per_uav_worst_eav_rate_bps: np.ndarray

    @property
    def total_secrecy_rate_bps(self) -> float:
        return float(self.per_uav_secrecy_rate_bps.sum())

    @property
    def mean_secrecy_rate_bps(self) -> float:
        return self.total_secrecy_rate_bps / len(self.gu_rate_bps)


def evaluate_security(
    a2g_state,
    a2a_state,
    scheduled_gu: np.ndarray,
    transmit_power_w: float | np.ndarray,
    *,
    config: ProjectConfig | None = None,
) -> SecurityMetrics:
    """Aggregate each non-colluding listener over the UAV's active RBs.

    The normalized network total is an aggregate throughput metric. It does
    not quantify the secrecy of each user's independently encoded message.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    hg = np.asarray(a2g_state.h, dtype=complex)
    if a2a_state is None and cfg.network.n_eav == 0:
        he = np.empty((cfg.network.n_uav, 0, cfg.security.n_tx_antennas), dtype=complex)
    else:
        he = np.asarray(a2a_state.uav_eav.h, dtype=complex)
    if (
        hg.ndim != 3 or he.ndim != 3
        or hg.shape[0] != he.shape[0]
        or hg.shape[2] != he.shape[2]
        or not np.all(np.isfinite(hg))
        or not np.all(np.isfinite(he))
    ):
        raise ValueError("Channels must be finite (N_UAV, N_GU/EAV, Nt) arrays")

    n_uav, n_gu, nt = hg.shape
    n_eav = he.shape[1]
    net = cfg.network
    if (
        n_uav < 1 or n_gu < 1 or nt < 1
        or (n_uav, n_gu, n_eav) != (net.n_uav, net.n_gu, net.n_eav)
        or nt != cfg.security.n_tx_antennas
    ):
        raise ValueError("Channel dimensions must match ProjectConfig")

    schedule = np.asarray(scheduled_gu)
    n_rb = cfg.a2g.n_resource_blocks
    if (
        schedule.shape != (n_uav, n_rb)
        or not np.issubdtype(schedule.dtype, np.integer)
        or np.any(schedule < 0) or np.any(schedule > n_gu)
        or not np.array_equal(
            np.sort(schedule[schedule > 0]), np.arange(1, n_gu + 1)
        )
    ):
        raise ValueError("Each GU must appear once in the configured RB schedule")

    power = np.asarray(transmit_power_w, dtype=float)
    if power.ndim == 0:
        power = np.full(n_uav, power.item())
    if (
        power.shape != (n_uav,) or not np.all(np.isfinite(power))
        or np.any(power < cfg.security.min_uav_power_w)
        or np.any(power > cfg.security.max_uav_power_w)
    ):
        raise ValueError("Transmitter power must lie within SecurityConfig bounds")

    counts = np.count_nonzero(schedule, axis=1)
    power_per_rb = np.divide(
        power, counts, out=np.zeros_like(power), where=counts > 0
    )

    # Fixed equal coefficients match the A2G/A2A channels; no adaptive beamforming.
    weights = np.zeros((n_uav, n_rb, nt), dtype=complex)
    weights[schedule > 0] = 1.0 / np.sqrt(nt)
    gain_gu = np.abs(hg.sum(axis=-1)) ** 2 / nt
    gain_eav = np.abs(he.sum(axis=-1)) ** 2 / nt

    bandwidth = cfg.a2g.resource_block_bandwidth_hz
    noise_gu = cfg.a2g.noise_power_w
    noise_eav = cfg.a2a.rb_noise_power_w
    # The schedule determines interference and bandwidth, but the listener
    # is selected once per transmitting UAV after accumulating every active RB.
    uav, rb = np.nonzero(schedule)
    gu = schedule[uav, rb] - 1
    serving_uav = np.empty(n_gu, dtype=int)
    serving_uav[gu] = uav
    rb_of_gu = np.empty(n_gu, dtype=int)
    rb_of_gu[gu] = rb
    active = schedule > 0
    received_gu = power_per_rb[:, None] * gain_gu * active[:, rb_of_gu]
    desired = received_gu[serving_uav, np.arange(n_gu)]
    interference = np.maximum(received_gu.sum(axis=0) - desired, 0.0)
    legitimate_sinr = desired / (interference + noise_gu)
    gu_rate = bandwidth * np.log2(1.0 + legitimate_sinr)
    per_uav_gu = np.bincount(serving_uav, weights=gu_rate, minlength=n_uav)

    received = (power_per_rb[:, None] * gain_eav)[:, :, None] * active[:, None, :]
    interference = np.maximum(received.sum(axis=0, keepdims=True) - received, 0.0)
    sinr = received / (interference + noise_eav)
    rb_rates = bandwidth * np.log2(1.0 + sinr)
    listener_rates = rb_rates.sum(axis=2)
    selected = np.full(n_uav, -1, dtype=int)
    leakage = np.zeros(n_uav)
    stream_leakage = np.zeros(n_gu)
    if n_eav:
        selected = listener_rates.argmax(axis=1)
        leakage = listener_rates[np.arange(n_uav), selected]
        stream_leakage[gu] = rb_rates[uav, selected[uav], rb]

    return SecurityMetrics(
        legitimate_sinr=legitimate_sinr,
        eav_sinr=sinr[serving_uav, :, rb_of_gu],
        gu_rate_bps=gu_rate,
        worst_eav_rate_bps=stream_leakage,
        secrecy_rate_bps=np.maximum(gu_rate - stream_leakage, 0.0),
        per_uav_secrecy_rate_bps=np.maximum(per_uav_gu - leakage, 0.0),
        transmit_weights=weights,
        per_uav_gu_rate_bps=per_uav_gu,
        per_uav_eav_rate_bps=listener_rates,
        worst_eav_index=selected,
        per_uav_worst_eav_rate_bps=leakage,
    )
