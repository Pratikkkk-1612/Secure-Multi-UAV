from dataclasses import dataclass

import numpy as np

from config import DEFAULT_CONFIG, ProjectConfig


@dataclass(frozen=True)
class SecurityMetrics:
    legitimate_sinr: np.ndarray
    eav_sinr: np.ndarray
    gu_rate_bps: np.ndarray
    worst_eav_rate_bps: np.ndarray
    secrecy_rate_bps: np.ndarray
    per_uav_secrecy_rate_bps: np.ndarray
    transmit_weights: np.ndarray

    @property
    def total_secrecy_rate_bps(self) -> float:
        return float(self.secrecy_rate_bps.sum())

    @property
    def mean_secrecy_rate_bps(self) -> float:
        return float(self.secrecy_rate_bps.mean())


def evaluate_security(
    a2g_state,
    a2a_state,
    scheduled_gu: np.ndarray,
    transmit_power_w: float | np.ndarray,
    *,
    config: ProjectConfig | None = None,
) -> SecurityMetrics:
    cfg = config if config is not None else DEFAULT_CONFIG
    hg = np.asarray(a2g_state.h, dtype=complex)
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
    legitimate_sinr = np.zeros(n_gu)
    eav_sinr = np.zeros((n_gu, n_eav))
    serving_uav = np.empty(n_gu, dtype=int)

    for rb in range(n_rb):
        active = np.flatnonzero(schedule[:, rb])
        if active.size == 0:
            continue
        gu = schedule[active, rb] - 1
        serving_uav[gu] = active

        # Only simultaneous transmissions on this RB contribute interference.
        received_gu = power_per_rb[active, None] * gain_gu[np.ix_(active, gu)]
        desired = np.diag(received_gu)
        interference = np.maximum(received_gu.sum(axis=0) - desired, 0.0)
        legitimate_sinr[gu] = desired / (interference + noise_gu)

        if n_eav:
            received_eav = power_per_rb[active, None] * gain_eav[active]
            interference_eav = np.maximum(
                received_eav.sum(axis=0, keepdims=True) - received_eav, 0.0
            )
            eav_sinr[gu] = received_eav / (interference_eav + noise_eav)

    gu_rate = bandwidth * np.log2(1.0 + legitimate_sinr)
    worst_eav_rate = (
        bandwidth * np.log2(1.0 + eav_sinr.max(axis=1))
        if n_eav else np.zeros(n_gu)
    )
    secrecy_rate = np.maximum(gu_rate - worst_eav_rate, 0.0)
    per_uav_rate = np.bincount(
        serving_uav, weights=secrecy_rate, minlength=n_uav
    )
    return SecurityMetrics(
        legitimate_sinr, eav_sinr, gu_rate, worst_eav_rate,
        secrecy_rate, per_uav_rate, weights,
    )
