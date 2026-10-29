"""Numerical and interface tests for the current A2G and A2A RF channels.

Model assumptions:
- Two transmit antennas use fixed equal complex weights (no adaptive beamforming).
- Each UAV divides its total transmit power equally among its scheduled RBs.
- A2G and scheduled EAV interception use matching RBs and per-RB noise.

A2S/FSO tests are deliberately not included: that channel has a separate interface
and should have its own test_a2s.py once finalized.
"""

from dataclasses import replace

import numpy as np
import pytest

from config import A2AConfig, A2GConfig
from channels.a2a_channel import (
    a2a_eav_sinr_rb,
    a2a_sinr,
    evaluate_a2a,
    sample_a2a,
)
from channels.a2g_channel import (
    a2g_sinr,
    evaluate_a2g,
    expected_a2g_gain,
    sample_a2g,
)


@pytest.fixture
def nodes():
    uav = np.array([[0., 0., 400.], [300., 400., 400.]])
    eav = np.array([[150., 0., 400.], [500., 200., 400.]])
    gu = np.array([[30., 0., 1.5], [200., 300., 1.5], [400., 0., 1.5]])
    uav_vel = np.array([[10., 0., 0.], [0., 0., 0.]])
    eav_vel = np.zeros_like(eav)
    return uav, eav, gu, uav_vel, eav_vel


@pytest.fixture
def a2g_case(nodes):
    uav, _, gu, _, _ = nodes
    cfg = A2GConfig(n_resource_blocks=2)
    # All three GUs scheduled exactly once; both UAVs are active on RB 0.
    schedule = np.array([[1, 2], [3, 0]], dtype=int)
    state = sample_a2g(uav, gu, config=cfg, rng=np.random.default_rng(7))
    return cfg, schedule, state


# ---------------------------------------------------------------------------
# A2A geometry, fading, and original full-band evaluation
# ---------------------------------------------------------------------------


def test_a2a_geometry_and_doppler(nodes):
    uav, eav, _, uav_vel, eav_vel = nodes
    cfg = A2AConfig()
    state = sample_a2a(
        uav, eav, uav_vel, eav_vel, 0., config=cfg,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_allclose(state.uav_uav.distance_m[0, 1], 500.)
    np.testing.assert_allclose(state.uav_uav.radial_velocity_mps[0, 1], -6.)
    np.testing.assert_allclose(state.uav_uav.doppler_hz[0, 1], -6. / cfg.wavelength_m)
    assert state.uav_eav.h.shape == (2, 2, cfg.n_tx_antennas)
    assert np.isnan(np.diag(state.uav_uav.gain)).all()


def test_a2a_fading_gain_and_reproducibility(nodes):
    uav, eav, _, uv, ev = nodes
    args = (uav, eav, uv, ev, 0.2)
    first = sample_a2a(*args, rng=np.random.default_rng(12))
    second = sample_a2a(*args, rng=np.random.default_rng(12))

    for a, b in (
        (first.uav_uav, second.uav_uav),
        (first.uav_eav, second.uav_eav),
    ):
        np.testing.assert_allclose(a.h, b.h, equal_nan=True)
        # Fixed equal weights: |sum(h)|^2 / Nt, NOT sum(|h|^2).
        expected_gain = np.abs(np.sum(a.h, axis=-1)) ** 2 / a.h.shape[-1]
        np.testing.assert_allclose(a.gain, expected_gain, equal_nan=True)
        expected_interference = (
            np.nansum(a.gain, axis=0, keepdims=True)
            - np.nan_to_num(a.gain, nan=0.)
        )
        valid = np.isfinite(a.gain)
        np.testing.assert_allclose(
            a.interference_gain[valid], expected_interference[valid]
        )
        valid_los = np.isfinite(a.los_probability)
        assert np.all((a.los_probability[valid_los] >= 0)
                      & (a.los_probability[valid_los] <= 1))


def test_a2a_nlos_diffraction_increases_loss(nodes):
    uav, eav, _, uv, ev = nodes
    # Artificial high-obstacle scenario to exercise the NLoS code path.
    cfg = replace(A2AConfig(), obstacle_height_scale_m=1000.,
                  obstacle_density_beta=1.)
    state = sample_a2a(
        uav, eav, uv, ev, 0., config=cfg, rng=np.random.default_rng(17)
    )
    link = state.uav_eav
    assert not np.any(link.is_los)
    fspl = 20 * np.log10(4 * np.pi * link.distance_m / cfg.wavelength_m)
    atmosphere = cfg.gas_attenuation_db_per_km + cfg.rain_attenuation_db_per_km
    assert np.all(link.path_loss_db > fspl + atmosphere * link.distance_m / 1000)


def test_a2a_full_band_sinr_with_individual_powers(nodes):
    """Legacy full-band formula; not the SINR to use for scheduled RL secrecy."""
    uav, eav, _, uv, ev = nodes
    uav = np.vstack((uav, [600., 400., 400.]))
    uv = np.vstack((uv, [0., 0., 0.]))
    cfg = A2AConfig()
    state = sample_a2a(
        uav, eav, uv, ev, 0., config=cfg, rng=np.random.default_rng(2)
    )
    powers = np.array([0.1, 0.4, 0.2])
    result = a2a_sinr(state, powers, config=cfg)

    for gain, actual in (
        (state.uav_uav.gain, result.uav_uav),
        (state.uav_eav.gain, result.uav_eav),
    ):
        received = powers[:, None] * gain
        interference = (
            np.nansum(received, axis=0, keepdims=True)
            - np.nan_to_num(received, nan=0.)
        )
        expected = received / (interference + cfg.noise_power_w)
        np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("power", [-0.1, [0.1], [0.1, np.nan]])
def test_a2a_rejects_invalid_full_band_power(nodes, power):
    uav, eav, _, uv, ev = nodes
    state = sample_a2a(uav, eav, uv, ev, 0., rng=np.random.default_rng(3))
    with pytest.raises(ValueError):
        a2a_sinr(state, power)


def test_a2a_monte_carlo_metrics(nodes):
    uav, eav, _, uv, ev = nodes
    result = evaluate_a2a(
        uav, eav, uv, ev, 0., n_realizations=3,
        rng=np.random.default_rng(4),
    )
    for outage, rate in (
        (result.outage_uav_uav, result.mean_rate_uav_uav_mbps),
        (result.outage_uav_eav, result.mean_rate_uav_eav_mbps),
    ):
        assert outage.shape == result.power_dbm.shape
        assert np.all((outage >= 0) & (outage <= 1))
        assert np.all(np.diff(outage) <= 1e-12)
        assert np.all(np.diff(rate) >= -1e-12)


# ---------------------------------------------------------------------------
# A2G geometry, fading, RB scheduling, and SINR
# ---------------------------------------------------------------------------


def test_a2g_geometry_and_path_loss(a2g_case):
    cfg, _, state = a2g_case
    fspl = 20 * np.log10(4 * np.pi * state.distance_m / cfg.wavelength_m)
    atmosphere = cfg.gas_attenuation_db_per_km + cfg.rain_attenuation_db_per_km
    expected_loss = (
        fspl + atmosphere * state.distance_m / 1000
        + np.where(state.is_los, cfg.los_excess_loss_db,
                   cfg.nlos_excess_loss_db)
    )
    np.testing.assert_allclose(state.path_loss_db, expected_loss)
    np.testing.assert_allclose(
        state.large_scale_gain,
        10 ** ((cfg.tx_antenna_gain_dbi + cfg.gu_rx_antenna_gain_dbi
                - state.path_loss_db) / 10)
        * np.exp(-cfg.blockage_beta_per_m * state.distance_m),
    )
    assert state.h.shape == (2, 3, cfg.n_tx_antennas)
    assert np.all((state.los_probability >= 0) & (state.los_probability <= 1))
    assert np.all((state.elevation_deg > 0) & (state.elevation_deg <= 90))


def test_a2g_seed_and_fixed_weight_nakagami_mean(nodes):
    uav, _, gu, _, _ = nodes
    cfg = A2GConfig()
    first = sample_a2g(uav, gu, config=cfg, rng=np.random.default_rng(5))
    second = sample_a2g(uav, gu, config=cfg, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(first.h, second.h)
    np.testing.assert_allclose(
        first.gain,
        np.abs(np.sum(first.h, axis=-1)) ** 2 / cfg.n_tx_antennas,
    )

    # Expected channel gain / large-scale gain is approximately 1, not Nt.
    many_gu = np.repeat(gu[:1], 2000, axis=0)
    sample = sample_a2g(
        uav[:1], many_gu, config=cfg, rng=np.random.default_rng(8)
    )
    assert np.mean(sample.gain / sample.large_scale_gain) == pytest.approx(
        1.0, abs=0.12
    )


def test_a2g_fixed_weight_gain(a2g_case):
    cfg, _, state = a2g_case
    expected_gain = np.abs(np.sum(state.h, axis=-1)) ** 2 / cfg.n_tx_antennas
    np.testing.assert_allclose(state.gain, expected_gain)


def test_a2g_sinr_matches_scheduled_rb_equations(a2g_case):
    cfg, schedule, state = a2g_case
    # Use a known effective channel-gain matrix to test interference exactly.
    gain = np.array([[2., 3., 4.], [5., 6., 7.]])
    powers = np.array([0.2, 0.1])
    actual = a2g_sinr(
        replace(state, gain=gain), schedule, powers, config=cfg
    )
    n = cfg.noise_power_w
    expected = np.array([
        0.2 / (0.5 + n),   # GU 1 on RB 0: UAV 1 interferes.
        0.3 / n,           # GU 2 on RB 1: no other UAV active.
        0.7 / (0.4 + n),   # GU 3 on RB 0: UAV 0 interferes.
    ])
    np.testing.assert_allclose(actual, expected)


def test_a2g_zero_power_produces_zero_sinr(a2g_case):
    cfg, schedule, state = a2g_case
    np.testing.assert_array_equal(
        a2g_sinr(state, schedule, [0., 0.], config=cfg),
        np.zeros(state.gain.shape[1]),
    )


@pytest.mark.parametrize("schedule", [
    [[1, 2], [1, 0]],  # GU 1 duplicated; GU 3 missing.
    [[0, 2], [3, 0]],  # GU 1 missing.
    [[1, 2], [4, 0]],  # Out-of-range GU index.
])
def test_a2g_rejects_invalid_schedule(a2g_case, schedule):
    cfg, _, state = a2g_case
    with pytest.raises(ValueError):
        a2g_sinr(state, np.asarray(schedule, dtype=int), [0.1, 0.2], config=cfg)


@pytest.mark.parametrize("power", [-0.1, [0.1], [0.1, np.nan]])
def test_a2g_rejects_invalid_power(a2g_case, power):
    cfg, schedule, state = a2g_case
    with pytest.raises(ValueError):
        a2g_sinr(state, schedule, power, config=cfg)


def test_a2g_monte_carlo_metrics(nodes, a2g_case):
    uav, _, gu, _, _ = nodes
    cfg, schedule, _ = a2g_case
    result = evaluate_a2g(
        uav, gu, schedule, config=cfg, n_realizations=3,
        rng=np.random.default_rng(6),
    )
    assert result.outage.shape == result.power_dbm.shape
    assert np.all((result.outage >= 0) & (result.outage <= 1))
    assert np.all(np.diff(result.outage) <= 1e-12)
    assert np.all(np.diff(result.mean_rate_mbps) >= -1e-12)


def test_expected_a2g_large_scale_gain():
    cfg = A2GConfig(blockage_beta_per_m=0.)
    uav = np.array([[0., 0., 400.]])
    gu = np.array([[300., 400., 1.5]])
    actual = expected_a2g_gain(uav, gu, config=cfg)

    distance = np.linalg.norm(uav[0] - gu[0])
    horizontal = np.linalg.norm(uav[0, :2] - gu[0, :2])
    elevation = np.degrees(np.arctan2(400. - 1.5, horizontal))
    p_los = 1. / (1. + cfg.urban_a
                  * np.exp(-cfg.urban_b * (elevation - cfg.urban_a)))
    fspl = 20 * np.log10(4 * np.pi * distance / cfg.wavelength_m)
    atmosphere = (cfg.gas_attenuation_db_per_km
                  + cfg.rain_attenuation_db_per_km) * distance / 1000
    common_gain = 10 ** ((cfg.tx_antenna_gain_dbi + cfg.gu_rx_antenna_gain_dbi
                          - fspl - atmosphere) / 10)
    g_los = common_gain * 10 ** (-cfg.los_excess_loss_db / 10)
    g_nlos = common_gain * 10 ** (-cfg.nlos_excess_loss_db / 10)
    expected = p_los * g_los + (1 - p_los) * g_nlos
    assert actual.shape == (1, 1)
    np.testing.assert_allclose(actual[0, 0], expected)
    np.testing.assert_array_equal(actual, expected_a2g_gain(uav, gu, config=cfg))


# ---------------------------------------------------------------------------
# The scheduled A2A EAV calculation used for A2G secrecy optimization
# ---------------------------------------------------------------------------


def test_a2a_eav_rb_sinr_matches_a2g_schedule(nodes):
    uav, eav, _, uv, ev = nodes
    cfg = A2AConfig(n_resource_blocks=2)
    state = sample_a2a(
        uav, eav, uv, ev, 0., config=cfg,
        rng=np.random.default_rng(19),
    )
    # Same schedule as the A2G test. UAV 1 transmits only on RB 0.
    schedule = np.array([[1, 2], [3, 0]], dtype=int)
    gain = np.array([[2., 3.], [5., 7.]])
    state = replace(state, uav_eav=replace(state.uav_eav, gain=gain))
    actual = a2a_eav_sinr_rb(
        state, schedule, [0.2, 0.1], config=cfg
    )
    n = cfg.rb_noise_power_w
    assert actual.shape == (2, 2, 2)
    expected = np.zeros((2, 2, 2))
    expected[0, :, 0] = [0.2 / (0.5 + n), 0.3 / (0.7 + n)]
    expected[1, :, 0] = [0.5 / (0.2 + n), 0.7 / (0.3 + n)]
    expected[0, :, 1] = [0.2 / n, 0.3 / n]
    np.testing.assert_allclose(actual, expected)


def test_a2a_eav_rb_noise_is_per_resource_block():
    cfg = A2AConfig(n_resource_blocks=100)
    # Under the same noise figure/margin, noise scales with bandwidth.
    np.testing.assert_allclose(
        cfg.rb_noise_power_w * cfg.n_resource_blocks,
        cfg.noise_power_w,
    )


@pytest.mark.parametrize("power", [-0.1, [0.1], [0.1, np.nan]])
def test_a2a_eav_rb_rejects_invalid_power(nodes, power):
    uav, eav, _, uv, ev = nodes
    cfg = A2AConfig(n_resource_blocks=2)
    state = sample_a2a(
        uav, eav, uv, ev, 0., config=cfg,
        rng=np.random.default_rng(21),
    )
    schedule = np.array([[1, 2], [3, 0]], dtype=int)
    with pytest.raises(ValueError):
        a2a_eav_sinr_rb(state, schedule, power, config=cfg)


def test_a2a_eav_rb_rejects_uav_without_scheduled_rb(nodes):
    uav, eav, _, uv, ev = nodes
    cfg = A2AConfig(n_resource_blocks=2)
    state = sample_a2a(
        uav, eav, uv, ev, 0., config=cfg,
        rng=np.random.default_rng(22),
    )
    schedule = np.array([[1, 2], [0, 0]], dtype=int)
    with pytest.raises(ValueError):
        a2a_eav_sinr_rb(state, schedule, [0.2, 0.1], config=cfg)

def test_a2a_rejects_duplicate_gu_schedule():
    cfg = A2AConfig(n_resource_blocks=2)

    uav = np.array([
        [0., 0., 400.],
        [300., 400., 400.]
    ])

    eav = np.array([[150., 0., 400.]])

    uv = np.zeros_like(uav)
    ev = np.zeros_like(eav)

    state = sample_a2a(
        uav, eav, uv, ev, 0.0,
        config=cfg,
        rng=np.random.default_rng(1)
    )

    # GU 1 appears twice — invalid scheduling.
    invalid_schedule = np.array([
        [1, 2],
        [1, 0]
    ])

    with pytest.raises(ValueError):
        a2a_eav_sinr_rb(
            state,
            invalid_schedule,
            np.array([0.1, 0.1]),
            config=cfg
        )
def test_a2g_a2a_rb_sinr_consistency():
    cfg_g = A2GConfig(n_resource_blocks=2)
    cfg_a = A2AConfig(n_resource_blocks=2)

    uav = np.array([
        [0., 0., 400.],
        [300., 400., 400.]
    ])

    gu = np.array([
        [30., 0., 1.5],
        [200., 300., 1.5],
        [400., 0., 1.5]
    ])

    eav = np.array([[150., 0., 400.]])

    schedule = np.array([
        [1, 2],
        [3, 0]
    ])

    power = np.array([0.4, 0.1])

    a2g = sample_a2g(
        uav, gu,
        config=cfg_g,
        rng=np.random.default_rng(10)
    )

    a2a = sample_a2a(
        uav, eav,
        np.zeros_like(uav),
        np.zeros_like(eav),
        0.0,
        config=cfg_a,
        rng=np.random.default_rng(11)
    )

    gu_sinr = a2g_sinr(
        a2g, schedule, power, config=cfg_g
    )

    eav_sinr = a2a_eav_sinr_rb(
        a2a, schedule, power, config=cfg_a
    )

    # Both channels must use identical noise per RB.
    np.testing.assert_allclose(
        cfg_g.noise_power_w,
        cfg_a.rb_noise_power_w
    )

    # UAV 0 uses 2 RBs; UAV 1 uses 1 RB.
    p0 = power[0] / 2
    p1 = power[1]

    g = a2g.gain
    e = a2a.uav_eav.gain

    ng = cfg_g.noise_power_w
    ne = cfg_a.rb_noise_power_w

    # Expected ground-user SINR.
    expected_gu = np.array([
        p0 * g[0, 0] / (p1 * g[1, 0] + ng),
        p0 * g[0, 1] / ng,
        p1 * g[1, 2] / (p0 * g[0, 2] + ng)
    ])

    np.testing.assert_allclose(
        gu_sinr, expected_gu
    )

    # Expected eavesdropper SINR per RB.
    expected_eav = np.zeros((2, 1, 2))

    expected_eav[0, 0, 0] = (
        p0 * e[0, 0] / (p1 * e[1, 0] + ne)
    )

    expected_eav[1, 0, 0] = (
        p1 * e[1, 0] / (p0 * e[0, 0] + ne)
    )

    expected_eav[0, 0, 1] = (
        p0 * e[0, 0] / ne
    )

    np.testing.assert_allclose(
        eav_sinr, expected_eav
    )
