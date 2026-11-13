"""Fast integration checks for network_model.py and the channel modules.

Run from the project root:  python -m pytest -q network_model_test.py
"""

from dataclasses import replace

import numpy as np
import pytest

from config import DEFAULT_CONFIG
from network_model import NetworkModel


@pytest.fixture
def config():
    # Small, feasible network keeps channel sampling and beamforming tests fast.
    network = replace(
        DEFAULT_CONFIG.network,
        n_uav=3,
        n_gu=12,
        n_eav=2,
        n_resource_blocks_per_uav=6,
        min_gu_per_uav=1,
        max_gu_per_uav=6,
        episode_duration_s=0.3,
    )
    return replace(
    DEFAULT_CONFIG,
    network=network,
    a2g=replace(
        DEFAULT_CONFIG.a2g,
        n_resource_blocks=6
    ),
    a2a=replace(
        DEFAULT_CONFIG.a2a,
        n_resource_blocks=6
    ),
)


@pytest.fixture
def model(config):
    simulator = NetworkModel(config)
    simulator.reset(seed=42)
    return simulator


def test_reset_and_reproducibility(model):
    net = model.net
    first = model.reset(seed=7)
    positions = model.uav_pos.copy()
    second = model.reset(seed=7)

    assert model.uav_pos.shape == (net.n_uav, 3)
    assert model.gu_pos.shape == (net.n_gu, 3)
    assert model.eav_pos.shape == (net.n_eav, 3)
    assert np.array_equal(model.uav_pos, positions)
    assert first.total_rate_bps == pytest.approx(second.total_rate_bps)
    assert np.all(model.uav_pos[:, 2] == net.fixed_2d_altitude_m)
    assert np.all(model.gu_pos[:, 2] == net.gu_altitude_m)
    assert model.step_count == 0


def test_user_association_and_scheduling(model):
    metrics = model.last_metrics
    net = model.net
    counts = np.bincount(metrics.association, minlength=net.n_uav)
    scheduled = metrics.schedule[metrics.schedule > 0]

    assert metrics.association.shape == (net.n_gu,)
    assert metrics.schedule.shape == (net.n_uav, net.n_resource_blocks_per_uav)
    assert np.all((counts >= net.min_gu_per_uav) & (counts <= net.max_gu_per_uav))
    assert np.array_equal(np.sort(scheduled), np.arange(1, net.n_gu + 1))
    assert np.array_equal(np.count_nonzero(metrics.schedule, axis=1), counts)


def test_mobility_and_power_actions(model):
    model.enforce_average_power_budget = False  # isolate the peak-power mapping
    before = model.uav_pos.copy()
    actions = np.zeros((model.net.n_uav, model.config.action_dimensions_per_uav))
    actions[:, 0] = 1.0
    actions[:, -1] = [-1.0, 0.0, 1.0]
    result = model.step(actions)

    distance = np.linalg.norm(model.uav_pos - before, axis=1)
    assert np.all(distance <= model.net.max_uav_step_m + 1e-9)
    assert np.all(model.uav_pos[:, 2] == model.net.fixed_2d_altitude_m)
    low, high = model.sec.min_uav_power_w, model.sec.max_uav_power_w
    assert np.allclose(result.power_w, [low, (low + high) / 2, high])
    assert model.step_count == 1


def test_boundary_clipping(model):
    model.uav_pos[0, 0] = model.net.area_x_m - 0.1
    actions = np.zeros((model.net.n_uav, model.config.action_dimensions_per_uav))
    actions[0, 0] = 1.0
    model.step(actions)

    assert model.boundary_violations[0]
    assert model.uav_pos[0, 0] == pytest.approx(model.net.area_x_m)
    assert np.all((model.uav_pos[:, :2] >= 0.0))
    assert np.all(model.uav_pos[:, 0] <= model.net.area_x_m)
    assert np.all(model.uav_pos[:, 1] <= model.net.area_y_m)


def test_rates_and_secrecy(model):
    metrics = model.last_metrics
    for values in (metrics.gu_sinr, metrics.gu_rate_bps,
                   metrics.eav_rate_bps, metrics.secrecy_rate_bps):
        assert values.shape == (model.net.n_gu,)
        assert np.all(np.isfinite(values) & (values >= 0))

    assert metrics.total_rate_bps == pytest.approx(metrics.gu_rate_bps.sum())
    expected = np.maximum(metrics.security.per_uav_gu_rate_bps - metrics.security.per_uav_eav_rate_bps.max(axis=1), 0.0)
    np.testing.assert_allclose(metrics.per_uav_secrecy_bps, expected)
    assert metrics.total_secrecy_bps == pytest.approx(metrics.per_uav_secrecy_bps.sum())
    assert np.all(metrics.secrecy_rate_bps <= metrics.gu_rate_bps + 1e-9)
    assert metrics.a2g_state is not None
    assert metrics.a2a_state is not None


def test_action_validation_and_episode_end(model):
    shape = (model.net.n_uav, model.config.action_dimensions_per_uav)
    with pytest.raises(ValueError):
        model.step(np.zeros((model.net.n_uav, 1)))
    with pytest.raises(ValueError):
        model.step(np.full(shape, np.nan))

    for _ in range(model.net.n_steps):
        model.step(np.zeros(shape))
    with pytest.raises(RuntimeError):
        model.step(np.zeros(shape))


def test_3d_mobility(config):
    network = replace(
        config.network,
        mobility_mode="3d",
        uav_3d_altitude_limits_m=(200.0, 600.0),
        eav_3d_altitude_limits_m=(200.0, 600.0),
    )
    model = NetworkModel(replace(config, network=network))
    model.reset(seed=42)
    before = model.uav_pos.copy()
    actions = np.zeros((network.n_uav, model.config.action_dimensions_per_uav))
    actions[:, 2] = 1.0
    model.step(actions)

    assert actions.shape[1] == 4  # dx, dy, dz, power
    assert np.all(model.uav_pos[:, 2] > before[:, 2])
    assert np.all(model.uav_pos[:, 2] <= network.uav_3d_altitude_limits_m[1])
    assert np.all(np.linalg.norm(model.uav_pos - before, axis=1)
                  <= network.max_uav_step_m + 1e-9)


def test_no_eavesdroppers(config):
    network = replace(config.network, n_eav=0)
    model = NetworkModel(replace(config, network=network))
    metrics = model.reset(seed=42)

    assert model.eav_pos.shape == (0, 3)
    assert metrics.a2a_state is None
    assert np.all(metrics.eav_rate_bps == 0)
    assert np.allclose(metrics.secrecy_rate_bps, metrics.gu_rate_bps)
