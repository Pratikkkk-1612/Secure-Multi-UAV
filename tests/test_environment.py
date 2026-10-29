"""Focused tests for the continuous-action Secure Multi-UAV SAGIN environment."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from config import DEFAULT_CONFIG
from environment import SAGINMultiUAVEnv


def _actions(env, power_action=-1.0):
    """Stationary actions, with a normalized power command for every UAV."""
    actions = {}
    for agent in env.agents:
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[-1] = power_action
        actions[agent] = action
    return actions


def test_reset_is_reproducible_and_observations_are_valid():
    env = SAGINMultiUAVEnv()
    first, _ = env.reset(seed=42)
    first_positions = env.uav_pos.copy()
    second, _ = env.reset(seed=42)

    np.testing.assert_array_equal(env.uav_pos, first_positions)
    assert set(second) == set(env.possible_agents)
    for agent in env.possible_agents:
        np.testing.assert_array_equal(first[agent], second[agent])
        assert env.observation_space(agent).contains(second[agent])
    assert np.all(np.isfinite(env.state()))
    env.close()


@pytest.mark.parametrize("mode", ["2d"])
def test_continuous_movement_and_power_bounds(mode):
    config = replace(DEFAULT_CONFIG, network=replace(DEFAULT_CONFIG.network, mobility_mode=mode))
    env = SAGINMultiUAVEnv(config=config, spawn_layout="spread")
    env.reset(seed=1)
    before = env.uav_pos.copy()

    actions = _actions(env, power_action=-1.0)
    assert env.action_dim == (3 if mode == "2d" else 4)
    actions["uav_0"][0] = 1.0
    if mode == "3d":
        actions["uav_1"][2] = 1.0
    env.step(actions)

    displacement = env.uav_pos - before
    assert np.all(np.linalg.norm(displacement, axis=1) <= config.network.max_uav_step_m + 1e-9)
    np.testing.assert_allclose(displacement[0], [config.network.max_uav_step_m, 0, 0], atol=1e-9)
    if mode == "2d":
        np.testing.assert_allclose(env.uav_pos[:, 2], config.network.fixed_2d_altitude_m)
    else:
        assert displacement[1, 2] == pytest.approx(config.network.max_uav_step_m)
    np.testing.assert_allclose(env.uav_power_w, config.security.min_uav_power_w)
    env.close()


def test_user_scheduling_and_step_metrics():
    env = SAGINMultiUAVEnv()
    env.reset(seed=3)
    actions = _actions(env, power_action=0.0)
    observations, rewards, terminations, truncations, infos = env.step(actions)

    allocated = env.scheduled_gu[env.scheduled_gu > 0]
    np.testing.assert_array_equal(np.sort(allocated), np.arange(1, env.n_gu + 1))
    counts = np.bincount(env.assigned_uav, minlength=env.n_uav)
    assert np.all((counts >= env.net.min_gu_per_uav) & (counts <= env.net.max_gu_per_uav))

    for agent in env.possible_agents:
        assert env.observation_space(agent).contains(observations[agent])
        assert np.isfinite(rewards[agent])
        assert not terminations[agent] and not truncations[agent]
        assert np.isfinite(infos[agent]["secrecy_sum_rate_bps"])
        assert 0.0 <= infos[agent]["coverage_fraction"] <= 1.0
    env.close()


def test_episode_end_and_average_power_budget():
    # A three-step episode keeps this test fast while covering termination logic.
    network = replace(DEFAULT_CONFIG.network, episode_duration_s=3 * DEFAULT_CONFIG.network.dt_s)
    env = SAGINMultiUAVEnv(config=replace(DEFAULT_CONFIG, network=network))
    env.reset(seed=5)

    for step in range(env.max_steps):
        observations, rewards, terminations, truncations, infos = env.step(
            _actions(env, power_action=1.0)
        )
        assert all(not finished for finished in terminations.values())
        assert all(finished == (step == env.max_steps - 1) for finished in truncations.values())


    assert env.agents == []
    assert set(observations) == set(env.possible_agents)

    for observation in observations.values():
        assert observation.shape == (40,)
        assert np.all(np.isfinite(observation))
        
    for agent in env.possible_agents:
        assert infos[agent]["own_mean_power_w"] <= env.sec.average_uav_power_budget_w + 1e-9
    with pytest.raises(RuntimeError):
        env.step({})
    env.close()


def test_invalid_actions_are_rejected():
    env = SAGINMultiUAVEnv()
    env.reset(seed=7)
    with pytest.raises(ValueError):
        env.step({})

    bad = _actions(env)
    bad["uav_0"][0] = np.nan
    with pytest.raises(ValueError):
        env.step(bad)
    env.close()

def test_eav_interception_rate():
    env = SAGINMultiUAVEnv()
    env.reset(seed=42)

    gains = np.array([
        [4e-10, 1e-10, 2e-10, 1e-10],
        [2e-10, 3e-10, 1e-10, 2e-10],
        [1e-10, 2e-10, 3e-10, 1e-10],
        [2e-10, 1e-10, 2e-10, 3e-10],
    ])

    w = np.ones(2) / np.sqrt(2)

    h_eav = np.sqrt(gains)[..., None] * w

    fake_state = SimpleNamespace(
        uav_eav=SimpleNamespace(h=h_eav)
    )
    h = fake_state.uav_eav.h
    fake_state.uav_eav.gain = (
        np.abs(np.sum(h, axis=-1)) ** 2 / h.shape[-1]
    )
    actual = env._eavesdropper_rates(fake_state)

    counts = np.bincount(
        env.assigned_uav,
        minlength=env.n_uav,
    )

    stream_power = env.uav_power_w / counts

    rb = 0
    u = 0

    active = np.flatnonzero(
        env.scheduled_gu[:, rb] > 0
    )

    desired = stream_power[u] * gains[u]

    interference = (
        np.sum(
            stream_power[active, None] * gains[active],
            axis=0,
        )
        - desired
    )

    noise = (
        env.config.a2a.noise_power_w
        * env.rb_bandwidth_hz
        / env.config.a2a.total_bandwidth_hz
    )

    eav_sinr = desired / (interference + noise)

    expected_rate = np.max(
        env.rb_bandwidth_hz * np.log2(1 + eav_sinr)
    )

    gu = env.scheduled_gu[u, rb] - 1

    np.testing.assert_allclose(
        actual[gu], expected_rate
    )

    env.close()

def test_secrecy_rate_calculation(monkeypatch):
    env = SAGINMultiUAVEnv()
    env.reset(seed=42)

    bandwidth = env.rb_bandwidth_hz

    # Every legitimate GU has SINR = 3.
    gu_sinr = np.full(env.n_gu, 3.0)

    # First 10 users have stronger EAV links.
    eav_rates = np.full(env.n_gu, bandwidth)
    eav_rates[:10] = 3.0 * bandwidth

    monkeypatch.setattr(
        "environment.a2g_sinr",
        lambda *args, **kwargs: gu_sinr,
    )

    monkeypatch.setattr(
        env,
        "_eavesdropper_rates",
        lambda *args, **kwargs: eav_rates,
    )

    metrics = env._calculate_metrics()

    gu_rates = bandwidth * np.log2(1 + gu_sinr)

    expected = np.maximum(
        gu_rates - eav_rates,
        0.0,
    )

    np.testing.assert_allclose(
        metrics["secrecy_sum_rate_bps"],
        np.sum(expected),
    )

    expected_per_uav = np.bincount(
        env.assigned_uav,
        weights=expected,
        minlength=env.n_uav,
    )

    np.testing.assert_allclose(
        env.secrecy_rate_per_uav_bps,
        expected_per_uav,
    )

    assert np.all(expected[:10] == 0)

    env.close()

def test_uav_power_budget():
    env = SAGINMultiUAVEnv(
        enforce_average_power_budget=True
    )

    env.reset(seed=42)

    energy_used = np.zeros(env.n_uav)

    while env.agents:
        actions = {
            agent: np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for agent in env.agents
        }

        env.step(actions)

        assert np.all(env.uav_power_w >= 0)

        assert np.all(
            env.uav_power_w
            <= env.sec.max_uav_power_w + 1e-10
        )

        energy_used += env.uav_power_w * env.dt_s

    max_energy = (
        env.sec.average_uav_power_budget_w
        * env.max_steps
        * env.dt_s
    )

    assert np.all(
        energy_used <= max_energy + 1e-10
    )

    env.close()

def test_3d_mode_is_rejected():
    config = replace(
        DEFAULT_CONFIG,
        network=replace(
            DEFAULT_CONFIG.network,
            mobility_mode="3d",
        ),
    )

    with pytest.raises(ValueError, match="2D mobility"):SAGINMultiUAVEnv(config=config)