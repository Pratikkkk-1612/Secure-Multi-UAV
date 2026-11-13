"""Focused tests for the continuous-action Secure Multi-UAV SAGIN environment."""

from dataclasses import replace
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


def test_partial_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        SAGINMultiUAVEnv(reward_mode="abs")


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
        assert all(finished == (step == env.max_steps - 1) for finished in terminations.values())
        assert not any(truncations.values())


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

def test_uav_level_secrecy_drives_environment_reward():
    env = SAGINMultiUAVEnv()
    env.reset(seed=42)
    _, rewards, _, _, infos = env.step(_actions(env, power_action=-0.5))
    security = env.network.last_metrics.security
    worst = security.per_uav_eav_rate_bps.max(axis=1)
    expected = np.maximum(security.per_uav_gu_rate_bps - worst, 0.0)
    np.testing.assert_allclose(env.secrecy_rate_per_uav_bps, expected)
    np.testing.assert_allclose(env.latest_metrics["eavesdropper_sum_rate_bps"], worst.sum())
    for i, agent in enumerate(env.possible_agents):
        assert infos[agent]["own_secrecy_rate_bps"] == pytest.approx(expected[i])
        power_cost = env.weights.power * env.uav_power_w[i] / env.sec.max_uav_power_w
        assert rewards[agent] == pytest.approx(expected.sum() / env.rate_scale_bps - power_cost)
    assert env.latest_metrics["mean_secrecy_rate_bps_per_gu"] == pytest.approx(expected.sum()/env.n_gu)


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