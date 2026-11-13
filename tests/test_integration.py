import numpy as np
from environment import SAGINMultiUAVEnv


def test_integration():
    env = SAGINMultiUAVEnv()
    observations, _ = env.reset(seed=42)

    assert env.state().shape == (358,)
    assert env._channel_features.shape == (12,)
    assert np.all(np.isfinite(env.state()))

    assert all(
        obs.shape == (40,)
        for obs in observations.values()
    )

    actions = {
        agent: np.zeros(3, dtype=np.float32)
        for agent in env.agents
    }

    env.step(actions)

    assert env.state().shape == (358,)
    assert np.all(np.isfinite(env.state()))

    env.close()

def test_network_and_environment_replay_same_seed_and_commands():
    from network_model import NetworkModel

    env = SAGINMultiUAVEnv(spawn_layout="spread", eav_policy="pursuit")
    env.reset(seed=42)
    model = NetworkModel(seed=42, spawn_layout="spread", eav_policy="pursuit")
    actions = np.random.default_rng(3).uniform(-1, 1, (4, 3))
    for _ in range(3):
        result = model.step(actions)
        env.step(dict(zip(env.possible_agents, actions)))
        np.testing.assert_array_equal(env.scheduled_gu, result.schedule)
        np.testing.assert_allclose(env.eav_pos, model.eav_pos)
        np.testing.assert_allclose(env.uav_power_w, result.power_w)
        np.testing.assert_allclose(env.secrecy_rate_per_uav_bps, result.per_uav_secrecy_bps)
