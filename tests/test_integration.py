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