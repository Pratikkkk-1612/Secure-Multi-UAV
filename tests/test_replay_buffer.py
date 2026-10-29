"""Tests for RF-only CTDE-MASAC replay_buffer.py.

Place this file at tests/test_replay_buffer.py and run:
    python -m pytest tests/test_replay_buffer.py -v
"""

import numpy as np
import pytest
import torch

from reinforcement_learning.replay_buffer import ReplayBuffer, combine_dones


AGENTS = ("uav_0", "uav_1", "uav_2", "uav_3")


def transition(value=1.0):
    """One valid four-UAV transition, with identifiable data per agent."""
    return dict(
        state=np.full(358, value, dtype=np.float32),
        observations={
            agent: np.full(40, value + i, dtype=np.float32)
            for i, agent in reversed(list(enumerate(AGENTS)))
        },
        actions={
            agent: np.full(3, i / 10.0, dtype=np.float32)
            for i, agent in reversed(list(enumerate(AGENTS)))
        },
        rewards={agent: value + i for i, agent in enumerate(AGENTS)},
        next_state=np.full(358, value + 0.5, dtype=np.float32),
        next_observations={
            agent: np.full(40, value + i + 0.5, dtype=np.float32)
            for i, agent in enumerate(AGENTS)
        },
        terminations={agent: False for agent in AGENTS},
        truncations={agent: False for agent in AGENTS},
    )


def test_buffer_starts_empty():
    buffer = ReplayBuffer(capacity=10, seed=7)
    assert len(buffer) == 0
    assert not buffer.can_sample(1)
    assert not buffer.can_sample(0)
    with pytest.raises(ValueError):
        buffer.sample(1)


def test_store_and_sample_shapes_and_agent_order():
    buffer = ReplayBuffer(capacity=10, seed=7)
    buffer.add(**transition())
    batch = buffer.sample(1)

    expected_shapes = {
        "states": (1, 358),
        "observations": (1, 4, 40),
        "actions": (1, 4, 3),
        "joint_actions": (1, 12),
        "rewards": (1, 4),
        "next_states": (1, 358),
        "next_observations": (1, 4, 40),
        "terminations": (1, 4),
        "truncations": (1, 4),
        "dones": (1, 4),
    }
    assert set(batch) == set(expected_shapes)
    for name, shape in expected_shapes.items():
        assert batch[name].shape == shape
        assert batch[name].dtype == torch.float32
        assert batch[name].device.type == "cpu"

    # Insertion order of the input dictionaries must NOT change UAV order.
    np.testing.assert_allclose(batch["observations"][0, :, 0].numpy(), [1, 2, 3, 4])
    np.testing.assert_allclose(batch["actions"][0, :, 0].numpy(), [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(batch["rewards"][0].numpy(), [1, 2, 3, 4])
    np.testing.assert_allclose(batch["joint_actions"][0].numpy(),
                               [0] * 3 + [0.1] * 3 + [0.2] * 3 + [0.3] * 3)
    assert torch.all(batch["next_states"] == 1.5)


def test_termination_and_truncation_are_combined():
    buffer = ReplayBuffer(capacity=2)
    record = transition()
    record["terminations"]["uav_0"] = True
    record["truncations"]["uav_1"] = True
    buffer.add(**record)
    batch = buffer.sample(1)
    np.testing.assert_array_equal(batch["terminations"][0].numpy(), [1, 0, 0, 0])
    np.testing.assert_array_equal(batch["truncations"][0].numpy(), [0, 1, 0, 0])
    np.testing.assert_array_equal(batch["dones"][0].numpy(), [1, 1, 0, 0])


def test_capacity_overwrites_oldest_transitions():
    buffer = ReplayBuffer(capacity=2, seed=1)
    for marker in (1.0, 2.0, 3.0):
        buffer.add(**transition(marker))
    assert len(buffer) == 2
    assert buffer.can_sample(2)
    values = sorted(buffer.sample(2)["states"][:, 0].tolist())
    assert values == [2.0, 3.0]
    with pytest.raises(ValueError):
        buffer.sample(3)


def test_rejects_wrong_shapes_and_nonfinite_values():
    buffer = ReplayBuffer(capacity=3)
    bad_state = transition()
    bad_state["state"] = np.zeros(858, dtype=np.float32)
    with pytest.raises(ValueError, match="state"):
        buffer.add(**bad_state)

    bad_obs = transition()
    bad_obs["observations"]["uav_0"] = np.zeros(39, dtype=np.float32)
    with pytest.raises(ValueError, match="observations"):
        buffer.add(**bad_obs)

    bad_action = transition()
    bad_action["actions"]["uav_1"] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="actions"):
        buffer.add(**bad_action)

    bad_reward = transition()
    bad_reward["rewards"]["uav_2"] = float("nan")
    with pytest.raises(ValueError, match="rewards"):
        buffer.add(**bad_reward)

    assert len(buffer) == 0  # Invalid transitions must not be stored.


def test_rejects_missing_agent_keys():
    buffer = ReplayBuffer(capacity=3)
    bad = transition()
    del bad["next_observations"]["uav_3"]
    with pytest.raises(ValueError, match="next_observations"):
        buffer.add(**bad)
    assert len(buffer) == 0


def test_sampling_is_reproducible_with_seed():
    buffer_a = ReplayBuffer(capacity=8, seed=19)
    buffer_b = ReplayBuffer(capacity=8, seed=19)
    for marker in range(1, 7):
        record = transition(float(marker))
        buffer_a.add(**record)
        buffer_b.add(**record)
    batch_a = buffer_a.sample(4)
    batch_b = buffer_b.sample(4)
    for name in batch_a:
        torch.testing.assert_close(batch_a[name], batch_b[name])


def test_combine_dones_helper():
    terminations = {agent: (agent == "uav_0") for agent in AGENTS}
    truncations = {agent: (agent == "uav_2") for agent in AGENTS}
    assert combine_dones(terminations, truncations) == {
        "uav_0": True, "uav_1": False, "uav_2": True, "uav_3": False,
    }
    with pytest.raises(ValueError):
        combine_dones(terminations, {"uav_0": False})


def test_real_environment_transition():
    """Integration check; runs when the project environment is installed."""
    environment = pytest.importorskip("environment")
    env = environment.SAGINMultiUAVEnv()
    try:
        observations, _ = env.reset(seed=42)
        state = env.state().copy()
        actions = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.agents
        }
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        next_state = env.state().copy()

        buffer = ReplayBuffer(capacity=4, agent_names=env.possible_agents, seed=42)
        buffer.add(state, observations, actions, rewards,
                   next_state, next_obs, terms, truncs)
        batch = buffer.sample(1)
        assert batch["states"].shape == (1, 358)
        assert batch["observations"].shape == (1, 4, 40)
        assert batch["actions"].shape == (1, 4, 3)
        assert batch["joint_actions"].shape == (1, 12)
    finally:
        env.close()
