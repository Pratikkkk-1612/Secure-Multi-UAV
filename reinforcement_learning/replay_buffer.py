"""Replay buffer for RF-only CTDE-MASAC in the multi-UAV SAGIN simulator.

Stores one joint transition per environment step:
    global state (358), four local observations (4 x 40),
    four continuous actions (4 x 3), and four rewards (4).

The dimensions are configurable, but the defaults match the current simulator.
Agent ordering is fixed by ``agent_names`` for every stored transition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int = 358,
        agent_names: Sequence[str] = ("uav_0", "uav_1", "uav_2", "uav_3"),
        device: str | torch.device = "cpu",
        seed: int | None = None,
        action_dim: int = 3,
        local_obs_dim: int = 40,
    ) -> None:
        if any(not isinstance(x, int) or x <= 0 for x in
               (capacity, state_dim, action_dim, local_obs_dim)):
            raise ValueError("capacity and all dimensions must be positive integers")

        self.agent_names = tuple(agent_names)
        if not self.agent_names or len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("agent_names must be nonempty and unique")

        self.capacity = capacity
        self.state_dim = state_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.n_agents = len(self.agent_names)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)

        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.observations = np.empty(
            (capacity, self.n_agents, local_obs_dim), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(
            (capacity, self.n_agents, action_dim), dtype=np.float32
        )
        self.rewards = np.empty((capacity, self.n_agents), dtype=np.float32)
        self.terminations = np.empty_like(self.rewards)
        self.truncations = np.empty_like(self.rewards)
        self.dones = np.empty_like(self.rewards)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def can_sample(self, batch_size: int) -> bool:
        return isinstance(batch_size, int) and batch_size > 0 and self._size >= batch_size

    def _vector(self, values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values with shape {shape}")
        return array

    def _agents(self, values: Mapping[str, object], name: str) -> None:
        if set(values) != set(self.agent_names):
            raise ValueError(f"{name} must contain exactly {self.agent_names}")

    def add(
        self,
        state: np.ndarray,
        observations: Mapping[str, np.ndarray],
        actions: Mapping[str, np.ndarray],
        rewards: Mapping[str, float],
        next_state: np.ndarray,
        next_observations: Mapping[str, np.ndarray],
        terminations: Mapping[str, bool],
        truncations: Mapping[str, bool],
    ) -> None:
        """Store one PettingZoo ParallelEnv transition (before the next reset)."""
        state_array = self._vector(state, (self.state_dim,), "state")
        next_state_array = self._vector(next_state, (self.state_dim,), "next_state")
        for name, values in (
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
            ("next_observations", next_observations),
            ("terminations", terminations),
            ("truncations", truncations),
        ):
            self._agents(values, name)

        obs_array = np.stack([
            self._vector(observations[a], (self.local_obs_dim,), f"observations[{a}]")
            for a in self.agent_names
        ])
        next_obs_array = np.stack([
            self._vector(next_observations[a], (self.local_obs_dim,), f"next_observations[{a}]")
            for a in self.agent_names
        ])
        action_array = np.stack([
            self._vector(actions[a], (self.action_dim,), f"actions[{a}]")
            for a in self.agent_names
        ])
        reward_array = self._vector(
            [rewards[a] for a in self.agent_names], (self.n_agents,), "rewards"
        )
        termination_array = np.asarray(
            [terminations[a] for a in self.agent_names], dtype=np.float32
        )
        truncation_array = np.asarray(
            [truncations[a] for a in self.agent_names], dtype=np.float32
        )
        for name, values in (("terminations", termination_array), ("truncations", truncation_array)):
            if not np.all((values == 0) | (values == 1)):
                raise ValueError(f"{name} must consist of boolean flags")

        idx = self._ptr
        self.states[idx] = state_array
        self.observations[idx] = obs_array
        self.actions[idx] = action_array
        self.rewards[idx] = reward_array
        self.next_states[idx] = next_state_array
        self.next_observations[idx] = next_obs_array
        self.terminations[idx] = termination_array
        self.truncations[idx] = truncation_array
        self.dones[idx] = np.maximum(termination_array, truncation_array)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a random batch of float32 tensors on the requested device."""
        if not self.can_sample(batch_size):
            raise ValueError(f"Cannot sample {batch_size} transitions (have {self._size})")

        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        result = {
            "states": self.states[indices],
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "next_observations": self.next_observations[indices],
            "terminations": self.terminations[indices],
            "truncations": self.truncations[indices],
            "dones": self.dones[indices],
        }
        batch = {key: torch.as_tensor(array, device=self.device)
                 for key, array in result.items()}
        batch["joint_actions"] = batch["actions"].reshape(batch_size, -1)
        return batch


def combine_dones(
    terminations: Mapping[str, bool], truncations: Mapping[str, bool]
) -> dict[str, bool]:
    """Utility retained from the reference file."""
    if set(terminations) != set(truncations):
        raise ValueError("terminations and truncations must have matching agents")
    return {name: bool(terminations[name] or truncations[name]) for name in terminations}
