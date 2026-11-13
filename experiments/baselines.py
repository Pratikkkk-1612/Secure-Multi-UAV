"""Reference policies using the same normalized action commands as MASAC."""

import numpy as np


def stationary_actions(env):
    low, high = env.sec.min_uav_power_w, env.sec.max_uav_power_w
    power = env.sec.average_uav_power_budget_w
    command = 2 * (power - low) / (high - low) - 1 if high > low else 0.0
    return {
        agent: np.array([0.0, 0.0, command], dtype=np.float32)
        for agent in env.agents
    }


def random_actions(env, rng):
    return {
        agent: rng.uniform(-1, 1, env.action_space(agent).shape).astype(np.float32)
        for agent in env.agents
    }
