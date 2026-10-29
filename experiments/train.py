
"""Short end-to-end MASAC training smoke test."""

from dataclasses import replace

import numpy as np

from config import DEFAULT_CONFIG
from environment import SAGINMultiUAVEnv
from reinforcement_learning.masac import MASAC
from reinforcement_learning.replay_buffer import ReplayBuffer


def main():
    rng = np.random.default_rng(42)

    # Short episodes for integration testing only
    network = replace(
        DEFAULT_CONFIG.network,
        episode_duration_s=6 * DEFAULT_CONFIG.network.dt_s,
    )
    config = replace(DEFAULT_CONFIG, network=network)

    env = SAGINMultiUAVEnv(config=config)
    masac = MASAC()

    buffer = ReplayBuffer(
        capacity=1000,
        agent_names=env.possible_agents, 
        seed=42,
    )

    episodes = 3
    batch_size = 8
    warmup_steps = 8
    total_steps = 0

    try:
        for episode in range(episodes):

            observations, _ = env.reset(seed=42 + episode)
            state = env.state().copy()

            episode_reward = 0.0
            updates = 0

            while env.agents:

                # Initial exploration
                if total_steps < warmup_steps:
                    actions = {
                        agent: rng.uniform(
                            -1.0, 1.0, size=3
                        ).astype(np.float32)
                        for agent in env.agents
                    }
                else:
                    actions = masac.select_actions(observations)

                # Execute UAV movements and RF power actions
                (
                    next_obs,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = env.step(actions)

                next_state = env.state().copy()

                print(
                    f"Step {env.current_step} | "
                    f"Power: {np.round(env.uav_power_w, 4)} W | "
                    f"GU rate: {env.latest_metrics['global_sum_rate_bps']:.2f} bps | "
                    f"EAV rate: {env.latest_metrics['eavesdropper_sum_rate_bps']:.2f} bps | "
                    f"Secrecy: {env.latest_metrics['secrecy_sum_rate_bps']:.2f} bps"
)


                # Preserve observations at episode end
                next_obs_complete = {}
                for agent in env.possible_agents:
                    if agent in next_obs:
                        next_obs_complete[agent] = next_obs[agent]
                    else:
                        next_obs_complete[agent] = (
                            infos[agent]["final_observation"]
                        )

                # Store real environment transition
                buffer.add(
                    state=state,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_state=next_state,
                    next_observations=next_obs_complete,
                    terminations=terminations,
                    truncations=truncations,
                )

                total_steps += 1
                episode_reward += np.mean(list(rewards.values()))

                # Train after collecting enough experiences
                if buffer.can_sample(batch_size):
                    metrics = masac.update(
                        buffer, batch_size=batch_size
                    )
                    updates += 1

                observations = next_obs_complete
                state = next_state

            secrecy_mbps = (
                env.latest_metrics["secrecy_sum_rate_bps"] / 1e6
            )

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {episode_reward:.4f} | "
                f"Final secrecy: {secrecy_mbps:.3f} Mbps | "
                f"Updates: {updates}"
            )

        print("\nMASAC training smoke test completed.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
