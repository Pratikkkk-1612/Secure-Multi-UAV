"""Train RF-only MASAC. Run from the project root with python -m experiments.train."""

import argparse
import csv
import sys
from dataclasses import asdict, replace
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from config import DEFAULT_CONFIG
from environment import SAGINMultiUAVEnv
from experiments.baselines import random_actions
from reinforcement_learning.masac import MASAC
from reinforcement_learning.replay_buffer import ReplayBuffer


def experiment_config(steps):
    network = DEFAULT_CONFIG.network
    return replace(DEFAULT_CONFIG, network=replace(network, episode_duration_s=steps * network.dt_s))


def add_training_arguments(parser):
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.network.n_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=50_000)


def train(args):
    if min(args.episodes, args.steps, args.batch_size) < 1 or args.capacity < args.batch_size:
        raise ValueError("Use positive episode, step and batch counts, with capacity >= batch size")
    if args.seed < 0 or args.warmup < 0:
        raise ValueError("Seed and warmup must be nonnegative")
    if args.episodes * args.steps < max(args.batch_size, args.warmup):
        raise ValueError("This run is too short for an update. Reduce --warmup and --batch-size.")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    config = experiment_config(args.steps)
    env = SAGINMultiUAVEnv(config=config)
    model_args = dict(n_agents=env.n_uav, state_dim=env.state_space.shape[0],
                      local_obs_dim=env.observation_space(env.possible_agents[0]).shape[0],
                      action_dim=env.action_dim)
    learner = MASAC(**model_args)
    buffer = ReplayBuffer(args.capacity, state_dim=model_args["state_dim"],
                          local_obs_dim=model_args["local_obs_dim"], action_dim=env.action_dim,
                          agent_names=env.possible_agents, seed=args.seed)
    total_steps = 0
    try:
        with (args.output / "training.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "seed", "return", "secrecy_mbps_per_gu",
                             "mean_power_w_per_uav", "updates"])
            for episode in range(args.episodes):
                seed = args.seed + episode
                obs, _ = env.reset(seed=seed)
                state = env.state()
                episode_return = secrecy = power = 0.0
                updates = 0
                while env.agents:
                    if total_steps < args.warmup:
                        actions = random_actions(env, rng)
                    else:
                        actions = learner.select_actions(obs)
                    next_obs, rewards, terms, truncs, _ = env.step(actions)
                    next_state = env.state()
                    buffer.add(state, obs, actions, rewards, next_state, next_obs, terms, truncs)
                    total_steps += 1
                    if total_steps >= args.warmup and buffer.can_sample(args.batch_size):
                        learner.update(buffer, batch_size=args.batch_size)
                        updates += 1
                    episode_return += np.mean(list(rewards.values()))
                    secrecy += env.latest_metrics["mean_secrecy_rate_bps_per_gu"] / 1e6
                    power += env.uav_power_w.mean()
                    obs, state = next_obs, next_state
                mean_secrecy = secrecy / env.current_step
                mean_power = power / env.current_step
                writer.writerow([episode + 1, seed, episode_return, mean_secrecy, mean_power, updates])
                handle.flush()
                print(f"Episode {episode + 1}/{args.episodes} | {mean_secrecy:.4f} Mbps/GU | "
                      f"{mean_power:.4f} W/UAV | {updates} updates", flush=True)
        checkpoint = args.output / "policy.pt"
        torch.save({"actors": learner.actors.state_dict(), "model_args": model_args,
                    "config": asdict(config), "steps": args.steps, "seed": args.seed,
                    "episodes": args.episodes}, checkpoint)
        return checkpoint
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    if len(sys.argv) == 1:
        # Preserve the original three-episode integration check.
        run_dir = Path("runs") / datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f")
        smoke = parser.parse_args(["--output", str(run_dir), "--episodes", "3",
                                   "--steps", "6", "--warmup", "8", "--batch-size", "8"])
        train(smoke)
    else:
        train(parser.parse_args())
