"""Compare a saved policy with stationary and random controls on paired held-out seeds."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from environment import SAGINMultiUAVEnv
from experiments.baselines import random_actions, stationary_actions
from experiments.train import experiment_config
from reinforcement_learning.masac import MASAC


def evaluate(checkpoint, episodes=10, seed=None, output=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = experiment_config(saved["steps"])
    if asdict(config) != saved["config"]:
        raise ValueError("The current simulation configuration differs from this checkpoint")
    train_start = saved["seed"]
    train_end = train_start + saved["episodes"]  # exclusive
    if seed is None:
        seed = train_end
    if episodes < 1 or seed < 0 or (seed < train_end and seed + episodes > train_start):
        raise ValueError("Use a positive episode count and held-out, nonnegative evaluation seeds")
    learner = MASAC(**saved["model_args"])
    learner.actors.load_state_dict(saved["actors"])
    learner.eval()
    rows = []
    env = SAGINMultiUAVEnv(config=config)
    try:
        for scenario in range(seed, seed + episodes):
            for policy in ("masac", "stationary", "random"):
                obs, _ = env.reset(seed=scenario)
                # Keep exploration draws independent of the scenario generator.
                rng = np.random.default_rng(np.random.SeedSequence([scenario, 1]))
                secrecy = power = 0.0
                while env.agents:
                    if policy == "masac":
                        actions = learner.select_actions(obs, deterministic=True)
                    elif policy == "stationary":
                        actions = stationary_actions(env)
                    else:
                        actions = random_actions(env, rng)
                    obs, _, _, _, _ = env.step(actions)
                    secrecy += env.latest_metrics["mean_secrecy_rate_bps_per_gu"] / 1e6
                    power += env.uav_power_w.mean()
                rows.append(dict(seed=scenario, policy=policy,
                                 secrecy_mbps_per_gu=secrecy / env.current_step,
                                 mean_power_w_per_uav=power / env.current_step))
    finally:
        env.close()
    if output is None:
        output = checkpoint.with_name("evaluation.csv")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for policy in ("masac", "stationary", "random"):
        runs = [row for row in rows if row["policy"] == policy]
        secrecy = np.mean([row["secrecy_mbps_per_gu"] for row in runs])
        power = np.mean([row["mean_power_w_per_uav"] for row in runs])
        print(f"{policy:12} {secrecy:.4f} Mbps/GU   {power:.4f} W/UAV")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.seed, args.output)
