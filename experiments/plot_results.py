"""Plot episode-mean secrecy and power on the paired evaluation scenarios."""

import argparse
import csv
from pathlib import Path


def plot_results(evaluation, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with evaluation.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for policy in ("masac", "stationary", "random"):
        runs = sorted((r for r in rows if r["policy"] == policy), key=lambda r: int(r["seed"]))
        seeds = [int(r["seed"]) for r in runs]
        for ax, name in zip(axes, ("secrecy_mbps_per_gu", "mean_power_w_per_uav")):
            ax.plot(seeds, [float(r[name]) for r in runs], marker="o", label=policy)
    axes[0].set_ylabel("Secrecy (Mbps per GU)")
    axes[1].set_ylabel("Mean transmit power (W per UAV)")
    axes[1].set_xlabel("Held-out scenario seed")
    if len(seeds) <= 12:
        axes[1].set_xticks(seeds)
    axes[0].legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_results(args.evaluation, args.output or args.evaluation.with_suffix(".png"))
