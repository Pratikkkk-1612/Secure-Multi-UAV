"""Train and evaluate with one command from the Python project root."""

import argparse

from experiments.evaluate import evaluate
from experiments.train import add_training_arguments, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    evaluate(train(args), episodes=args.eval_episodes)
