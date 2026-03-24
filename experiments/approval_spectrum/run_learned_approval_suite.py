"""CLI for the learned-approval PPO experiment suite."""

from __future__ import annotations

import argparse

from approval_spectrum.runner import run_learned_approval_suite


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-root",
      default="experiments/outputs/learned_approval",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()
  run_learned_approval_suite(
      output_root=args.output_root,
      seed=args.seed,
      force=args.force,
  )


if __name__ == "__main__":
  main()
