"""CLI for the scripted PPO reproduction slice."""

from __future__ import annotations

import argparse

from approval_spectrum.runner import run_scripted_ppo_reproduction


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-root",
      default="experiments/outputs/ppo_reproduction",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()
  run_scripted_ppo_reproduction(
      output_root=args.output_root,
      seed=args.seed,
      force=args.force,
  )


if __name__ == "__main__":
  main()
