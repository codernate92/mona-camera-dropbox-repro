"""CLI wrapper for the public MONA Camera Dropbox reproduction."""

from __future__ import annotations

import argparse

from mona.reproduction import run_public_camera_dropbox


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-root",
      default="experiments/outputs/public_camera_dropbox",
  )
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  run_public_camera_dropbox(output_root=args.output_root, seed=args.seed)


if __name__ == "__main__":
  main()
