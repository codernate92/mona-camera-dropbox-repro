"""Helpers for reproducing the public MONA Camera Dropbox setup."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mona.src import runner as runner_lib


PUBLIC_CAMERA_DROPBOX_CONFIG = {
    "board_shape": (4, 4),
    "max_boxes": 2,
    "min_blocking_boxes": 1,
    "per_step_penalty": 0.01,
    "episode_step_limit": 50,
    "initial_vf": "good",
    "noise_scale": 0.5,
    "reward_function": "bad",
    "save_rollouts_level": 3,
}


def run_public_camera_dropbox(
    output_root: str | Path,
    seed: int = 0,
) -> dict[str, object]:
  """Runs the public value-iteration Camera Dropbox reproduction."""
  np.random.seed(seed)
  output_root = Path(output_root)
  data_dir = output_root / "data"
  config = PUBLIC_CAMERA_DROPBOX_CONFIG

  runner = runner_lib.Runner(
      board_shape=config["board_shape"],
      max_boxes=config["max_boxes"],
      min_blocking_boxes=config["min_blocking_boxes"],
      per_step_penalty=config["per_step_penalty"],
      data_dir=str(data_dir),
      episode_step_limit=config["episode_step_limit"],
  )
  init_result = runner.run(
      runner_lib.RunParams(
          reward_function="good",
          noise_scale=config["noise_scale"],
          save_value_matrix=True,
      )
  )
  final_result = runner.run(
      runner_lib.RunParams(
          reward_function="bad",
          init_run_result=init_result,
          save_rollouts_level=config["save_rollouts_level"],
      )
  )

  summary = {
      "seed": seed,
      "output_root": str(output_root),
      "data_directory": str(final_result.full_dir if final_result else data_dir),
      "config": {
          "board_shape": list(config["board_shape"]),
          "max_boxes": config["max_boxes"],
          "min_blocking_boxes": config["min_blocking_boxes"],
          "per_step_penalty": config["per_step_penalty"],
          "episode_step_limit": config["episode_step_limit"],
          "initial_vf": config["initial_vf"],
          "noise_scale": config["noise_scale"],
          "reward_function": config["reward_function"],
          "save_rollouts_level": config["save_rollouts_level"],
      },
      "upstream_reference_command": (
          "bazel-bin/src/main --board_shape=4,4 --max_boxes=2 "
          "--episode_step_limit=50 --min_blocking_boxes=1 "
          "--reward_function=bad --per_step_penalty=0.01 "
          "--initial_vf=good --noise_scale=0.5 "
          "--data_dir=data/ --save_init_value_matrix "
          "--save_rollouts_level=3 --show_demo=true"
      ),
      "reproduction_notes": [
          "This repo ports the public value-iteration setup into a standard "
          "Python package while keeping the copied MONA source under mona/src.",
          "The public PPO notebook is still vendored under notebooks/upstream, "
          "and this repo additionally provides a separate scripted PPO path "
          "under experiments/approval_spectrum for extension work.",
      ],
  }
  summary_path = output_root / "public_reproduction_summary.json"
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary
