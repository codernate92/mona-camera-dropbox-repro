# Public Reproduction Notes

## Reference

- Paper: `MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking`
- Public repo: `https://github.com/google-deepmind/mona`

## Reproduced Setup

This repo reproduces the public Camera Dropbox value-iteration path with the same public configuration:

- `board_shape=4,4`
- `max_boxes=2`
- `min_blocking_boxes=1`
- `episode_step_limit=50`
- `reward_function=bad`
- `per_step_penalty=0.01`
- `initial_vf=good`
- `noise_scale=0.5`
- `save_rollouts_level=3`

## Command Used Here

```bash
python -m experiments.approval_spectrum.run_public_reproduction --output-root experiments/outputs/public_camera_dropbox --seed 0
```

## Output Artifacts

- Initial value matrix:
  `experiments/outputs/public_camera_dropbox/data/R=good-noise_init=0.5-step_penalty=0.01-shape=4_4-boxes=2/value_matrix.npy`
- Final rollout protobuf:
  `experiments/outputs/public_camera_dropbox/data/R=bad-V_init=good-noise_init=0.5-step_penalty=0.01-shape=4_4-boxes=2/rollouts.pb`
- Run summary:
  `experiments/outputs/public_camera_dropbox/public_reproduction_summary.json`

## Fidelity

- Match: copied environment, training, rollout, and file-handling logic from the public MONA repo.
- Match: reproduced the public value-iteration Camera Dropbox configuration.
- Divergence: packaged the code as a normal Python project instead of relying on Bazel as the primary entrypoint.
- Divergence: kept the public PPO notebook under `notebooks/upstream/ppo.ipynb` but did not convert it into a scripted experiment runner.
