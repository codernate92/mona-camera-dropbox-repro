# MONA Camera Dropbox Reproduction

This repository is a Python-first reproduction of the public MONA Camera Dropbox codebase:

- Paper: [MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking](https://arxiv.org/abs/2501.13011)
- Public repo: [google-deepmind/mona](https://github.com/google-deepmind/mona)

The copied reference implementation lives under `mona/src/` and `mona/proto/`. The goal here is not to redesign the project, but to make the public value-iteration setup runnable as a normal Python package with a clean reproduction command.

## Scope

- Included: public Camera Dropbox value-iteration code, a scripted reproduction command, the upstream notebooks under `notebooks/upstream/`, and lightweight sanity tests.
- Not yet ported: the public PPO notebook as a fully scripted experiment pipeline.

## Install

```bash
pip install -e .[dev]
```

The generated protobuf binding `mona/proto/rollout_pb2.py` is committed in this repo after setup. If you ever need to regenerate it manually:

```bash
python -m grpc_tools.protoc -I. --python_out=. mona/proto/rollout.proto
```

## Reproduce The Public Camera Dropbox Run

```bash
python -m experiments.approval_spectrum.run_public_reproduction --output-root experiments/outputs/public_camera_dropbox --seed 0
```

This reproduces the public value-iteration configuration:

- `board_shape=4,4`
- `max_boxes=2`
- `min_blocking_boxes=1`
- `episode_step_limit=50`
- `reward_function=bad`
- `per_step_penalty=0.01`
- `initial_vf=good`
- `noise_scale=0.5`
- `save_rollouts_level=3`

The run writes a `public_reproduction_summary.json` plus the reproduced value matrix and rollout artifacts under the chosen output directory.

## Fidelity Notes

- Match: the value-iteration environment and training logic are copied from the public MONA repository.
- Divergence: the original public repo is Bazel-first; this repo wraps the same code in a standard Python package.
- Divergence: PPO remains notebook-only here, matching the public notebook artifact rather than a new scripted runner.

## Validate

```bash
pytest
```
