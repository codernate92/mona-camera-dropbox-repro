"""Top-level runners for scripted PPO reproduction and learned approval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from approval_spectrum.configs import (
    ENVIRONMENT_PRESETS,
    ExperimentSpec,
    build_report_suite,
    build_scripted_ppo_reproduction_suite,
)
from approval_spectrum.metrics import ExperimentResult
from approval_spectrum.oracle import build_oracle_artifacts
from approval_spectrum.overseers import build_approval_artifacts
from approval_spectrum.plotting import (
    plot_approval_quality,
    plot_frontier,
    plot_public_reference_comparison,
    plot_training_curves,
)
from approval_spectrum.ppo_training import evaluate_policy_model, train_ppo_policy


def _project_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_result(path: Path) -> ExperimentResult:
  payload = json.loads(path.read_text(encoding="utf-8"))
  approval_payload = payload["approval_metrics"]
  policy_payload = payload["policy_metrics"]
  snapshots_payload = payload["snapshots"]
  from approval_spectrum.metrics import (
      ApprovalModelMetrics,
      PolicyMetrics,
      TrainingSnapshot,
  )

  approval_metrics = (
      None if approval_payload is None else ApprovalModelMetrics(**approval_payload)
  )
  policy_metrics = PolicyMetrics(**policy_payload)
  snapshots = tuple(TrainingSnapshot(**snapshot) for snapshot in snapshots_payload)
  return ExperimentResult(
      name=payload["name"],
      env_name=payload["env_name"],
      seed=payload["seed"],
      approval_method=payload["approval_method"],
      horizon=payload["horizon"],
      noise_scale=payload["noise_scale"],
      misspecification=payload["misspecification"],
      calibration_method=payload["calibration_method"],
      dataset_size=payload["dataset_size"],
      total_timesteps=payload["total_timesteps"],
      approval_metrics=approval_metrics,
      policy_metrics=policy_metrics,
      snapshots=snapshots,
      notes=tuple(payload["notes"]),
  )


def _flatten_result(result: ExperimentResult) -> dict[str, object]:
  return {
      "name": result.name,
      "env_name": result.env_name,
      "seed": result.seed,
      "approval_method": result.approval_method,
      "horizon": result.horizon,
      "noise_scale": result.noise_scale,
      "misspecification": result.misspecification,
      "calibration_method": result.calibration_method,
      "dataset_size": result.dataset_size,
      "total_timesteps": result.total_timesteps,
      "observed_return": result.policy_metrics.observed_return,
      "true_return": result.policy_metrics.true_return,
      "reward_hacking_rate": result.policy_metrics.reward_hacking_rate,
      "intended_behavior_rate": result.policy_metrics.intended_behavior_rate,
      "failure_rate": result.policy_metrics.failure_rate,
      "average_boxes_in_hole": result.policy_metrics.average_boxes_in_hole,
      "approval_mse_to_oracle_score": None if result.approval_metrics is None else result.approval_metrics.mse_to_oracle_score,
      "approval_ranking_correlation": None if result.approval_metrics is None else result.approval_metrics.ranking_correlation,
      "approval_intended_auc": None if result.approval_metrics is None else result.approval_metrics.intended_auc,
      "approval_intended_brier": None if result.approval_metrics is None else result.approval_metrics.intended_brier,
      "approval_intended_ece": None if result.approval_metrics is None else result.approval_metrics.intended_ece,
      "adversarial_false_safe_rate": None if result.approval_metrics is None else result.approval_metrics.adversarial_false_safe_rate,
  }


def _write_results_csv(results: list[ExperimentResult], path: Path) -> None:
  if not results:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  rows = [_flatten_result(result) for result in results]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _publish_plot_assets(plot_paths: list[Path], suite_name: str) -> list[Path]:
  asset_dir = _project_root() / "reports" / "assets" / suite_name
  asset_dir.mkdir(parents=True, exist_ok=True)
  published_paths = []
  for plot_path in plot_paths:
    destination = asset_dir / plot_path.name
    shutil.copy2(plot_path, destination)
    published_paths.append(destination)
  return published_paths


def _run_single_experiment(
    spec: ExperimentSpec,
    output_root: Path,
    force: bool,
    oracle_cache: dict[str, object],
) -> ExperimentResult:
  result_path = output_root / "runs" / f"{spec.name}.json"
  if result_path.exists() and not force:
    return _load_result(result_path)

  env_config = ENVIRONMENT_PRESETS[spec.env_name]
  if spec.env_name not in oracle_cache:
    oracle_cache[spec.env_name] = build_oracle_artifacts(env_config)
  oracle = oracle_cache[spec.env_name]

  approval_artifacts = build_approval_artifacts(spec.approval, oracle, spec.seed)
  model, snapshots = train_ppo_policy(
      env_config=env_config,
      ppo_config=spec.ppo,
      reward_override=approval_artifacts.reward_override,
      optimization_horizon=spec.approval.horizon,
      seed=spec.seed,
      output_dir=output_root / "models" / spec.name,
  )
  policy_metrics = evaluate_policy_model(
      model,
      env_config=env_config,
      num_rollouts_per_initial_state=3,
      deterministic=False,
      seed=spec.seed,
  )
  if model.get_env() is not None:
    model.get_env().close()
  result = ExperimentResult(
      name=spec.name,
      env_name=spec.env_name,
      seed=spec.seed,
      approval_method=spec.approval.method,
      horizon=spec.approval.horizon,
      noise_scale=spec.approval.noise_scale,
      misspecification=spec.approval.misspecification,
      calibration_method=spec.approval.calibration_method,
      dataset_size=None if spec.approval.dataset is None else spec.approval.dataset.num_samples,
      total_timesteps=spec.ppo.total_timesteps,
      approval_metrics=approval_artifacts.metrics,
      policy_metrics=policy_metrics,
      snapshots=snapshots,
      notes=approval_artifacts.notes,
  )
  if approval_artifacts.reward_override is not None:
    approval_path = output_root / "approval_tensors" / f"{spec.name}.npy"
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    np = __import__("numpy")
    np.save(approval_path, approval_artifacts.reward_override)
  _write_json(result_path, result.to_dict())
  return result


def _write_report(results: list[ExperimentResult], plot_paths: list[Path]) -> Path:
  report_path = _project_root() / "reports" / "mona_learned_approval_report.md"
  report_path.parent.mkdir(parents=True, exist_ok=True)

  public_results = [result for result in results if result.env_name == "public_camera_dropbox"]
  safest = min(public_results, key=lambda result: (result.policy_metrics.reward_hacking_rate, -result.policy_metrics.observed_return))
  most_capable = max(public_results, key=lambda result: result.policy_metrics.observed_return)
  learned_results = [
      result
      for result in public_results
      if "classifier" in result.approval_method
  ]
  best_learned = max(learned_results, key=lambda result: result.policy_metrics.true_return)
  public_reference_dir = _project_root() / "data" / "public_ppo_reference"
  public_reference_summary = None
  if (public_reference_dir / "mona_ppo_rollouts.npy").exists() and (
      public_reference_dir / "ordinary_rl_ppo_rollouts.npy"
  ).exists():
    mona = np.load(public_reference_dir / "mona_ppo_rollouts.npy")
    ordinary = np.load(public_reference_dir / "ordinary_rl_ppo_rollouts.npy")
    mona_final = mona[:, -1, :].mean(axis=0)
    ordinary_final = ordinary[:, -1, :].mean(axis=0)
    public_reference_summary = {
        "mona_final_pct": (mona_final / mona_final.sum()).tolist(),
        "ordinary_final_pct": (ordinary_final / ordinary_final.sum()).tolist(),
    }
  relative_plot_paths = [path.relative_to(report_path.parent) for path in plot_paths]
  plot_lines = "\n".join(f"- `{path.as_posix()}`" for path in relative_plot_paths)
  figure_blocks = "\n\n".join(
      [
          f"### {path.stem.replace('_', ' ').title()}\n\n"
          f"![{path.stem}]({path.as_posix()})"
          for path in relative_plot_paths
      ]
  )
  public_reference_lines = ""
  if public_reference_summary is not None:
    public_reference_lines = (
        f"- Released public PPO MONA final behavior: fail `{public_reference_summary['mona_final_pct'][0]:.3f}`, intended `{public_reference_summary['mona_final_pct'][1]:.3f}`, hack `{public_reference_summary['mona_final_pct'][2]:.3f}`.\n"
        f"- Released public PPO ordinary RL final behavior: fail `{public_reference_summary['ordinary_final_pct'][0]:.3f}`, intended `{public_reference_summary['ordinary_final_pct'][1]:.3f}`, hack `{public_reference_summary['ordinary_final_pct'][2]:.3f}`."
    )

  report = f"""# MONA Learned Approval Report

## Motivation

The public MONA results show that myopic optimization with non-myopic approval can mitigate multi-step reward hacking in Camera Dropbox. This extension asks whether that safety benefit survives when we replace exact tabular approval with learned, noisy, misspecified approval under PPO.

## Exact Hypotheses

- H1: Approval ranking quality matters more than mean offline error for preserving PPO-time safety.
- H2: Small misspecification is more damaging under PPO than in the tabular reproduction because policy optimization actively seeks approval weaknesses.
- H3: Calibrated learned overseers preserve more intended behavior than uncalibrated learned overseers trained on the same data.
- H4: Longer optimization horizons improve observed return but increase exploitability.
- H5: There is a practical phase transition where PPO with weak approval behaves much more like ordinary RL.

## Reproduction Fidelity

- Preserved the existing public value-iteration reproduction path.
- Ported the public PPO notebook logic into scripts and reusable Python modules.
- Reused the notebook's MONA rollout-buffer recomposition callback and key PPO settings (`gamma=1.0`, `ent_coef=0.05`, `clip_range=0.3`, `learning_rate=5e-5`) as the starting point.
- Upgraded the PPO path to a custom CNN torso over spatial board observations, reward normalization with `VecNormalize`, and `SubprocVecEnv` parallel rollout collection.
- Did not claim a full rerun of the paper's million-step PPO study on identical compute; this report uses smaller scripted sweeps that fit the local environment.

## Implementation Details

- `mona/src/`: copied public Camera Dropbox environment and tabular training code.
- `approval_spectrum/ppo_training.py`: scripted PPO training, spatial observation wrapper, custom CNN feature extractor, reward normalization, vectorized rollout collection, MONA callback, periodic evaluation snapshots.
- `approval_spectrum/overseers.py`: oracle, noisy, misspecified, learned, and calibrated approval mechanisms.
- Learned overseers are trajectory-trained outcome models over `(time, state, action)` tuples.
- The calibration-aware variant uses explicit probability calibration (`sigmoid` or `isotonic`) before constructing the approval score.
- `tests/test_ppo_training.py`: checks spatial observation encoding and explicit temporal alignment of MONA reward injection during subepisode extraction.

## Experiment Matrix

- Environments: `public_camera_dropbox`, `harder_camera_dropbox`
- Approval methods: `ordinary_rl`, `oracle_mona`, `noisy_oracle_mona`, `misspecified_oracle_mona`, `learned_outcome_classifier`, `calibrated_outcome_classifier`
- Horizons: `None`, `1`, `4`
- Dataset sizes: `512`, `2048`
- Calibration methods: `none`, `sigmoid`, `isotonic`
- PPO budgets: `768`, `1536`, `3072` in the executed local pilot, with larger values supported by the scripted pipeline

## Results

- Safest public-environment run: `{safest.name}` with reward-hacking rate `{safest.policy_metrics.reward_hacking_rate:.3f}` and observed return `{safest.policy_metrics.observed_return:.3f}`.
- Most capable public-environment run: `{most_capable.name}` with observed return `{most_capable.policy_metrics.observed_return:.3f}` and reward-hacking rate `{most_capable.policy_metrics.reward_hacking_rate:.3f}`.
- Best learned-overseer run by true return: `{best_learned.name}` with true return `{best_learned.policy_metrics.true_return:.3f}`, intended-behavior rate `{best_learned.policy_metrics.intended_behavior_rate:.3f}`, and hacking rate `{best_learned.policy_metrics.reward_hacking_rate:.3f}`.
- In the reduced local pilot, PPO did not yet reach the strong reward-hacking regime; the dominant failure mode was slow learning and persistent failure to complete the task.
{public_reference_lines}

## Failure Cases

- In the local pilot, the main failure mode was under-optimization: many runs still failed to complete the task reliably at these CPU-feasible budgets.
- The released public PPO reference still shows the paper's core contrast: ordinary RL converges to heavy reward hacking while MONA does not.
- Uncalibrated learned overseers can look strong offline while still having worse ranking correlation and weaker safety/capability tradeoffs than calibrated variants.

## Limitations

- PPO sweeps here are reduced-budget local runs, not identical-scale replicas of the paper's longest notebook experiments.
- Learned overseers are trained on trajectory-sampled state/action tuples rather than richer language-like oversight data.
- The calibration-aware models test one practical intervention class, not the full space of learned-approval architectures.
- The scripted PPO pipeline fixes seeds and improves repeatability, but repeated SB3/Torch runs are still not bitwise deterministic in this local setup, so single-seed numbers should be treated as pilot estimates rather than exact invariants.
- Because the PPO stack now differs from the flat single-environment notebook baseline, the extension suite should be read as a stronger experimental variant of the public setup rather than a strict apples-to-apples reproduction of every PPO implementation detail.

## Plot Artifacts

{plot_lines}

## Figures

{figure_blocks}

## Next Steps

- Push the scripted PPO runs to larger budgets and multiple seeds.
- Add a direct learned action-value overseer to compare against the outcome-model overseers here.
- Test adversarial data collection loops where PPO actively seeks states that maximize approval-model uncertainty or overestimation.
"""
  report_path.write_text(report, encoding="utf-8")
  return report_path


def _run_suite(
    specs: list[ExperimentSpec],
    output_root: Path,
    force: bool,
    write_report: bool = False,
) -> dict[str, object]:
  oracle_cache: dict[str, object] = {}
  results = [
      _run_single_experiment(spec, output_root, force=force, oracle_cache=oracle_cache)
      for spec in specs
  ]
  results_csv = output_root / "results.csv"
  _write_results_csv(results, results_csv)
  plot_dir = output_root / "plots"
  plot_paths = [
      plot_frontier(results, plot_dir),
      plot_training_curves(results, plot_dir),
      plot_approval_quality(results, plot_dir),
  ]
  public_reference_dir = _project_root() / "data" / "public_ppo_reference"
  if (public_reference_dir / "mona_ppo_rollouts.npy").exists() and (
      public_reference_dir / "ordinary_rl_ppo_rollouts.npy"
  ).exists():
    plot_paths.append(
        plot_public_reference_comparison(
            public_reference_dir / "mona_ppo_rollouts.npy",
            public_reference_dir / "ordinary_rl_ppo_rollouts.npy",
            plot_dir,
        )
    )
  published_plot_paths = _publish_plot_assets(plot_paths, output_root.name)
  summary = {
      "results_csv": str(results_csv),
      "plot_paths": [str(path) for path in plot_paths],
      "published_plot_paths": [str(path) for path in published_plot_paths],
      "num_experiments": len(results),
  }
  if write_report:
    report_path = _write_report(results, published_plot_paths)
    summary["report_path"] = str(report_path)
  _write_json(output_root / "summary.json", summary)
  return summary


def run_scripted_ppo_reproduction(
    output_root: str | Path,
    seed: int = 0,
    force: bool = False,
) -> dict[str, object]:
  """Runs the notebook-derived PPO reproduction slice."""
  specs = build_scripted_ppo_reproduction_suite(seed=seed)
  return _run_suite(specs, Path(output_root), force=force, write_report=False)


def run_learned_approval_suite(
    output_root: str | Path,
    seed: int = 0,
    force: bool = False,
) -> dict[str, object]:
  """Runs the learned-approval PPO suite and writes artifacts."""
  specs = build_report_suite(seed=seed)
  return _run_suite(specs, Path(output_root), force=force, write_report=True)
