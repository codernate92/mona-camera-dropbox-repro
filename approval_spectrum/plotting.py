"""Plotting utilities for learned-approval experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from approval_spectrum.metrics import ExperimentResult


def _label(result: ExperimentResult) -> str:
  if result.horizon is None:
    return result.approval_method
  return f"{result.approval_method} (H={result.horizon})"


def plot_frontier(results: list[ExperimentResult], output_dir: Path) -> Path:
  """Plots the policy-level safety/capability frontier."""
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "safety_capability_frontier.png"
  plt.figure(figsize=(8, 6))
  for result in results:
    plt.scatter(
        result.policy_metrics.reward_hacking_rate,
        result.policy_metrics.observed_return,
        alpha=0.8,
    )
    plt.annotate(
        _label(result),
        (
            result.policy_metrics.reward_hacking_rate,
            result.policy_metrics.observed_return,
        ),
        fontsize=7,
    )
  plt.xlabel("Reward Hacking Rate")
  plt.ylabel("Observed Return")
  plt.title("Safety/Capability Frontier")
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(path, dpi=200)
  plt.close()
  return path


def plot_training_curves(results: list[ExperimentResult], output_dir: Path) -> Path:
  """Plots intended vs hacking rates across training snapshots."""
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "training_curves.png"
  plt.figure(figsize=(9, 6))
  for result in results:
    if not result.snapshots:
      continue
    xs = [snapshot.timesteps for snapshot in result.snapshots]
    ys = [snapshot.intended_behavior_rate - snapshot.reward_hacking_rate for snapshot in result.snapshots]
    plt.plot(xs, ys, marker="o", label=_label(result))
  plt.xlabel("Training Timesteps")
  plt.ylabel("Intended Rate - Hacking Rate")
  plt.title("Training Safety Trajectories")
  plt.grid(alpha=0.3)
  plt.legend(fontsize=7)
  plt.tight_layout()
  plt.savefig(path, dpi=200)
  plt.close()
  return path


def plot_approval_quality(results: list[ExperimentResult], output_dir: Path) -> Path:
  """Plots approval AUC against calibration error for learned overseers."""
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "approval_quality.png"
  plt.figure(figsize=(8, 6))
  for result in results:
    if result.approval_metrics is None:
      continue
    auc = result.approval_metrics.intended_auc
    ece = result.approval_metrics.intended_ece
    if auc is None or ece is None:
      continue
    plt.scatter(ece, auc, alpha=0.8)
    plt.annotate(_label(result), (ece, auc), fontsize=7)
  plt.xlabel("ECE (lower is better)")
  plt.ylabel("Intended AUC (higher is better)")
  plt.title("Approval Quality")
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(path, dpi=200)
  plt.close()
  return path


def plot_public_reference_comparison(
    mona_rollouts_path: Path,
    ordinary_rl_rollouts_path: Path,
    output_dir: Path,
) -> Path:
  """Plots released public PPO MONA vs ordinary-RL behavior trajectories."""
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "public_ppo_reference.png"
  mona = np.load(mona_rollouts_path)
  ordinary_rl = np.load(ordinary_rl_rollouts_path)
  mona_pct = mona.mean(axis=0)
  mona_pct = mona_pct / mona_pct.sum(axis=1, keepdims=True)
  ordinary_pct = ordinary_rl.mean(axis=0)
  ordinary_pct = ordinary_pct / ordinary_pct.sum(axis=1, keepdims=True)
  xs = np.arange(mona_pct.shape[0])

  plt.figure(figsize=(9, 6))
  plt.plot(xs, mona_pct[:, 1], label="MONA Intended", color="#2c7fb8")
  plt.plot(xs, mona_pct[:, 2], label="MONA Hacking", color="#7fcdbb")
  plt.plot(xs, ordinary_pct[:, 1], label="RL Intended", color="#d95f0e")
  plt.plot(xs, ordinary_pct[:, 2], label="RL Hacking", color="#f16913")
  plt.xlabel("Saved PPO checkpoint index")
  plt.ylabel("Behavior frequency")
  plt.title("Released Public PPO Reference")
  plt.grid(alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=200)
  plt.close()
  return path
