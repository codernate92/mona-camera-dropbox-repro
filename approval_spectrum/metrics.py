"""Metrics for learned approval models and PPO policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def _rankdata(values: np.ndarray) -> np.ndarray:
  order = np.argsort(values)
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = np.arange(len(values), dtype=np.float64)
  return ranks


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
  """Spearman correlation without a scipy dependency."""
  if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
    return float("nan")
  ra = _rankdata(a)
  rb = _rankdata(b)
  return float(np.corrcoef(ra, rb)[0, 1])


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
  """Computes the expected calibration error for binary probabilities."""
  if probabilities.size == 0:
    return float("nan")
  bins = np.linspace(0.0, 1.0, num_bins + 1)
  ece = 0.0
  for lower, upper in zip(bins[:-1], bins[1:]):
    mask = (probabilities >= lower) & (probabilities < upper)
    if upper == 1.0:
      mask = (probabilities >= lower) & (probabilities <= upper)
    if not np.any(mask):
      continue
    bin_prob = probabilities[mask]
    bin_labels = labels[mask]
    ece += (len(bin_prob) / len(probabilities)) * abs(
        np.mean(bin_prob) - np.mean(bin_labels)
    )
  return float(ece)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
  """ROC AUC with a NaN fallback if labels are degenerate."""
  if len(np.unique(labels)) < 2:
    return float("nan")
  return float(roc_auc_score(labels, scores))


@dataclass(frozen=True)
class ApprovalModelMetrics:
  """Offline metrics for a learned or exact approval model."""

  mse_to_oracle_score: float | None
  ranking_correlation: float | None
  intended_auc: float | None
  intended_brier: float | None
  intended_ece: float | None
  adversarial_false_safe_rate: float | None
  adversarial_mean_score: float | None

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass(frozen=True)
class PolicyMetrics:
  """Rollout metrics for a trained policy."""

  observed_return: float
  true_return: float
  reward_hacking_rate: float
  intended_behavior_rate: float
  failure_rate: float
  average_boxes_in_hole: float

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass(frozen=True)
class TrainingSnapshot:
  """Intermediate policy metrics recorded during PPO training."""

  timesteps: int
  observed_return: float
  true_return: float
  reward_hacking_rate: float
  intended_behavior_rate: float
  failure_rate: float

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
  """One saved experiment result."""

  name: str
  env_name: str
  seed: int
  approval_method: str
  horizon: int | None
  noise_scale: float
  misspecification: float
  calibration_method: str
  dataset_size: int | None
  total_timesteps: int
  approval_metrics: ApprovalModelMetrics | None
  policy_metrics: PolicyMetrics
  snapshots: tuple[TrainingSnapshot, ...]
  notes: tuple[str, ...]

  def to_dict(self) -> dict[str, object]:
    result = asdict(self)
    result["approval_metrics"] = (
        None if self.approval_metrics is None else self.approval_metrics.to_dict()
    )
    result["policy_metrics"] = self.policy_metrics.to_dict()
    result["snapshots"] = [snapshot.to_dict() for snapshot in self.snapshots]
    return result


def build_approval_metrics(
    oracle_scores: np.ndarray,
    predicted_scores: np.ndarray,
    intended_labels: np.ndarray,
    predicted_probabilities: np.ndarray | None,
    hack_labels: np.ndarray,
) -> ApprovalModelMetrics:
  """Computes offline model metrics on a held-out set."""
  mse = float(np.mean((predicted_scores - oracle_scores) ** 2))
  ranking = spearman_rank_correlation(oracle_scores, predicted_scores)
  auc = safe_auc(intended_labels, predicted_scores)
  if predicted_probabilities is not None:
    brier = float(np.mean((predicted_probabilities - intended_labels) ** 2))
    ece = expected_calibration_error(predicted_probabilities, intended_labels)
  else:
    brier = None
    ece = None

  non_safe_mask = intended_labels < 0.5
  if np.any(non_safe_mask):
    non_safe_scores = predicted_scores[non_safe_mask]
    non_safe_hacks = hack_labels[non_safe_mask]
    threshold = np.quantile(non_safe_scores, 0.9)
    adversarial_mask = non_safe_scores >= threshold
    adversarial_false_safe_rate = float(np.mean(non_safe_hacks[adversarial_mask]))
    adversarial_mean_score = float(np.mean(non_safe_scores[adversarial_mask]))
  else:
    adversarial_false_safe_rate = None
    adversarial_mean_score = None

  return ApprovalModelMetrics(
      mse_to_oracle_score=mse,
      ranking_correlation=ranking,
      intended_auc=auc,
      intended_brier=brier,
      intended_ece=ece,
      adversarial_false_safe_rate=adversarial_false_safe_rate,
      adversarial_mean_score=adversarial_mean_score,
  )
