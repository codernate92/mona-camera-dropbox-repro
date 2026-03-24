"""Tests for oracle construction and learned overseers."""

from __future__ import annotations

from approval_spectrum.configs import ApprovalConfig, DatasetConfig, EnvironmentConfig
from approval_spectrum.oracle import build_misspecified_oracle, build_oracle_artifacts
from approval_spectrum.overseers import build_approval_artifacts


def _small_env() -> EnvironmentConfig:
  return EnvironmentConfig(
      name="tiny_camera_dropbox",
      board_shape=(3, 4),
      max_boxes=2,
      episode_step_limit=20,
  )


def test_oracle_artifacts_have_expected_shapes():
  oracle = build_oracle_artifacts(_small_env())
  step_limit = oracle.env_config.episode_step_limit
  num_states, num_actions = oracle.transition_matrix.shape
  assert oracle.oracle_approval.shape == (step_limit, num_states, num_actions)
  assert oracle.intended_label.shape == (step_limit, num_states, num_actions)
  assert oracle.hack_label.shape == (step_limit, num_states, num_actions)


def test_misspecified_oracle_changes_the_score_tensor():
  oracle = build_oracle_artifacts(_small_env())
  misspecified, _ = build_misspecified_oracle(oracle, 0.25)
  assert misspecified.shape == oracle.oracle_approval.shape
  assert not (misspecified == oracle.oracle_approval).all()


def test_learned_outcome_classifier_builds_reward_override():
  oracle = build_oracle_artifacts(_small_env())
  artifacts = build_approval_artifacts(
      ApprovalConfig(
          method="learned_outcome_classifier",
          horizon=1,
          dataset=DatasetConfig(num_samples=64, hidden_layer_sizes=(16,), max_iter=100),
      ),
      oracle,
      seed=0,
  )
  assert artifacts.reward_override is not None
  assert artifacts.predicted_probabilities is not None
  assert artifacts.metrics is not None
