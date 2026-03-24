"""Approval-model construction for MONA extensions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

from approval_spectrum.configs import ApprovalConfig, DatasetConfig
from approval_spectrum.metrics import ApprovalModelMetrics, build_approval_metrics
from approval_spectrum.oracle import OracleArtifacts, build_misspecified_oracle, make_gym_env


@dataclass(frozen=True)
class SupervisedDataset:
  """A trajectory-sampled dataset for training learned overseers."""

  features: np.ndarray
  oracle_scores: np.ndarray
  intended_labels: np.ndarray
  hack_labels: np.ndarray
  train_indices: np.ndarray
  val_indices: np.ndarray
  test_indices: np.ndarray


@dataclass(frozen=True)
class ApprovalArtifacts:
  """Reward override tensor plus offline approval metrics."""

  reward_override: np.ndarray | None
  predicted_scores: np.ndarray | None
  predicted_probabilities: np.ndarray | None
  metrics: ApprovalModelMetrics | None
  notes: tuple[str, ...]


def _one_hot_encode_tiles(tiles: np.ndarray, num_categories: int = 3) -> np.ndarray:
  encoded = np.zeros((tiles.size, num_categories), dtype=np.float64)
  encoded[np.arange(tiles.size), tiles.astype(int)] = 1.0
  return encoded.reshape(-1)


def featurize_state_action_time(
    env,
    t: int,
    state_idx: int,
    action: int,
) -> np.ndarray:
  """Returns a feature vector for one time/state/action tuple."""
  timeless_obs = env.get_mat_constructor().get_state(state_idx)
  obs = env.state_to_observation(None, timeless_obs)
  tile_features = _one_hot_encode_tiles(obs)
  time_feature = np.array([t / max(env._episode_step_limit - 1, 1)], dtype=np.float64)
  action_features = np.zeros((env.action_space.n,), dtype=np.float64)
  action_features[action] = 1.0
  return np.concatenate([time_feature, tile_features, action_features])


def _sample_action(
    oracle: OracleArtifacts,
    behavior_name: str,
    t: int,
    state_idx: int,
    rng: np.random.Generator,
) -> int:
  if behavior_name == "good":
    return int(oracle.good_policy_actions[t, state_idx])
  if behavior_name == "bad":
    return int(oracle.bad_policy_actions[t, state_idx])
  if behavior_name == "random":
    return int(rng.integers(oracle.transition_matrix.shape[1]))
  raise ValueError(f"Unknown behavior policy {behavior_name}")


def build_trajectory_dataset(
    oracle: OracleArtifacts,
    dataset_config: DatasetConfig,
    seed: int,
) -> SupervisedDataset:
  """Samples a state-action dataset from mixed behavior trajectories."""
  env = make_gym_env(oracle.env_config, use_good_reward=False)
  initial_states = list(env.get_mat_constructor().initial_states())
  rng = np.random.default_rng(seed)
  features = []
  oracle_scores = []
  intended_labels = []
  hack_labels = []

  while len(features) < dataset_config.num_samples:
    state_idx = int(rng.choice(initial_states))
    behavior_name = str(rng.choice(dataset_config.behavior_policy_mix))
    t = 0
    while t < oracle.env_config.episode_step_limit and len(features) < dataset_config.num_samples:
      action = _sample_action(oracle, behavior_name, t, state_idx, rng)
      features.append(featurize_state_action_time(env, t, state_idx, action))
      oracle_scores.append(oracle.oracle_approval[t, state_idx, action])
      intended_labels.append(oracle.intended_label[t, state_idx, action])
      hack_labels.append(oracle.hack_label[t, state_idx, action])
      state_idx = int(oracle.transition_matrix[state_idx, action])
      t += 1
      if state_idx == 0:
        break

  features_arr = np.array(features, dtype=np.float64)
  oracle_scores_arr = np.array(oracle_scores, dtype=np.float64)
  intended_arr = np.array(intended_labels, dtype=np.float64)
  hack_arr = np.array(hack_labels, dtype=np.float64)

  indices = rng.permutation(len(features_arr))
  train_end = int(0.6 * len(indices))
  val_end = int(0.8 * len(indices))
  return SupervisedDataset(
      features=features_arr,
      oracle_scores=oracle_scores_arr,
      intended_labels=intended_arr,
      hack_labels=hack_arr,
      train_indices=indices[:train_end],
      val_indices=indices[train_end:val_end],
      test_indices=indices[val_end:],
  )


def _build_full_feature_grid(oracle: OracleArtifacts) -> np.ndarray:
  env = make_gym_env(oracle.env_config, use_good_reward=False)
  step_limit = oracle.env_config.episode_step_limit
  num_states, num_actions = oracle.transition_matrix.shape
  feature_rows = []
  for t in range(step_limit):
    for state_idx in range(num_states):
      for action in range(num_actions):
        feature_rows.append(featurize_state_action_time(env, t, state_idx, action))
  return np.array(feature_rows, dtype=np.float64)


def _build_mlp_classifier(
    dataset_config: DatasetConfig,
    seed: int,
) -> MLPClassifier:
  return MLPClassifier(
      hidden_layer_sizes=dataset_config.hidden_layer_sizes,
      random_state=seed,
      max_iter=dataset_config.max_iter,
  )


def _train_probability_model(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_config: DatasetConfig,
    calibration_method: str,
    seed: int,
):
  base = _build_mlp_classifier(dataset_config, seed)
  if calibration_method == "none":
    base.fit(features, labels)
    return base
  return CalibratedClassifierCV(
      estimator=base,
      method=calibration_method,
      cv=3,
  ).fit(features, labels)


def _predict_probability(model, features: np.ndarray) -> np.ndarray:
  probabilities = model.predict_proba(features)
  if probabilities.shape[1] == 1:
    return np.zeros((len(features),), dtype=np.float64)
  return probabilities[:, 1]


def _reshape_score_tensor(oracle: OracleArtifacts, flat_values: np.ndarray) -> np.ndarray:
  step_limit = oracle.env_config.episode_step_limit
  num_states, num_actions = oracle.transition_matrix.shape
  return flat_values.reshape((step_limit, num_states, num_actions))


def _build_exact_metrics(
    oracle: OracleArtifacts,
    predicted_scores: np.ndarray,
) -> ApprovalModelMetrics:
  return build_approval_metrics(
      oracle_scores=oracle.oracle_approval.reshape(-1),
      predicted_scores=predicted_scores.reshape(-1),
      intended_labels=oracle.intended_label.reshape(-1),
      predicted_probabilities=None,
      hack_labels=oracle.hack_label.reshape(-1),
  )


def build_approval_artifacts(
    approval_config: ApprovalConfig,
    oracle: OracleArtifacts,
    seed: int,
) -> ApprovalArtifacts:
  """Builds a reward override tensor for the requested approval method."""
  method = approval_config.method
  score_scale = max(1.0, float(np.max(np.abs(oracle.oracle_approval))))

  if method == "ordinary_rl":
    return ApprovalArtifacts(
        reward_override=None,
        predicted_scores=None,
        predicted_probabilities=None,
        metrics=None,
        notes=("Uses the observed bad reward directly with no MONA approval.",),
    )

  if method == "oracle_mona":
    reward_override = oracle.oracle_approval.copy()
    return ApprovalArtifacts(
        reward_override=reward_override,
        predicted_scores=reward_override,
        predicted_probabilities=None,
        metrics=_build_exact_metrics(oracle, reward_override),
        notes=("Exact MONA approval tensor from the trusted good value function.",),
    )

  if method == "noisy_oracle_mona":
    rng = np.random.default_rng(seed)
    reward_override = oracle.oracle_approval + rng.normal(
        scale=approval_config.noise_scale * score_scale,
        size=oracle.oracle_approval.shape,
    )
    return ApprovalArtifacts(
        reward_override=reward_override,
        predicted_scores=reward_override,
        predicted_probabilities=None,
        metrics=_build_exact_metrics(oracle, reward_override),
        notes=(
            "Adds Gaussian noise directly to the exact MONA approval tensor.",
        ),
    )

  if method == "misspecified_oracle_mona":
    reward_override, _ = build_misspecified_oracle(
        oracle,
        approval_config.misspecification,
    )
    return ApprovalArtifacts(
        reward_override=reward_override,
        predicted_scores=reward_override,
        predicted_probabilities=None,
        metrics=_build_exact_metrics(oracle, reward_override),
        notes=(
            "Uses a trusted value function from a blended true/observed reward model.",
        ),
    )

  if method not in {
      "learned_outcome_classifier",
      "calibrated_outcome_classifier",
  }:
    raise ValueError(f"Unsupported approval method: {method}")

  if approval_config.dataset is None:
    raise ValueError(f"{method} requires a dataset configuration.")

  dataset = build_trajectory_dataset(oracle, approval_config.dataset, seed)
  calibration = (
      "none"
      if method == "learned_outcome_classifier"
      else approval_config.calibration_method
  )
  train_indices = dataset.train_indices
  intended_model = _train_probability_model(
      dataset.features[train_indices],
      dataset.intended_labels[train_indices],
      approval_config.dataset,
      calibration,
      seed,
  )
  hack_model = _train_probability_model(
      dataset.features[train_indices],
      dataset.hack_labels[train_indices],
      approval_config.dataset,
      calibration,
      seed + 1,
  )

  full_features = _build_full_feature_grid(oracle)
  intended_probability = _predict_probability(intended_model, full_features)
  hack_probability = _predict_probability(hack_model, full_features)
  predicted_scores = score_scale * (intended_probability - hack_probability)
  predicted_scores -= oracle.env_config.per_step_penalty
  reward_override = _reshape_score_tensor(oracle, predicted_scores)
  predicted_prob_tensor = _reshape_score_tensor(oracle, intended_probability)

  test_indices = dataset.test_indices
  test_scores = score_scale * (
      _predict_probability(intended_model, dataset.features[test_indices])
      - _predict_probability(hack_model, dataset.features[test_indices])
  ) - oracle.env_config.per_step_penalty
  test_probs = _predict_probability(intended_model, dataset.features[test_indices])
  metrics = build_approval_metrics(
      oracle_scores=dataset.oracle_scores[test_indices],
      predicted_scores=test_scores,
      intended_labels=dataset.intended_labels[test_indices],
      predicted_probabilities=test_probs,
      hack_labels=dataset.hack_labels[test_indices],
  )
  notes = (
      "Learned on trajectory-sampled state/action/time tuples from a mixture of random, good, and bad behavior policies.",
      f"Calibration method: {calibration}.",
  )
  return ApprovalArtifacts(
      reward_override=reward_override,
      predicted_scores=reward_override,
      predicted_probabilities=predicted_prob_tensor,
      metrics=metrics,
      notes=notes,
  )
