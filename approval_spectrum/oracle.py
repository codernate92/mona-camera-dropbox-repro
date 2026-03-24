"""Exact environment quantities used by learned-approval experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from approval_spectrum.configs import EnvironmentConfig
from mona.src import block_push_gym_env
from mona.src import policy_constructor
from mona.src import train


@dataclass(frozen=True)
class OracleArtifacts:
  """Exact dynamic-programming objects for one environment."""

  env_config: EnvironmentConfig
  transition_matrix: np.ndarray
  good_reward_state: np.ndarray
  bad_reward_state: np.ndarray
  good_value: np.ndarray
  bad_value: np.ndarray
  oracle_approval: np.ndarray
  good_policy_actions: np.ndarray
  bad_policy_actions: np.ndarray
  intended_label: np.ndarray
  hack_label: np.ndarray
  failure_label: np.ndarray
  true_return_if_follow_good: np.ndarray
  observed_return_if_follow_good: np.ndarray


def make_gym_env(
    env_config: EnvironmentConfig,
    use_good_reward: bool,
    reward_override: np.ndarray | None = None,
) -> block_push_gym_env.BlockPushGymEnv:
  """Builds a gym environment from an EnvironmentConfig."""
  return block_push_gym_env.BlockPushGymEnv(
      board_shape=env_config.board_shape,
      max_boxes=env_config.max_boxes,
      min_blocking_boxes=env_config.min_blocking_boxes,
      per_step_penalty=env_config.per_step_penalty,
      episode_step_limit=env_config.episode_step_limit,
      use_good_reward=use_good_reward,
      reward_override=reward_override,
  )


def _policy_actions_from_tq(tq: np.ndarray) -> np.ndarray:
  policy = policy_constructor.TPolicy.from_TQ(tq)
  num_steps, num_states = tq.shape[:2]
  actions = np.zeros((num_steps, num_states), dtype=np.int32)
  for t in range(num_steps):
    for state_idx in range(num_states):
      actions[t, state_idx] = policy[(t, state_idx)]
  return actions


def _continuation_returns(
    transition_matrix: np.ndarray,
    reward_state: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns cumulative reward and box counts under a fixed time-dependent policy."""
  step_limit, num_states = actions.shape
  cumulative_reward = np.zeros((step_limit + 1, num_states), dtype=np.float64)
  cumulative_boxes = np.zeros((step_limit + 1, num_states), dtype=np.int32)
  state_indices = np.arange(num_states)

  for t in reversed(range(step_limit)):
    chosen_actions = actions[t]
    next_states = transition_matrix[state_indices, chosen_actions]
    immediate_reward = reward_state[next_states]
    cumulative_reward[t] = immediate_reward + cumulative_reward[t + 1, next_states]
    cumulative_boxes[t] = (
        (immediate_reward > 0).astype(np.int32) + cumulative_boxes[t + 1, next_states]
    )
  return cumulative_reward[:step_limit], cumulative_boxes[:step_limit]


def _candidate_outcomes(
    transition_matrix: np.ndarray,
    reward_state: np.ndarray,
    continuation_reward: np.ndarray,
    continuation_boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns outcomes for each action then continuation policy."""
  step_limit = continuation_reward.shape[0]
  num_states, num_actions = transition_matrix.shape
  reward_tensor = np.zeros((step_limit, num_states, num_actions), dtype=np.float64)
  boxes_tensor = np.zeros((step_limit, num_states, num_actions), dtype=np.int32)

  for t in range(step_limit):
    next_states = transition_matrix
    reward_tensor[t] = reward_state[next_states]
    boxes_tensor[t] = (reward_state[next_states] > 0).astype(np.int32)
    if t + 1 < step_limit:
      reward_tensor[t] += continuation_reward[t + 1][next_states]
      boxes_tensor[t] += continuation_boxes[t + 1][next_states]
  return reward_tensor, boxes_tensor


def finite_horizon_values(
    transition_matrix: np.ndarray,
    reward_state: np.ndarray,
    step_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Dynamic-programming values for a custom state reward."""
  num_states, num_actions = transition_matrix.shape
  values = np.zeros((step_limit + 1, num_states), dtype=np.float64)
  q_values = np.zeros((step_limit, num_states, num_actions), dtype=np.float64)
  reward_sa = reward_state[transition_matrix]
  for t in reversed(range(step_limit)):
    q_values[t] = reward_sa
    if t + 1 < step_limit:
      q_values[t] += values[t + 1][transition_matrix]
    values[t] = np.max(q_values[t], axis=-1)
  return values[:step_limit], q_values


def add_public_style_noise(
    value_matrix: np.ndarray,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
  """Adds per-state Gaussian noise the same way the public code does."""
  if noise_scale <= 0:
    return value_matrix.copy()
  rng = np.random.default_rng(seed)
  noise_vector = rng.normal(scale=noise_scale, size=value_matrix.shape[1])
  return value_matrix + np.where(np.abs(value_matrix) > 1e-5, noise_vector, 0.0)


def build_oracle_artifacts(env_config: EnvironmentConfig) -> OracleArtifacts:
  """Builds exact MONA quantities for one environment."""
  trainer = train.Trainer(
      board_shape=env_config.board_shape,
      max_boxes=env_config.max_boxes,
      min_blocking_boxes=env_config.min_blocking_boxes,
      per_step_penalty=env_config.per_step_penalty,
      episode_step_limit=env_config.episode_step_limit,
  )
  good_result = trainer.get_training_result("good")
  bad_result = trainer.get_training_result("bad")

  good_policy_actions = _policy_actions_from_tq(good_result.TQ)
  bad_policy_actions = _policy_actions_from_tq(bad_result.TQ)

  good_true_returns, good_box_counts = _continuation_returns(
      good_result.P, good_result.R, good_policy_actions
  )
  good_observed_returns, observed_boxes = _continuation_returns(
      good_result.P, bad_result.R, good_policy_actions
  )
  candidate_true_returns, candidate_box_counts = _candidate_outcomes(
      good_result.P, good_result.R, good_true_returns, good_box_counts
  )
  candidate_observed_returns, _ = _candidate_outcomes(
      good_result.P, bad_result.R, good_observed_returns, observed_boxes
  )

  intended_label = (candidate_box_counts == 1).astype(np.float64)
  hack_label = (candidate_box_counts >= 2).astype(np.float64)
  failure_label = (candidate_box_counts == 0).astype(np.float64)

  oracle_approval = train.get_advantage_function(
      good_result.P,
      bad_result.R,
      good_result.TV,
      env_config.episode_step_limit,
  )

  return OracleArtifacts(
      env_config=env_config,
      transition_matrix=good_result.P,
      good_reward_state=good_result.R,
      bad_reward_state=bad_result.R,
      good_value=good_result.TV,
      bad_value=bad_result.TV,
      oracle_approval=oracle_approval,
      good_policy_actions=good_policy_actions,
      bad_policy_actions=bad_policy_actions,
      intended_label=intended_label,
      hack_label=hack_label,
      failure_label=failure_label,
      true_return_if_follow_good=candidate_true_returns,
      observed_return_if_follow_good=candidate_observed_returns,
  )


def build_misspecified_oracle(
    oracle: OracleArtifacts,
    misspecification: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Builds a misspecified MONA reward tensor and the trusted value it came from."""
  blended_reward = (
      (1.0 - misspecification) * oracle.good_reward_state
      + misspecification * oracle.bad_reward_state
  )
  mixed_value, _ = finite_horizon_values(
      oracle.transition_matrix,
      blended_reward,
      oracle.env_config.episode_step_limit,
  )
  approval = train.get_advantage_function(
      oracle.transition_matrix,
      oracle.bad_reward_state,
      mixed_value,
      oracle.env_config.episode_step_limit,
  )
  return approval, mixed_value
