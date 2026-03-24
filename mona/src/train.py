# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a policy in the gridworld and returns the results."""

from collections.abc import Sequence
import dataclasses
from typing import Tuple

import numpy as np
from numpy import typing as np_typing

from mona.src import data
from mona.src import matrix_constructor
from mona.src import policy_constructor
from mona.src import state as state_lib


State = state_lib.State
Action = data.Action
NDArray = np_typing.NDArray

# pylint: disable=invalid-name


def get_advantage_function(
    P: NDArray[np.int32],
    R: NDArray[np.float64],
    trusted_TV: NDArray[np.float64] | None,
    step_limit: int,
) -> NDArray[np.float64]:
  """Returns the advantage function for the given matrices.

  Args:
    P: The transition matrix, with shape (num_states, num_actions).
    R: The reward matrix, with shape (num_states,).
    trusted_TV: The trusted time-dependent value function to use to derive the
      advantage function, with shape (step_limit, num_states). If None, the
      value function is initialized to all zeros.
    step_limit: The maximum number of steps in a rollout.
  """
  num_states, num_actions = P.shape

  TA = np.zeros((step_limit, num_states, num_actions))

  next_step_rewards = R[P]

  # The last timestep always has an advantage of zero, because the action
  # cannot affect the reward.
  if trusted_TV is None:
    # If trusted_TV is None, the advantage is the reward at the next state.
    TA[: step_limit - 1] = next_step_rewards
  else:
    # Calculate next-step values for all states and actions.
    next_step_values = np.zeros((step_limit - 1, num_states, num_actions))
    for t in range(step_limit - 1):
      next_step_values[t] = trusted_TV[t + 1][P]

    # Compute the advantage function.
    TA[: step_limit - 1] = (
        next_step_rewards
        + next_step_values
        - trusted_TV[: step_limit - 1, :, None]
    )

  return TA


@dataclasses.dataclass(frozen=True)
class TrainingResult:
  """The result of running value iteration.

  Attributes:
    P: The transition matrix.
    R: The reward matrix.
    TV: A list of time-dependent value functions, each representing a timestep
      in a rollout.
    TQ: A list of time-dependent Qs, each representing a timestep in a rollout.
    intermediate_TQs: A list of lists of time-dependent Qs, each representing
      the Q functions at a certain point in training. Shape: (iterations, steps,
      states, actions).
    reward_function: A name which identifies the reward function.
  """

  P: NDArray[np.int32] | None
  R: NDArray[np.float64] | None
  TV: NDArray[np.float64] | None
  TQ: NDArray[np.float64] | None
  intermediate_TQs: np_typing.NDArray[np.float64] | None
  reward_function: str | None


class Trainer:
  """Runs value iteration and returns the results."""

  def __init__(
      self,
      board_shape: Tuple[int, int],
      max_boxes: int,
      min_blocking_boxes: int,
      per_step_penalty: float,
      episode_step_limit: int = 50,
  ):
    self._board_shape = board_shape
    self._max_boxes = max_boxes
    self._min_blocking_boxes = min_blocking_boxes
    self._per_step_penalty = per_step_penalty
    self._step_limit = episode_step_limit

    self._mat_constructor = matrix_constructor.MatrixConstructor(
        board_shape=self._board_shape,
        max_boxes=self._max_boxes,
        min_blocking_boxes=self._min_blocking_boxes,
        per_step_penalty=self._per_step_penalty,
        verbose=True,
    )

    self._num_states = self._mat_constructor.num_states
    self._num_actions = len(data.AGENT_ACTIONS)

  def _validate_matrices(
      self,
      P: NDArray[np.int32],
      R: NDArray[np.float64],
      trusted_TV: NDArray[np.float64] | None,
  ) -> None:
    """Validates the sizes of the matrices."""
    assert P.shape == (self._num_states, self._num_actions)
    assert P.dtype == np.int32
    assert R.shape == (self._num_states,)
    assert R.dtype == np.float64
    if trusted_TV is not None:
      assert trusted_TV.shape == (self._step_limit, self._num_states)
      assert trusted_TV.dtype == np.float64

  def _rollout_with_exploration(
      self,
      P: NDArray[np.int32],
      policy: policy_constructor.Policy | policy_constructor.TPolicy,
      init_state: int,
      explore_prob: float,
  ) -> Tuple[Sequence[int], Sequence[Action]]:
    """Rolls out from `init_state` with exploration.

    Args:
      P: The transition matrix.
      policy: The policy to use.
      init_state: The initial state to roll out from.
      explore_prob: The probability of choosing a random action instead of the
        policy's action at each step.

    Returns:
      A tuple of the rollout states and actions. There will be one more state
      than action, since we include both the initial state and the final state.
    """
    s_rollout = [init_state]
    a_rollout = []
    for t in range(self._step_limit):
      s = s_rollout[-1]
      if s == 0:
        # Stop early if we've reached the end state.
        break
      # With probability explore_prob, choose a random action.
      if np.random.random() < explore_prob:
        a = np.random.choice(data.AGENT_ACTIONS)
      elif isinstance(policy, policy_constructor.Policy):
        a = policy[s]
      else:
        a = policy[(t, s)]
      a_rollout.append(a)
      s_rollout.append(P[s, a])
    return s_rollout, a_rollout

  def _value_iteration(
      self,
      P: NDArray[np.int32],
      R: NDArray[np.float64],
      trusted_TV: NDArray[np.float64] | None = None,
  ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Performs value iteration to find the optimal value function.

    Args:
      P: The transition matrix.
      R: The reward matrix.
      trusted_TV: The trusted time-dependent value function to use to derive the
        advantage function. If None, the value function is initialized to all
        zeros.

    Returns:
      A list of value matrices, where the first element is the initial value
      function and the last element is the final value function.
    """
    self._validate_matrices(P=P, R=R, trusted_TV=trusted_TV)
    TA = get_advantage_function(P, R, trusted_TV, self._step_limit)

    # Initialize an empty array of TV and TQ functions.
    TVs = np.zeros((self._step_limit + 1, self._step_limit, self._num_states))
    TQs = np.zeros((
        self._step_limit + 1,
        self._step_limit,
        self._num_states,
        self._num_actions,
    ))

    for M in range(1, self._step_limit + 1):
      for t in reversed(range(self._step_limit)):
        TQs[M, t] = TA[t]
        if t + 1 < self._step_limit:
          TQs[M, t] += TVs[M - 1, t + 1, P]
        # Take the max of Q over all actions to get the value function.
        TVs[M, t] = np.max(TQs[M, t], axis=-1)

    return TVs, TQs

  def _add_noise_to_TV(
      self,
      TV: NDArray[np.float64],
      noise_scale: float,
  ) -> NDArray[np.float64]:
    """Returns a copy of `TV` with Gaussian noise added.

    Doesn't add noise to near-zero values in order to ensure fast convergence.

    Args:
      TV: The time-dependent value function to add noise to. Expected to have
        shape (step_limit, num_states).
      noise_scale: The standard deviation of the Gaussian noise to add.
    """
    assert TV.shape == (self._step_limit, self._num_states)
    # For any given state, add the same noise to it on each timestep.
    # This noise will be broadcasted to all timesteps.
    noise_vector = np.random.normal(scale=noise_scale, size=self._num_states)
    # Add the noise to the TV function, without adding noise to zero values.
    return TV + np.where(np.abs(TV) > 1e-5, noise_vector, 0.0)

  def get_matrix_constructor(self) -> matrix_constructor.MatrixConstructor:
    return self._mat_constructor

  def get_training_result(
      self,
      reward_function: str,
      noise_scale: float = 0.0,
      init_training_result: TrainingResult | None = None,
  ) -> TrainingResult:
    """Runs training and returns the matrices and value functions."""
    print('Constructing transition matrix...')
    P = self._mat_constructor.transition_matrix

    print('Constructing reward matrix...')
    R = (
        self._mat_constructor.construct_good_reward_matrix()
        if reward_function == 'good'
        else self._mat_constructor.construct_bad_reward_matrix()
    )

    print('\nRunning value iteration...')

    trusted_TV = (
        init_training_result.TV if init_training_result is not None else None
    )
    TVs, TQs = self._value_iteration(P, R, trusted_TV=trusted_TV)
    TV = TVs[-1]
    intermediate_TQs, TQ = TQs[:-1], TQs[-1]

    if noise_scale > 0 and TV is not None:
      print(f'Adding Gaussian noise with standard deviation {noise_scale}...')
      TV = self._add_noise_to_TV(TV, noise_scale)

    # Make the matrices read-only so we don't accidentally change them.
    P.flags.writeable = False
    R.flags.writeable = False
    if TV is not None:
      TV.flags.writeable = False
    TQ.flags.writeable = False

    return TrainingResult(
        P=P,
        R=R,
        TV=TV,
        TQ=TQ,
        intermediate_TQs=intermediate_TQs,
        reward_function=reward_function,
    )

  @property
  def num_states(self) -> int:
    return self._mat_constructor.num_states

  @property
  def num_initial_states(self) -> int:
    return len(self._mat_constructor.initial_states())
