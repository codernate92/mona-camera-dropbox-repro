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

"""Constructs a policy."""

from collections.abc import Iterable, Sequence
import enum

import numpy as np
from numpy import typing as np_typing

from mona.src import data

Action = data.Action
NDArray = np_typing.NDArray

# pylint: disable=invalid-name


class Policy(Iterable):
  """A mapping of state indices to actions."""

  class Source(enum.Enum):
    # Deterministic policy from P and V matrices, with cached responses.
    V_MATRIX = 1
    # Deterministic policy from P and Q matrices, with cached responses.
    Q_MATRIX = 2
    # Deterministic policy from a list of actions.
    ACTIONS = 3
    # Stochastic policy from a list of action probabilities.
    ACTION_PROBS = 4

  def __init__(
      self,
      P: NDArray[np.int32] | None = None,
      V: NDArray[np.float64] | None = None,
      Q: NDArray[np.float64] | None = None,
      actions: NDArray[np.int32] | None = None,
      action_probs: NDArray[np.float64] | None = None,
  ):
    """Initializes the policy.

    Args:
      P: The transition matrix, with shape (num_states, num_actions).
      V: The value matrix, with shape (num_states,).
      Q: The Q matrix, with shape (num_states, num_actions).
      actions: A list of actions, with shape (num_states,).
      action_probs: A list of action probabilities, with shape (num_states,
        num_actions).
    """
    sources = []  # Temp array to ensure that there is exactly one data source.
    if P is not None or V is not None:
      assert P is not None and V is not None
      self._P = P
      self._V = V
      self._cache = [None] * V.shape[0]
      sources.append(Policy.Source.V_MATRIX)
    if Q is not None:
      self._Q = Q
      self._cache = [None] * Q.shape[0]
      sources.append(Policy.Source.Q_MATRIX)
    if actions is not None:
      self._actions = actions
      sources.append(Policy.Source.ACTIONS)
    if action_probs is not None:
      action_probs = [_normalize_distribution(probs) for probs in action_probs]
      self._action_probs = action_probs
      sources.append(Policy.Source.ACTION_PROBS)
    assert len(sources) == 1
    self._source = sources[0]

  def is_deterministic(self) -> bool:
    return self._source in [
        Policy.Source.ACTIONS,
        Policy.Source.V_MATRIX,
        Policy.Source.Q_MATRIX,
    ]

  def __getitem__(self, index):
    match self._source:
      case Policy.Source.V_MATRIX:
        if self._cache[index] is None:
          self._cache[index] = get_policy_action_from_v(self._P, self._V, index)
        return self._cache[index]
      case Policy.Source.Q_MATRIX:
        if self._cache[index] is None:
          self._cache[index] = get_policy_action_from_Q(self._Q, index)
        return self._cache[index]
      case Policy.Source.ACTIONS:
        return self._actions[index]
      case Policy.Source.ACTION_PROBS:
        return np.random.choice(data.AGENT_ACTIONS, p=self._action_probs[index])

  def __len__(self):
    match self._source:
      case Policy.Source.V_MATRIX:
        return len(self._cache)
      case Policy.Source.Q_MATRIX:
        return len(self._cache)
      case Policy.Source.ACTIONS:
        return len(self._actions)
      case Policy.Source.ACTION_PROBS:
        return len(self._action_probs)

  def __str__(self):
    match self._source:
      case Policy.Source.V_MATRIX:
        return str([self[i] for i in range(len(self))])
      case Policy.Source.Q_MATRIX:
        return str([self[i] for i in range(len(self))])
      case Policy.Source.ACTIONS:
        return str(self._actions)
      case Policy.Source.ACTION_PROBS:
        return str(self._action_probs)

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def to_numpy(self) -> NDArray[np.float64]:
    result = np.zeros((len(self), len(data.AGENT_ACTIONS)))
    if self.is_deterministic():
      for s, a in enumerate(self):
        result[s, a] = 1.0
    else:
      assert self._source == Policy.Source.ACTION_PROBS
      for s in range(len(self)):
        result[s, :] = self._action_probs[s]
    return result

  @property
  def num_states(self) -> int:
    return len(self)


def get_policy_action_from_v(
    P: NDArray[np.int32],
    V: NDArray[np.float64],
    s_idx: int,
) -> Action:
  """Gets the action the policy would take at s_idx, given P and V matrices."""
  return _get_policy_action_from_q_list(np.array([V[s] for s in P[s_idx, :]]))


class TPolicy:
  """A policy that changes over time."""

  def __init__(self, policies: Sequence[Policy]):
    self._policies = policies

  def __getitem__(self, args):
    t, i = args
    if t < 0 or t >= len(self._policies):
      raise IndexError(f"Index {t} is out of range [0, {len(self._policies)})")
    return self._policies[t][i]

  def is_deterministic(self) -> bool:
    # Even if the policy is deterministic, it is not deterministic solely with
    # respect to the state, since the timestep also affects the policy.
    return False

  def get_TPolicy_for_M(self, M: int) -> "TPolicy":
    """Returns the TV or TQ function for a given value of M.

    The last M steps will be the same as in the original. All other steps
    will be the same as the Mth step from the end.

    Args:
      M: The number of steps from the end to get the TFunction for.
    """
    T = self.max_steps
    assert M <= T
    # There is implicitly an all-zero value function after the last step.
    # A policy where all V = 0 is essentially equivalent to a random policy.
    if M == 0:
      return random_t_policy(self.num_states, T)
    new_policies = [self._policies[-M]] * (T - M) + list(self._policies[-M:])
    return TPolicy(new_policies)

  def to_numpy(self) -> NDArray[np.float64]:
    return np.array([p.to_numpy() for p in self._policies])

  @property
  def max_steps(self) -> int:
    return len(self._policies)

  @property
  def num_states(self) -> int:
    return self._policies[0].num_states

  @classmethod
  def from_TQ(cls, TQ: NDArray[np.float64]) -> "TPolicy":
    """Returns a TPolicy from a TQ matrix.

    Each policy in the TPolicy is built from a Q matrix with shape
    (num_states, num_actions).

    Args:
      TQ: The TQ matrix, with shape (max_steps, num_states, num_actions).
    """
    assert TQ.ndim == 3
    return TPolicy([Policy(Q=Q) for Q in TQ])


def get_policy_action_from_Q(
    Q: NDArray[np.float64],
    s_idx: int,
) -> Action:
  """Gets the action the policy would take at s_idx, given the Q matrix."""
  assert Q.ndim == 2
  return _get_policy_action_from_q_list(Q[s_idx, :])


def random_policy(num_states: int) -> Policy:
  actions = np.random.choice(data.AGENT_ACTIONS, size=num_states)
  return Policy(actions=actions)


def random_t_policy(num_states: int, rollout_step_limit: int) -> TPolicy:
  return TPolicy([random_policy(num_states) for _ in range(rollout_step_limit)])


def _get_policy_action_from_q_list(qs: NDArray[np.float64]) -> Action:
  # Choose a random index from the actions with the maximum Q value.
  # (This is essentially argmax, but with ties broken randomly.)
  assert qs.ndim == 1
  a = data.AGENT_ACTIONS[np.random.choice(np.flatnonzero(qs == np.max(qs)))]
  return a


def _normalize_distribution(
    distribution: Sequence[float],
) -> Sequence[float]:
  """Normalizes the distribution, making sure the sum equals 1."""
  # Copy the distribution
  distribution = list(distribution)
  # Assert that the sum is close to 1.
  distribution_sum = np.sum(distribution)
  assert np.abs(1.0 - distribution_sum) < 0.001
  # Make it sum to 1 if it doesn't quite already.
  distribution[np.argmax(distribution)] += 1.0 - distribution_sum
  return distribution
