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

"""Given a policy, get a rollout of the policy for each initial state.

Records statistics about each rollout and/or the aggregate set of rollouts.
"""

from collections.abc import Sequence

import numpy as np

from mona.proto import rollout_pb2
from mona.src import matrix_constructor
from mona.src import policy_constructor


class RolloutHandler:
  """Get rollout data for a given policy."""

  def __init__(self, mat_constructor: matrix_constructor.MatrixConstructor):
    self._mat_constructor = mat_constructor
    self._transition_matrix = mat_constructor.transition_matrix
    self._bad_reward_matrix = mat_constructor.construct_bad_reward_matrix()

  def get_rollout(
      self,
      initial_state: int,
      policy: policy_constructor.TPolicy,
      granularity: int = 3,
  ) -> rollout_pb2.Rollout:
    """Get a rollout for each initial state."""
    steps_to_box_milestones = dict()
    num_boxes_in_hole = 0
    trajectory = [initial_state]
    s_idx = initial_state

    for t in range(policy.max_steps):
      a = policy[(t, s_idx)]
      sn_idx = self._transition_matrix[s_idx, a]
      trajectory.append(sn_idx)
      sn = self._mat_constructor.get_state(sn_idx)

      if sn.ended:
        break

      # Reward can only come from pushing a box into the hole.
      if self._bad_reward_matrix[sn_idx] > 0:
        num_boxes_in_hole += 1
        steps_to_box_milestones[num_boxes_in_hole] = len(trajectory) - 1
      s_idx = sn_idx

    return rollout_pb2.Rollout(
        initial_state=initial_state,
        steps_to_box_milestones=steps_to_box_milestones,
        num_boxes_in_hole=num_boxes_in_hole,
        trajectory=trajectory if granularity >= 3 else None,
    )

  def get_rollout_iteration(
      self,
      policy: policy_constructor.TPolicy,
      max_rollouts: int | None = None,
      granularity: int = 1,
  ) -> rollout_pb2.RolloutIteration:
    """Get a rollout for each initial state."""
    if granularity < 1:
      raise ValueError("Granularity must be at least 1.")
    iteration = rollout_pb2.RolloutIteration()

    # Get a rollout for each initial state.
    initial_states = self._mat_constructor.initial_states()
    # Choose `max_rollouts` initial states.
    if max_rollouts is not None:
      initial_states = np.random.choice(
          list(initial_states),
          size=min(max_rollouts, len(initial_states)),
          replace=False,
      )
    for initial_state in initial_states:
      rollout = self.get_rollout(initial_state, policy, granularity)
      iteration.rollouts.append(rollout)

    # Get rollout stats.
    box_milestones: list[rollout_pb2.RolloutStats.BoxMilestone] = []
    for rollout in iteration.rollouts:
      highest_milestone = max(
          [0] + list(rollout.steps_to_box_milestones.keys())
      )
      while len(box_milestones) < highest_milestone + 1:
        box_milestones.append(rollout_pb2.RolloutStats.BoxMilestone())
      box_milestones[highest_milestone].num_initial_states += 1

      for num_boxes, num_steps in rollout.steps_to_box_milestones.items():
        box_milestone = box_milestones[num_boxes]
        while len(box_milestone.step_counts) < num_steps + 1:
          box_milestone.step_counts.append(0)
        box_milestone.step_counts[num_steps] += 1

    average_boxes_in_hole = sum(
        i * milestone.num_initial_states
        for i, milestone in enumerate(box_milestones)
    ) / len(initial_states)

    iteration.stats.CopyFrom(
        rollout_pb2.RolloutStats(
            num_initial_states=len(initial_states),
            average_boxes_in_hole=average_boxes_in_hole,
            box_milestones=box_milestones,
        )
    )
    if granularity <= 1:
      iteration.ClearField("rollouts")

    return iteration

  def get_rollout_iterations(
      self,
      policies: Sequence[policy_constructor.TPolicy],
      max_rollouts: int | None = None,
      granularity: int = 1,
      verbose: bool = False,
  ) -> rollout_pb2.RolloutIterations:
    """Get a rollout for each initial state for each policy.

    The level of detail in the rollouts is determined by granularity.
      - granularity >= 1: Summary statistics for each iteration are included.
      - granularity >= 2: Summary statistics for each rollout are included.
      - granularity >= 3: The state trajectory of each rollout is included.

    Args:
      policies: The policies to get rollouts for.
      max_rollouts: The maximum number of rollouts to create, capped by the
        number of initial states. If None, there is no maximum.
      granularity: The level of detail in the rollouts. See above.
      verbose: Whether to print after each iteration.

    Returns:
      A RolloutIterations proto containing the rollouts for each policy.
    """
    iterations = []
    if verbose:
      print(f"Getting {len(policies)} rollout iterations:", end="", flush=True)
    for i, policy in enumerate(policies):
      iterations.append(
          self.get_rollout_iteration(
              policy,
              max_rollouts=max_rollouts,
              granularity=granularity,
          )
      )
      if verbose:
        print(f" {i+1}", end="", flush=True)
    if verbose:
      print()
    return rollout_pb2.RolloutIterations(iterations=iterations)
