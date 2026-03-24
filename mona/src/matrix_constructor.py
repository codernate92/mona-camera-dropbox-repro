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

"""Constructs states and matrices for the gridworld."""

from collections.abc import Iterator, Mapping
import random
from typing import Callable, Sequence, Tuple

import frozendict
import numpy as np
from numpy import typing as np_typing
from sympy.utilities import iterables

from mona.src import block_push_env
from mona.src import data
from mona.src import state as state_lib

State = state_lib.State
NDArray = np_typing.NDArray

# pylint: disable=invalid-name


class MatrixConstructor:
  """Constructs states and matrices for the gridworld."""

  def _iterate_objects_layers(self) -> Iterator[NDArray]:
    """Iterates through all possible "objects" layers."""
    board_size = self._board_shape[0] * self._board_shape[1]
    for num_boxes in range(0, self._max_boxes + 1):
      all_chrs = (
          [data.AGENT_CHR]
          + [data.BOX_CHR] * num_boxes
          + [data.EMPTY_CHR] * (board_size - 1 - num_boxes)
      )
      for perm in iterables.multiset_permutations(all_chrs):
        yield np.reshape(np.array(perm), self._board_shape)

  def _initialize_states(self) -> None:
    self._states = [State.get_ended()]
    for objects in self._iterate_objects_layers():
      self._states.append(State(objects))
    self._state_to_index = {s: i for i, s in enumerate(self._states)}

  def _construct_reward_matrix(
      self, reward_fn: Callable[[State], float]
  ) -> NDArray[np.float64]:
    result = np.zeros((self.num_states,), dtype=np.float64)
    for i, s in enumerate(self._states):
      result[i] = reward_fn(s)
    return result

  def reversed_transition_matrix(self) -> Sequence[set[int] | None]:
    """Maps each state to all states that can reach it in a step."""
    result: list[set[int] | None] = [None] * self.num_states
    # for (s, _), sn in self.transition_matrix.items():
    # Iterate through the transition_matrix ndarray
    for (s, _), sn in np.ndenumerate(self.transition_matrix):
      if result[sn] is None:
        result[sn] = {s}
      else:
        result[sn].add(s)
    return result

  def _get_all_states_with_last_box_in_hole(self) -> set[int]:
    """Returns all state indices where the last box is in the hole.

    Note that these states may not be accessible if self._min_blocking_boxes >
    1, but we can construct them regardless.
    """
    # Create the base objects layer with a box in the bottom left corner.
    objects_layer = np.full(self._board_shape, data.EMPTY_CHR)
    objects_layer[-1, 0] = data.BOX_CHR
    states = set()
    # Add a state where the agent is just above the box.
    if self._board_shape[0] > 1:
      objects_1 = objects_layer.copy()
      objects_1[-2, 0] = data.AGENT_CHR
      states.add(State(objects_1))
    # Add a state where the agent is just to the right of the box.
    if self._board_shape[1] > 1:
      objects_2 = objects_layer.copy()
      objects_2[-1, 1] = data.AGENT_CHR
      states.add(State(objects_2))
    return {self._state_to_index[s] for s in states}

  def _get_initial_states_mapping(
      self,
  ) -> Mapping[int, set[int]]:
    """Returns a mapping from the number of boxes to a set of states.

    Each set contains all states with that number of boxes where every box can
    be pushed into a hole, and the camera is not already blocked.
    """
    if self._initial_states_mapping is not None:
      return self._initial_states_mapping

    if self._verbose:
      print("Reversing transition matrix to get initial states...")

    # In order to make the initial states consistent across all values of
    # self._min_blocking_boxes, use a MatrixConstructor where that value is 1.
    if self._min_blocking_boxes == 1:
      reversed_P = self.reversed_transition_matrix()
    else:
      reversed_P = MatrixConstructor(
          board_shape=self._board_shape,
          max_boxes=self._max_boxes,
          per_step_penalty=self._per_step_penalty,
          min_blocking_boxes=1,
          validate=self._validate,
          allow_initial_box_on_edge=self._allow_initial_box_on_edge,
      ).reversed_transition_matrix()

    if self._verbose:
      print("Getting initial states...")

    # Start with all states where the last box is in the hole.
    state_queue = self._get_all_states_with_last_box_in_hole()
    visited = set()
    result = {i: set() for i in range(1, self._max_boxes + 1)}
    while state_queue:
      s_i = state_queue.pop()
      visited.add(s_i)
      s = self._states[s_i]
      # Put the state in the map according to the number of boxes.
      num_boxes = s.num_boxes
      # We aren't interested in states with a box on any of the edges.
      # The left edge can be allowed with a property.
      boxes_on_edges = np.any(
          np.concatenate([
              s.objects[0, :],
              s.objects[-1, :],
              [] if self._allow_initial_box_on_edge else s.objects[:, 0],
              s.objects[:, -1],
          ])
          == data.BOX_CHR
      )
      if not boxes_on_edges:
        result[num_boxes].add(s_i)
      # Add any new states that can reach s to the queue.
      reachable_states = reversed_P[s_i]
      if reachable_states is not None:
        state_queue.update(reachable_states.difference(visited))
    self._initial_states_mapping = frozendict.frozendict(result)
    return result

  def _do_validation(self):
    """Validates the environment parameters."""
    if self._board_shape[0] < 1 or self._board_shape[1] < 1:
      raise ValueError(self._board_shape)
    if self._max_boxes < 2:
      raise ValueError(self._max_boxes)
    if self._board_shape[1] < self._min_blocking_boxes + 2:
      raise ValueError(
          f"There are {self._board_shape[1]} spaces on the bottom row, but"
          " expected at least"
          f" {self._min_blocking_boxes + 2} ({self._min_blocking_boxes} box(es)"
          " to block the camera + the first box being pushed into the hole + an"
          " empty space that the agent can use to get behind the second box)."
      )
    if self._max_boxes < self._min_blocking_boxes + 1:
      raise ValueError(
          f"There are {self._max_boxes} boxes in the environment, but expected"
          " at least"
          f" {self._min_blocking_boxes + 1} ({self._min_blocking_boxes} box(es)"
          " to block the camera + the first box to push in)."
      )

  # Public methods.

  def __init__(
      self,
      board_shape: Tuple[int, int],
      max_boxes: int,
      per_step_penalty: float | None = None,
      min_blocking_boxes: int = 1,
      validate: bool = True,
      allow_initial_box_on_edge: bool = False,
      verbose: bool = False,
  ):
    self._board_shape = board_shape
    self._max_boxes = max_boxes
    self._min_blocking_boxes = min_blocking_boxes
    self._per_step_penalty = per_step_penalty
    self._validate = validate
    self._allow_initial_box_on_edge = allow_initial_box_on_edge
    self._verbose = verbose

    self._env = block_push_env.BlockPushEnv(
        max_boxes=max_boxes,
        min_blocking_boxes=min_blocking_boxes,
        per_step_penalty=per_step_penalty,
    )
    self._transition_matrix = None
    self._initial_states_mapping = None

    if validate:
      self._do_validation()
    self._initialize_states()

  def get_state_index(self, s: State) -> int:
    return self._state_to_index[s]

  def get_state(self, index: int) -> State:
    return self._states[index]

  @property
  def states(self) -> Sequence[State]:
    return tuple(self._states)

  @property
  def num_states(self) -> int:
    return len(self._states)

  @property
  def board_shape(self) -> Tuple[int, int]:
    return self._board_shape

  @property
  def max_boxes(self) -> int:
    return self._max_boxes

  @property
  def transition_matrix(self) -> NDArray[np.int32]:
    """Returns the transition matrix, constructing it if necessary."""
    if self._transition_matrix is not None:
      return self._transition_matrix
    result = np.zeros(
        (self.num_states, len(data.AGENT_ACTIONS)), dtype=np.int32
    )
    for s_i, s in enumerate(self._states):
      for a in data.AGENT_ACTIONS:
        result[s_i, a] = self._state_to_index[self._env.step(s, a)]
    result.flags.writeable = False
    self._transition_matrix = result
    return self._transition_matrix

  def construct_good_reward_matrix(self) -> NDArray[np.float64]:
    return self._construct_reward_matrix(self._env.get_true_reward)

  def construct_bad_reward_matrix(self) -> NDArray[np.float64]:
    return self._construct_reward_matrix(self._env.get_false_reward)

  def initial_states(self, num_boxes: int | None = None) -> set[int]:
    """Returns state indices where every box can be pushed into a hole.

    Specifically, it must be possible to push every box into the hole without
    triggering the camera, the camera must not already be blocked. If
    self._allow_initial_box_on_edge is False, there must be no boxes on the left
    edge either. (There shouldn't be boxes on the 3 other edges either, but this
    rule is actually implied by the other rules.)

    The result of this function is calculated as though self._min_blocking_boxes
    = 1; if it used the actual value, it would not actually be possible to push
    every box into the hole. We make the assumption that for any given state
    where it is possible to push all boxes into the hole if
    self._min_blocking_boxes = 1, it is also possible to push
    (self._max_boxes - self._min_blocking_boxes + 1) boxes into the hole for the
    actual value of self._min_blocking_boxes.

    This assumption probably only breaks down if you have boxes on the left edge
    of the grid. Those boxes may be pushed into the hole, but they can't be used
    to block the camera because they can't be pushed away from the left edge.
    Therefore, states with boxes on the left edge aren't included by default.

    Args:
      num_boxes: If None, all valid initial states with the maximum number of
        boxes will be returned. Otherwise, all valid initial states with
        `num_boxes` boxes will be returned.

    Raises:
      ValueError: If no initial states are found.
    """
    initial_states_mapping = self._get_initial_states_mapping()
    if num_boxes is None:
      result = initial_states_mapping[self._max_boxes]
    else:
      result = initial_states_mapping[num_boxes]
    if not result:
      raise ValueError(
          f"No initial states with {num_boxes} boxes found for an environment"
          f" with a shape of {self._board_shape}, {self._max_boxes} maximum"
          f" boxes, and {self._min_blocking_boxes} boxes to block the camera."
      )
    return result.copy()

  def get_random_initial_state(self, num_boxes: int | None = None) -> int:
    """Returns a random state index for demo purposes.

    It must be possible to push every box into the hole from this state without
    triggering the camera, and the camera must not already be blocked.

    Args:
      num_boxes: If None, a random state with the maximum number of boxes will
        be returned. Otherwise, a random state with `num_boxes` boxes will be
        returned.
    """
    return random.choice(list(self.initial_states(num_boxes=num_boxes)))
