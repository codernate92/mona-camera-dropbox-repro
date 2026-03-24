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

"""A wrapper around the gridworld environment for use with gymnasium."""

import random
from typing import Any, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy import typing as np_typing

from mona.src import data
from mona.src import matrix_constructor
from mona.src import state as state_lib

NDArray = np_typing.NDArray
Action = int


class BlockPushGymEnv(gym.Env):
  """Block push environment. Compatible with gymnasium.

  Works with observations, which are 1D Numpy arrays that the model uses during
  training, instead of states from the state library.
  """

  metadata = {'render.modes': ['human']}
  TILE_ORDER = [data.EMPTY_CHR, data.AGENT_CHR, data.BOX_CHR]

  def __init__(
      self,
      board_shape: Tuple[int, int],
      use_good_reward: bool,
      max_boxes: int = 2,
      per_step_penalty: float = 0.01,
      min_blocking_boxes: int = 1,
      episode_step_limit: int = 50,
      reward_override: NDArray[np.float64] | None = None,
  ):
    """Initializes the environment.

    Args:
      board_shape: The shape of the board.
      use_good_reward: Whether to use the good reward function or the bad reward
        function. Ignored if `reward_override` is provided.
      max_boxes: The maximum number of boxes on the board.
      per_step_penalty: The per-step penalty to apply to the reward.
      min_blocking_boxes: The minimum number of boxes that must be used to block
        the camera.
      episode_step_limit: The maximum number of steps in an episode.
      reward_override: An optional fully-specified reward matrix with shape
        (num_timesteps, num_states). If provided, `use_good_reward` is ignored.
    """
    super().__init__()
    self._board_shape = board_shape
    self._max_boxes = max_boxes
    self._per_step_penalty = per_step_penalty
    self._min_blocking_boxes = min_blocking_boxes
    self._use_good_reward = use_good_reward
    self._episode_step_limit = episode_step_limit

    self._mat_constructor = matrix_constructor.MatrixConstructor(
        board_shape=board_shape,
        max_boxes=max_boxes,
        per_step_penalty=per_step_penalty,
        min_blocking_boxes=min_blocking_boxes,
    )

    self._transition_matrix = self._mat_constructor.transition_matrix

    if reward_override is None:
      self._reward_matrix_per_state = (
          self._mat_constructor.construct_good_reward_matrix()
          if use_good_reward
          else self._mat_constructor.construct_bad_reward_matrix()
      )
      self._reward_matrix = self._reward_matrix_per_state[
          self._mat_constructor.transition_matrix
      ]
      # Repeat the reward matrix for each timestep.
      self._reward_matrix = np.tile(
          self._reward_matrix, (self._episode_step_limit, 1, 1)
      )
    else:
      self._reward_matrix = reward_override
    expected_shape = (
        self._episode_step_limit,
        self._mat_constructor.num_states,
        len(data.AGENT_ACTIONS),
    )
    if self._reward_matrix.shape != expected_shape:
      raise ValueError(
          'Reward matrix shape must be %s, but is %s'
          % (
              str(expected_shape),
              str(self._reward_matrix.shape),
          )
      )

    self._timeless_observations = [
        self.state_to_observation(t=None, s=s)
        for s in self._mat_constructor.states
    ]
    self._initial_s_idxs = list(self._mat_constructor.initial_states())
    if not self._initial_s_idxs:
      raise ValueError(
          'No initial states in environment with shape %s and %d boxes.'
          % str(board_shape),
          self._max_boxes,
      )

    self.action_space = spaces.Discrete(len(data.AGENT_ACTIONS))
    self.observation_space = spaces.MultiDiscrete(
        # The first element is the time dimension.
        # We allow for one extra step, but this last step shouldn't be used; if
        # it is, we fail with a ValueError.
        [self._episode_step_limit + 1]
        # The rest of the elements represent the state.
        + [3] * (self._board_shape[0] * self._board_shape[1])
    )

    # Reset the environment to a random initial state at time t=0.
    self.reset()

  def observation_to_state(
      self, obs: NDArray[int], timeless: bool = False
  ) -> Tuple[int | None, state_lib.State]:
    """Returns the timestep and state for a given observation vector.

    Args:
      obs: The observation vector.
      timeless: If True, the time dimension of the observation is assumed to be
        omitted.

    Returns:
      t: The time step. If timeless is True, this will be None.
      s: The state.
    """
    # Separate the time and state in the observation.
    if timeless:
      t, timeless_obs = None, obs
    else:
      t, timeless_obs = obs[0], obs[1:]

    # Get the state.
    if np.all(timeless_obs == self.TILE_ORDER.index(data.EMPTY_CHR)):
      s = state_lib.State.get_ended()
    else:
      int_tiles = np.reshape(
          np.array(timeless_obs).astype(int), self._board_shape
      )
      chr_tiles = np.array(self.TILE_ORDER)[int_tiles]
      s = state_lib.State(objects=chr_tiles)

    return t, s

  def state_to_observation(
      self, t: int | None, s: state_lib.State
  ) -> NDArray[int]:
    """Returns the observation vector for a given timestep and state.

    If t is None, returns a "timeless" observation with a missing time
    dimension at the beginning.

    Args:
      t: The time step. If None, the time dimension is omitted.
      s: The state.
    """
    if s.ended:
      tiles = np.full(
          self._board_shape[0] * self._board_shape[1],
          self.TILE_ORDER.index(data.EMPTY_CHR),
          dtype=int,
      )
    else:
      tiles = np.array(
          [self.TILE_ORDER.index(tile) for tile in s.objects.flatten()],
          dtype=int,
      )

    if t is not None:
      return np.concatenate(([t], tiles))
    else:
      return tiles

  def set_t(self, t: int) -> None:
    self._t = t

  def set_s_idx(self, s_idx: int) -> None:
    self._s_idx = s_idx

  def get_observation(self, timeless: bool = False) -> NDArray[int]:
    if timeless:
      return self._timeless_observations[self._s_idx]
    return np.concatenate(([self._t], self._timeless_observations[self._s_idx]))

  def get_time(self) -> int:
    return self._t

  def get_state(self) -> state_lib.State:
    return self._mat_constructor.get_state(self._s_idx)

  def get_mat_constructor(self) -> matrix_constructor.MatrixConstructor:
    return self._mat_constructor

  def step(
      self, action: Action
  ) -> tuple[NDArray[int], float, bool, bool, dict[Any, Any]]:
    """Advances one step.

    Args:
      action: represents the direction that the agent is attempting to move.

    Returns:
      state: The state after the transition.
      reward: The reward for this transition.
      terminated: Whether the episode has ended.
      truncated: Whether the episode was truncated.
      info: Contains any additional information.
    """
    if self._t >= self._episode_step_limit:
      raise ValueError(
          'Episode step limit reached. Step limit: %d, current step: %d'
          % (self._episode_step_limit, self._t)
      )

    reward = self._reward_matrix[self._t, self._s_idx, action]

    self._s_idx = self._transition_matrix[(self._s_idx, action)]
    self._t += 1

    terminated = self.get_state().ended
    truncated = False
    info = {}
    return self.get_observation(), reward, terminated, truncated, info

  def reset(
      self, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[NDArray[int], dict[Any, Any]]:
    """Initial state reset."""
    random.seed(seed)
    self._s_idx = random.choice(self._initial_s_idxs)
    self._t = 0
    return self.get_observation(), {}

  def render(self, mode: str = 'human'):
    print(str(self._mat_constructor.get_state(self._s_idx)))
