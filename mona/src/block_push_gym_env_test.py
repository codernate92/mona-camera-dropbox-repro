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

import unittest
from gymnasium import spaces
import numpy as np
from mona.src import block_push_gym_env
from mona.src import data
from mona.src import state as state_lib

Action = data.Action


class BlockPushGymEnvTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = block_push_gym_env.BlockPushGymEnv(
        board_shape=(3, 4),
        use_good_reward=True,
    )

  def test_step(self):
    # Reset with a fixed seed to make the initial state deterministic.
    observation, info = self.env.reset(seed=0)

    self.assertEqual(
        self.env.observation_space,
        spaces.MultiDiscrete([51, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
    )

    t, state = self.env.observation_to_state(observation)

    self.assertEqual(t, 0)
    self.assertEqual(
        state,
        state_lib.State.from_string("""
                                    ######
                                    #    #
                                    # BB #
                                    #H A #
                                    ######
                                    """),
    )
    self.assertDictEqual(info, {})

    observation, reward, terminated, truncated, info = self.env.step(
        Action.UP.value
    )

    t, state = self.env.observation_to_state(observation)
    self.assertEqual(t, 1)
    self.assertEqual(
        state,
        state_lib.State.from_string("""
                                    ######
                                    #  B #
                                    # BA #
                                    #H   #
                                    ######
                                    """),
    )
    self.assertEqual(reward, -0.01)
    self.assertFalse(terminated)
    self.assertFalse(truncated)
    self.assertDictEqual(info, {})

  def test_step_with_reward_override(self):
    reward_override = np.ones_like(self.env._reward_matrix) * 0.12345
    env = block_push_gym_env.BlockPushGymEnv(
        board_shape=(3, 4),
        use_good_reward=False,
        reward_override=reward_override,
    )
    _, reward, _, _, _ = env.step(Action.UP.value)
    self.assertEqual(reward, 0.12345)

  def test_observation_to_state(self):
    with self.subTest("ended"):
      observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      _, state = self.env.observation_to_state(observation)
      self.assertEqual(state.ended, True)

    with self.subTest("not_ended"):
      observation = np.array(
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0]
      )  # Agent and Box
      _, state = self.env.observation_to_state(observation)
      self.assertEqual(state.ended, False)

  def test_state_to_observation(self):
    state = state_lib.State.get_ended()
    obs = self.env.state_to_observation(t=0, s=state)
    self.assertTrue(
        np.all(obs == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    )

    state = state_lib.State.from_string("""
                                    ######
                                    #    #
                                    # BB #
                                    #H A #
                                    ######
                                    """)
    obs = self.env.state_to_observation(t=5, s=state)
    expected_obs = np.array([5, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0])
    self.assertTrue(np.all(obs == expected_obs))

    timeless_obs = self.env.state_to_observation(t=None, s=state)
    expected_timeless_obs = np.array([0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 0])
    self.assertTrue(np.all(timeless_obs == expected_timeless_obs))

  def test_step_after_last_timestep_fails(self):
    for _ in range(self.env._episode_step_limit):
      self.env.step(Action.UP.value)
    with self.assertRaisesRegex(ValueError, "Episode step limit reached"):
      self.env.step(Action.UP.value)

  def test_set_and_get_time(self):
    self.env.set_t(5)
    self.assertEqual(self.env.get_time(), 5)

  def test_set_and_get_observation(self):
    self.env.reset(seed=0)
    initial_observation = self.env.get_observation()
    self.env.set_s_idx(0)
    new_observation = self.env.get_observation()
    self.assertFalse(np.all(initial_observation == new_observation))
    self.env.set_s_idx(self.env._s_idx)  # set the state to the current one
    timeless_obs = self.env.get_observation(timeless=True)
    self.assertEqual(len(timeless_obs), 12)


if __name__ == "__main__":
  unittest.main()
