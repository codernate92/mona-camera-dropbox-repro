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

import numpy as np

from mona.src import state as state_lib
from mona.src import train

State = state_lib.State

# pylint: disable=invalid-name


class TrainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.trainer = train.Trainer(
        board_shape=(3, 3),
        max_boxes=2,
        min_blocking_boxes=1,
        per_step_penalty=0.01,
        episode_step_limit=50,
    )
    self.mat_constructor = self.trainer.get_matrix_constructor()
    self.good_result = self.trainer.get_training_result(reward_function="good")
    self.bad_result = self.trainer.get_training_result(reward_function="bad")
    self.good_to_bad_result = self.trainer.get_training_result(
        reward_function="bad", init_training_result=self.good_result
    )

  def str_to_index(self, s: str) -> int:
    return self.mat_constructor.get_state_index(State.from_string(s))

  def test_get_matrices_PRV_good_1(self):
    self.assertAlmostEqual(
        self.good_result.TV[
            0,
            self.str_to_index("""
              #####
              #  A#
              #BB #
              #H  #
              #####
              """),
        ],
        # 3 steps until the box is about to be in the hole.
        1 - 0.03,
    )

  def test_get_matrices_PRV_good_2(self):
    self.assertAlmostEqual(
        self.good_result.TV[
            0,
            self.str_to_index("""
              #####
              #A  #
              #BB #
              #H  #
              #####
              """),
        ],
        1 - 0.01,
    )

  def test_get_matrices_PRV_bad_1(self):
    self.assertAlmostEqual(
        self.bad_result.TV[
            0,
            self.str_to_index("""
              #####
              #  A#
              #BB #
              #H  #
              #####
              """),
        ],
        # 9 steps until the second box is in the hole.
        2 - 0.09,
    )

  def test_get_matrices_PRV_bad_2(self):
    self.assertAlmostEqual(
        self.bad_result.TV[
            0,
            self.str_to_index("""
              #####
              #A  #
              #B  #
              #H  #
              #####
              """),
        ],
        1 - 0.01,
    )

  def test_get_matrices_PRV_TV_last_step(self):
    self.assertEqual(
        self.good_result.TV[
            -1,
            self.str_to_index("""
              #####
              #A  #
              #B  #
              #H  #
              #####
              """),
        ],
        0,
    )

  def test_get_matrices_PRV_TV_last_step_ending(self):
    self.assertEqual(
        self.good_result.TV[
            -1,
            self.str_to_index("""
              #####
              #   #
              #A  #
              #B  #
              #####
              """),
        ],
        0,
    )

  # The value of the state should transition from the value under the good
  # reward function to the value under the bad reward function.
  def test_get_matrices_PRV_good_to_bad(self):
    actual_values = []
    for V in self.good_to_bad_result.TV:
      actual_values.append(V[self.str_to_index("""
          #####
          #  A#
          #BB #
          #H  #
          #####
          """)])
    # After 3 steps, one box can get pushed into the hole. After 9 steps, both
    # boxes can get pushed into the hole.
    bad_value = (2 - 0.09) - (1 - 0.03)
    expected_values = [bad_value] * (50 - 9) + [0.0] * 9
    for i, (actual_v, expected_v) in enumerate(
        zip(actual_values, expected_values)
    ):
      self.assertAlmostEqual(
          actual_v,
          expected_v,
          None,
          f"Index {i} failed.\n"
          f"Actual: {np.array(actual_values).tolist()}\n"
          f"Expected: {np.array(expected_values).tolist()}",
      )

  def test_get_advantage_function_correct_shape(self):
    P = self.good_result.P
    R = self.good_result.R
    trusted_TV = self.good_result.TV

    TA = train.get_advantage_function(P, R, trusted_TV, 50)

    num_states = self.mat_constructor.num_states
    self.assertEqual(TA.shape, (50, num_states, 4))

  def test_get_advantage_function_good_approval(self):
    P = self.good_result.P
    R = self.bad_result.R
    trusted_TV = self.good_result.TV

    TA = train.get_advantage_function(P, R, trusted_TV, 50)

    actual_advantages = TA[
        0,
        self.str_to_index("""
          #####
          #   #
          # B #
          #HBA#
          #####
          """),
    ]
    # Going up would delay the good policy from getting the second box in the
    # hole by two steps, and hitting the wall would delay it by one step.
    expected_advantages = np.array(
        [-0.02, -0.01, 0.0, -0.01],  # UP, DOWN, LEFT, RIGHT
    )
    self.assertTrue(
        np.allclose(actual_advantages, expected_advantages),
        msg=(
            f"Actual: {actual_advantages.tolist()}\n"
            f"Expected: {expected_advantages.tolist()}"
        ),
    )

  def test_get_advantage_function_bad_approval(self):
    P = self.good_result.P
    R = self.bad_result.R
    trusted_TV = self.bad_result.TV

    TA = train.get_advantage_function(P, R, trusted_TV, 50)

    actual_advantages = TA[
        0,
        self.str_to_index("""
          #####
          #   #
          # B #
          #HBA#
          #####
          """),
    ]
    # The bad policy wants to push the second box into the hole, but it can't do
    # that if we go left here, so it would lose 1 reward.
    # (Although it takes fewer steps, which increases its reward by 0.08.)
    expected_advantages = np.array(
        [0.00, -0.01, -0.92, -0.01],  # UP, DOWN, LEFT, RIGHT
    )
    self.assertTrue(
        np.allclose(actual_advantages, expected_advantages),
        msg=(
            f"Actual: {actual_advantages.tolist()}\n"
            f"Expected: {expected_advantages.tolist()}"
        ),
    )

  def test_get_advantage_function_approval_conflicts_with_reward(self):
    P = self.good_result.P
    R = self.bad_result.R
    trusted_TV = self.good_result.TV

    TA = train.get_advantage_function(P, R, trusted_TV, 50)

    actual_advantages = TA[
        0,
        self.str_to_index("""
          #####
          #A  #
          #B  #
          #H  #
          #####
          """),
    ]
    # The current value of this state is -0.49, because the good value function
    # believes that it has to wait for 49 more steps. The actual reward when
    # pushing the box down is 1, and the value in the following state is -0.01.
    # So the advantage for DOWN is -0.01 - (-0.49) + 1 = 1.48. All other actions
    # have zero advantage, because the good policy is indifferent between them.
    expected_advantages = np.array(
        [0.0, 1.48, 0.0, 0.0],  # UP, DOWN, LEFT, RIGHT
    )
    self.assertTrue(
        np.allclose(actual_advantages, expected_advantages),
        msg=(
            f"Actual: {actual_advantages.tolist()}\n"
            f"Expected: {expected_advantages.tolist()}"
        ),
    )


if __name__ == "__main__":
  unittest.main()
