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

from mona.src import data
from mona.src import policy_constructor

# pylint: disable=invalid-name


class PolicyConstructorTest(unittest.TestCase):

  def test_get_policy_from_matrices(self):
    policy = policy_constructor.Policy(
        # Simple environment with 4 states:
        # 0 1 2 3
        P=np.array(
            [
                # Moving left and right moves the agent to the adjacent state.
                # Moving up and down moves the agent back to 0.
                # All actions from the rightmost state move the agent to 0,
                # except for ACTION_LEFT, which moves the agent to 2.
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 1, 3],
                [0, 0, 2, 0],
            ],
            dtype=np.int32,
        ),
        V=np.array([0.4, 0.6, 0.8, 1.0]),
    )
    # The agent should move back and forth between the rightmost and
    # second-to-right states.
    self.assertEqual(
        [policy[i] for i in range(4)],
        [
            data.Action.RIGHT.value,
            data.Action.RIGHT.value,
            data.Action.RIGHT.value,
            data.Action.LEFT.value,
        ],
    )

  def test_make_policy_stochastic_with_stochastic_input(self):
    policy = policy_constructor.Policy(
        action_probs=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    self.assertEqual(
        policy._action_probs,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

  def test_make_policy_stochastic_adjusts_probabilities(self):
    policy = policy_constructor.Policy(
        action_probs=[
            [0.9999, 0.0, 0.0, 0.0],
            [0.0, 1.0001, 0.0, 0.0],
            [0.1, 0.1, 0.6999, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    for actual_a_probs, expected_a_probs in zip(
        policy._action_probs,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.1, 0.7, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ):
      self.assertTrue(np.allclose(actual_a_probs, expected_a_probs))

  def test_make_policy_fn_with_deterministic_input(self):
    policy = policy_constructor.Policy(
        actions=[
            data.Action.UP.value,
            data.Action.LEFT.value,
            data.Action.DOWN.value,
            data.Action.RIGHT.value,
        ]
    )
    self.assertEqual(
        [policy[s] for s in range(4)],
        [0, 2, 1, 3],
    )

  def test_make_policy_fn_with_stochastic_input(self):
    policy = policy_constructor.Policy(
        action_probs=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    self.assertEqual(
        [policy[s] for s in range(4)],
        [0, 2, 1, 3],
    )


if __name__ == "__main__":
  unittest.main()
