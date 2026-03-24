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


class StateTest(unittest.TestCase):

  def test_state_from_string(self):
    self.assertEqual(
        state_lib.State(
            np.array([
                [" ", "A", " "],
                ["B", "B", " "],
                [" ", " ", " "],
            ])
        ),
        state_lib.State.from_string("""
        #####
        # A #
        #BB #
        #H  #
        #####
        """),
    )

  def test_ended_state_from_string(self):
    self.assertEqual(
        state_lib.State.get_ended(),
        state_lib.State.from_string("ENDED"),
    )

  def test_state_equality(self):
    state_1 = state_lib.State.from_string("""
        #####
        # A #
        #BB #
        #H  #
        #####
        """)
    state_2 = state_lib.State.from_string("""
        #####
        # A #
        #B  #
        #HB #
        #####
        """)
    ended_state = state_lib.State.get_ended()

    self.assertEqual(state_1, state_1)
    self.assertEqual(ended_state, ended_state)
    self.assertNotEqual(state_1, state_2)
    self.assertNotEqual(state_1, ended_state)


if __name__ == "__main__":
  unittest.main()
