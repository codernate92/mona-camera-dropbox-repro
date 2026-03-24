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

from typing import Any, Iterable
import unittest
from mona.src import matrix_constructor


def _state_indices_to_tuples(
    state_indices: Iterable[int],
    constructor: matrix_constructor.MatrixConstructor,
) -> set[tuple[tuple[str, ...], ...]]:
  """Returns the OBJECTS layer of each state, converted to a nested tuple."""
  return {
      tuple(
          map(
              tuple,
              constructor.get_state(s_i).objects,
          )
      )
      for s_i in state_indices
  }


class MatrixConstructorTest(unittest.TestCase):

  def assertContainsSubset(self, subset: set[Any], superset: set[Any]):
    self.assertSetEqual(subset.difference(superset), set())

  def assertNoCommonElements(self, a: set[Any], b: set[Any]):
    self.assertSetEqual(a.intersection(b), set())

  def test_initial_states(self):
    constructor = matrix_constructor.MatrixConstructor(
        board_shape=(3, 5), max_boxes=2
    )
    result = _state_indices_to_tuples(constructor.initial_states(), constructor)
    self.assertContainsSubset(
        {
            # Make sure that the set includes only valid states with 2 boxes.
            (
                (' ', 'A', ' ', ' ', ' '),
                (' ', 'B', 'B', ' ', ' '),
                (' ', ' ', ' ', ' ', ' '),
            ),
            (
                (' ', ' ', ' ', ' ', ' '),
                (' ', ' ', 'B', 'B', ' '),
                (' ', ' ', ' ', 'A', ' '),
            ),
        },
        result,
    )
    # Check that ineligible states are not included.
    self.assertNoCommonElements(
        result,
        {
            # The topmost box can't be pushed into the hole.
            (
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', ' ', 'A', ' '),
            ),
            # The leftmost box is on the left edge.
            (
                (' ', ' ', ' ', ' ', ' '),
                ('B', ' ', 'B', ' ', ' '),
                (' ', ' ', ' ', 'A', ' '),
            ),
            # The bottommost box is blocking the camera.
            (
                (' ', 'A', ' ', ' ', ' '),
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', 'B', ' ', ' '),
            ),
            # There is only one box.
            (
                (' ', ' ', ' ', 'A', ' '),
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', ' ', ' ', ' '),
            ),
        },
    )

    # There are 3 valid configurations with 2 boxes, since the boxes can be
    # anywhere in the middle 3 spaces. In each of these configurations, the
    # agent can be in any of the 15 - 2 = 13 positions unoccupied by a box.
    # 13 * 3 = 39 states.
    self.assertEqual(len(result), 39)

  def test_initial_states_with_allow_initial_box_on_edge(self):
    constructor = matrix_constructor.MatrixConstructor(
        board_shape=(3, 5),
        max_boxes=2,
        allow_initial_box_on_edge=True,
    )
    result = _state_indices_to_tuples(constructor.initial_states(), constructor)

    # Assert that there can be a box on the left edge.
    self.assertIn(
        (
            (' ', ' ', ' ', ' ', ' '),
            ('B', ' ', 'B', ' ', ' '),
            (' ', ' ', ' ', 'A', ' '),
        ),
        result,
    )
    # However, the other edges should still be disallowed for other reasons.
    self.assertNoCommonElements(
        result,
        {
            # The topmost box can't be pushed into the hole.
            (
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', ' ', 'A', ' '),
            ),
            # The rightmost box can't be pushed into the hole.
            (
                (' ', ' ', ' ', ' ', ' '),
                (' ', ' ', 'B', ' ', 'B'),
                (' ', ' ', ' ', 'A', ' '),
            ),
            # The bottommost box is blocking the camera.
            (
                (' ', ' ', ' ', ' ', ' '),
                (' ', ' ', 'B', ' ', ' '),
                (' ', ' ', 'B', 'A', ' '),
            ),
        },
    )

  def test_initial_states_with_num_boxes(self):
    constructor = matrix_constructor.MatrixConstructor(
        board_shape=(3, 4), max_boxes=3
    )
    self.assertSetEqual(
        {
            (
                (' ', 'A', ' ', ' '),
                (' ', 'B', 'B', ' '),
                (' ', ' ', ' ', ' '),
            ),
            (
                (' ', ' ', 'A', ' '),
                (' ', 'B', 'B', ' '),
                (' ', ' ', ' ', ' '),
            ),
            (
                (' ', ' ', ' ', 'A'),
                (' ', 'B', 'B', ' '),
                (' ', ' ', ' ', ' '),
            ),
            (
                (' ', ' ', ' ', ' '),
                (' ', 'B', 'B', 'A'),
                (' ', ' ', ' ', ' '),
            ),
            (
                (' ', ' ', ' ', ' '),
                (' ', 'B', 'B', ' '),
                (' ', ' ', ' ', 'A'),
            ),
            (
                (' ', ' ', ' ', ' '),
                (' ', 'B', 'B', ' '),
                (' ', ' ', 'A', ' '),
            ),
            (
                (' ', ' ', ' ', ' '),
                (' ', 'B', 'B', ' '),
                (' ', 'A', ' ', ' '),
            ),
            (
                (' ', ' ', ' ', ' '),
                (' ', 'B', 'B', ' '),
                ('A', ' ', ' ', ' '),
            ),
            (
                (' ', ' ', ' ', ' '),
                ('A', 'B', 'B', ' '),
                (' ', ' ', ' ', ' '),
            ),
            (
                ('A', ' ', ' ', ' '),
                (' ', 'B', 'B', ' '),
                (' ', ' ', ' ', ' '),
            ),
        },
        _state_indices_to_tuples(
            constructor.initial_states(num_boxes=2),
            constructor,
        ),
    )

  def test_initial_states_with_min_blocking_boxes(self):
    constructor = matrix_constructor.MatrixConstructor(
        board_shape=(3, 5), max_boxes=3, min_blocking_boxes=2
    )
    # The states where the final box is being pushed into the hole are not
    # accessible in this MatrixConstructor because the min_blocking_boxes means
    # that at least one box must remain. However, because we use a temporary
    # MatrixConstructor with min_blocking_boxes=1, this is okay.
    self.assertIn(
        (
            (' ', 'A', ' ', ' ', ' '),
            (' ', 'B', 'B', 'B', ' '),
            (' ', ' ', ' ', ' ', ' '),
        ),
        _state_indices_to_tuples(
            constructor.initial_states(),
            constructor,
        ),
    )


if __name__ == '__main__':
  unittest.main()
