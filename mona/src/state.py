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

"""Represents a gridworld state.

This class is used to represent the state of the gridworld. It is used to
easily get useful information about a state, including a string representation.

Internally, it is a wrapper around a numpy array of characters, where each
character represents an object on the grid. The possible characters are defined
in data.py.
"""

from collections.abc import Sequence
from typing import Tuple
import numpy as np
from mona.src import data


def box_array(arr: np.ndarray) -> np.ndarray:
  boxed_arr = np.full((arr.shape[0] + 2, arr.shape[1] + 2), data.BORDER_CHR)
  boxed_arr[1:-1, 1:-1] = arr  # Place the original array in the center
  return boxed_arr


class State:
  """Represents a gridworld state."""

  def __str__(self):
    if self._objects is None:
      return 'ENDED'
    str_arr = self._objects
    if str_arr[self.hole_pos[0], self.hole_pos[1]] == data.EMPTY_CHR:
      str_arr = str_arr.copy()
      str_arr[self.hole_pos[0], self.hole_pos[1]] = data.HOLE_CHR
    boxed_arr = box_array(str_arr)
    return '\n'.join(''.join(l) for l in boxed_arr)

  def __repr__(self):
    if self.ended:
      inner_str = 'ENDED'
    else:
      inner_str = '\n' + str(self._objects) + '\n'
    return f'State({inner_str})'

  def __eq__(self, other: 'State') -> bool:
    if self._objects is not None and other._objects is not None:
      return np.array_equal(self._objects, other._objects)
    return self._objects is None and other._objects is None

  def __hash__(self):
    if self._objects is None:
      return 0
    return hash(self._objects.tobytes())

  # Public methods

  def __init__(self, objects: np.ndarray | None):
    self._objects = objects

  @property
  def hole_pos(self) -> Tuple[int, int]:
    return self.shape[0] - 1, 0

  @property
  def objects(self) -> np.ndarray:
    if self._objects is None:
      raise ValueError('State is ended, so objects are not available.')
    return self._objects

  @property
  def num_boxes(self) -> int:
    return np.count_nonzero(self._objects == data.BOX_CHR)

  @property
  def shape(self) -> tuple[int, int] | None:
    if self._objects is None:
      return None
    return self._objects.shape

  @property
  def ended(self) -> bool:
    return self._objects is None

  @classmethod
  def get_ended(cls) -> 'State':
    return cls(None)

  @classmethod
  def from_string(cls, string: str) -> 'State':
    """Returns a state from a string representation."""
    if string.lower() == 'ended':
      return cls.get_ended()
    # Ignore the hole if it appears.
    string = string.replace(data.HOLE_CHR, data.EMPTY_CHR)
    lines = string.split('\n')
    # Remove any leading or trailing whitespace in each line.
    lines = [l.strip() for l in lines]
    # Remove the bounding box.
    lines = [l.replace(data.BORDER_CHR, '') for l in lines]
    # Remove any empty lines (these might appear at the top and bottom if there
    # was a bounding box) and convert to a numpy array of characters.
    objects = np.array([list(l) for l in lines if l])
    return cls(objects)

  def find_many_in_objects(self, char: str) -> Sequence[Tuple[int, int]]:
    rows, cols = np.where(self._objects == char)
    return list(zip(rows, cols))

  def find_one_in_objects(self, char: str) -> Tuple[int, int]:
    results = self.find_many_in_objects(char)
    if len(results) != 1:
      raise ValueError(f'Expected 1 result, got {len(results)}')
    return results[0]
