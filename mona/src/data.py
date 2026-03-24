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

"""Provides basic data structures for the gridworld."""

import enum
from typing import Tuple

Pos = Tuple[int, int]


class Action(enum.Enum):
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3


AGENT_ACTIONS: Tuple[int, ...] = tuple(
    [a.value for a in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]]
)

AGENT_CHR = "A"
BORDER_CHR = "#"
BOX_CHR = "B"
HOLE_CHR = "H"
EMPTY_CHR = " "
