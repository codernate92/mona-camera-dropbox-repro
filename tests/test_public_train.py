"""Sanity checks ported from the public MONA training setup."""

from __future__ import annotations

from mona.src import state as state_lib
from mona.src import train


def _str_to_index(mat_constructor, state: str) -> int:
  return mat_constructor.get_state_index(state_lib.State.from_string(state))


def test_good_value_function_matches_public_reference_state_value():
  trainer = train.Trainer(
      board_shape=(3, 3),
      max_boxes=2,
      min_blocking_boxes=1,
      per_step_penalty=0.01,
      episode_step_limit=50,
  )
  mat_constructor = trainer.get_matrix_constructor()
  good_result = trainer.get_training_result(reward_function="good")
  state_index = _str_to_index(
      mat_constructor,
      """
      #####
      #  A#
      #BB #
      #H  #
      #####
      """,
  )
  assert good_result.TV[0, state_index] == 1 - 0.03
