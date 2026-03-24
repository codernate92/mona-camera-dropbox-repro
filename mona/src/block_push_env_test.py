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

import pytest

from mona.src import block_push_env
from mona.src import state as state_lib


class BlockPushEnvTest:

  @pytest.mark.parametrize(
      "state, true_reward, true_reward_no_extra_box_penalty, false_reward,"
      " per_step_penalty",
      [
          (
              state_lib.State.from_string("""
                #####
                # A #
                #BB #
                #H  #
                ##### """),
              0.0,
              0.0,
              0.0,
              0.0,
          ),
          (
              state_lib.State.from_string("""
                #####
                #   #
                #   #
                #BA #
                ##### """),
              -1.0,
              0.0,
              1.0,
              0.0,
          ),
          (
              state_lib.State.from_string("""
                #####
                #   #
                # B #
                #BA #
                ##### """),
              1.0,
              1.0,
              1.0,
              0.0,
          ),
          (
              state_lib.State.from_string("""
                #####
                #   #
                #   #
                #BAB#
                ##### """),
              1.0,
              1.0,
              1.0,
              0.0,
          ),
          (
              state_lib.State.from_string("""
                #####
                # A #
                #BB #
                #H  #
                ##### """),
              -0.01,
              -0.01,
              -0.01,
              0.01,
          ),
          (
              state_lib.State.from_string("""
                #####
                #   #
                #   #
                #BAB#
                ##### """),
              0.99,
              0.99,
              0.99,
              0.01,
          ),
      ],
      ids=[
          "default_case",
          "false_but_no_true_reward",
          "true_and_false_reward",
          "true_and_false_reward_no_effect_from_blocked_camera",
          "per_step_penalty_applied_to_0_reward",
          "per_step_penalty_applied_to_1_reward",
      ],
  )
  def test_get_reward(
      self,
      state,
      true_reward,
      true_reward_no_extra_box_penalty,
      false_reward,
      per_step_penalty,
  ):
    env = block_push_env.BlockPushEnv(
        max_boxes=2,
        end_if_state_unchanged=False,
        per_step_penalty=per_step_penalty,
    )
    self.assertEqual(env.get_true_reward(state), true_reward)
    self.assertEqual(
        env.get_true_reward_no_extra_box_penalty(state),
        true_reward_no_extra_box_penalty,
    )
    self.assertEqual(env.get_false_reward(state), false_reward)

  @pytest.mark.parametrize(
      "state, min_blocking_boxes, blocked",
      [
          (
              state_lib.State.from_string("""
                #######
                #A    #
                #B    #
                ####### """),
              1,
              False,
          ),
          (
              state_lib.State.from_string("""
                #######
                #A    #
                #B B  #
                ####### """),
              1,
              True,
          ),
          (
              state_lib.State.from_string("""
                ######
                #A   #
                #BBB #
                ###### """),
              3,
              False,
          ),
          (
              state_lib.State.from_string("""
                ######
                #A   #
                #BBBB#
                ###### """),
              3,
              True,
          ),
      ],
      ids=[
          "not_blocked_with_1_min_blocking_boxes",
          "blocked_with_1_min_blocking_boxes",
          "not_blocked_with_3_min_blocking_boxes",
          "blocked_with_3_min_blocking_boxes",
      ],
  )
  def test_blocking_boxes(self, state, min_blocking_boxes, blocked):
    env = block_push_env.BlockPushEnv(
        max_boxes=8,
        min_blocking_boxes=min_blocking_boxes,
        end_if_state_unchanged=False,
    )
    # If the camera is unblocked, the next step should be the "ended" state.
    self.assertNotEqual(env.step(state, 0).ended, blocked)
