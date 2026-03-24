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
from mona.src import runner as runner_lib


class RunnerTest:

  @pytest.mark.parametrize(
      "reward_function, expected_policy_name",
      [
          ("good", "good"),
          ("bad", "bad"),
      ],
      ids=[
          "good_reward",
          "bad_reward",
      ],
  )
  def test_run_name_is_correct(
      self, reward_function: str, expected_policy_name: str
  ):
    runner = runner_lib.Runner(
        board_shape=(3, 3),
        max_boxes=2,
        min_blocking_boxes=1,
        per_step_penalty=0.01,
        data_dir="",
    )
    result = runner.run(runner_lib.RunParams(reward_function=reward_function))
    self.assertEqual(result.policy_name, expected_policy_name)
