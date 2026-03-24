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

from mona.proto import rollout_pb2
from mona.src import data
from mona.src import matrix_constructor
from mona.src import policy_constructor
from mona.src import rollout_handler as rollout_handler_lib
from mona.src import state as state_lib
from mona.src import train


class RolloutHandlerTest(unittest.TestCase):

  def assertLen(self, container, expected_len):
    self.assertEqual(len(container), expected_len)

  def assertEmpty(self, container):
    self.assertLen(container, 0)

  def assertNotEmpty(self, container):
    self.assertNotEqual(len(container), 0)

  def get_rollout_with_initial_state(self, rollouts, initial_state_index):
    for rollout in rollouts.rollouts:
      if rollout.initial_state == initial_state_index:
        return rollout
    self.fail(f"Could not find initial state index {initial_state_index}.")

  def test_get_rollout_iteration_opt(self):
    trainer = train.Trainer(
        per_step_penalty=0.01,
        board_shape=(3, 4),
        max_boxes=2,
        min_blocking_boxes=1,
    )
    mat_constructor = trainer.get_matrix_constructor()
    result = trainer.get_training_result(
        "bad", init_training_result=trainer.get_training_result("good")
    )

    opt_policy = policy_constructor.TPolicy.from_TQ(result.TQ)

    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)
    opt_rollouts = rollout_handler.get_rollout_iteration(
        opt_policy, granularity=2
    )

    initial_state_index = mat_constructor.get_state_index(
        state_lib.State.from_string("""
            ######
            #    #
            # BB #
            #H A #
            ######
            """)
    )

    # Assert that the rollout takes 9 steps to get one
    # box in the hole and 15 steps to get two boxes in the hole.
    rollout = self.get_rollout_with_initial_state(
        opt_rollouts, initial_state_index
    )
    self.assertLen(rollout.steps_to_box_milestones, 2)
    self.assertEqual(rollout.steps_to_box_milestones[1], 9)
    self.assertEqual(rollout.steps_to_box_milestones[2], 15)

    stats = opt_rollouts.stats
    # There are 12-2=10 spaces for the agent, and only one arrangement of boxes.
    self.assertEqual(stats.num_initial_states, 10)
    self.assertLen(stats.box_milestones, 3)
    expected_box_milestones = [
        rollout_pb2.RolloutStats.BoxMilestone(
            num_initial_states=0, step_counts=[]
        ),
        rollout_pb2.RolloutStats.BoxMilestone(
            num_initial_states=0, step_counts=[0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1]
        ),
        rollout_pb2.RolloutStats.BoxMilestone(
            num_initial_states=10,
            step_counts=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1],
        ),
    ]
    for i, expected in enumerate(expected_box_milestones):
      actual = stats.box_milestones[i]
      self.assertSequenceEqual(actual.step_counts, expected.step_counts)
      self.assertEqual(actual.num_initial_states, expected.num_initial_states)
    self.assertAlmostEqual(stats.average_boxes_in_hole, 2)

  def test_get_rollout_iteration_subopt(self):
    trainer = train.Trainer(
        per_step_penalty=0.01,
        board_shape=(3, 4),
        max_boxes=2,
        min_blocking_boxes=1,
    )
    mat_constructor = trainer.get_matrix_constructor()
    result = trainer.get_training_result(
        "bad", init_training_result=trainer.get_training_result("good")
    )

    # This policy can maximize its reward as long as it takes 14 steps or less.
    # The step with index 0 is step 1.
    subopt_policy_14 = policy_constructor.TPolicy.from_TQ(
        result.intermediate_TQs[14],
    )
    # This policy can maximize its reward as long as it takes 15 steps or less.
    subopt_policy_15 = policy_constructor.TPolicy.from_TQ(
        result.intermediate_TQs[15],
    )

    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)
    subopt_rollouts_14 = rollout_handler.get_rollout_iteration(
        subopt_policy_14, granularity=2
    )
    subopt_rollouts_15 = rollout_handler.get_rollout_iteration(
        subopt_policy_15, granularity=2
    )

    initial_state_index = mat_constructor.get_state_index(
        state_lib.State.from_string("""
            ######
            #    #
            # BB #
            #H A #
            ######
            """)
    )

    with self.subTest("subopt_14"):
      # Assert that the 14-step policy takes 1 step to get one box in the hole.
      rollout = self.get_rollout_with_initial_state(
          subopt_rollouts_14, initial_state_index
      )
      self.assertLen(rollout.steps_to_box_milestones, 1)
      self.assertEqual(rollout.steps_to_box_milestones[1], 5)

    with self.subTest("subopt_15"):
      # Assert that the 15-step policy takes 9 steps to get one
      # box in the hole and 15 steps to get two boxes in the hole.
      rollout = self.get_rollout_with_initial_state(
          subopt_rollouts_15, initial_state_index
      )
      self.assertLen(rollout.steps_to_box_milestones, 2)
      self.assertEqual(rollout.steps_to_box_milestones[1], 9)
      self.assertEqual(rollout.steps_to_box_milestones[2], 15)

    with self.subTest("stats"):
      for rollouts in [subopt_rollouts_14, subopt_rollouts_15]:
        stats = rollouts.stats
        self.assertEqual(stats.num_initial_states, 10)
        self.assertLen(stats.box_milestones, 3)
        self.assertEqual(
            sum(m.num_initial_states for m in stats.box_milestones), 10
        )

  def test_get_rollout_iteration_granularity_0_fails(self):
    vi_runner = train.Trainer(
        per_step_penalty=0.01,
        board_shape=(3, 4),
        max_boxes=2,
        min_blocking_boxes=1,
    )
    mat_constructor = vi_runner.get_matrix_constructor()
    result = vi_runner.get_training_result("good")
    policy = policy_constructor.TPolicy.from_TQ(result.TQ)
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)

    with self.assertRaises(ValueError):
      rollout_handler.get_rollout_iteration(policy, granularity=0)

  def test_get_rollout_iteration_granularity_1_succeeds(self):
    trainer = train.Trainer(
        per_step_penalty=0.01,
        board_shape=(3, 4),
        max_boxes=2,
        min_blocking_boxes=1,
    )
    mat_constructor = trainer.get_matrix_constructor()
    result = trainer.get_training_result("good")
    policy = policy_constructor.TPolicy.from_TQ(result.TQ)
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)

    rollouts = rollout_handler.get_rollout_iteration(policy, granularity=1)
    self.assertNotEmpty(rollouts.stats.box_milestones)
    self.assertEmpty(rollouts.rollouts)

  def test_get_rollout_iteration_granularity_2_succeeds(self):
    trainer = train.Trainer(
        per_step_penalty=0.01,
        board_shape=(3, 4),
        max_boxes=2,
        min_blocking_boxes=1,
    )
    mat_constructor = trainer.get_matrix_constructor()
    result = trainer.get_training_result("good")
    policy = policy_constructor.TPolicy.from_TQ(result.TQ)
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)

    rollouts = rollout_handler.get_rollout_iteration(policy, granularity=2)
    self.assertNotEmpty(rollouts.stats.box_milestones)
    self.assertNotEmpty(rollouts.rollouts)
    self.assertEmpty(rollouts.rollouts[0].trajectory)

  def test_get_rollout_iteration_granularity_3_succeeds(self):
    trainer = train.Trainer(
        per_step_penalty=0.01,
        board_shape=(3, 4),
        max_boxes=2,
        min_blocking_boxes=1,
    )
    mat_constructor = trainer.get_matrix_constructor()
    result = trainer.get_training_result("good")
    policy = policy_constructor.TPolicy.from_TQ(result.TQ)
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)

    rollouts = rollout_handler.get_rollout_iteration(policy, granularity=3)
    self.assertNotEmpty(rollouts.stats.box_milestones)
    self.assertNotEmpty(rollouts.rollouts)
    self.assertNotEmpty(rollouts.rollouts[0].trajectory)

  def get_down_policy(self, num_states: int) -> policy_constructor.TPolicy:
    return policy_constructor.TPolicy([
        policy_constructor.Policy(actions=[data.Action.DOWN.value] * num_states)
        for _ in range(num_states)
    ])

  def test_get_rollout_iterations(self):
    mat_constructor = matrix_constructor.MatrixConstructor(
        per_step_penalty=0.01,
        board_shape=(4, 2),
        max_boxes=1,
        # Allow only one initial box.
        validate=False,
        # Allow a box on the left edge.
        allow_initial_box_on_edge=True,
    )
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)

    # Get a policy that always chooses DOWN.
    down_policy = self.get_down_policy(mat_constructor.num_states)

    iterations = rollout_handler.get_rollout_iterations(
        [down_policy, down_policy], granularity=2
    ).iterations

    # Contains the states that are expected to get one box in the hole, along
    # with the number of steps to get there.
    expected_states_with_num_steps = {
        state_lib.State.from_string(
            """
            ####
            #A #
            #B #
            #  #
            #H #
            ####
            """
        ): 2,
        state_lib.State.from_string(
            """
            ####
            #A #
            #  #
            #B #
            #H #
            ####
            """
        ): 2,
        state_lib.State.from_string(
            """
            ####
            #  #
            #A #
            #B #
            #H #
            ####
            """
        ): 1,
    }
    expected_state_idxs_with_num_steps = {
        mat_constructor.get_state_index(state): num_steps
        for state, num_steps in expected_states_with_num_steps.items()
    }

    for iteration in iterations:
      # Assert on individual rollouts.
      self.assertLen(iteration.rollouts, 14)
      for rollout in iteration.rollouts:
        if rollout.initial_state in expected_state_idxs_with_num_steps:
          # Assert that the rollout has the expected number of steps to get one
          # box in the hole.
          self.assertLen(rollout.steps_to_box_milestones, 1)
          self.assertEqual(
              rollout.steps_to_box_milestones[1],
              expected_state_idxs_with_num_steps[rollout.initial_state],
          )
          # Assert that the rollout got one box in the hole.
          self.assertEqual(rollout.num_boxes_in_hole, 1)
        else:
          # All other rollouts should get zero boxes in the hole.
          self.assertEmpty(rollout.steps_to_box_milestones)
          self.assertEqual(rollout.num_boxes_in_hole, 0)
      # Assert on the aggregated rollout stats.
      self.assertEqual(iteration.stats.num_initial_states, 14)
      self.assertAlmostEqual(iteration.stats.average_boxes_in_hole, 3 / 14)
      self.assertEqual(iteration.stats.box_milestones[0].num_initial_states, 11)
      self.assertEqual(iteration.stats.box_milestones[1].num_initial_states, 3)
      self.assertEqual(iteration.stats.box_milestones[0].step_counts, [])
      self.assertEqual(iteration.stats.box_milestones[1].step_counts, [0, 1, 2])

  def test_get_rollout_iteration_max_rollouts_lt_initial_states_len(self):
    mat_constructor = matrix_constructor.MatrixConstructor(
        board_shape=(4, 4),
        max_boxes=2,
        per_step_penalty=0.01,
    )
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)
    max_rollouts = 3
    assert len(mat_constructor.initial_states()) > max_rollouts

    iteration = rollout_handler.get_rollout_iteration(
        self.get_down_policy(mat_constructor.num_states),
        max_rollouts=max_rollouts,
        granularity=2,
    )
    self.assertLen(iteration.rollouts, max_rollouts)

  def test_get_rollout_iteration_max_rollouts_gt_initial_states_len(self):
    mat_constructor = matrix_constructor.MatrixConstructor(
        board_shape=(4, 4),
        max_boxes=2,
        per_step_penalty=0.01,
    )
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)
    max_rollouts = 1000000
    assert len(mat_constructor.initial_states()) < max_rollouts

    iteration = rollout_handler.get_rollout_iteration(
        self.get_down_policy(mat_constructor.num_states),
        max_rollouts=max_rollouts,
        granularity=2,
    )
    self.assertLen(iteration.rollouts, len(mat_constructor.initial_states()))


if __name__ == "__main__":
  unittest.main()
