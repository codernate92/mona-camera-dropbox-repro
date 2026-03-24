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

"""Wraps around Trainer and handles file I/O."""

import dataclasses
import os
from typing import Sequence, Tuple

from mona.src import file_handler as file_handler_lib
from mona.src import file_system as file_system_lib
from mona.src import matrix_constructor
from mona.src import policy_constructor
from mona.src import rollout_handler as rollout_handler_lib
from mona.src import train

TrainingResult = train.TrainingResult
Trainer = train.Trainer

# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class RunResult:
  """The result of a training run, including file handling information.

  This may only contain the policy name and noise scale, if it is being
  generated with init_run_result_is_minimal=True in main.py. This is so we can
  create a minimal initial run result to determine the directory to check for
  existing results. In this case, all other fields will be None.

  Attributes:
    training_result: The result of the training run.
    vf_name: A name which identifies the policy.
    noise_scale: The standard deviation of the Gaussian noise added to V.
    full_dir: The directory where the data may have been saved or loaded from.
    intermediate_policies: The intermediate policies during training.
    final_policy: The final policy after training is complete.
  """

  training_result: TrainingResult | None = None
  vf_name: str | None = None
  noise_scale: float | None = None
  full_dir: str | None = None
  intermediate_policies: Sequence[policy_constructor.TPolicy] | None = None
  final_policy: (
      policy_constructor.TPolicy | policy_constructor.Policy | None
  ) = None


@dataclasses.dataclass(frozen=True)
class RunParams:
  """The parameters for a training run.

  These can be used either to determine whether an experiment result already
  exists in the file system, or to run a new experiment.

  Attributes:
    reward_function: The reward function to use.
    noise_scale: The standard deviation of the Gaussian noise added to V.
    init_run_result: The result of the previous run, if any.
    trial_idx: The trial index.
    save_rollouts_level: The level of detail to save rollouts at.
    save_value_matrix: Whether to save the value matrix as a Numpy array.
  """

  reward_function: str
  noise_scale: float = 0.0
  init_run_result: RunResult | None = None
  trial_idx: int = 1
  save_rollouts_level: int = 0
  save_value_matrix: bool = False


class Runner:
  """Reads any existing results, runs training, and saves new results."""

  def __init__(
      self,
      board_shape: Tuple[int, int],
      max_boxes: int,
      min_blocking_boxes: int,
      per_step_penalty: float,
      data_dir: str,
      save_proto_encoding: file_handler_lib.Encoding = file_handler_lib.Encoding.BINARYPB,
      episode_step_limit: int = 50,
  ):
    self._board_shape = board_shape
    self._max_boxes = max_boxes
    self._min_blocking_boxes = min_blocking_boxes
    self._per_step_penalty = per_step_penalty
    self._data_dir = data_dir
    self._episode_step_limit = episode_step_limit

    self._trainer = Trainer(
        board_shape=self._board_shape,
        max_boxes=self._max_boxes,
        min_blocking_boxes=self._min_blocking_boxes,
        per_step_penalty=self._per_step_penalty,
        episode_step_limit=self._episode_step_limit,
    )
    self._file_handler = file_handler_lib.FileHandler(
        save_encoding=save_proto_encoding,
    )

  def get_file_handler(self) -> file_handler_lib.FileHandler:
    return self._file_handler

  def load_run_result(
      self,
      filepath: str,
      vf_name: str,
  ) -> RunResult:
    """Returns a RunResult loaded from a ValueMatrices file.

    Most of the fields in the RunResult will be None, since we only use this
    function to initialize another run, which doesn't require all the fields.

    Args:
      filepath: The path to the ValueMatrices file to load.
      vf_name: The name of the value function to use.
    """
    TV = self._file_handler.load_value_matrix(filepath)
    assert TV.ndim == 2  # (T, S)
    full_dir = os.path.dirname(filepath)
    return RunResult(
        TrainingResult(
            P=None,
            R=None,
            TV=TV,
            TQ=None,
            intermediate_TQs=None,
            reward_function=None,
        ),
        vf_name=vf_name,
        noise_scale=None,
        full_dir=full_dir,
    )

  def _get_policies(
      self,
      training_result: TrainingResult,
  ) -> Tuple[Sequence[policy_constructor.TPolicy], policy_constructor.TPolicy]:
    """Returns the policies to use for rollouts."""

    def TQ_to_policy(TQ):
      return policy_constructor.TPolicy(
          [policy_constructor.Policy(Q=Q) for Q in TQ]
      )

    if training_result.intermediate_TQs is None:
      intermediate_policies = []
    else:
      intermediate_policies = [
          TQ_to_policy(TQ) for TQ in training_result.intermediate_TQs
      ]

    final_policy = TQ_to_policy(training_result.TQ)

    return intermediate_policies, final_policy

  def _save_rollouts(
      self,
      intermediate_policies: Sequence[policy_constructor.TPolicy],
      final_policy: policy_constructor.TPolicy,
      mat_constructor: matrix_constructor.MatrixConstructor,
      trial_idx: int,
      save_rollouts_level: int,
      full_dir: str,
      max_rollouts: int | None,
  ):
    """Roll out all policies and save the results."""
    if save_rollouts_level == 0:
      return
    rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)
    print("\nGetting rollouts for every M...")
    policies = list(intermediate_policies) + [final_policy]
    iterations = rollout_handler.get_rollout_iterations(
        policies,
        max_rollouts=max_rollouts,
        granularity=save_rollouts_level,
        verbose=True,
    )
    print("Saving rollouts...")
    self._file_handler.save_rollouts(full_dir, iterations, trial_idx=trial_idx)

  def get_full_dir_for_run(
      self,
      params: RunParams,
  ) -> str:
    """Returns the full directory that a run with the given params would use."""
    if params.init_run_result is None:
      init_vf_name = None
      init_noise_scale = params.noise_scale
    else:
      init_vf_name = params.init_run_result.vf_name
      init_noise_scale = params.init_run_result.noise_scale
    return self._file_handler.get_full_directory(
        data_dir=self._data_dir,
        reward_function=params.reward_function,
        board_shape=self._board_shape,
        max_boxes=self._max_boxes,
        per_step_penalty=self._per_step_penalty,
        min_blocking_boxes=self._min_blocking_boxes,
        init_vf_name=init_vf_name,
        init_noise_scale=init_noise_scale,
        episode_step_limit=self._episode_step_limit,
    )

  def run(
      self,
      params: RunParams,
  ) -> RunResult | None:
    """Runs training, saves results, and returns the result.

    Args:
      params: The parameters for the training run. See RunParams for details.

    Returns:
      The result of the training run, or None if the data already exists and
      --skip_existing_results=true.
    """
    full_dir = self.get_full_dir_for_run(params)
    trial_idx = params.trial_idx

    # Delete the directory if there will be results saved, but only on the first
    # trial. If --skip_existing_results=true, this will never be reached.
    if trial_idx == 1 and (
        params.save_value_matrix or params.save_rollouts_level > 0
    ):
      self.get_file_handler().delete_dir_if_exists(full_dir)

    init_training_result = (
        params.init_run_result.training_result
        if params.init_run_result is not None
        else None
    )

    training_result = self._trainer.get_training_result(
        reward_function=params.reward_function,
        noise_scale=params.noise_scale,
        init_training_result=init_training_result,
    )

    intermediate_policies, final_policy = self._get_policies(training_result)

    self._save_rollouts(
        intermediate_policies,
        final_policy,
        self._trainer.get_matrix_constructor(),
        trial_idx,
        params.save_rollouts_level,
        full_dir,
        max_rollouts=None,
    )

    if params.save_value_matrix:
      TV = training_result.TV
      self._file_handler.save_value_matrix(full_dir, TV, trial_idx)
      print(f"Saved value matrix to {full_dir}")

    return RunResult(
        training_result=training_result,
        vf_name=params.reward_function,
        noise_scale=params.noise_scale,
        full_dir=full_dir,
        intermediate_policies=intermediate_policies,
        final_policy=final_policy,
    )

  def get_matrix_constructor(self) -> matrix_constructor.MatrixConstructor:
    return self._trainer.get_matrix_constructor()

  @property
  def num_states(self) -> int:
    return self._trainer.num_states

  @property
  def num_initial_states(self) -> int:
    return self._trainer.num_initial_states
