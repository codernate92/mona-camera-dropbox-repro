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

"""Train a policy in the gridworld environment."""

from collections.abc import Sequence
import functools

from absl import app
from absl import flags

from mona.src import file_handler as file_handler_lib
from mona.src import matrix_constructor
from mona.src import policy_constructor
from mona.src import rollout_handler as rollout_handler_lib
from mona.src import runner as runner_lib

DATA_DIR = flags.DEFINE_string(
    'data_dir',
    None,
    'The directory to save and load data from.',
    required=True,
)

SAVE_ROLLOUTS_LEVEL = flags.DEFINE_integer(
    'save_rollouts_level',
    0,
    'Determines whether to save data from the rollouts of the final policy, and'
    ' if so with what level of detail.'
    ' 0: No rollout data.'
    ' 1: Iteration summary statistics only.'
    ' 2: Iteration and rollout summary statistics.'
    ' 3: Iteration and rollout summary statistics and full state trajectories.',
)

SAVE_PROTO_ENCODING = flags.DEFINE_enum(
    'save_proto_encoding',
    file_handler_lib.Encoding.TEXTPROTO.name,
    [e.name for e in file_handler_lib.Encoding],
    'The encoding to use when saving the rollouts proto.',
)

SAVE_INIT_VALUE_MATRIX = flags.DEFINE_bool(
    'save_init_value_matrix',
    False,
    'Whether to save the initial value matrix as a Numpy array to --data_dir.'
    ' The value matrix will match initial_vf with noise added based on'
    ' noise_scale. --initial_vf must be "good" or "bad". Note that'
    ' --reward_function has no effect on the saved value matrix file.',
)

SKIP_EXISTING_RESULTS = flags.DEFINE_bool(
    'skip_existing_results',
    False,
    'Whether to skip experiments where data already exists.',
)

SHOW_DEMO = flags.DEFINE_bool(
    'show_demo',
    True,
    'Whether to demo the completed policy starting from a random state.',
)

BOARD_SHAPE = flags.DEFINE_list(
    'board_shape',
    ['4', '4'],
    'The shape of the board of the gridworld.',
)

MAX_BOXES = flags.DEFINE_integer(
    'max_boxes',
    2,
    'The maximum number of boxes in the gridworld.',
)

MIN_BLOCKING_BOXES = flags.DEFINE_integer(
    'min_blocking_boxes',
    1,
    'The minimum number of boxes that must be in front of the camera for it to'
    ' be blocked.',
)

INITIAL_VF = flags.DEFINE_string(
    'initial_vf',
    None,
    'The initial value function to begin training with. If "good" or "bad", the'
    ' initial value function is trained via value iteration. If "zero", the'
    ' value function is set to all zeros. To load the value function from a'
    ' specific file, use the --initial_vf_path flag and set this flag to the'
    ' name to use in the directory name.',
)

INITIAL_VF_PATH = flags.DEFINE_string(
    'initial_vf_path',
    None,
    'The path to the value function to load to create the initial policy used'
    ' to derive the trusted value function. If this is set, --initial_vf should'
    ' be set to the name to use in the directory name.',
)

REWARD_FUNCTION = flags.DEFINE_enum(
    'reward_function',
    'bad',
    ['good', 'bad'],
    'The reward function to use.',
)

PER_STEP_PENALTY = flags.DEFINE_float(
    'per_step_penalty',
    0.01,
    'The reward penalty to apply for each step in the rollout.',
)

NOISE_SCALE = flags.DEFINE_float(
    'noise_scale',
    0.0,
    'The standard deviation of the Gaussian noise added to the value function.'
    ' This simulates a human being mistaken about the value of each state.',
)

EPISODE_STEP_LIMIT = flags.DEFINE_integer(
    'episode_step_limit',
    50,
    'The maximum number of steps before the episode ends.',
)

NUM_TRIALS = flags.DEFINE_integer(
    'num_trials',
    1,
    'After this script is complete, there should be at least this number of'
    ' saved results (value matrices + rollouts) from experiment trials with'
    ' these parameters. If --skip_existing_results=true, any existing trials'
    ' will be counted towards this total. For example, if --num_trials=10 and'
    ' there are already 5 trials, this script will run 5 more trials to reach'
    ' 10 trials in total.',
)

# pylint: disable=invalid-name


def demo_from_random_state(
    policy: policy_constructor.Policy | policy_constructor.TPolicy,
    mat_constructor: matrix_constructor.MatrixConstructor,
) -> None:
  """Prints a demo of a rollout of the policy from a random initial state.

  In the initial state, all boxes can be pushed into the hole. The demo will
  end early if the environment terminates.

  Args:
    policy: The policy to demo.
    mat_constructor: The matrix constructor for the environment.
  """
  rollout_handler = rollout_handler_lib.RolloutHandler(mat_constructor)
  initial_s_idx = mat_constructor.get_random_initial_state()
  rollout = rollout_handler.get_rollout(initial_s_idx, policy)
  states = [mat_constructor.get_state(s_idx) for s_idx in rollout.trajectory]
  for state in states:
    print(f'\n{state}')
  if states[-1].ended:
    print(f'\nFinished in {len(states) - 1} steps.')
  else:
    print('\nEnded early.')


def get_initial_run_result(
    runner: runner_lib.Runner,
    initial_vf: str | None,
    initial_vf_path: str | None,
    noise_scale: float,
    save_value_matrix: bool,
    trial_idx: int,
    is_minimal: bool,
) -> runner_lib.RunResult | None:
  """Returns the initial run result to use for training.

  Args:
    runner: The runner to use to run the initial policy.
    initial_vf: The name of the initial policy to use.
    initial_vf_path: The path to the initial policy to load.
    noise_scale: The standard deviation of the Gaussian noise added to the value
      function.
    save_value_matrix: Whether to save the value matrix as a Numpy array to
      --data_dir.
    trial_idx: The trial index to use for the initial policy.
    is_minimal: If True, the return value will only contain the policy name and
      noise scale. This is so we can create a minimal initial run result to
      determine the directory to check for existing results.
  """
  initial_vf_is_unique_name = initial_vf not in ['good', 'bad', 'zero', None]
  if initial_vf_path is not None:
    if not initial_vf_is_unique_name:
      raise ValueError(
          'If --initial_vf_path is set, --initial_vf must be set to a string'
          ' representing the name of the value function to load, which cannot'
          ' be "good", "bad", "zero", or None. Got'
          f' --initial_vf_path={initial_vf_path} and --initial_vf={initial_vf}.'
      )

  if initial_vf_path is None and initial_vf_is_unique_name:
    raise ValueError(
        'If --initial_vf_path is not set, --initial_vf must be set to "good",'
        ' "bad", "zero", or None. Got'
        f' --initial_vf_path={initial_vf_path} and --initial_vf={initial_vf}.'
    )

  if initial_vf not in ['good', 'bad']:
    if noise_scale != 0.0:
      raise ValueError(
          'If --initial_vf is not "good" or "bad", --noise_scale must be 0.0,'
          f' but got {noise_scale}.'
      )
    if save_value_matrix:
      raise ValueError(
          'If --initial_vf is not "good" or "bad", --save_init_value_matrix'
          " must be False, since it doesn't make sense to save the value"
          " matrix if we're loading it or setting it to all zeros."
      )

  match initial_vf:
    case 'zero':
      print('\nInitial V will be all zero values.')
      return None
    case 'good' | 'bad':
      if is_minimal:
        return runner_lib.RunResult(vf_name=initial_vf, noise_scale=noise_scale)
      print(f'\nInitial V will be the {initial_vf} value function.')
      return runner.run(
          runner_lib.RunParams(
              reward_function=initial_vf,
              noise_scale=noise_scale,
              trial_idx=trial_idx,
              save_value_matrix=save_value_matrix,
          )
      )
    case _:
      if is_minimal:
        return runner_lib.RunResult(vf_name=initial_vf)
      print(
          f'\nInitial V with name {initial_vf} will be loaded from'
          f' {initial_vf_path}.'
      )
      return runner.load_run_result(initial_vf_path, initial_vf)


def get_final_run_params(
    runner: runner_lib.Runner,
    initial_vf: str | None,
    initial_vf_path: str | None,
    reward_function: str,
    noise_scale: float,
    trial_idx: int,
    save_rollouts_level: int,
    save_init_value_matrix: bool,
    init_run_result_is_minimal: bool,
) -> runner_lib.RunParams:
  """Returns the final run params to use for training."""
  init_run_result = get_initial_run_result(
      runner=runner,
      initial_vf=initial_vf,
      initial_vf_path=initial_vf_path,
      noise_scale=noise_scale,
      trial_idx=trial_idx,
      save_value_matrix=save_init_value_matrix,
      is_minimal=init_run_result_is_minimal,
  )
  return runner_lib.RunParams(
      reward_function=reward_function,
      init_run_result=init_run_result,
      trial_idx=trial_idx,
      save_rollouts_level=save_rollouts_level,
  )


def run_trial(runner: runner_lib.Runner, trial_idx: int = 1):
  """Runs a single trial of an experiment."""

  # Use a partial function here so we can use two versions of the initial run
  # result: one with only the minimal data needed to check for existing results
  # and one with the full data needed to run training.
  get_final_run_params_partial = functools.partial(
      get_final_run_params,
      runner=runner,
      initial_vf=INITIAL_VF.value,
      initial_vf_path=INITIAL_VF_PATH.value,
      reward_function=REWARD_FUNCTION.value,
      noise_scale=NOISE_SCALE.value,
      trial_idx=trial_idx,
      save_rollouts_level=SAVE_ROLLOUTS_LEVEL.value,
      save_init_value_matrix=SAVE_INIT_VALUE_MATRIX.value,
  )

  # Check whether we should skip this trial; if so, return None.
  if SKIP_EXISTING_RESULTS.value:
    full_dir = runner.get_full_dir_for_run(
        get_final_run_params_partial(init_run_result_is_minimal=True)
    )
    if runner.get_file_handler().rollouts_exist(full_dir, trial_idx):
      print(f'Skipping trial {trial_idx} in {full_dir}; data already exists.')
      return None

  # After determining that we shouldn't skip existing results, we can train or
  # load the initial policy, then use it to train the final policy.
  print('\nGetting final V...')
  final_result = runner.run(
      get_final_run_params_partial(
          init_run_result_is_minimal=False,
      )
  )

  # If requested, demonstrate the final policy starting from a random state.
  if SHOW_DEMO.value and final_result is not None:
    print('\nRunning the policy from a random state...')
    demo_from_random_state(
        final_result.final_policy, runner.get_matrix_constructor()
    )


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

  board_shape = tuple(map(int, BOARD_SHAPE.value))

  print('Initializing runner and generating all states...')
  runner = runner_lib.Runner(
      board_shape=board_shape,
      max_boxes=MAX_BOXES.value,
      min_blocking_boxes=MIN_BLOCKING_BOXES.value,
      per_step_penalty=PER_STEP_PENALTY.value,
      data_dir=DATA_DIR.value,
      save_proto_encoding=file_handler_lib.Encoding[SAVE_PROTO_ENCODING.value],
      episode_step_limit=EPISODE_STEP_LIMIT.value,
  )
  print(
      f'Initialized runner with {runner.num_states} states'
      f' ({runner.num_initial_states} initial states).'
  )

  if NUM_TRIALS.value > 1:
    assert SAVE_ROLLOUTS_LEVEL.value != 0, (
        'If --num_trials > 1, --save_rollouts_level must be at least 1. If no'
        ' data is saved, running multiple trials is pointless.'
    )
    for trial_idx in range(1, NUM_TRIALS.value + 1):
      print(f'\nTrial {trial_idx} of {NUM_TRIALS.value}')
      run_trial(runner, trial_idx)
  else:
    run_trial(runner)


if __name__ == '__main__':
  app.run(main)
