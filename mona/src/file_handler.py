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

"""Saves and loads matrices to/from files."""

import enum
import os
from typing import Any, Tuple

from google.protobuf import text_format
import numpy as np
from numpy import typing as np_typing

from mona.proto import rollout_pb2
from mona.src import file_system

# pylint: disable=invalid-name


@enum.unique
class Encoding(enum.Enum):
  """The encoding to use when saving and loading proto files."""
  BINARYPB = '.pb'
  TEXTPROTO = '.textproto'


ENCODINGS = [Encoding.BINARYPB, Encoding.TEXTPROTO]


def _get_full_filename(
    full_dir: str,
    basename: str,
    encoding: Encoding | None = None,
    ext: str | None = None,
) -> str:
  """Returns a full filename from a directory, basename, and encoding/extension.

  Exactly one of encoding or ext must be specified.

  Example:
    _get_full_filename('/tmp/dir', 'file', encoding=Encoding.BINARYPB)
    # Returns '/tmp/dir/file.pb'

  Args:
    full_dir: The full directory to save the file in.
    basename: The basename of the file.
    encoding: The encoding, which maps to a file extension.
    ext: The exact extension to use when saving the file; it doesn't matter if
      it has a leading dot or not.
  """
  if ext is not None and encoding is not None:
    raise ValueError('Only one of encoding or ext can be specified.')
  if ext is not None:
    ext = '.' + ext.removeprefix('.')
  elif encoding is not None:
    ext = encoding.value
  else:
    raise ValueError('Either encoding or ext must be specified.')
  return os.path.join(full_dir, basename + ext)


def _add_trial_idx_to_basename(basename: str, trial_idx: int) -> str:
  """Adds the trial index after the basename of a file."""
  assert trial_idx > 0
  if trial_idx == 1:
    return basename
  return basename + f'_{trial_idx}'


def _get_rollouts_basename(trial_idx: int) -> str:
  return _add_trial_idx_to_basename('rollouts', trial_idx)


class FileHandler:
  """Handles saving and loading matrices to/from files."""

  def __init__(
      self,
      save_encoding: Encoding = Encoding.BINARYPB,
      filesystem: file_system.FileSystem = file_system.LocalFileSystem(),
  ):
    self._save_encoding = save_encoding
    self._filesystem = filesystem

  def get_full_directory(
      self,
      data_dir: str,
      reward_function: str,
      board_shape: Tuple[int, int],
      max_boxes: int,
      per_step_penalty: float,
      min_blocking_boxes: int = 1,
      init_vf_name: str | None = None,
      init_noise_scale: float | None = None,
      num_reward_steps: int | None = None,
      explore_prob: float | None = None,
      episode_step_limit: int = 50,
  ) -> str:
    """Returns the full directory for an experiment with the given parameters.

    Most parameters have a default value, and if the parameter matches the
    default, it is omitted from the directory name.

    Args:
      data_dir: The base directory for the experiment.
      reward_function: The name of the reward function.
      board_shape: The shape of the board.
      max_boxes: The maximum number of boxes on the board.
      per_step_penalty: The per-step penalty.
      min_blocking_boxes: The minimum number of boxes that must be used to block
        the camera.
      init_vf_name: The name of the initial policy.
      init_noise_scale: The noise scale for the initial policy.
      num_reward_steps: The number of steps to take for the reward function.
      explore_prob: The exploration probability.
      episode_step_limit: The maximum number of steps in an episode.
    """
    subdir = f'R={reward_function}'
    if init_vf_name is not None:
      # If the policy has a complex name, add # around it for clarity.
      if '=' in init_vf_name or '-' in init_vf_name:
        init_vf_name = f'#{init_vf_name}#'
      subdir += f'-V_init={init_vf_name}'
    if init_noise_scale is not None and init_noise_scale > 0:
      subdir += f'-noise_init={init_noise_scale}'
    if num_reward_steps is not None and num_reward_steps > 1:
      subdir += f'-R_steps={num_reward_steps}'
    if explore_prob is not None:
      subdir += f'-explore={explore_prob}'
    if min_blocking_boxes != 1:
      subdir += f'-blocking_boxes={min_blocking_boxes}'
    subdir += f'-step_penalty={per_step_penalty}'
    if episode_step_limit != 50:
      subdir += f'-step_limit={episode_step_limit}'
    subdir += (
        f'-shape={board_shape[0]}_{board_shape[1]}' + f'-boxes={max_boxes}'
    )
    return os.path.join(data_dir, subdir)

  def _load_proto(
      self,
      directory: str,
      basename: str,
      proto_class,
  ) -> Any:
    """Returns a proto with the given basename from the given directory.

    Attempts to load the proto from multiple formats, in the order of ENCODINGS.

    Args:
      directory: The directory to load the proto from.
      basename: The basename of the proto file.
      proto_class: The proto class to load.

    Raises:
      FileNotFoundError: If no file exists in either format.
    """
    possible_filepaths = [
        _get_full_filename(directory, basename, encoding=e) for e in ENCODINGS
    ]
    for filepath, encoding in zip(possible_filepaths, ENCODINGS):
      try:
        with self._filesystem.open(filepath, 'rb') as f:
          if encoding == Encoding.TEXTPROTO:
            return text_format.Parse(f.read(), proto_class())
          elif encoding == Encoding.BINARYPB:
            return proto_class.FromString(f.read())
          else:
            raise ValueError(f'Unsupported encoding: {encoding}')
      except FileNotFoundError:
        pass
    raise FileNotFoundError(
        f'Could not find any matrix file: {possible_filepaths}'
    )

  def delete_dir_if_exists(self, full_dir: str) -> None:
    if self._filesystem.is_directory(full_dir):
      print(
          'Overwriting existing directory:',
          full_dir,
      )
      self._filesystem.delete_recursively(full_dir)

  def load_rollouts(
      self, directory: str, trial_idx: int | None = None
  ) -> rollout_pb2.RolloutIterations | list[rollout_pb2.RolloutIterations]:
    """Returns loaded rollout iterations from a directory.

    If trial_idx is not None, returns the rollouts from each iteration of that
    trial. If trial_idx is None, returns all rollout iterations from all trials
    in the directory. If no trials exist in the directory, raises a
    FileNotFoundError.

    Args:
      directory: The directory to load the rollouts from.
      trial_idx: The trial index to load the rollouts from. If None, loads all
        trials in the directory.
    """
    if trial_idx is not None:
      return self._load_proto(
          directory,
          _get_rollouts_basename(trial_idx),
          rollout_pb2.RolloutIterations,
      )
    trial_idx = 1
    result = []
    while True:
      try:
        result.append(self.load_rollouts(directory, trial_idx))
      except FileNotFoundError as e:
        if result:
          return result
        raise e
      trial_idx += 1

  def _save_proto(self, proto, full_dir, basename):
    """Saves a proto to a file, either as a binary proto or a text proto."""
    self._filesystem.makedirs(full_dir)
    full_filename = _get_full_filename(
        full_dir, basename, encoding=self._save_encoding
    )
    with self._filesystem.open(full_filename, 'wb') as f:
      if self._save_encoding == Encoding.TEXTPROTO:
        f.write(text_format.MessageToString(proto).encode('utf-8'))
      elif self._save_encoding == Encoding.BINARYPB:
        f.write(proto.SerializeToString())
      else:
        raise ValueError(f'Unsupported encoding: {self._save_encoding}')
    print('Wrote file at', full_filename)

  def rollouts_exist(self, full_dir: str, trial_idx: int) -> bool:
    """Returns whether a rollouts file exists for the given trial."""
    return any(
        self._filesystem.exists(
            _get_full_filename(
                full_dir, _get_rollouts_basename(trial_idx), encoding=e
            )
        )
        for e in ENCODINGS
    )

  def save_rollouts(
      self,
      full_dir: str,
      rollouts: rollout_pb2.RolloutIterations,
      trial_idx: int = 1,
  ) -> None:
    self._save_proto(rollouts, full_dir, _get_rollouts_basename(trial_idx))

  def save_value_matrix(
      self,
      full_dir: str,
      value_matrix: np_typing.NDArray[np.float64],
      trial_idx: int = 1,
  ) -> None:
    """Saves the value matrix as a Numpy array."""
    self._filesystem.makedirs(full_dir)
    with self._filesystem.open(
        _get_full_filename(
            full_dir,
            _add_trial_idx_to_basename('value_matrix', trial_idx),
            ext='npy',
        ),
        'wb',
    ) as f:
      np.save(f, value_matrix)

  def load_value_matrix(
      self,
      filepath: str,
  ) -> np_typing.NDArray[np.float64]:
    """Loads the value matrix as a Numpy array."""
    with self._filesystem.open(filepath, 'rb') as f:
      return np.load(f)
