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

"""Abstract base class for file systems."""

import abc
import os
import shutil
from typing import Any


class FileSystem(abc.ABC):
  """Abstract base class for file systems."""

  @abc.abstractmethod
  def exists(self, filepath: str) -> bool:
    """Returns whether a file exists."""

  @abc.abstractmethod
  def is_directory(self, filepath: str) -> bool:
    """Returns whether a file is a directory."""

  @abc.abstractmethod
  def open(self, filepath: str, mode: str) -> Any:
    """Opens a file."""

  @abc.abstractmethod
  def delete_recursively(self, filepath: str) -> None:
    """Deletes a file or directory recursively."""

  @abc.abstractmethod
  def makedirs(self, filepath: str) -> None:
    """Makes directories."""


class LocalFileSystem(FileSystem):
  """File system implementation using regular Python file operations."""

  def exists(self, filepath: str) -> bool:
    return os.path.exists(filepath)

  def is_directory(self, filepath: str) -> bool:
    return os.path.isdir(filepath)

  def open(self, filepath: str, mode: str) -> Any:
    return open(filepath, mode)

  def delete_recursively(self, filepath: str) -> None:
    shutil.rmtree(filepath)

  def makedirs(self, filepath: str) -> None:
    os.makedirs(filepath, exist_ok=True)
