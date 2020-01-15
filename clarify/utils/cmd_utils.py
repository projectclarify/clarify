# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Subprocess command utilities."""

import os
import subprocess

import tensorflow as tf


def _maybe_decode(raw):

  if isinstance(raw, str):
    return raw

  elif isinstance(raw, bytes):
    return raw.decode()

  return raw.encode('ascii', 'ignore').decode('ascii')


def run_and_output(command, cwd=None, env=None):
  """Run a system command.

  Args:
    command (list): The command to run as a string array.
    cwd (str): The directory to set as the current working directory
    env (list): Variables to make available in environment of command

  Returns:
    str: A string storing the stdout produced by running the command.

  Raises:
    CalledProcessError: If the command run exits with a system error.

  """

  tf.logging.info("Running: %s \ncwd=%s", " ".join(command), cwd)

  if not env:

    env = os.environ

  try:

    output = subprocess.check_output(command,
                                     cwd=cwd,
                                     env=env,
                                     stderr=subprocess.STDOUT).decode("utf-8")

    output = _maybe_decode(output)

    tf.logging.info("\n=======\nSubprocess output:%s\n=====" % output)

  except subprocess.CalledProcessError as e:

    output = _maybe_decode(e.output)

    tf.logging.info(
        "\n=======\nCommand failed, subprocess output:\n%s\n=======", output)

    raise

  return output
