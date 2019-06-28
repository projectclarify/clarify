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

"""Utilities for working with Bazel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

from pcml.utils.fs_utils import get_pcml_root


def bazel_command(command_string):
  """Wrap Bazel commands with the context to allow them to work.

  Notably we're activating the necessary python environment,
  discovering the root of the Bazel workspace, and executing
  the provided command with the latter as the cwd.

  """

  cwd = get_pcml_root()

  wrapped_cmd = []
  wrapped_cmd.append("source activate py2")
  wrapped_cmd.append(command_string)
  # TODO: Generalize from `source activate py2` which assumes there
  # is a conda env by that name on the host system to something
  # that works more generally.

  cmd = "bash -c '%s'" % "; ".join(wrapped_cmd)

  proc = subprocess.Popen(cmd, shell=True, cwd=cwd)
  proc.wait()
