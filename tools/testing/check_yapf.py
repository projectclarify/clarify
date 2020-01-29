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

"""Assert yapf --diff is clean"""

import subprocess
import argparse
import sys

    
def yapf_diff(path):

  command = ["yapf", "-r", "--diff", path]

  process = subprocess.Popen(
    command, stdout=subprocess.PIPE
  )

  output = []

  for line in process.stdout:
    line = line.decode("utf-8")
    sys.stdout.write(line)
    output.append(line)

  return output


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Check yapf --diff is clean.')

  parser.add_argument('--path', type=str, default=None,
                      required=True,
                      help='The path to yapf')

  args = parser.parse_args()

  diff = yapf_diff(args.path)

  if diff:
    msg = "yapf --diff is nonempty, run yapf -r --in-pace [path]"
    raise Exception(msg)
