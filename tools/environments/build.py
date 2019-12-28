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

"""Workspace, trainer, and test container building."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import subprocess
import os
import shutil
import sys


def run_and_output(command, cwd=None, env=None):
  
  process = subprocess.Popen(
    command, cwd=cwd, env=env,
    stdout=subprocess.PIPE
  )

  output = []

  for line in process.stdout:
    line = line.decode("utf-8")
    sys.stdout.write(line)
    output.append(line)

  return output


def get_image_id(registry="gcr.io",
                 project="clarify",
                 name="workspace"):
  #output = run_and_output(
  #  ["curl", "--silent",
  #   "https://api.github.com/repos/{}/releases/latest".format(
  #   "projectclarify/pcml"
  #   )])
  tag = "v0.1.0"

  image_id = "{}/{}/{}:{}".format(registry, project, name, tag)

  return image_id


def prepare_build_context():

  ctx_dir = os.path.dirname(os.path.abspath(__file__))

  tmpdir = tempfile.mkdtemp()

  # Copy Dockerfile to build context
  dockerfile_path = os.path.join(
    ctx_dir, "Dockerfile")

  dockerfile_dest_path = os.path.join(
    tmpdir, "Dockerfile")

  shutil.copyfile(
    dockerfile_path,
    dockerfile_dest_path)

  # Copy tools to build context
  src = os.path.join(ctx_dir, "../../tools")

  run_and_output(["cp", "-r", src, tmpdir])

  return tmpdir


def build_and_push(mode):

  if mode not in ["local", "gcb"]:
    raise ValueError("Unrecognized mode: {}".format(mode))

  image_id = get_image_id()

  print("Building with image id: {}".format(image_id))

  tmpdir = prepare_build_context()
  os.chdir(tmpdir)

  # Log build context
  run_and_output(["find", "."])

  if mode == "local":

    run_and_output(
      ["which", "docker"])

    run_and_output(
      ["docker", "build", "-t", image_id, "."])

    run_and_output(
      ["gcloud", "docker", "--", "push", image_id]
    )

  elif mode == "gcb":
    
    run_and_output(
      ["gcloud", "builds", "submit", "-t", image_id, "."]
    )


if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser(description='Build PCML containers.')

  parser.add_argument('--mode', type=str, default="gcb",
                      help='Build using GCB or local Docker.')

  args = parser.parse_args()

  build_and_push(mode=args.mode)
