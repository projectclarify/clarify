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
                 name="workspace",
                 version="v0.1.0",
                 generate_tag=True):

  tag = version

  if generate_tag:
    # TODO: Get unique ID of code tree from Bazel
    import uuid
    tag = "{}-{}".format(version, str(uuid.uuid4())[0:4])

  image_id = "{}/{}/{}:{}".format(registry, project, name, tag)

  return image_id


def prepare_build_context(container_type):

  ctx_dir = os.path.dirname(os.path.abspath(__file__))

  tmpdir = tempfile.mkdtemp()

  if container_type == "workspace":
    suffix = "workspace"
  elif container_type == "runtime":
    suffix = "runtime"
  else:
    raise ValueError("Unrecognized Dockerfile suffix.")

  # Copy Dockerfile to build context
  dockerfile_path = os.path.join(
    ctx_dir, "Dockerfile.{}".format(suffix))

  dockerfile_dest_path = os.path.join(
    tmpdir, "Dockerfile")

  shutil.copyfile(
    dockerfile_path,
    dockerfile_dest_path)

  # Copy tools to build context
  src = os.path.join(ctx_dir, "../../")

  run_and_output(["cp", "-r", src, tmpdir])

  return tmpdir


def build(mode, and_push=False, container_type="workspace",
          static_image_id=None):

  if mode not in ["local", "gcb"]:
    raise ValueError("Unrecognized mode: {}".format(mode))

  image_id = get_image_id(
    registry="gcr.io",
    project="clarify",
    name="test-container",
    version="v0.1.0",
    generate_tag=True
  )
  if static_image_id:
    image_id = static_image_id

  print("Building with image id: {}".format(image_id))

  tmpdir = prepare_build_context(container_type)
  os.chdir(tmpdir)
    
  # Log build context
  run_and_output(["find", "."])

  if mode == "local":

    run_and_output(
      ["which", "docker"])

    run_and_output(
      ["docker", "build", "-t", image_id, "."])

    if and_push:
      run_and_output(
        ["gcloud", "docker", "--", "push", image_id]
      )

  elif mode == "gcb":
    
    run_and_output(
      ["gcloud", "builds", "submit", "-t", image_id, "."]
    )

  return image_id


if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser(description='Build PCML containers.')

  parser.add_argument('--build_mode', type=str, default="gcb",
                      help='Build using GCB or local Docker.')

  parser.add_argument('--container_type', type=str,
                      default="workspace",
                      help='Build type, `workspace` or `runtime` container.')

  parser.add_argument('--and_push', type=bool, default=False,
                      help='If building with docker, whether to push result.')

  parser.add_argument('--static_image_id', type=str,
                      default=None, required=False,
                      help='A static image id to use such as when building on circle.')

  args = parser.parse_args()

  build(mode=args.build_mode, and_push=args.and_push,
        container_type=args.container_type,
        static_image_id=args.static_image_id)
