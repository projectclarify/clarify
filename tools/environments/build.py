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
import datetime
import yaml
import uuid
import dateutil
import dateutil.parser
from absl import logging

import googleapiclient.discovery
from googleapiclient import _auth


# TODO: Discover from a centralized location.
BUILD_VERSION="v0.1.0"


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


def get_image_id(project="clarify", generate_tag=True):

  registry = "gcr.io"
  name = "runtime-base"
  version = BUILD_VERSION
  tag = version

  if generate_tag:
    # TODO: Get unique ID of code tree from Bazel
    import uuid
    tag = "{}-{}".format(version, str(uuid.uuid4())[0:4])

  image_id = "{}/{}/{}:{}".format(registry, project, name, tag)

  return image_id


def prepare_build_context(environment):

  ctx_dir = os.path.dirname(os.path.abspath(__file__))

  tmpdir = tempfile.mkdtemp()

  # Copy Dockerfile to build context
  dockerfile_path = os.path.join(
    ctx_dir, environment, "Dockerfile")

  dockerfile_dest_path = os.path.join(
    tmpdir, "Dockerfile")

  shutil.copyfile(
    dockerfile_path,
    dockerfile_dest_path)

  print("ctx_dir: {}".format(ctx_dir))

  # Copy tools to build context
  src = os.path.join(ctx_dir, "../../")

  # Or 
  # src = "/home/jovyan"

  run_and_output(["cp", "-r", src, tmpdir])

  return tmpdir


def write_gcb_config(image_id, cache_from, tmp_dir):

  cfg_path = os.path.join(tmp_dir, "build.yaml")

  build_config = {
      "steps": [
          {
              "name": "gcr.io/cloud-builders/docker",
              "args": ["pull", cache_from]
          },
          {
              "name": "gcr.io/cloud-builders/docker",
              "args": [
                  "build",
                  "--cache-from",
                  cache_from,
                  "-t", image_id,
                  "."
              ]
          }
      ],
      "images": [image_id],
      "timeout": "80000s"
  }

  with open(cfg_path, "w") as f:
    f.write(yaml.dump(build_config))

  return cfg_path


def latest_successful_build(image_id):
  """Given an image URI get the most recent green cloudbuild."""

  credentials = _auth.default_credentials()

  uri_prefix = image_id.split(":")[0]
  
  project_id = uri_prefix.split("/")[1]
  cloudbuild = googleapiclient.discovery.build(
    'cloudbuild', 'v1', credentials=credentials,
    cache_discovery=False)
  builds = cloudbuild.projects().builds().list(
      projectId=project_id).execute()

  latest_time = None
  latest = None

  for build in builds["builds"]:
    if build["status"] == "SUCCESS":
      images = build["images"]
      if len(images) == 1:
        if images[0].startswith(uri_prefix):
          finish_time = dateutil.parser.parse(build["finishTime"])
          if not latest:
            latest_time = finish_time
          if finish_time >= latest_time:
            latest = images[0]
            latest_time = finish_time

  if latest:
    logging.info("Found a latest successful build: {}".format(
      latest
    ))

  return latest


def build(mode, and_push=False, static_image_id=None,
          project="clarify", environment="base"):

  if mode not in ["local", "gcb"]:
    raise ValueError("Unrecognized mode: {}".format(mode))

  image_id = get_image_id(
    project=project,
    generate_tag=True
  )
  if static_image_id:
    image_id = static_image_id

  print("Building with image id: {}".format(image_id))

  tmpdir = prepare_build_context(environment)
  os.chdir(tmpdir)

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

    cache_from = latest_successful_build(image_id)

    cfg_path = write_gcb_config(
        image_id=image_id,
        cache_from=cache_from,
        tmp_dir=tmpdir)

    run_and_output(
      ["gcloud", "builds", "submit",
       "--config={}".format("build.yaml"),
       "."],
      cwd = tmpdir
    )

  return image_id


if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser(description='Build PCML containers.')

  parser.add_argument('--environment', type=str, default="base",
                      help='a subdir of tools/environments.')

  parser.add_argument('--project', type=str, default="clarify",
                      help='gcr.io project to which to push.')

  parser.add_argument('--build_mode', type=str, default="gcb",
                      help='Build using GCB or local Docker.')

  parser.add_argument('--and_push', type=bool, default=False,
                      help='If building with docker, whether to push result.')

  parser.add_argument('--static_image_id', type=str,
                      default=None, required=False,
                      help='A static image id to use such as when building on circle.')

  args = parser.parse_args()

  if args.environment not in ["base", "primary"]:
    raise ValueError("Unrecognized env name {}".format(
      args.environment
    ))

  build(mode=args.build_mode,
        and_push=args.and_push,
        static_image_id=args.static_image_id,
        project=args.project,
        environment=args.environment)
