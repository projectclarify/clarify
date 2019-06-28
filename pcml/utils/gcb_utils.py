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

"""Google Container Builder utils"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import yaml
import uuid

import tensorflow as tf

from pcml.utils.cmd_utils import run_and_output


def generate_image_tag(project_or_user, app_name,
                       registry_target="gcr.io"):
  """Tags of fmt {registry_target}/{project_or_user}/{app_name}:{uid}"""

  now = datetime.datetime.now()

  build_id = now.strftime("%m%d-%H%M") + "-" + uuid.uuid4().hex[0:4]

  return "%s/%s/%s:%s" % (
      registry_target, project_or_user, app_name, build_id
  )


def gcb_build_and_push(image_tag, build_dir,
                       cache_from="tensorflow/tensorflow:1.6.0",
                       dry_run=False):
  """Generate GCB config and build container, caching from `cache_from`.

  A Google Container Builder build.yaml config will be produced in
  `build_dir` and if not `dry_run` a system call will be made from
  `build_dir` to 'gcloud container build submit --config build.yaml .'.

  Blocks until completion.

  Args:
    image_tag (str): The tag to apply to the newly built image.
    build_dir (str): Path to directory to use as '.' during GCB build.
    cache_from (str): A container image string identifier.
    dry_run (bool): Whether to actually trigger the build on GCB.

  TODO: Use GCB REST API.

  """

  if not isinstance(build_dir, str):
    raise ValueError("Paths must be of type str, saw: %s" % build_dir)
  if not build_dir.startswith("/"):
    raise ValueError("Expected an absolute path, saw: %s" % build_dir)
  if not tf.gfile.Exists(build_dir):
    raise ValueError("Path does not exist: %s" % build_dir)

  os.chdir(build_dir)

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
                  "-t", image_tag,
                  "."
              ]
          }
      ],
      "images": [image_tag]
  }

  output_config_path = "build.yaml"

  if build_dir is not None:
    output_config_path = os.path.join(build_dir, output_config_path)

  with open("build.yaml", "w") as f:
    f.write(yaml.dump(build_config))

  build_config = os.path.join(build_dir, "build.yaml")
  tf.logging.info("Generated build config: %s" % build_config)

  if not dry_run:
    tf.logging.info("Triggering build...")
    os.chdir(build_dir)
    _ = run_and_output(["gcloud", "container", "builds",
                        "submit", "--config", "build.yaml", "."])

  return build_config
