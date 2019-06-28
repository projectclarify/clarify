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

"""General utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import shutil
import logging
import datetime
import uuid
import pprint
import yaml

from pcml.utils.cmd_utils import run_and_output


def stage_workspace(train_dir, workspace_root):
  workspace_mount_path = os.path.join(train_dir, "workspace")
  if os.path.exists(workspace_mount_path):
    tf.logging.info(
        ("Staged workspace already exists, skipping creation for path: "
         "%s" % workspace_mount_path))
  shutil.copytree(workspace_root, workspace_mount_path)
  logging.info("Successfully staged workspace to %s" % workspace_mount_path)
  return workspace_mount_path


def object_as_dict(obj):
  if hasattr(obj, "__dict__"):
    obj = obj.__dict__
  if isinstance(obj, dict):
    data = {}
    for key, value in obj.items():
      data[key] = object_as_dict(value)
    return data
  elif isinstance(obj, list):
    return [object_as_dict(item) for item in obj]
  else:
    return obj


def show_object(obj):
  pp = pprint.PrettyPrinter(indent=0)
  pp.pprint(object_as_dict(obj))


def object_as_yaml(obj):
  d = object_as_dict(obj)
  return yaml.dump(d, default_flow_style=False)


def expect_type(obj, ty):
  """Check that `obj` is of type `ty`.

  Raises:
      ValueError: If `obj` is not of type `ty`.

  """
  if not isinstance(obj, ty):
    raise ValueError("Expected type %s, saw object %s of type %s" % (
        ty, obj, type(obj)))


def gen_timestamped_uid():
  """Generate a string uid of the form MMDD-HHMM-UUUU."""
  now = datetime.datetime.now()
  return now.strftime("j%m%d-%H%M") + "-" + uuid.uuid4().hex[0:4]


def maybe_mkdir(path):
  """Single interface to multiple ways of mkdir -p.

  Looks like os.makedirs(..., exist_ok=True) doesn't work with
  python 2.7. Changing interface once.

  """
  return run_and_output(["mkdir", "-p", path])


def dict_prune_private(d):
  """Return a copy of a dict removing subtrees w/ keys starting w/ '_'."""

  if isinstance(d, dict):
    data = {}
    for key, value in d.items():
      if not key.startswith("_"):
        data[key] = dict_prune_private(value)
    return data
  else:
    return d


def generate_job_name(base_name):
  """Generate a unique job name from a study ID."""
  job_id = gen_timestamped_uid()
  job_name = "%s-%s" % (base_name, job_id)
  job_name = job_name.replace("_", "-")
  return job_name


def hack_dict_to_cli_args(d):
  cmd = []
  for key, value in d.items():
    cmd.append("--%s=%s" % (key, value))
  return cmd


def _compress_and_stage(local_app_root, remote_app_root):
  """Bundle the pcml codebase in `local_app_root` into a tgz and stage.

  Args:
    local_app_root(str): A path on local fs to pcml codebase root.
    remote_app_root(str): A path on GCS to which the resulting wheel
      should be staged.

  """

  _ = run_and_output(["python", "setup.py", "sdist"],
                     cwd=local_app_root)

  gz_file = "pcml-0.0.1.tar.gz"

  local_gz_file_path = os.path.join(local_app_root, "dist", gz_file)

  tf.gfile.Copy(
      local_gz_file_path,
      os.path.join(remote_app_root, gz_file),
      overwrite=True)
