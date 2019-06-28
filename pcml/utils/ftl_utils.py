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

"""FTL utilities.

E.g.

ftl_build_and_push(image_target_name="gcr.io/clarify/trainer:dev-0.0.1",
                   image_base_tag="gcr.io/clarify/clarify-base:latest",
                   source_build_dir="/home/jovyan/work/pcml",
                   destination="/home/jovyan/work/build",
                   destination_path="/home/jovyan/work/build",
                   virtualenv_path="/opt/conda/envs/dev/bin/virtualenv",
                   python_cmd="/opt/conda/envs/dev/bin/python")

"""

import tempfile
import os
import subprocess
import tensorflow as tf

from pcml.utils.cmd_utils import run_and_output


def fetch_ftl():
  """Download the FTL par file.

  Returns:
    str: The path to the obtained ftl.par file.

  """

  ftl_url = ("https://storage.googleapis.com/"
             "gcp-container-tools/ftl/python/"
             "latest/ftl.par")

  ftl_dir = tempfile.mkdtemp()

  os.chdir(ftl_dir)

  run_and_output(["wget", ftl_url])

  ftl_path = os.path.join(ftl_dir, "ftl.par")

  if not os.path.exists(ftl_path):
    tf.logging.error("Failed to obtain FTL.")
    return None

  return ftl_path


def try_lookup_venv_command():
  """Try looking up the virtualenv command path."""

  try:
    output = run_and_output(["which", "virtualenv"]).strip()

  except subprocess.CalledProcessError as e:
    tf.logging.warning("'virtualenv' not found, please install it.")
    raise e

  if not output:
    raise ValueError("Couldn't find a virtualenv binary in current env.")

  tf.logging.info("Found virtualenv command at path: %s" % output)

  return output


def ftl_build_and_push(image_target_name,
                       source_build_dir,
                       destination,
                       image_base_tag,
                       virtualenv_path=None,
                       ftl_path=None,
                       python_cmd=None):
  """Use FTL to build and push a container image.

  Args:
    image_target_name (str): The full image identifier string for the
      resulting image (e.g. gcr.io/project/name:tag)
    source_build_dir (str): The directory containing a requirements.txt
      file to use as the build directory which will result in those
      requirements being installed and the contents of that directory
      being staged to container /srv or an alternative path specified
      by --directory.
    image_base_tag (str): The full image identifier string for the image
      on which the build should be based (i.e. the analog of what is
      specified via FROM {base_name} in a Dockerfile).
    virtualenv_path (str): The string path to a virtualenv executable.
    ftl_path (str): The string path to an ftl.par executable.

  """

  tf.logging.info("calling build and push with args: %s" % locals())

  if ftl_path is None:
    ftl_path = fetch_ftl()

  if virtualenv_path is None:
    virtualenv_path = try_lookup_venv_command()

  # TODO: If python_cmd is not specified look it up from the system
  # default.
  if python_cmd is None:
    python_cmd = run_and_output("which python")

  cmd = ["python2", ftl_path,
         "--virtualenv-cmd", virtualenv_path,
         "--cache",
         "--name", image_target_name,
         "--directory", source_build_dir,
         "--base", image_base_tag,
         "--destination", destination,
         "--python-cmd", python_cmd]

  tf.logging.info("calling command: %s" % " ".join(cmd))

  output = run_and_output(cmd)

  return output
