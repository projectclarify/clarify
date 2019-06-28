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

"""General test utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
import tensorflow as tf

from pcml.utils.fs_utils import upsearch


def maybe_get_tfms_path():
  """Try to obtain path to tensorflow_model_server."""
  
  key = "TENSORFLOW_MODEL_SERVER_PATH"

  if key in os.environ:

    path = os.environ[key]

    if not path.endswith("tensorflow_model_server"):
      raise ValueError("Saw malformed tfms path from env var, %s" % path)

  else:
    path = get_tfms_path_from_testing()

  if path is None:
    tf.logging.warning(
        "Could not obtain tfms path from env var or pcml testing."
    )

  return path


def get_tfms_path_from_testing():
  """Obtain tensorflow_model_server path from PCML serving tools dir."""

  d = os.path.dirname(
      os.path.abspath(inspect.getfile(inspect.currentframe())))

  pcml_root = upsearch(d, query="WORKSPACE")

  tfms_path = os.path.join(pcml_root, "tools", "serving",
                           "tensorflow_model_server")

  if not os.path.exists(tfms_path):
    raise ValueError("Could not find tensorflow_model_server.")

  return os.path.realpath(tfms_path)
