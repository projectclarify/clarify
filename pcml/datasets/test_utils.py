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

"""Common test utils related to data generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import inspect

import tensorflow as tf


def PIL_Image():  # pylint: disable=invalid-name
  """PIL Image importer.

  For cases where we want to avoid importing PIL unless specifically needed.
  """

  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    if not self.name.startswith("/tmp"):
      raise ValueError("Safety first!")
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)


def get_dataset_data_root(name):
  """Given named dataset and TEST_DATA_ROOT from env constructs local path.
  
  Args:
    name(str): The name of a dataset, will be created if necessary at
      $(TEST_DATA_ROOT)/`name`.
  """
  if "TEST_DATA_ROOT" not in os.environ:
    raise ValueError("Variable TEST_DATA_ROOT not found in environment.")
  tdr = os.environ["TEST_DATA_ROOT"]
  dataset_root = os.path.join(tdr, name)
  tf.gfile.MakeDirs(dataset_root)
  return dataset_root


def _verify_local_given_manifest(local_path,
                                 remote_stem,
                                 remote_manifest):
  """Verify files in manifest are present in `local_path`.

  Trim remote_stem from prefix of remote_manifest to determine paths
  within local_path where files are expected to reside.

  Args:
    local_path(str): A local directory path beneath which the suffix
      of elements in `remote_manifest` after subtracting `remote_stem`
      should be present. E.g. gs://my/path/to/file.txt with `remote_stem`
      gs://my/path should be present in `local_path`/to/file.txt.
    remote_stem(str): The remote GCS bucket path to which the manifest
      refers, as described in previous.
    remote_manifest(str): A list of GCS file paths that should be present
      in `local_path` with structure as described above.

  """

  with TemporaryDirectory() as tmp_dir:
    local_manifest_file = os.path.join(tmp_dir, "manifest.txt")
    tf.gfile.Copy(remote_manifest, local_manifest_file)
    with open(local_manifest_file, "r") as f:
      for remote_fname in f:
        remote_fname = remote_fname.strip()
        suffix = remote_fname.split(remote_stem)[1]
        if suffix.startswith("/"):
          suffix = suffix[1:]
        expected_local = os.path.join(local_path, suffix)

        if not tf.gfile.Exists(expected_local):
          msg = ("Could not verify local tree given manifest. ",
                 "The manifest %s was used " % remote_manifest,
                 "to determine files in local path %s " % local_path,
                 "do not match the expected file structure. Please ",
                 "use the manifest as a reference and re-structure ",
                 "your local files to match. Specifically could not ",
                 "find local version of %s " % remote_fname,
                 "(at expected location %s)." % expected_local)
          raise Exception("".join(msg))

  return True


def get_test_data_path():
  """Gets the path to a folder named 'test_data' in same dir as curr. file."""
  return os.path.abspath(os.path.join(
      inspect.getfile(inspect.currentframe()), "../test_data"))


@registry.register_model
class TrivialModel(t2t_model.T2TModel):
  def body(self, features):
    return features["targets"]
  def infer(self, features=None, **kwargs):
    del kwargs
    predictions, _ = self(features)
    return predictions
