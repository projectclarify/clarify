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

"""File-system utilities."""

import os
import inspect
import tensorflow as tf
import shutil
import tempfile


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    if not self.name.startswith("/tmp"):
      raise ValueError("Safety first!")
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)


def upsearch(directory, query, max_levels=3):
  """Recursively search parent dirs for one containing `query` file.

  The root of the Bazel workspace is indicated by the presence of
  a WORKSPACE file and when we want to run commands specifically
  from the root of that workspace we first need to discover its
  path.

  Args:
    directory(str): A filesystem directory path.
    query(str): The name of a file for which to search.
    max_levels(int): The maximum depth of recursion.

  """
  
  tf.logging.info("Upsearched to {}".format(directory))

  if max_levels < 0:
    return None

  directory_contents = tf.gfile.ListDirectory(directory)
  tf.logging.info("Directory contents {}".format(directory_contents))

  if query in directory_contents:
    return directory

  parent_dir = "/".join(directory.split("/")[:-1])

  return upsearch(parent_dir, query, max_levels=(max_levels-1))


def get_pcml_root():

  this_dir = os.path.dirname(
      os.path.abspath(inspect.getfile(inspect.currentframe())))
  tf.logging.info("Searching from pcml root from {}".format(this_dir))

  return upsearch(this_dir, "MANIFEST.in")


def expect_path(path):
    """Check that a path exists (and is a valid).
    
    Args:
        path (str): An absolute, user-space path (e.g. /mnt/nfs/foo).
    
    Raises:
        ValueError: If the path is not a string that starts with '/'
          referring to extant path.

    """

    if not isinstance(path, str):
        raise ValueError("Paths must be of type string, saw: %s" % path)

    if not path.startswith("/"):
        raise ValueError(
            "Expected an absolute, user-space path, saw: %s" % path)

    if not tf.gfile.Exists(path):
        raise ValueEror("Path does not exist: %s" % path)
