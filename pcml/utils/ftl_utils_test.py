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

"""Tests of FTL utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.utils import ftl_utils


class TestFTLUtils(tf.test.TestCase):

  def test_build_and_push(self):

    # TODO: Works as long as we just want to build a container that
    # includes our code are are cool with pip installing everything at
    # runtime instead of at container build time. Including the
    # installation of PCML by including the following in a
    # requiremets.txt file
    # file:///home/jovyan/work/pcml#egg=pcml --no-index
    # causes a permissions error suggesting installation with --user.

    # TODO: Paths are specific to jlab workspace and not necessarily
    # test environment.

    ftl_utils.ftl_build_and_push(
        image_target_name="gcr.io/clarify/trainer:dev-0.0.1",
        image_base_tag="gcr.io/clarify/clarify-base:latest",
        source_build_dir="/home/jovyan/work/pcml",
        destination="/home/jovyan/work/build",
        virtualenv_path="/opt/conda/envs/py2/bin/virtualenv",
        python_cmd="/opt/conda/envs/py2/bin/python")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
