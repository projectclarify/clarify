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
"""Tests of Bazel utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.utils import bazel_utils


class TestBazelUtils(tf.test.TestCase):

    def test_run_command(self):
        """Test we can run a simple bazel command."""

        bazel_utils.bazel_command("bazel clean")

    def test_build_and_push(self):
        """Test we can build and push docker container with Bazel."""

        bazel_utils.bazel_command("; ".join([
            "bazel clean", "bazel build //:trainer_image",
            "bazel run //:push_trainer"
        ]))

        # Requires the test container to have push permissions
        # to gcr.io/clarify

        # So for now this just puts the whl and the installation
        # script in /build and that needs to be run when the trainer
        # starts up.


if __name__ == "__main__":
    tf.test.main()
