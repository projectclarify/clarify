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

"""Tests of Google Container Builder utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.utils import gcb_utils


class TestGCBUtils(tf.test.TestCase):

  def test_build_and_push(self):
    """Test we can build and push a trainer image."""

    image_tag = "gcr.io/clarify/debug:0.0.1"
    build_dir = "/home/jovyan/work/pcml/build"

    gcb_utils.gcb_build_and_push(
        image_tag,
        build_dir,
        cache_from="tensorflow/tensorflow:1.6.0",
        dry_run=False)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
