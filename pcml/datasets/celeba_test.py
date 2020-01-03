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
"""Tests of CelebA problem definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.utils.dev_utils import T2TDevHelper
from pcml.datasets.test_utils import TrivialModel
from pcml.datasets import celeba


class TestCelebaProblemTiny(tf.test.TestCase):

  def test_e2e(self):

    helper = T2TDevHelper("trivial_model", "image_celeba_tiny",
                          "transformer_tiny", None)

    helper.datagen()

    example = helper.eager_get_example()["targets"]

    #vsize = helper.problem.vocab_size
    # Enforce that labels approximtely in range [0, vocab_size)
    #self.assertTrue(tf.reduce_all([
    #    tf.reduce_all(tf.less(example, tf.zeros_like(example) + vsize)),
    #    tf.reduce_all(tf.greater_equal(example, tf.zeros_like(example)))]).numpy())


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()
