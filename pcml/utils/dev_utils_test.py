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

"""Tests of the development helper utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import algorithmic

from pcml.utils.dev_utils import T2TDevHelper


@registry.register_problem
class TinyAlgoProblem(algorithmic.AlgorithmicIdentityBinary40):
  """A tiny algorithmic problem to aid testing (quickly)."""

  @property
  def num_symbols(self):
    return 2

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 40

  @property
  def train_size(self):
    return 10

  @property
  def dev_size(self):
    return 10

  @property
  def num_shards(self):
    return 1


@registry.register_model
class TrivialModelT2tdh(t2t_model.T2TModel):
  def body(self, features):
    return features["inputs"]


class TestDevHelper(tf.test.TestCase):

  def test_finds_tfms_path(self):
    """Test of the maybe_lookup_tfms_path method."""

    helper = T2TDevHelper("trivial_model_t2tdh",
                          "tiny_algo_problem",
                          "transformer_tiny",
                          [["1 0 0 1"]])

    helper.maybe_lookup_tfms_path()

    self.assertTrue(helper.tf_model_server_path is not None)

  def test_e2e(self):
    """End-to-end test of the dev helper utility."""

    helper2 = T2TDevHelper("trivial_model_t2tdh",
                           "tiny_algo_problem",
                           "transformer_tiny",
                           [["1 0 0 1"]])

    helper2.run_e2e()


if __name__ == "__main__":
  tf.test.main()
