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

import tensorflow as tf

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import algorithmic

from clarify.utils.dev_utils import T2TDevHelper

from clarify.utils.cfg_utils import Config

TEST_CONFIG = Config()


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

  def test_e2e(self):
    """End-to-end test of the dev helper utility."""

    tfms_path = TEST_CONFIG.get("tfms_path")

    import tensor2tensor.models
    helper2 = T2TDevHelper("trivial_model_t2tdh",
                           "tiny_algo_problem",
                           "transformer_tiny", [["1 0 0 1"]],
                           tfms_path=tfms_path)

    helper2.datagen()
    helper2.train()
    # TODO: Re-include e2e if retaining T2TDevHelper, note this
    # currently has removed the model export, serve, and query
    # steps.


if __name__ == "__main__":
  tf.test.main()
