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

"""FEC model tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.models import percep_similarity_emb
from tensor2tensor.utils import registry


class TestPercepSimilarityModel(tf.test.TestCase):
  
  def setUp(self):
    self.model_name = "percep_similarity_triplet_emb"
    self.problem_name = "fec_tiny"
    self.hparams_name = "percep_similarity_triplet_emb_tiny"

  def test_registry_lookups(self):

    model_names = [
      "percep_similarity_triplet_emb",
      "percep_similarity_triplet_classify"
    ]

    for model_name in model_names:
      _ = registry.model(model_name)

      
    hparams_sets = [
      "percep_similarity_triplet_emb",
      "percep_similarity_triplet_emb_tiny"
    ]
    
    for hparams_set in hparams_sets:
      _ = registry.hparams(hparams_set)

  def test_tiny_e2e(self):
    pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
