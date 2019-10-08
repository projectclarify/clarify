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

"""Perceptual similarity embedding network and related from Vemulapalli and Agarwala (2019).

Vemulapalli, Raviteja, and Aseem Agarwala. "A Compact Embedding for Facial Expression Similarity." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensor2tensor.utils import t2t_model

from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry


@registry.register_hparams
def percep_similarity_triplet_emb():
  hparams = common_hparams.basic_params1()
  # TODO
  return hparams


@registry.register_hparams
def percep_similarity_triplet_emb_tiny():
  hparams = percep_similarity_triplet_emb()
  # TODO
  return hparams


@registry.register_model
class PercepSimilarityTripletEmb(t2t_model.T2TModel):
  """Triplet perceptual similarity embedding model."""

  def body(self, features):
    hp = self.hparams
    return features["inputs"]


@registry.register_model
class PercepSimilarityTripletClassify(PercepSimilarityTripletEmb):
  """Classify & use classification loss, for comparsion to model trained to label AffectNet."""

  def body(self, features):
    hp = self.hparams
    return features["inputs"]
