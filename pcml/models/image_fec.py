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

from pcml.models.dev import resnet_wrapper
from pcml.models.dev import MCLDev3
from pcml.models import modality_correspondence as mcl


@registry.register_hparams
def percep_similarity_triplet_emb():
  hparams = mcl.mcl_res_ut()
  hparams.batch_size = 128
  return hparams


@registry.register_hparams
def percep_similarity_triplet_emb_tiny():
  hparams = percep_similarity_triplet_emb()
  # TODO
  return hparams


def _l2_distance(e1, e2):
  diff = e1 - e2
  diff = tf.multiply(diff, diff)
  return tf.sqrt(tf.reduce_sum(diff, -1))


def _ordered_triplet_loss_partial(e1, e2, e3, delta=0.1):

  k = tf.constant(delta, dtype=tf.float32)

  loss = tf.clip_by_value(
      _l2_distance(e1, e2) - _l2_distance(e1, e3) + k, 0.0, 2.0)

  loss += tf.clip_by_value(
      _l2_distance(e1, e2) - _l2_distance(e2, e3) + k, 0.0, 2.0)

  return loss


def _triplet_loss_partial(e1, e2, e3, similarity_condition, delta=0.1):

  similarity_condition = tf.cast(similarity_condition, tf.int64)

  key1 = tf.ones_like(similarity_condition, dtype=tf.int64)
  mask1 = tf.cast(tf.equal(similarity_condition, key1), tf.float32)

  key2 = tf.ones_like(similarity_condition, dtype=tf.int64) * 2
  mask2 = tf.cast(tf.equal(similarity_condition, key2), tf.float32)

  key3 = tf.ones_like(similarity_condition, dtype=tf.int64) * 3
  mask3 = tf.cast(tf.equal(similarity_condition, key3), tf.float32)

  otlp1 = _ordered_triplet_loss_partial(e2, e3, e1, delta=delta)
  loss1 = tf.multiply(otlp1, mask1)

  otlp2 = _ordered_triplet_loss_partial(e1, e3, e2, delta=delta)
  loss2 = tf.multiply(otlp2, mask2)

  otlp3 = _ordered_triplet_loss_partial(e1, e2, e3, delta=delta)
  loss3 = tf.multiply(otlp3, mask3)

  return loss1 + loss2 + loss3


def obtain_triplet_embeddings(num, data_dir, ckpt_dir, moode, hparams_set_name,
                              model_name, problem_name):

  registered_model = registry.model(model_name)
  hparams = registry.hparams(hparams_set_name)
  hparams.mode = mode

  problem_instance = registry.problem(problem_name)
  problem_hparams = problem_instance.get_hparams(hparams)

  dataset = problem_instance.dataset(mode=mode, data_dir=data_dir)

  dataset = dataset.repeat(None).batch(batch_size)
  dataset_iterator = tfe.Iterator(dataset)

  embeddings = {}

  with tfe.restore_variables_on_create(ckpt_dir):

    model_instance = registered_model(hparams, mode, problem_hparams)

    eval_examples = dataset_iterator.next()

    predictions, _ = model_instance(eval_examples)

    examples_a = eval_examples["image/a/noaug"].numpy()
    examples_b = eval_examples["image/b/noaug"].numpy()
    examples_c = eval_examples["image/c/noaug"].numpy()
    examples = np.append(examples_a, examples_b, axis=0)
    examples = np.append(examples, examples_c, axis=0)

    predictions = predictions.numpy()
    predictions_a = predictions[0]
    predictions_b = predictions[1]
    predictions_c = predictions[2]
    predictions = np.append(predictions_a, predictions_b, axis=0)
    predictions = np.append(predictions, predictions_c, axis=0)

  data = {}
  for i, _ in enumerate(predictions):
    data[i] = {"emb": predictions[i], "img": examples[i]}

  X = np.asarray([np.asarray(thing) for thing in predictions])
  kdt = KDTree(X, leaf_size=30, metric='euclidean')

  return data, kdt


def make_and_show_img_similarity_query(data, kdt, idx):

  k = 8
  return_distance = True
  query = [data[idx]["emb"]]
  dist, ind = kdt.query(query, k=k, return_distance=return_distance)

  plt.figure()
  f, axarr = plt.subplots(k, 2, figsize=(4, 12), dpi=100)

  img = data[ind[0][0]]["img"].astype(np.int32)
  axarr[0, 0].imshow(img)
  axarr[0, 0].set_title("query")
  axarr[0, 0].axis('off')

  for i in range(k):
    axarr[i, 0].axis("off")
    axarr[i, 1].axis("off")

  for i in range(k - 1):
    img = data[ind[0][i + 1]]["img"].astype(np.int32)
    axarr[i, 1].imshow(img)
    d = int(dist[0][i + 1] * 1000) / 1000.0
    axarr[i, 1].set_title("d={}".format(d))
    axarr[i, 1].axis("off")

  similarity_condition = tf.cast(similarity_condition, tf.int64)

  key1 = tf.ones_like(similarity_condition, dtype=tf.int64)
  mask1 = tf.cast(tf.equal(similarity_condition, key1), tf.float32)

  key2 = tf.ones_like(similarity_condition, dtype=tf.int64) * 2
  mask2 = tf.cast(tf.equal(similarity_condition, key2), tf.float32)

  key3 = tf.ones_like(similarity_condition, dtype=tf.int64) * 3
  mask3 = tf.cast(tf.equal(similarity_condition, key3), tf.float32)

  otlp1 = _ordered_triplet_loss_partial(e2, e3, e1, delta=delta)
  loss1 = tf.multiply(otlp1, mask1)

  otlp2 = _ordered_triplet_loss_partial(e1, e3, e2, delta=delta)
  loss2 = tf.multiply(otlp2, mask2)

  otlp3 = _ordered_triplet_loss_partial(e1, e2, e3, delta=delta)
  loss3 = tf.multiply(otlp3, mask3)

  return loss1 + loss2 + loss3


@registry.register_model
class PercepSimilarityTripletEmb(MCLDev3):
  """Triplet perceptual similarity embedding model.

  From the FEC dataset documentation:

  "Each annotation is an integer from the set {1, 2, 3}. A value of 1 means the expressions on
  second and third faces in the triplet are visually more similar to each other when compared to
  the expression on the first face. A value of 2 means the expressions on the first and third faces
  in the triplet are visually more similar to each other when compared to the expression on the
  second face. A value of 3 means the expressions on the first and second faces in the triplet are
  visually more similar to each other when compared to the expression on the third face."

  I.e.

  1: d(2,3) < d(2,1) and
     d(2,3) < d(3,1)

  2: d(1,3) < d(1,2) and
     d(1,3) < d(3,2)

  3: d(1,2) < d(1,3) and
     d(1,2) < d(2,3)

  L1(e1, e2, e3) = max(0, d(e1, e2) - d(e1, e3) + k) + max(0, d(e1, e2) - d(e2, e3) + k)

  """

  @property
  def modality_embedding_size(self):
    return 512

  def infer(self, features=None, **kwargs):
    del kwargs
    predictions, _ = self(features)
    return predictions

  def eager_eval(self, features):

    predictions, _ = self(features)
    predictions = predictions.numpy()

    e1 = tf.convert_to_tensor(predictions[0])
    e2 = tf.convert_to_tensor(predictions[1])
    e3 = tf.convert_to_tensor(predictions[2])

    partial = _triplet_loss_partial(e1,
                                    e2,
                                    e3,
                                    features["triplet_code"],
                                    delta=0.0)

    # A vector of either 1 or 0 depending on whether each example in the
    # batch is either accuracte or not accurate
    unreduced_accuracy = tf.equal(partial, tf.constant(0.0))

    # The total number of accuracte predictions devided by the number
    # of predictions
    accuracy = tf.reduce_mean(tf.cast(unreduced_accuracy, tf.float32))

    metrics = {"triplet_accuracy": accuracy.numpy()}

    return metrics

  def body(self, features):

    e1 = self.embed_image(features["image/a"])
    e2 = self.embed_image(features["image/b"])
    e3 = self.embed_image(features["image/c"])

    if "image/b" in features:
      partial = _triplet_loss_partial(e1,
                                      e2,
                                      e3,
                                      features["triplet_code"],
                                      delta=0.1)
      loss = tf.reduce_mean(partial)
    else:
      loss = 0.0

    return tf.stack([e1, e2, e3], 0), {"training": loss}


@registry.register_model
class PercepSimilarityTripletClassify(PercepSimilarityTripletEmb):
  """Classify & use classification loss, for comparsion to model trained to label AffectNet."""

  def body(self, features):
    hp = self.hparams

    if "image/b" in features:
      # Placeholder
      loss = tf.losses.mean_squared_error(tf.squeeze(features["image/a"]),
                                          tf.squeeze(features["image/b"]))
    else:
      loss = 0.0

    return features["image/a"], {"training": loss}
