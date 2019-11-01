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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KDTree

import tensorflow as tf

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name

from pcml.utils.dev_utils import T2TDevHelper
import tempfile

from tensor2tensor.utils import registry

temp = tempfile.mkdtemp()

from pcml.datasets.fec import FacialExpressionCorrespondence as FEC

from sklearn.neighbors import KDTree
import numpy as np

@registry.register_problem
class FECNoAug(FEC):

  @property
  def name_override(self):
    return "facial_expression_correspondence"

  def preprocess_example(self, example, mode, unused_hparams):
    example["image/a/noaug"] = example["image/a"]
    example["image/b/noaug"] = example["image/b"]
    example["image/c/noaug"] = example["image/c"]
    return super(FECNoAug, self).preprocess_example(example,
                                                    mode,
                                                    unused_hparams)


def obtain_triplet_embeddings(batch_size, data_dir, ckpt_dir, mode, hparams_set_name,
                              model_name, problem_name, num_batches):

  registered_model = registry.model(model_name)
  hparams = registry.hparams(hparams_set_name)
  hparams.mode = mode

  problem_instance = registry.problem(problem_name)
  problem_hparams = problem_instance.get_hparams(hparams)

  dataset = problem_instance.dataset(mode=mode, data_dir=data_dir)

  dataset = dataset.repeat(None).batch(batch_size)
  dataset_iterator = tfe.Iterator(dataset)

  data = {}
  num_failed = 0
  max_idx = 0

  with tfe.restore_variables_on_create(ckpt_dir):

    model_instance = registered_model(
      hparams, mode, problem_hparams
    )

    for j in range(num_batches):

      tf.logging.info("embedding for batch {} of {}".format(
        j, num_batches
      ))

      eval_examples = dataset_iterator.next()
      try:
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

          for i, _ in enumerate(predictions):
            idx = (j-num_failed)*batch_size*3 + i
            if idx > max_idx:
              max_idx = idx
            data[str(idx)] = {
              "emb": predictions[i].tolist(), "img": examples[i].tolist()}

      except:
        # TODO: Specifically except CBT deadline exceeded.
        num_failed += 1

  all_predictions = [None for _ in range(max_idx)]
  for i in range(max_idx):
    all_predictions[i] = data[str(i)]["emb"]

  return data, all_predictions


def restore_embedding_data(path):

  with open("/tmp/embedding.json", "r") as f:
    data, predictions = json.loads(f.read())

  tf.logging.info("Computing kdtree...")
  predictions = np.asarray([np.asarray(thing) for thing in predictions])
  kdt = KDTree(predictions, leaf_size=30, metric='euclidean')

  for key, value in data.items():
    data[key] = {
        "emb": np.asarray(data[key]["emb"]),
        "img": np.asarray(data[key]["img"])
    }

  return data, predictions, kdt


def make_and_show_img_similarity_query(query_data, ref_data, kdt, query_idx, num_hits=4):

  from matplotlib import pyplot as plt

  num_column_major = 5

  k = num_hits
  return_distance = True

  plt.figure()
  f, axarr = plt.subplots(k, 2*num_column_major, figsize=(num_column_major*4,12), dpi=150)

  for j in range(num_column_major):

    query = [query_data[str(query_idx + j)]["emb"]]
    dist, ind = kdt.query(query, k=k, return_distance=return_distance)

    img = ref_data[str(ind[0][0])]["img"].astype(np.int32)
    axarr[0,2*j].imshow(img)
    axarr[0,2*j].set_title("query")
    axarr[0,2*j].axis('off')

    for i in range(k):
      axarr[i,2*j].axis("off")
      axarr[i,2*j+1].axis("off")

    for i in range(k-1):
      img = ref_data[str(ind[0][i+1])]["img"].astype(np.int32)
      axarr[i, 2*j+1].imshow(img)
      d = int(dist[0][i+1]*1000)/1000.0
      axarr[i, 2*j+1].set_title("d={}".format(d))
      axarr[i, 2*j+1].axis("off")


def write_embeddings(data, predictions, path):
  with open(path, "w") as f:
    f.write(json.dumps([data, predictions]))
