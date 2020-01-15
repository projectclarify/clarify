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

import json
from sklearn.neighbors import KDTree
import numpy as np
import os
import tempfile

from google.cloud import firestore
import numpy as np
import datetime

checkpoint_path = "...?"

client = firestore.Client()


class DataPacket:

  def __init__(self, raw_data):

    self.audio_data = raw_data["audioData"]["stringValue"]
    self.video_data = raw_data["videoData"]["stringValue"]
    self.audio_meta = {}
    self.video_meta = {}
    self.meta = {}

    for key, value in raw_data["audioMeta"]["mapValue"]["fields"].items():
      self.audio_meta[key] = value["integerValue"]

    for key, value in raw_data["videoMeta"]["mapValue"]["fields"].items():
      self.video_meta[key] = value["integerValue"]

    for key, value in raw_data["meta"]["mapValue"]["fields"].items():
      self.meta[key] = value["integerValue"]


class LabeledEmbeddingSpace(object):
  """Usage:
  
  space = LabeledEmbeddingSpace(given_data="/path/to/data")
  space.build_kdtree()

  query = [0.1, 0.3, 0.2, 0.9]
  consensus_labels = space.consensus_query(query)
  
  """

  def __init__(self, given_data, generate_random=False):
    if generate_random:
      self.generate_random(given_data)
    self.emb_data = self.load_emb_data(given_data)

  def generate_random(self,
                      emb_data_path,
                      embedding_length=2048,
                      num_embeddings=100):

    ref = []

    for _ in range(num_embeddings):
      emb = np.random.random((embedding_length,)).tolist()
      ref.append({
          "embedding": emb,
          "labels": {
              "happiness": float(np.random.random()),
              "calm": float(np.random.random()),
              "confidence": float(np.random.random()),
              "kindness": float(np.random.random()),
              "focus": float(np.random.random()),
              "posture": float(np.random.random()),
              "flow": float(np.random.random())
          }
      })

    serialized_data = json.dumps(ref)

    with open(emb_data_path, "w") as f:
      f.write(serialized_data)

    return serialized_data

  def load_emb_data(self, emb_data_path):
    with open(emb_data_path, "r") as f:
      self.data = json.loads(f.read())

  def build_kdtree(self):

    X = np.asarray([np.asarray(thing["embedding"]) for thing in self.data])

    self.kdt = KDTree(X, leaf_size=30, metric='euclidean')

  def consensus_labels(self, result_indices):

    consensus = None
    num_results = len(result_indices)

    err = "expected all label sets to have the same keys"

    for idx in result_indices:
      labels = self.data[idx]["labels"]
      if not consensus:
        consensus = labels
      else:
        for key, value in labels.items():
          if key not in consensus:
            raise Exception(err)
          else:
            consensus[key] += value

    for key, value in consensus.items():
      consensus[key] = consensus[key] / num_results

    return consensus

  def consensus_query(self, query, k=10, return_distance=False):
    results = self.kdt.query(query, k=k, return_distance=return_distance)
    return self.consensus_labels(results[0])


space = None


def label_state(data, context):
  """Glue between video in Firestore and queries to KFServing.

  Notes:
  * Function is triggered by a change to a Firestore document.
  * When called, function validates changed data and constructs
    a serving query.
  * Query is made against tensorflow model deployed on Kubernetes
    via KFServing.
  * The model's prediction is received and written to the
    Firebase database.

  Args:
    data (dict): The event payload.
    context (google.cloud.functions.Context): Metadata for the event.

  """

  global space

  # Lazy-load the embedding space object on first invocation instead of on
  # cold-start to deal with unclear issue of gcloud.functions.deploy not being
  # able to find label_state function if this code is in the global scope above
  # as well as because the docs recommend this pattern.
  if not space:

    tmpdir = tempfile.gettempdir()

    # NOTE: For dev purposes initially this is using random embedding vectors.
    space = LabeledEmbeddingSpace(given_data="{}/emb.json".format(tmpdir),
                                  generate_random=True)

    space.build_kdtree()

  path_parts = context.resource.split('/documents/')[1].split('/')
  collection_path = path_parts[0]
  document_path = '/'.join(path_parts[1:])
  uid = path_parts[1]

  affected_doc = client.collection(collection_path).document(document_path)

  data_packet = DataPacket(data["value"]["fields"])

  # ========
  # NOTE: This is currently mocked data.
  image_data = np.zeros((28, 28, 1)).astype(np.float32)
  out = np.random.random((1,2048))
  # ========

  #out = model.eager_embed_single_image(image_data)

  # (1, 2048)
  query = np.asarray([out])

  logging.info(query)

  consensus_labels = space.consensus_query(query)

  update = {
      u'state': {
          u'data': [],
          u'meta': {
              u'timestamp': datetime.datetime.utcfromtimestamp(0)
          }
      }
  }

  targets = {
      "happiness": 0.5,
      "calm": 0.5,
      "confidence": 0.5,
      "kindness": 0.5,
      "focus": 0.5,
      "posture": 0.5,
      "flow": 0.5
  }

  for key, value in consensus_labels.items():
    update[u'state'][u'data'].append({
        "label": key,
        "current": value,
        "target": targets[key]
    })

  target_doc = client.collection(collection_path).document(uid)
  target_doc.set(update)
