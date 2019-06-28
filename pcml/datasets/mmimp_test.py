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

"""Tests of multi-modal neuro-imaging multi-problems."""

import tensorflow as tf
import os
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensor2tensor.serving import serving_utils

from pcml.utils.dev_utils import T2TDevHelper
from pcml.datasets import mmimp


class TestMMIMP(tf.test.TestCase):

  def test_e2e(self):

    helper = T2TDevHelper(
      "multi_modal_dev_model",
      "multi_modal_imaging_multi_problem_dev",
      "multi_modal_dev_model_tiny",
      None,
      tmp_dir="/tmp/celeba") # HACK: Remove this?

    helper.run_e2e()

    export_dir = helper.export_dir
    saved_model_path = os.path.join(export_dir, tf.gfile.ListDirectory(export_dir)[0])
    
    template = mmimp.MultiModalImagingMultiProblemDev().example_specification

    video = tf.reshape(template.fields["video"].mock_one(), (4*4*4*3,))
    image = tf.reshape(template.fields["image"].mock_one(zeros=True), (4*4*3,))
    eeg = tf.reshape(template.fields["eeg"].mock_one(zeros=True), (3200,))

    ex = tf.train.Example(features=tf.train.Features(feature={
      "audio": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * 100)),
      "video": tf.train.Feature(int64_list=tf.train.Int64List(value=video)),
      "image": tf.train.Feature(int64_list=tf.train.Int64List(value=image)),
      "eeg": tf.train.Feature(int64_list=tf.train.Int64List(value=eeg)),
      "problem_code": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * 1)),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * 12))
    }))

    def _make_grpc_request(examples, server_stub, servable_name, timeout_secs=5):

      request = predict_pb2.PredictRequest()
      request.model_spec.name = servable_name
      request.inputs["input"].CopyFrom(
        tf.make_tensor_proto(
                [ex.SerializeToString() for ex in examples], shape=[len(examples)]))
      response = stub.Predict(request, timeout_secs)
      outputs = tf.make_ndarray(response.outputs["outputs"])
        
      return [{
        "outputs": outputs[i]
      } for i in range(len(outputs))]

    server = helper.serve()
    stub = serving_utils._create_stub(server)
    servable_name = "multi_modal_dev_model"
    response = _make_grpc_request(examples=[ex], server_stub=stub,
                                  servable_name=servable_name)

    self.assertTrue(isinstance(response, list))
    self.assertTrue(isinstance(response[0], dict))
    self.assertTrue("outputs" in response[0].keys())
    self.assertEqual(type(response[0]["outputs"]), type(np.array([None])))
    self.assertEqual(response[0]["outputs"].shape, (12,1,1))
    self.assertEqual(response[0]["outputs"].dtype, np.float32)


if __name__ == "__main__":
  tf.test.main()
