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

"""Tests of modality correspondence learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.launcher.experiment import configure_experiment
from pcml.launcher.kube_test import _testing_run_poll_and_check_tfjob

from tensor2tensor.utils import registry

from pcml.utils.fs_utils import get_pcml_root

from pcml.utils import dev_utils

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


def tag_simplifier(input_string):
  out = []
  a = input_string.split("_")
  for string_component in a:
    if len(string_component) < 3:
      out.append(string_component)
      continue
    out.append(string_component[:3])
  return "-".join(out)


class TestModel(tf.test.TestCase):
  
  def setUp(self):
    self.model_name = "modality_correspondence_learner"
    self.problem_name = "vox_celeb_cbt"
    self.hparams_name = "mcl_res_ut_tiny"

  def test_lookup_model(self):
    registry.model(self.model_name)

  def test_local_small_hparams_one_step(self):
    pass
    """
    dev_utils.T2TDevHelper(
      self.model_name,
      self.problem_name,
      self.hparams_name,
      None
    ).eager_train_one_step()
    """

  def test_local_small_hparams_e2e(self):
    pass

    #dev_utils.T2TDevHelper(
    #  self.model_name, self.problem_name, self.hparams_name, None
    #).run_e2e()

    """
    TODO: Add this

    def _make_grpc_request(examples):

        request = predict_pb2.PredictRequest()
        request.model_spec.name = servable_name
        request.inputs["input"].CopyFrom(
            tf.make_tensor_proto(
                [ex.SerializeToString() for ex in examples], shape=[len(examples)]))
        response = stub.Predict(request, timeout_secs)
        outputs = tf.make_ndarray(response.outputs["outputs"])
        return [{
            "outputs": outputs[i],
        } for i in range(len(outputs))]

    export_dir = helper.export_dir
    saved_model_path = os.path.join(export_dir, tf.gfile.ListDirectory(export_dir)[0])

    example = helper.eager_get_example()

    single_video_unflat = tf.cast(example["video"], tf.int64).numpy()[0]

    encoded_video = array2gif(single_video_unflat)

    ex = tf.train.Example(features=tf.train.Features(feature={
      "audio": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * 100)),
      "frames/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_video] * 1)),
      "frames/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=['GIF'.encode()] * 1)),
      "problem_code": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * 1)),
      "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * 12))
    }))

    server = helper.serve()

    timeout_secs = 5
    stub = serving_utils._create_stub(server)
    servable_name = "multi_modal_dev_model"
    _make_grpc_request([ex])
  
    """


  def test_batch_tpu(self):

    tag = tag_simplifier("test-{}".format(self.problem_name))

    experiment = configure_experiment(
      base_name=tag,
      problem=self.problem_name,
      model=self.model_name,
      hparams_set=self.hparams_name,
      num_gpu_per_worker=0,
      num_train_steps=30000,
      num_eval_steps=30,
      local_eval_frequency=10,
      trainer_memory="4Gi",
      trainer_cpu=1,
      app_root="/home/jovyan/work/pcml",
      base_image="gcr.io/clarify/basic-runtime:0.0.4",
      schedule="train",

      # Need mod to avoid need for data dir
      data_dir="gs://clarify-models-us-central1/experiments/example-scaleup18",

      remote_base="gs://clarify-models-us-central1/experiments/cbtdev",
      use_tpu=True,
      num_tpu_cores=8,
      tpu_tf_version="1.13",
      selector_labels={"type": "tpu-host"})

    experiment.batch_run()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
