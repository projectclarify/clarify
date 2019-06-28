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

"""Tests of utilities for constructing Katib StudyJob's."""

import tensorflow as tf

from tensor2tensor.utils import registry

from pcml.launcher.study import T2TKubeStudy
from pcml.launcher.experiment import configure_experiment


class TestStudyJob(tf.test.TestCase):

  def test_instantiate_study_job(self):
    """Test that we can instantaite a T2TKubeStudy."""

    # TODO: Disable linter's warning about unused variable for
    # study_job_test_rhp

    @registry.register_ranged_hparams
    def study_job_test_rhp(rhp):
      rhp.set_categorical("recurrence_type", ["act", "basic"])
      rhp.set_discrete("num_heads", [4, 8, 16, 32])
      rhp.set_discrete("filter_size", [128, 256, 512, 1024, 2048, 4096])
      rhp.set_discrete("hidden_size", [128, 256, 512, 1024, 2048, 4096])
      rhp.set_discrete("batch_size", [16, 32, 48, 64, 96, 128])

    experiment = configure_experiment(
        base_name="tpu_ut_tune_2",
        problem="multi_modal_imaging_multi_problem",
        model="multi_modal_model_ut",
        hparams_set="multi_modal_model_tiny_v2",
        num_gpu_per_worker=0,
        num_train_steps=1500,
        num_eval_steps=30,
        local_eval_frequency=10,
        trainer_memory="7Gi",
        trainer_cpu=4,
        extra_hparams={},
        app_root="/home/jovyan/work/pcml",
        base_image="tensorflow/tensorflow:1.13.1-py3",
        schedule="train",
        data_dir="gs://clarify-dev/dummy",
        remote_base="gs://clarify-dev/dummy",
        use_tpu=True,
        num_tpu_cores=8,
        tpu_tf_version="1.13",
        selector_labels={"type": "tpu-host"},
        use_katib=True)

    _ = T2TKubeStudy(
        study_name="foo",
        study_ranged_hparams="study_job_test_rhp",
        experiment=experiment)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()
