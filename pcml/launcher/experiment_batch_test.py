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

import tensorflow as tf

from pcml.launcher.experiment import configure_experiment
from pcml.launcher.kube_test import _testing_run_poll_and_check_tfjob


class TestT2TExperiment(tf.test.TestCase):

    def test_e2e_gpu_dev(self):
        """E2E tests of dev problem and model on a K80 GPU."""

        experiment = configure_experiment(
            base_name="test_e2e_gpu_dev",
            problem="multi_modal_imaging_multi_problem_dev",
            model="multi_modal_dev_model",
            hparams_set="multi_modal_dev_model_tiny",
            num_gpu_per_worker=1,
            num_train_steps=600,
            num_eval_steps=100,
            local_eval_frequency=10,
            num_workers=0,
            num_ps=0,
            ps_gpu=1,
            log_device_placement=False,
            profile=False,
            dbgprofile=False,
            trainer_memory="32Gi",
            trainer_cpu=7,
            app_root="/home/jovyan/work/pcml",
            base_image="gcr.io/clarify/clarify-base:0.0.13",
            reuse_output_dir=None,
            schedule="train_and_evaluate",
            #data_dir="/mnt/disks/ssd0",
            data_dir="gs://clarify-models-us-central1/experiments",
            remote_base="gs://clarify-models-us-central1/experiments",
            selector_labels={
                "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
            })

        # TODO: No longer works since refactor. Maybe fix.
        # i.e. this combination of model, problem, and hparams.
        """
    create_response, job_dict = experiment.batch_run()

    _testing_run_poll_and_check_tfjob(
        test_object=self,
        create_response=create_response,
        expect_in_logs=[
            'Loss for final step',
            'physical_device_desc: "device: XLA_GPU device"'
        ])
    """

    def test_e2e_tpu_dev(self):
        """E2E tests of dev problem and model on a v3 TPU.

    Runs to completion.

    """

        # Currently only works with train schedule,
        # "TPU system" not available when coming back from eval
        # https://github.com/tensorflow/tensor2tensor/issues/1202
        _ = configure_experiment(
            base_name="test_e2e_tpu_dev",
            problem="multi_modal_imaging_multi_problem_dev",
            model="multi_modal_dev_model",
            hparams_set="multi_modal_dev_model_tiny",
            num_gpu_per_worker=0,
            num_train_steps=600,
            num_eval_steps=100,
            local_eval_frequency=10,
            num_workers=0,
            num_ps=0,
            ps_gpu=0,
            log_device_placement=False,
            profile=False,
            dbgprofile=False,
            trainer_memory="7Gi",
            trainer_cpu=4,
            app_root="/home/jovyan/work/pcml",
            base_image="tensorflow/tensorflow:1.13.1-py3",
            reuse_output_dir=None,
            schedule="train",
            data_dir="gs://clarify-models-us-central1/experiments",
            remote_base="gs://clarify-models-us-central1/experiments",
            use_tpu=True,
            num_tpu_cores=8,
            tpu_tf_version="1.13",
            selector_labels={"type": "tpu-host"})
        """
    create_response, job_dict = experiment.batch_run()

    _testing_run_poll_and_check_tfjob(test_object=self,
                                      create_response=create_response,
                                      expect_in_logs=[
                                          'Loss for final step',
                                          'Found TPU system',
                                          'Shutdown TPU system'
                                      ])
    """

    def test_e2e_gpu(self):
        """E2E tests of non-dev problem and non-dev model on a K80 GPU.

    Runs to completion.

    TODO: Saw an instance of what looked like the trainer node DOS'ing the
    notebook container so may need to think more carefully about network
    usage patterns.

    """

        _ = configure_experiment(
            base_name="test_e2e_gpu",
            problem="multi_modal_imaging_multi_problem",
            model="multi_modal_model",
            hparams_set="multi_modal_model_tiny_v2",
            num_gpu_per_worker=1,
            num_train_steps=6000,
            num_eval_steps=100,
            local_eval_frequency=10,
            num_workers=0,
            num_ps=0,
            ps_gpu=1,
            log_device_placement=False,
            profile=False,
            dbgprofile=False,
            trainer_memory="40Gi",
            trainer_cpu=7,
            extra_hparams={"batch_size": 32},
            app_root="/home/jovyan/work/pcml",
            base_image="gcr.io/clarify/clarify-base:0.0.13",
            reuse_output_dir=None,
            schedule="train_and_evaluate",
            data_dir=("gs://clarify-models-us-central1/experiments/"
                      "example-scaleup4"),
            remote_base=("gs://clarify-models-us-central1/experiments/"
                         "example-scaleup4"),
            selector_labels={
                "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
            })
        """

    create_response, job_dict = experiment.batch_run()

    _testing_run_poll_and_check_tfjob(
        test_object=self,
        create_response=create_response,
        expect_in_logs=[
            'Loss for final step',
            'physical_device_desc: "device: XLA_GPU device"'
        ])

    """

    def test_e2e_tpu(self):
        """E2E tests of full problem and model on a v3 TPU.

    Runs to point of export (completes training, 6x faster than K80) then
    fails during export.

    """

        _ = configure_experiment(
            base_name="test_e2e_tpu",
            problem="multi_modal_imaging_multi_problem",
            model="multi_modal_model",
            hparams_set="multi_modal_model_tiny_v2",
            num_gpu_per_worker=0,
            num_train_steps=6000,
            num_eval_steps=30,
            local_eval_frequency=10,
            num_workers=0,
            num_ps=0,
            ps_gpu=0,
            log_device_placement=False,
            profile=False,
            dbgprofile=False,
            trainer_memory="7Gi",
            trainer_cpu=4,
            extra_hparams={"batch_size": 32},
            app_root="/home/jovyan/work/pcml",
            base_image="tensorflow/tensorflow:1.13.1-py3",
            reuse_output_dir=None,
            schedule="train",
            data_dir=("gs://clarify-models-us-central1/experiments/"
                      "example-scaleup"),
            remote_base=("gs://clarify-models-us-central1/experiments/"
                         "example-scaleup"),
            use_tpu=True,
            num_tpu_cores=8,
            tpu_tf_version="1.13",
            selector_labels={"type": "tpu-host"})
        """

    create_response, job_dict = experiment.batch_run()

    _testing_run_poll_and_check_tfjob(test_object=self,
                                      create_response=create_response,
                                      expect_in_logs=[
                                          'Loss for final step',
                                          'Found TPU system',
                                          'Shutdown TPU system'
                                      ])
    """

    def test_e2e_gpu_ut(self):
        """E2E tests of UT model on a K80 GPU.

    Runs to completion.

    """

        _ = configure_experiment(
            base_name="test_e2e_gpu_ut",
            problem="multi_modal_imaging_multi_problem",
            model="multi_modal_model_ut",
            hparams_set="multi_modal_model_tiny_v2",
            num_gpu_per_worker=1,
            num_train_steps=3000,
            num_eval_steps=300,
            local_eval_frequency=10,
            num_workers=0,
            num_ps=0,
            ps_gpu=1,
            log_device_placement=False,
            profile=False,
            dbgprofile=False,
            trainer_memory="40Gi",
            trainer_cpu=7,
            extra_hparams={"batch_size": 32},
            app_root="/home/jovyan/work/pcml",
            base_image="gcr.io/clarify/clarify-base:0.0.13",
            reuse_output_dir=None,
            schedule="train_and_evaluate",
            data_dir=("gs://clarify-models-us-central1/experiments/"
                      "example-scaleup"),
            remote_base=("gs://clarify-models-us-central1/experiments/"
                         "example-scaleup"),
            selector_labels={
                "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
            })
        """

    create_response, job_dict = experiment.batch_run()

    _testing_run_poll_and_check_tfjob(
        test_object=self,
        create_response=create_response,
        expect_in_logs=[
            'Loss for final step',
            'physical_device_desc: "device: XLA_GPU device"'
        ])

    """

    def test_e2e_tpu_ut(self):
        """E2E tests of full problem and model on a v3 TPU with UT variant.

    Runs to point of export (completes training, 6x faster than K80) then
    fails during export.

    """

        experiment = configure_experiment(
            base_name="test_e2e_tpu_ut",
            problem="multi_modal_imaging_multi_problem",
            model="multi_modal_model_ut",
            hparams_set="multi_modal_model_tiny_v2",
            num_gpu_per_worker=0,
            num_train_steps=100,
            num_eval_steps=30,
            local_eval_frequency=10,
            num_workers=0,
            num_ps=0,
            ps_gpu=0,
            log_device_placement=False,
            profile=False,
            dbgprofile=False,
            trainer_memory="7Gi",
            trainer_cpu=4,
            extra_hparams={"batch_size": 48},
            app_root="/home/jovyan/work/pcml",
            base_image="tensorflow/tensorflow:1.13.1-py3",
            reuse_output_dir=None,
            schedule="train",
            data_dir=("gs://clarify-models-us-central1/experiments/"
                      "example-scaleup5"),
            remote_base=("gs://clarify-models-us-central1/experiments/"
                         "example-scaleup5"),
            use_tpu=True,
            num_tpu_cores=8,
            tpu_tf_version="1.13",
            selector_labels={"type": "tpu-host"})

        create_response, _ = experiment.batch_run()

        _testing_run_poll_and_check_tfjob(test_object=self,
                                          create_response=create_response,
                                          expect_in_logs=[
                                              'Loss for final step',
                                              'Found TPU system',
                                              'Shutdown TPU system'
                                          ])


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()
