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

"""Development helper utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import atexit
import subprocess
import socket
import shlex
import grpc
import time

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry
from tensor2tensor.utils import decoding

#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name

RPC_TIMEOUT = 15
WAIT_FOR_SERVER_READY_INT_SECS = 60


def pick_unused_port():
  s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
  s.bind(('', 0))
  port = s.getsockname()[1]
  s.close()
  return port


def run_server(model_name, model_path, tfms_path):
  """Call tensorflow_model_server to serve """

  port = pick_unused_port()
  rest_api_port = pick_unused_port()

  tf.logging.info('Starting test server on port: {} for model_name: '
                  '{}'.format(port, model_name))

  command = tfms_path
  command += ' --port=' + str(port)
  command += ' --model_name=' + model_name
  command += ' --model_base_path=' + model_path
  tf.logging.info(command)

  proc = subprocess.Popen(shlex.split(command), stderr=None)
  atexit.register(proc.kill)
  tf.logging.info('Server started')

  hostports = (
      proc,
      'localhost:' + str(port),
      'localhost:' + str(rest_api_port),
  )

  return hostports


def wait_for_server_ready(server):
  """Waits for a server on the localhost to become ready."""

  for _ in range(0, WAIT_FOR_SERVER_READY_INT_SECS):

    time.sleep(1)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'intentionally_missing_model'

    tf.logging.info('Waiting for server to become ready...')

    try:
      # Send empty request to missing model
      channel = grpc.insecure_channel(server)
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      stub.Predict(request, RPC_TIMEOUT)
    except grpc.RpcError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details():
        tf.logging.info('Server is ready')
        return

  raise Exception("Timed out waiting for service %s" % server)


class T2TDevHelper(object):
  """Datagen, train, export, and query served model."""

  def __init__(self, model_name, problem_name, hparams_set, queries,
               output_dir=None, data_dir=None, model_dir=None,
               tmp_dir=None, export_dir=None, decode_hparams="",
               default_tmp=None, tfms_path=None, mode="train"):

    self.model_name = model_name
    self.model = registry.model(model_name)

    self.problem_name = problem_name
    self.hparams_set = hparams_set

    self.queries = queries

    self.problem = registry.problem(self.problem_name)
    self.problem.mode = mode

    tmp = tempfile.mkdtemp() if default_tmp is None else default_tmp

    self.output_dir = output_dir if output_dir is not None else tmp
    self.data_dir = data_dir if data_dir is not None else tmp
    self.model_dir = model_dir if model_dir is not None else tmp
    self.tmp_dir = tmp_dir if tmp_dir is not None else tmp
    self.export_dir = (
        export_dir if export_dir is not None else tmp + "/export"
    )

    self.decode_hparams = decode_hparams

    # HACK
    self.tf_model_server_path = tfms_path

    self.train_dataset = None

    self.has_run_datagen = False

    self._lookup_hparams()

  def datagen(self):
    """Run datagen."""
    self.problem.generate_data(self.data_dir, self.tmp_dir)
    self.has_run_datagen = True

  def train(self, use_tpu=False, schedule="train"):
    """Run training."""

    exp_fn = trainer_lib.create_experiment_fn(
        self.model_name,
        self.problem_name,
        self.data_dir,
        train_steps=10,
        eval_steps=1,
        min_eval_frequency=9,
        use_tpu=use_tpu,
        schedule=schedule)

    run_config = trainer_lib.create_run_config(
        model_name=self.model_name,
        model_dir=self.model_dir,
        num_gpus=0,
        use_tpu=use_tpu,
        schedule=schedule)

    hparams = registry.hparams(self.hparams_set)

    exp = exp_fn(run_config, hparams)

    exp.train()

    self.hparams = hparams

    return hparams

  def export(self, hparams):
    """Run model export for serving."""

    hparams.no_data_parallelism = True

    problem = hparams.problem

    run_config = trainer_lib.create_run_config(
        model_name=self.model_name,
        model_dir=self.model_dir,
        num_gpus=0,
        use_tpu=False
    )

    estimator = trainer_lib.create_estimator(
        self.model_name,
        hparams,
        run_config,
        decode_hparams=decoding.decode_hparams(self.decode_hparams)
    )

    exporter = tf.estimator.FinalExporter(
        "exporter",
        lambda: problem.serving_input_fn(hparams),
        as_text=True
    )

    exporter.export(
        estimator,
        self.export_dir,
        checkpoint_path=tf.train.latest_checkpoint(self.model_dir),
        eval_result=None,
        is_the_final_export=True
    )

  def serve(self):
    """Serve a SavedModel."""

    if not self.tf_model_server_path:
      raise ValueError("Need to set `tf_model_server_path` attr.")

    _, server, _ = run_server(
        self.model_name,
        self.export_dir,
        self.tf_model_server_path
    )

    wait_for_server_ready(server)

    self.server = server

    return server

  def batch_query_server(self, queries, server, hparams, timeout_secs=5):
    """Query a served model with a batch of multiple queries."""

    problem = hparams.problem

    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=self.model_name,
        server=server,
        timeout_secs=timeout_secs)

    responses = []

    for query in queries:
      response = serving_utils.predict(query,
                                       problem,
                                       request_fn)
      responses.append(response)

    return responses

  def run_e2e(self):
    """Run end-to-end datagen, training, export, serving, and query."""

    self.datagen()

    hparams = self.train()

    self.export(hparams)

    server = self.serve()

    if self.queries is None:
      tf.logging.info("self.queries unset, skipping query of server.")
      return None

    return self.batch_query_server(self.queries, server, hparams)

  def eager_get_example(self, batch_size=2, mode=Modes.TRAIN):
    """Instantiate tf.data.Dataset if necessary and get one example."""

    train_dataset = self.problem.dataset(mode, data_dir=self.data_dir)

    train_dataset = train_dataset.repeat(None).batch(batch_size)

    return tfe.Iterator(train_dataset).next()

  def eager_get_nth_example(self, n, batch_size=2):

    if self.train_dataset is None:
      train_dataset = self.problem.dataset(Modes.TRAIN,
                                           data_dir=self.data_dir)
      self.train_dataset = train_dataset.repeat(None).batch(batch_size)

    for _ in range(n):
      ex = tfe.Iterator(self.train_dataset).next()

    return ex
  
  def _lookup_hparams(self):

    self.hparams = registry.hparams(self.hparams_set)
    self.hparams.data_dir = self.data_dir
    self.p_hparams = self.problem.get_hparams(self.hparams)

  def get_model_instance(self, mode=Modes.TRAIN):
    """Get an instantiated instance of self.model."""

    model = self.model(
        self.hparams,
        mode,
        self.p_hparams
    )

    return model

  def eager_train_one_step(self, extra_example_preprocess_fn=None):
    """Perform a single step of training with tf.Eager."""

    model = self.get_model_instance()

    @tfe.implicit_value_and_gradients
    def loss_fn(features):
      _, losses = model(features)
      return losses["training"]

    example = self.eager_get_example()

    optimizer = tf.train.AdamOptimizer()

    if extra_example_preprocess_fn is not None:
      example = extra_example_preprocess_fn(example)

    logits, gv = loss_fn(example)

    optimizer.apply_gradients(gv)

    return example, logits, gv

  def eager_train_many_steps(self, steps=1, save_each_step=False, batch_size=8):
    """Perform a single step of training with tf.Eager."""

    saved = {}

    model = self.get_model_instance()

    @tfe.implicit_value_and_gradients
    def loss_fn(features):
      _, losses = model(features)
      return losses["training"]

    optimizer = tf.train.AdamOptimizer()

    for i in range(steps):

      example = self.eager_get_example(batch_size=batch_size)

      logits, gv = loss_fn(example)

      optimizer.apply_gradients(gv)

      if save_each_step:
        saved[str(i)] = {"logits": logits, "gv": gv, "example": example}

    return saved

  def eager_infer(self, features, ckpt_path):

    model = self.get_model_instance()

    with tfe.restore_variables_on_create(ckpt_path):
      out = model.infer(features)

    return out
