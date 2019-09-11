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
import time
import os
import numpy as np
import uuid
import json

from pcml.launcher.kube import CronJob
from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid
from pcml.launcher.kube import Resources

from tensor2tensor.utils import registry

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name

from pcml.utils.cmd_utils import run_and_output

# ==============
# HACK: In order to be able to access CBT.
#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
# ==============

Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name


class EvalJob(CronJob, PCMLJob):

  def __init__(self, ckpt_dir, staging_path=None,
               schedule="* * * * *",
               job_name_prefix="eval-job",
               num_cpu=1,
               memory="7Gi",
               image="gcr.io/clarify/basic-runtime:0.0.4",
               *args, **kwargs):
    """Periodically run eval for ckpt dir."""

    cmd = "python -m pcml.operations.eval "
    cmd += "--ckpt_dir=%s" % ckpt_dir

    command = ["/bin/sh", "-c"]
    command_args = [cmd]

    job_name = "%s-%s" % (job_name_prefix, gen_timestamped_uid())
    
    super(EvalJob, self).__init__(
      job_name=job_name,
      command=command,
      command_args=command_args,
      namespace="kubeflow",
      image=image,
      staging_path=staging_path,
      schedule=schedule,
      resources=Resources(limits={"cpu": num_cpu,
                                  "memory": memory}),
      *args, **kwargs)

def _metrics_given_threshold(predictions, targets, threshold):

  TP = 0.0
  TN = 0.0
  FP = 0.0
  FN = 0.0

  for i, prediction in enumerate(predictions):
    if prediction > threshold:
      # Prediction is a positive prediction
      if targets[i] == 1.0:
        # Prediction is a true positive
        TP += 1.0
        #tf.logging.debug("TP: {},{}".format(prediction, targets[i]))
      else:
        # Prediction is a false positive
        FP += 1.0
        #tf.logging.debug("FP: {},{}".format(prediction, targets[i]))

    else:
      # Prediction is a negative prediction
      if targets[i] == 1.0:
        # Prediction is a false negative
        FN += 1.0
        #tf.logging.debug("FN: {},{}".format(prediction, targets[i]))
      else:
        # Prediction is a true negative
        TN += 1.0
        #tf.logging.debug("TN: {},{}".format(prediction, targets[i]))

  def _safe_fraction(numerator, denominator):
    if denominator == 0:
      return 0
    return numerator/denominator

  # Thanks, Wikipedia!

  tpr = _safe_fraction(TP, TP + FN)
  tnr = _safe_fraction(TN, TN + FP)
  ppv = _safe_fraction(TP, TP + FP)
  npv = _safe_fraction(TN, TN + FN)
  accuracy = _safe_fraction(TP + TN, TP + TN + FP + FN)
  f1 = _safe_fraction(2*TP, 2*TP + FP + FN)
  informativeness = (tpr + tnr - 1)
  markedness = (ppv + npv - 1)
  balanced_accuracy = (tpr + tnr)/2

  return {
    "tpr": tpr,
    "tnr": tnr,
    "ppv": ppv,
    "npv": npv,
    "accuracy": accuracy,
    "f1": f1,
    "informativeness": informativeness,
    "markedness": markedness,
    "tp": TP,
    "tn": TN,
    "fp": FP,
    "fn": FN,
    "balanced_accuracy": balanced_accuracy
  }


def _auc_for_metrics_set(metrics_set):
  """
  
  Assuming the set is in threshold sorted order.
  
  """

  last_precision = None
  last_recall = None
  total_area = 0.0

  for m in metrics_set:
    precision = float(m["tpr"])
    recall = float(m["ppv"])
    if last_precision and last_recall:
      total_area += (recall - last_recall)*precision
    last_precision = precision
    last_recall = recall

  return total_area


def compute_metrics(problem_name, model_name, hparams_name, 
                    ckpt_dir, data_dir="/tmp", eval_batch_size=32,
                    eval_steps=100, extra_hparams=[], mode=Modes.EVAL,
                    num_threshold_bins=100):

  if not isinstance(num_threshold_bins, int) and num_threshold_bins > 0:
    msg = "Num threshold bins should be int > 0, saw {}".format(num_threshold_bins)
    raise ValueError(msg)

  registered_model = registry.model(model_name)

  hparams = registry.hparams(hparams_name)
  hparams.mode = mode

  for extra_hparam in extra_hparams:
    assert len(extra_hparam) == 2
    if extra_hparam[0] == "mode":
      continue
    hparams.setattr(extra_hparam[0], extra_hparam[1])

  problem_instance = registry.problem(problem_name)
  problem_hparams = problem_instance.get_hparams(hparams)

  # Build the eval dataset and get the examples
  eval_dataset = problem_instance.dataset(mode=Modes.EVAL,
                                          data_dir=data_dir)

  '''eval_dataset = problem_instance.dataset(mode=Modes.TRAIN,
                                          mode_override=mode,
                                          data_dir=data_dir)'''

  eval_dataset = eval_dataset.repeat(None).batch(eval_batch_size)
  eval_dataset_iterator = tfe.Iterator(eval_dataset)

  with tfe.restore_variables_on_create(ckpt_dir):

    model_instance = registered_model(
      hparams, mode, problem_hparams
    )
  
    predictions = np.array([], dtype=np.float32)
    targets = np.array([], dtype=np.float32)

    for i in range(eval_steps):

      try:

        eval_examples = eval_dataset_iterator.next()

        prediction = model_instance.infer(eval_examples)

        # We've concatenated the two embedding vectors followed
        # by the label so we can obtain the label just by looking
        # at the last value
        prediction = np.array([thing[-1] for thing in prediction], dtype=np.float32)

        target = tf.squeeze(eval_examples["targets"]).numpy().astype(np.float32)

        predictions = np.concatenate([predictions, prediction])
        targets = np.concatenate([targets, target])
        
        if i % 10 == 0:
          msg = "Finished collecting predictions for eval step {}.".format(i)
          tf.logging.info(msg)

      except Exception as e:
        # Seeing rare CBT deadline exceeded errors and don't know how to modfiy
        # the deadline... More likely to run into error with more iterations,
        # wasn't seeing it with 10 and almost always seeing it with 100.
        # Could conceivably have to do with running out of examples in the eval
        # set... but there should be over 20k and this would only go through
        # 3200.
        msg = "HACK: Squashing inference error."
        tf.logging.info(msg)

  metrics_set = []

  for i in range(num_threshold_bins):
    threshold = i/num_threshold_bins
    metrics = _metrics_given_threshold(predictions, targets,
                                       threshold=threshold)
    metrics["at_threshold"] = threshold
    metrics["num_threshold_bins"] = num_threshold_bins
    metrics_set.append(metrics)

  midpoint_metrics = _metrics_given_threshold(predictions, targets,
                                              threshold=0.5)
  midpoint_metrics["auc"] = _auc_for_metrics_set(metrics_set)

  return midpoint_metrics, metrics_set, predictions, targets


def lookup_config(ckpt_path, max_retries=10, retry_delay=10):
  """Lookup t2t_config and retry until available or timeout."""

  if not ckpt_path.startswith("gs://"):
    raise ValueError("ckpt path isn't a GCS path.")

  config = {}

  err = None

  for _ in range(max_retries):

    try:

      with tf.gfile.Open(os.path.join(ckpt_path, "flags_t2t.txt")) as f:

        for line in f:
          line = line.strip()
          key = line.split("=")[0][2:]
          value = line.split("=")[1]
          config[key] = value

      return config

    except Exception as e:
      err = e
      time.sleep(retry_delay)

  raise err


def parse_extra_hparams(hparams_string):
  hp = {}
  tmp = hparams_string.split(",")
  for kv in tmp:
    arr = kv.split("=")
    if len(arr) == 2:
      hp[k] = v
  return hp


def report_metrics_summaries(metrics, global_step, ckpt_dir, mode):
  
  tf.logging.info("Reporting metrics: {},{}".format(metrics,
                                                    global_step))

  global_step = tf.convert_to_tensor(global_step, dtype=tf.int64)

  summary_writer = tf.contrib.summary.create_file_writer(
    os.path.join(ckpt_dir, mode))

  with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    for key, value in metrics.items():
      tf.contrib.summary.scalar(key, value, step=global_step)


def latest_step_for_ckpt_dir(ckpt_dir):

  max_global_step = None

  files = tf.gfile.ListDirectory(ckpt_dir)
  for filename in files:
    if filename.endswith("index") and filename.startswith("model"):
      ckpt_tag = filename.split(".")[1]

      if not ckpt_tag.startswith("ckpt"):
        err = "Unrecognized ckpt file name format, {}".format(
          filename
        )
        raise ValueError(err)
    
      global_step = int(ckpt_tag.split("-")[1])

      if not max_global_step or global_step > max_global_step:
        max_global_step = global_step

  if not max_global_step:
    err = ("A max global step could not be determined, " +
           "returning zero.")
    tf.logging.info(err)
    max_global_step = 0

  tf.logging.info("Last global step: {}".format(max_global_step))

  return max_global_step


def log_all_metric_data(predictions, targets, metrics_set, ckpt_dir, global_step, mode):
  plen = len(predictions)
  tlen = len(targets)
  if plen != tlen:
    msg = "Should have same number of predictions and targets, saw {}, {}.".format(
      plen, tlen
    )
    raise ValueError(msg)
  uid = str(uuid.uuid4())[0:8]
  fname = "predictions-{}-{}".format(str(global_step), uid)

  for i, m in enumerate(metrics_set):
    m["global_step"] = global_step
    metrics_set[i] = m

  data = {
    "metrics_set": metrics_set,
    "predictions": predictions.flatten().tolist()
  }

  fpath = os.path.join(ckpt_dir, mode, "metric_data", fname)
  with tf.gfile.Open(fpath, "w") as f:
    f.write(json.dumps(data))


def eval_for_ckpt_dir(ckpt_dir, delay_seconds=5, mock_acc=False,
                      mock_step=False, continuous=True, eval_steps=10,
                      data_dir=None):
  """Run Eager-mode eval with the option to repeat periodically."""

  tf.logging.set_verbosity(tf.logging.INFO)

  if continuous:
    msg = ("Running eval continuously with delay {}.".format(delay_seconds))
    tf.logging.info(msg)
  else:
    msg = ("Running a single round of eval.")
    tf.logging.info(msg)

  # This will poll until the t2t_hparams.txt file is present.
  config = lookup_config(ckpt_dir)
  extra_hparams = parse_extra_hparams(config["hparams"])

  # HACK
  #config["model"] = "res_ut_conv3d"
  #config["problem"] = "vox_celeb_problem_v3_preaugmentation_dev"
  #config["hparams_set"] = "ut_tpu"

  last_global_step = None

  while True:

    global_step = latest_step_for_ckpt_dir(ckpt_dir)

    if mock_step:
      if not last_global_step:
        last_global_step = 0
      global_step = last_global_step + 1

    if global_step == last_global_step:
      msg = ("Didn't find a new global step to evaluate, will retry "
             "in {} seconds.".format(delay_seconds))
      tf.logging.info(msg)
      time.sleep(delay_seconds)

      continue

    for mode in [Modes.EVAL]: #, Modes.TRAIN]:

      midpoint_metrics, metrics_set, predictions, targets = compute_metrics(
        problem_name=config["problem"],
        model_name=config["model"],
        hparams_name=config["hparams_set"],
        ckpt_dir=ckpt_dir,
        extra_hparams=extra_hparams,
        mode=mode,
        eval_steps=eval_steps,
        data_dir=data_dir)

      report_metrics_summaries(metrics=midpoint_metrics,
                               ckpt_dir=ckpt_dir,
                               global_step=global_step,
                               mode=mode)

      log_all_metric_data(predictions=predictions,
                          targets=targets,
                          metrics_set=metrics_set,
                          ckpt_dir=ckpt_dir,
                          global_step=global_step,
                          mode=mode)

    tf.logging.info("Delaying {}s until next eval".format(
      delay_seconds
    ))

    last_global_step = global_step

    if not continuous:
      tf.logging.info("Finished single round of eval.")
      return

    time.sleep(delay_seconds)


def trigger_eval(ckpt_dir, data_dir=None, eval_steps=30):
  """Convenience wrapper for triggering functionally."""

  if isinstance(ckpt_dir, tuple) or isinstance(ckpt_dir, list):
    ckpt_dir = ckpt_dir[0]
    if not isinstance(ckpt_dir, str):
      raise ValueError("Problem receiving thread args.")

  """
  
  HACK: Having an issue with restoring two different checkpoints
  with Eager and finding conflicts due to the previous cached
  variables not being expunged upon exiting the context in which
  the variables were restored (i.e. globals are modified). This
  hack runs only a single round of eval each in a separate system
  process to get around this, aptly named for its hackiness.`
  
  """

  # Wait 10min before computing first metrics
  time.sleep(300)

  while True:

    cmd = ("python -m pcml.operations.eval " +
           "--ckpt_dir={} --continuous=False " +
           "--eval_steps={} ").format(ckpt_dir, eval_steps)

    if data_dir:
      cmd += "--data_dir={}".format(data_dir)

    try:

      out = run_and_output(cmd.split())

    except Exception as e:

      tf.logging.info("Squashing failed eval call, will retry...")
      # ...
    
    time.sleep(60)

  return out


def main(_):

  if FLAGS.ckpt_dir is None:
    raise ValueError("Please provide a ckpt_dir with --ckpt_dir.")

  eval_for_ckpt_dir(ckpt_dir=FLAGS.ckpt_dir,
                    delay_seconds=FLAGS.delay_seconds,
                    continuous=FLAGS.continuous,
                    eval_steps=FLAGS.eval_steps,
                    data_dir=FLAGS.data_dir)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string('ckpt_dir', None,
                      'Path to ckpts to eval.')

  flags.DEFINE_boolean('continuous', True,
                       'Whether to run eval continuously.')

  # TODO: Make use of this flag instead of hard coding (currently 60).
  flags.DEFINE_integer('delay_seconds', 300,
                       'Delay in seconds between evals.')

  # TODO: Make use of this flag instead of hard coding (currently 60).
  flags.DEFINE_integer('eval_steps', 10,
                       'Number of steps of eval to perform.')

  flags.DEFINE_string('data_dir', None,
                      'Directory where examples are stored (incl. GCS).')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
