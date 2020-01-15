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
"""Utilities supporting data generation code."""

import math
import os
import io
from io import BytesIO
import numpy as np
from scipy import signal
from PIL import Image

import tensorflow as tf
from tensorflow.python.framework import ops

from tensor2tensor.layers import common_video


def get_random_index_subinterval(interval_length, subinterval_length,
                                 num_samples):
  """Given length, sample a pair of indicies in [0,length].
  
  Args:
    interval_length(int): The length of an interval to sample.
    subinterval_length(int): The length of the interval to sample,
      e.g. a length=2 sub-interval from a parent interval of
      length=10
    num_samples(int): The number of samples to generate.
    
  Returns:
    list: The sampled sub-interval start and end indices.
      
  """

  samples = []

  if subinterval_length > interval_length:
    raise ValueError(
        "Can't sample interval of length %s % " % subinterval_length,
        "from a larger interval of length %s." % duration)

  for _ in range(num_samples):
    start = np.random.randint(0, interval_length - subinterval_length)
    end = start + subinterval_length
    samples.append([start, end])

  return samples


def get_temporal_subinterval(duration, subinterval_length):

  if subinterval_length > duration:
    raise ValueError("Can sample interval of length %s % " % subinterval_length,
                     "from a larger interval of length %s." % duration)

  start_seconds = np.random.uniform(0, duration - subinterval_length)
  end_seconds = start_seconds + subinterval_length

  return start_seconds, end_seconds


def get_input_file_paths(glob_or_file_path, is_training, training_fraction):
  """Get a list of input files from `tmp_dir`."""

  #if not glob_or_file_path.endswith("*"):
  #  glob_or_file_path += "/*"
  all_files = tf.gfile.Glob(glob_or_file_path)

  dividing_index = int(math.floor(len(all_files) * training_fraction))

  if is_training:
    return all_files[:dividing_index]
  else:
    return all_files[dividing_index:]


def array2gif(raw_frames):
  frames = []
  for frame in raw_frames:
    frames.append(Image.fromarray(np.uint8(frame)))

  with io.BytesIO() as output:

    frames[0].save(output,
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True)

    return output.getvalue()


def evaluate_tensor(tensor):
  if tf.executing_eagerly():
    return tensor.numpy()
  else:
    sess = ops.get_default_session()
    if sess is None:
      with tf.Session() as sess:
        return sess.run(tensor)
    else:
      return sess.run(tensor)


def gen_gif_encoded_from_field_spec(field_spec, zeros=False):
  mocked = evaluate_tensor(field_spec.mock_one(zeros=zeros))
  encoded = array2gif(mocked)
  return encoded


def gen_dummy_schedule(num_problems, num_steps=10):
  steps = []
  pwf = []
  steps.append(num_steps)
  prob_pwf = [1 / num_problems for _ in range(num_problems)]
  pwf.append(prob_pwf)
  schedule = ('step', tuple(steps), tuple(tuple(thing) for thing in pwf))
  return schedule


# -----
# Old, probably remove.


def normalize_example_multimodal(example,
                                 task_id,
                                 example_norm_spec,
                                 dtype=tf.int64):
  """
  
  This expects to receive a single example not a batch of examples.
  
  """

  # Hack
  max_num_classes = example_norm_spec["targets"]
  xlen = tf.shape(example["targets"])[0]
  x = example["targets"]
  #example["targets"] = tf.pad(x, [[0, max_num_classes - xlen]])

  for key, shape in example_norm_spec["modalities"].items():
    if key not in example:
      example[key] = tf.zeros(shape, dtype=dtype)

  example["task_id"] = tf.constant([task_id], dtype=dtype)

  return example


def sim_dummy_tensor(shape, vocab_size):
  """Simulate a dummy tensor of specified `shape` and `vocab_size`.
  
  Zero-based, uses [0, vocab_size).
  """
  x = np.random.uniform(0, 1, shape) * vocab_size
  x = x.astype(np.int32).tolist()
  return x


def gen_dummy_encoded_video(shape, vocab_size, sample_fps=1):

  def make_frame():
    return (np.random.uniform(0, 1, shape[1:]) * vocab_size).astype(np.int32)

  frame_list = np.asarray([make_frame() for _ in range(shape[0])])

  return array2gif(frame_list)


def extend_hparams_with_hparams(hp1, hp2):
  """Given one set of hparams add to that another."""
  for k, v in hp2.__dict__.items():
    setattr(hp1, k, v)
  return hp1


# --------
