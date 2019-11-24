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
import tempfile
import cv2
import time

from google.cloud import firestore
from PIL import Image
import io
import base64

import random

from gtts import gTTS

import numpy as np
from sklearn.neighbors import KDTree

import tensorflow as tf
tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name

from faced import FaceDetector

from tensor2tensor.utils import registry

from pcml.utils.dev_utils import T2TDevHelper
from pcml.datasets.fec import _normalize_dimensions
from pcml.datasets.fec import _read_image
import pcml

from pcml.datasets import image_aug

from IPython.display import (
    Audio, display, clear_output)
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import pylab

detector = FaceDetector()
detect_threshold = 0.7

problem_name = "ext_dev2"
model_name = "percep_similarity_triplet_emb"
hparams_set_name = "percep_similarity_triplet_emb"

ckpt_dir = "gs://clarify-public/models/fec-train-j1030-0136-3a8f/output"
data_dir = "gs://clarify-public/models/fec-train-j1030-0136-3a8f/output"

temp = tempfile.mkdtemp()

problem_name = "facial_expression_correspondence"
mode = "predict"

hparams = registry.hparams(hparams_set_name)
hparams.data_dir = data_dir

problem_obj = registry.problem(problem_name)

p_hparams = problem_obj.get_hparams(hparams)

model_obj = registry.model(model_name)

rate = 16000.
duration = .25
t = np.linspace(
    0., duration, int(rate * duration))


def load_image(img_string):
  binary = base64.b64decode(img_string.split(',')[1])
  filename = "/tmp/stream.jpeg"
  with open(filename, 'wb') as f:
    f.write(binary)
  img = Image.open(filename)
  return np.array(img)

def synth(f):
    x = np.sin(f * 2. * np.pi * t)
    display(Audio(x, rate=rate, autoplay=True))


def _random_crop_square(image):

  x,y,c = image.shape

  x_crop_before = 0
  x_crop_after = 0
  y_crop_before = 0
  y_crop_after = 0

  if x > y:
    x_crop = x - y
    x_crop_before = np.random.randint(0, x_crop)
    x_crop_after = x_crop - x_crop_before
  elif y > x:
    y_crop = y - x
    y_crop_before = np.random.randint(0, y_crop)
    y_crop_after = y_crop - y_crop_before

  x_start = x_crop_before
  x_end = x - x_crop_after
  y_start = y_crop_before
  y_end = y - y_crop_after

  return image[x_start:x_end, y_start:y_end, :]


def _normalize_dimensions(image, target_shape):

  image = _random_crop_square(image)

  mn, mx = np.amin(image), np.amax(image)
  if mn >=0 and mx <= 255:
    image = image / 255.0

  source_shape = image.shape
  scale_x_factor = target_shape[0]/source_shape[0]
  scale_y_factor = target_shape[1]/source_shape[1]
  scale_x_first = (scale_x_factor <= scale_y_factor)

  if scale_x_first:

    new_x = target_shape[0]
    new_y = int(source_shape[1]*scale_x_factor)
    resize_dim = (new_x, new_y)
    newimg = cv2.resize(image, resize_dim)
    pad_width = target_shape[1] - new_y
    if pad_width > 0:
      # Pad in Y direction
      newimg = np.pad(newimg, [(0,pad_width),(0,0),(0,0)], mode="mean")

  else:

    new_y = target_shape[1]
    new_x = int(source_shape[0]*scale_y_factor)
    resize_dim = (new_x, new_y)
    newimg = cv2.resize(image, resize_dim)
    pad_width = target_shape[0] - new_x
    if pad_width > 0:
      # Pad in X direction
      newimg = np.pad(newimg, [(0,0),(0,pad_width),(0,0)], mode="mean")

  newimg = (newimg*255.0).astype(np.int64)

  return newimg


def detect_and_preprocess(image):

  detector = FaceDetector()
  detect_threshold = 0.5
  predictions = detector.predict(image, detect_threshold)

  xcenter = predictions[0][0]
  ycenter = predictions[0][1]
  width = predictions[0][2]*1.80
  height = predictions[0][3]*1.80

  xmax = image.shape[1]
  ymax = image.shape[0]

  ystart = max(0,int(ycenter-height/2))
  yend = min(ymax,int(ycenter+height/2))
  xstart = max(0,int(xcenter-width/2))
  xend = min(xmax,int(xcenter+width/2))

  img_with_face = image[ystart:yend,xstart:xend,:]

  image_shape = (64,64,3)
  img_post = _normalize_dimensions(img_with_face, target_shape=image_shape)

  return img_post


def get_example():

  # Add a new document
  db = firestore.Client()
  doc_ref = db.collection(u'users/2LbhP63ADQfo5XkmKeVVEtPWvAD2/modalities').document(u'av')

  image_stats = {"mean": [0.330, 0.537, -0.242], "sd": [0.220, 0.169, 1.156]}
  shape = (64,64,3)
  mode = "eval"

  data = doc_ref.get().to_dict()["videoData"]

  img = np.asarray(load_image(data))
  
  img = detect_and_preprocess(img)

  # Convert to int32

  example = {
    "image/a": img,
    "image/b": img,
    "image/c": img,
    "image/a/noaug": img,
    "image/b/noaug": img,
    "image/c/noaug": img,
    "triplet_code": [0],
    "type": [1],
    "targets": img
  }

  def _preproc(image):

    image = image_aug.preprocess_image(
      image, mode,
      resize_size=shape,
      normalize=True,
      image_statistics=image_stats,
      crop_area_min=1,
      contrast_lower=0.45,
      contrast_upper=0.55,
      brightness_delta_min=-0.01,
      brightness_delta_max=0.01)

    image.set_shape(shape)

    return image

  example["image/a"] = tf.expand_dims(_preproc(example["image/a"]),0)
  example["image/b"] = tf.expand_dims(_preproc(example["image/b"]),0)
  example["image/c"] = tf.expand_dims(_preproc(example["image/c"]),0)
  example["triplet_code"] = tf.expand_dims(tf.cast(example["triplet_code"], tf.int64),0)

  return example


def _goal_from_goals(goals):
  return goals[0]


def _tts(msg):
  filename = "/tmp/clarify-tmp.mp3"
  tts = gTTS(msg)
  tts.save(filename)
  display(Audio(filename, autoplay=True))


def play_instructions():
  message = "Goal definition will begin in twenty seconds following a tone and will "
  message += "last for ten seconds. Please assume a mental state that you want to "
  message += "be your goal. The definition period will begin in seven seconds."
  _tts(message)
  time.sleep(20)
  synth(600)


def play_beginning_session():
  _tts("Goal definition complete. Beginning session.")
  time.sleep(3)


def play_establishing_baseline():
  _tts("Establishing baseline. Please demonstrate a variety of expressions and poses.")
  time.sleep(3)


def play_baseline_complete():
  _tts("Finished establishing baseline.")
  time.sleep(3)


def optimize():

  play_instructions()

  query_data = {}

  synth_min = 200
  synth_max = 600
  synth_range = synth_max - synth_min

  mn = None
  mx = None

  goals = []
  goal = None

  distance_threshold = 1.3

  num_sampled = 0

  baseline = True

  distances = []
  examples = []
  num_baseline_steps = 0

  with tfe.restore_variables_on_create(tf.train.latest_checkpoint(ckpt_dir)):

    model = model_obj(hparams, mode, p_hparams)

    while True:

      print("Doing step...")

      try:

        example = get_example()
        current, _ = model(example)

        num_sampled += 1

        if num_sampled <= 3:

          goals.append(current)
          distances.append(0)
          examples.append(example)

        else:

          if goal is None:
            goal = _goal_from_goals(goals)
            play_beginning_session()

          dist = np.linalg.norm(current - goal)
          print(dist)

          distances.append(dist)
          examples.append(example)

          if not mn:
            mn = dist
          if not mx:
            mx = dist

          if dist < mn:
            mn = dist

          if dist > mx:
            mx = dist

          if num_sampled == 4:
            play_establishing_baseline()

          synth_level = synth_min + synth_range*((dist-mn)/(mx-mn))

          #if distances:
          #  distance_threshold = np.mean(distances)

          print(synth_level)
          if dist > distance_threshold:
            synth(synth_level)

          if num_sampled == 20:
            play_baseline_complete()
            baseline = False
            num_baseline_steps = num_sampled

      except (KeyboardInterrupt, SystemExit):
        break

      except:
        raise
        print("there was an exception but we're not worried ;D")

  return locals()