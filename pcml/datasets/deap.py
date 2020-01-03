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
"""DEAP problem definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile
import os
import math
import json
import uuid
import pickle

import mne

import numpy as np

import tensorflow as tf

from tensor2tensor.data_generators import generator_utils

from pcml.datasets.utils import get_input_file_paths

from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

from tensor2tensor.layers import modalities

DEFAULT_DEAP_ROOT = "gs://clarify-data/requires-eula/deap/"


def _raw_data_verifier(tmp_dir, test_mode=False):
  """Raw data verifier for DEAP dataset."""
  preproc_python_dir = os.path.join(DEFAULT_DEAP_ROOT, "preproc_python/*")
  files = get_input_file_paths(preproc_python_dir, True, 1)
  for file in files:
    if not file.endswith(".dat"):
      raise ValueError("Expected file with suffix .dat, saw %s" % file)
  if not len(files) == 32:
    raise ValueError(("Expected 32 subjects in preproc_python subdir, "
                      "saw %s" % len(files)))


def maybe_get_deap_data(tmp_dir,
                        deap_root=DEFAULT_DEAP_ROOT,
                        is_training=True,
                        training_fraction=0.9):
  """Maybe download DEAP data.

  Args:
    tmp_dir(str): A local path where temporary files can
      be stored.
    deap_root(str): The root path to a directory containing
      deap data wherein subdirectories meta/, biosemi/,
      face_video/, preproc_matlab/, and preproc_python/
      store respective sub-datasets. Currently we only
      make use of that in preproc_python/.

  """

  preproc_python_dir = os.path.join(deap_root, "preproc_python/*")
  input_paths = get_input_file_paths(preproc_python_dir, is_training,
                                     training_fraction)

  paths = []
  for i, path in enumerate(input_paths):
    fname = path.split("/")[-1]
    res_path = generator_utils.maybe_download(tmp_dir, fname, path)
    yield res_path


def load_deap_meta(local_deap_root):

  participant_ratings = pd.read_csv(
      os.path.join(local_deap_root, "meta", "participant_ratings.csv"))
  online_ratings = pd.read_csv(
      os.path.join(local_deap_root, "meta", "online_ratings.csv"))
  video_list = pd.read_csv(
      os.path.join(local_deap_root, "meta", "video_list.csv"))

  return {
      "participant_ratings": participant_ratings,
      "online_ratings": online_ratings,
      "video_list": video_list
  }


def _parse_deap_metadata(local_deap_root):
  """Build a participant ratings"""

  raw_meta = load_deap_meta(local_deap_root)

  dataset = {}

  for i, pid in enumerate(raw_meta["participant_ratings"]["Participant_id"]):
    tid = raw_meta["participant_ratings"]["Trial"][i]
    exp = raw_meta["participant_ratings"]["Experiment_id"][i]
    start = raw_meta["participant_ratings"]["Start_time"][i]
    data = {
        "pid": pid,
        "tid": tid,
        "exp": exp,
        "start": start,
        "valence": raw_meta["participant_ratings"]["Valence"][i],
        "arousal": raw_meta["participant_ratings"]["Arousal"][i],
        "dominance": raw_meta["participant_ratings"]["Dominance"][i],
        "liking": raw_meta["participant_ratings"]["Liking"][i],
        "familiarity": raw_meta["participant_ratings"]["Familiarity"][i]
    }
    if pid not in dataset:
      dataset[pid] = {}
    dataset[pid][tid] = data


def _preprocess_trial_data(acquired_data, layout, channel_data):
  """Reconcile acquired labels with layout; clip and standardize."""

  # Build a lookup dict for acquired data keyed on channel name
  acquired_data_lookup = {}
  for i, d in enumerate(acquired_data):
    key = channel_data[i][0]
    min_val, max_val = channel_data[i][1]
    d_standardized = clip_and_standardize(d, min_val, max_val)
    acquired_data_lookup[key] = d_standardized

  # Build a lookup dict for layout positions keyed on channel name
  layout_data_lookup = {}
  for i, k in enumerate(layout.names):
    layout_data_lookup[k] = layout.pos[i]

  sensor_traces = []
  sensor_positions = []
  channels = []

  for key, data in acquired_data_lookup.items():
    if key in layout_data_lookup:
      sensor_traces.append(data)
      sensor_positions.append(layout_data_lookup[key])
      channels.append(key)

  sensor_traces = np.asarray(sensor_traces)
  sensor_positions = np.asarray(sensor_positions)
  channels = np.asarray(channels)

  return (sensor_traces, sensor_positions, acquired_data_lookup, channels)


# All of the channels expected to be present in DEAP bdf files
# along with the ranges to which the signal from those channels
# should be clipped.
DEAP_BDF_CHANNELS = [('Fp1', (-20, 20)), ('AF3', (-20, 20)), ('F3', (-20, 20)),
                     ('F7', (-20, 20)), ('FC5', (-20, 20)), ('FC1', (-20, 20)),
                     ('C3', (-20, 20)), ('T7', (-20, 20)), ('CP5', (-20, 20)),
                     ('CP1', (-20, 20)), ('P3', (-20, 20)), ('P7', (-20, 20)),
                     ('PO3', (-20, 20)), ('O1', (-20, 20)), ('Oz', (-20, 20)),
                     ('Pz', (-20, 20)), ('Fp2', (-20, 20)), ('AF4', (-20, 20)),
                     ('Fz', (-20, 20)), ('F4', (-20, 20)), ('F8', (-20, 20)),
                     ('FC6', (-20, 20)), ('FC2', (-20, 20)), ('Cz', (-20, 20)),
                     ('C4', (-20, 20)), ('T8', (-20, 20)), ('CP6', (-20, 20)),
                     ('CP2', (-20, 20)), ('P4', (-20, 20)), ('P8', (-20, 20)),
                     ('PO4', (-20, 20)), ('O2', (-20, 20)), ('hEOG', (-400, 0)),
                     ('vEOG', (0, 500)), ('zEMG', (-120, 120)),
                     ('tEMG', (-100, 100)), ('GSR', (-10000, 10000)),
                     ('rAMP', (-2000, 2000)),
                     ('plethysmograph', (-20000, 20000)), ('temp', (-0.1, 0.1))]

# The channels of DEAP_BDF_CHANNELS we expect to be able to relate
# to a the Biosemi layout from mne's biosemi.lay.
DEAP_EXPECTED_MAPPED_EEG_CHANNELS = [
    "P3", "Pz", "O2", "O1", "FC5", "P4", "T8", "Fz", "AF4", "PO4", "AF3", "FC1",
    "FC2", "P7", "FC6", "T7", "P8", "PO3", "C3", "Fp1", "Oz", "Fp2", "F3", "F4",
    "F7", "F8", "Cz", "CP1", "CP2", "C4", "CP5", "CP6"
]


def clip_and_standardize(data,
                         min_val,
                         max_val,
                         vocab_size=256,
                         dtype=np.int32):
  """Given a signal on [min_val, max_val] yield signal on [-1, 1]."""

  assert min_val < max_val

  # Compute the width of the clipping range.
  width = (max_val - min_val)

  # If [min_val, max_val] is not centered compute the delta
  # that would be needed to center it.
  delta = (max_val - width / 2.0)

  # Shift the clipping range to be centered
  min_val, max_val = min_val - delta, max_val - delta

  std = data.astype(np.float32)

  # Shift the data the same amount as the clipping range
  std -= delta

  # Apply the clip
  std = np.clip(std, min_val, max_val)

  # Transform [a,b] to [-1,1]
  std = std / (width / 2)

  std *= vocab_size / 2

  std = std.astype(dtype)

  return std


def load_preprocessed_deap_data(deap_acquisition_path):
  """
  
  Notes:
    * Load raw data from file.
    * Clip and standardize EEG signal data to [-1,1] from [-20,20].
    * Clip and standardize label data to [-1,1] from [0,10].

  """

  with open(deap_acquisition_path, "rb") as f:
    raw_data = pickle.load(f, encoding="latin1")

  # Labels start with an intensity value in [0,10]
  labels = clip_and_standardize(raw_data["labels"], 0.0, 10.0)

  biosemi_layout = mne.channels.layout.read_layout("biosemi.lay")

  rec_positions = None
  reconciled_acquisition_data = []
  raw_data_lookups = []
  matched_channels = []
  for trial_data in raw_data["data"]:
    rec_data, rec_positions, raw_data_lookup, chan = _preprocess_trial_data(
        acquired_data=trial_data,
        layout=biosemi_layout,
        channel_data=DEAP_BDF_CHANNELS)
    reconciled_acquisition_data.append(rec_data)
    raw_data_lookups.append(raw_data_lookup)
    matched_channels.append(chan)
  return (reconciled_acquisition_data, rec_positions, raw_data_lookups, labels,
          matched_channels)


def _tiled_subsample_example(example, subsample_width, subsample_step):
  subrange_keys = [
      "eeg/raw", "physio/hEOG", "physio/vEOG", "physio/zEMG", "physio/tEMG",
      "physio/GSR", "physio/rAMP", "physio/plethysmograph", "physio/temp"
  ]
  max_len = len(example["physio/temp"])  # HACK
  if subsample_width <= 0 or subsample_step <= 0:
    raise ValueError("subsample width and step must be natural numbers")
  for i in range(max_len):
    # The most we'll ever iterate is max_len.
    # zero-based.
    interval_start = i * subsample_step
    interval_end = interval_start + subsample_width
    if interval_end > max_len:
      return
    subsampled_example = example.copy()
    for key in subrange_keys:
      feature = example[key]
      shape = np.asarray(feature).shape
      if len(shape) == 1:
        subsampled_example[key] = feature[interval_start:interval_end]
      elif len(shape) == 2:
        subsampled_example[key] = [
            thing[interval_start:interval_end] for thing in feature
        ]
      else:
        # There's probably a better way to slice by the last dimension using
        # numpy... idk maybe something like this?
        # subsampled_example[key] = feature[..., interval_start:interval_end]
        raise NotImplementedError()
    yield subsampled_example


def _generator(tmp_dir,
               subsample_width=100,
               subsample_step=10,
               how_many=None,
               is_training=True,
               training_fraction=0.9,
               vocab_size=256):
  """Generator for base training examples from the DEAP dataset.

  Notes:

    Definitions of acronymns and composite features:
      * hEOG1 (to the left of left eye)
      * hEOG2 (to the right of right eye)
      * vEOG1 (above right eye)
      * vEOG4 (below right eye)
      * zEMG1 (Zygomaticus Major, +/- 1cm from left corner of mouth)
      * zEMG2 (Zygomaticus Major, +/- 1cm from zEMG1)
      * tEMG1 (Trapezius, left shoulder blade)
      * tEMG2 (Trapezius, +/- 1cm below tEMG1)
      * Resp Respiration belt
      * Plet Plethysmograph, left thumb
      * Temp Temperature, left pinky
      * Status Status channel containing markers 
      * hEOG (horizontal EOG, hEOG1 - hEOG2)
      * vEOG (vertical EOG, vEOG1 - vEOG2)
      * zEMG (Zygomaticus Major EMG, zEMG1 - zEMG2)
      * tEMG (Trapezius EMG, tEMG1 - tEMG2)
      * GSR (values from Twente converted to Geneva format (Ohm))

    Status code  Event duration  Event Description:
      1 (First occurence)  N/A  start of experiment (participant pressed key to start)
      1 (Second occurence)  120000 ms  start of baseline recording
      1 (Further occurences)  N/A  start of a rating screen
      2 1000 ms  Video synchronization screen (before first trial, before and after break, after last trial)
      3 5000 ms  Fixation screen before beginning of trial
      4 60000 ms  Start of music video playback
      5 3000 ms  Fixation screen after music video playback
      7 N/A  End of experiment

    See: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html for
      more details.

  Args:
    tmp_dir(str): Directory to which to download raw data prior to
      processing into examples.
    subsample_width(int): Number of sensor measurements per sub-sample.
    subsample_step(int): Step size of sensor measurements between
      subsequent tiled sub-sampled, i.e. for width w and step s, we
      will index sub-intervals as follows: [(i-1)*s, (i-1)*s + w]
    is_training(bool): Whether to generate examples from the
      `training_fraction` (or 1-`training_fraction`) portion of input
      training files.
    training_fraction(float): The fraction of input files to use for
      training.

  """
  ct = 0

  def _flatten(list_or_array):
    return np.asarray(list_or_array).flatten().tolist()

  for acquisition_path in maybe_get_deap_data(
      tmp_dir, is_training=is_training, training_fraction=training_fraction):
    trials, positions, raw, labels, chan = load_preprocessed_deap_data(
        acquisition_path)
    for i, trial_reconciled in enumerate(trials):
      example = {
          "eeg/raw": trial_reconciled,
          "eeg/positions": positions,
          "eeg/channels": chan[i],

          # -----
          # TODO (maybe):
          # Currently foregoing encoding of EEG as GIFs until we
          # actually need to combine datasets with two different
          # electrode placements - averaging signal and producing
          # a traditional-looking cortical heatmap loses all the
          # fine detail of the signal. Even when needing to combine
          # electrode placements we might do better to just provide
          # the vector of placements together with the raw sensor data
          # instead of mapping this into a common image.
          # "eeg/encoded": None,
          # -----
          "physio/hEOG": raw[i]["hEOG"],
          "physio/vEOG": raw[i]["vEOG"],
          "physio/zEMG": raw[i]["zEMG"],
          "physio/tEMG": raw[i]["tEMG"],
          "physio/GSR": raw[i]["GSR"],
          "physio/rAMP": raw[i]["rAMP"],
          "physio/plethysmograph": raw[i]["plethysmograph"],
          "physio/temp": raw[i]["temp"],

          # valence, arousal, dominance, liking
          "affect/trial_selfreport": labels[i],
      }

      for subsampled in _tiled_subsample_example(example, subsample_width,
                                                 subsample_step):
        for key, value in subsampled.items():
          subsampled[key] = _flatten(value)
        yield subsampled
        ct += 1
        if isinstance(how_many, int) and ct >= how_many:
          return


@registry.register_problem
class DeapProblemBase(problem.Problem):

  @property
  def problem_code(self):
    return 1234

  @property
  def trial_subsample_hparams(self):
    return {"width": 100, "step": 10}

  @property
  def training_fraction(self):
    return 0.9

  def generator(self, tmp_dir, max_nbr_cases, is_training):
    w, s = self.trial_subsample_hparams.values()
    return _generator(tmp_dir=tmp_dir,
                      subsample_width=w,
                      subsample_step=s,
                      how_many=max_nbr_cases,
                      is_training=is_training,
                      training_fraction=self.training_fraction)

  @property
  def train_size(self):
    return 100

  @property
  def dev_size(self):
    return 100

  @property
  def num_shards(self):
    return 1

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    generator_utils.generate_dataset_and_shuffle(
        self.generator(tmp_dir, self.train_size, is_training=True),
        self.training_filepaths(data_dir, self.num_shards, shuffled=True),
        self.generator(tmp_dir, self.dev_size, is_training=False),
        self.dev_filepaths(data_dir, 1, shuffled=True),
        shuffle=True)

  @property
  def num_self_report_affect_classes(self):
    return 4

  def feature_encoders(self, data_dir):
    del data_dir

    return {
        #"video": text_encoder.ImageEncoder(channels=self.video_shape[3]),
    }

  def get_eeg_shape(self):
    return (len(DEAP_EXPECTED_MAPPED_EEG_CHANNELS),
            self.trial_subsample_hparams["width"])

  def example_reading_spec(self):

    num_classes = self.num_self_report_affect_classes
    eeg_shape = self.get_eeg_shape()

    data_fields = {
        "eeg/raw":
            tf.FixedLenFeature((32 * 100,), dtype=tf.int64),
        "affect/trial_selfreport":
            tf.FixedLenFeature([num_classes], dtype=tf.int64),
    }

    data_items_to_decoders = {
        "eeg":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="eeg/raw"),
        "targets":
            tf.contrib.slim.tfexample_decoder.Tensor(
                tensor_key="affect/trial_selfreport")
    }

    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"eeg": "IdentityModality", "targets": "SymbolModality"}

    # Ugh this isn't necessary #HACK clarify how to remove it
    p.vocab_size = {
        "eeg": 256,
        "targets": 256,
    }

  def preprocess_example(self, example, mode, hparams):
    return example
