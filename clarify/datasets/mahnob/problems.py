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
"""MAHNOB-HCI problem definitions."""

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
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry
from tensor2tensor.layers import modalities

from pcml.datasets.utils import get_input_file_paths

DEFAULT_DATA_ROOT = "gs://clarify-data/requires-eula/hcitagging/"


def _raw_data_verifier(remote_data_root):
  """Raw data verifier for MAHNOB-HCI dataset."""
  video_regex = os.path.join(remote_data_root,
                             "videotag/Sessions/*/*C1\ trigger*")
  audio_regex = os.path.join(remote_data_root, "videotag/Sessions/*/*Audio*")
  guide_cut_regex = os.path.join(remote_data_root,
                                 "videotag/Sessions/*/*Guide-Cut*")
  all_data_regex = os.path.join(remote_data_root,
                                "videotag/Sessions/*/*All-Data*")
  for query in [video_regex, audio_regex, guide_cut_regex, all_data_regex]:
    files = get_input_file_paths(query, True, 1)
    l = len(files)
    if l != 870:
      raise ValueError("Expected 870 files returned for query %s, saw %s" %
                       (query, l))


def maybe_get_data(tmp_dir,
                   remote_data_root,
                   is_training=True,
                   training_fraction=0.9):
  """Maybe download MAHNOB-HCI data."""

  video_regex = os.path.join(remote_data_root,
                             "videotag/Sessions/*/*C1\ trigger*")
  audio_regex = os.path.join(remote_data_root, "videotag/Sessions/*/*Audio*")
  guide_cut_regex = os.path.join(remote_data_root,
                                 "videotag/Sessions/*/*Guide-Cut*")
  all_data_regex = os.path.join(remote_data_root,
                                "videotag/Sessions/*/*All-Data*")

  video_paths = get_input_file_paths(video_regex,
                                     is_training=is_training,
                                     training_fraction=training_fraction)
  audio_paths = get_input_file_paths(audio_regex,
                                     is_training=is_training,
                                     training_fraction=training_fraction)
  guide_cut_paths = get_input_file_paths(guide_cut_regex,
                                         is_training=is_training,
                                         training_fraction=training_fraction)
  all_data_paths = get_input_file_paths(all_data_regex,
                                        is_training=is_training,
                                        training_fraction=training_fraction)

  # This is valid in part because these lists have been verified to have the same length
  # TODO: Can we trust they will always have the same *order*?? Perhaps get_input_file_paths
  # should have this property explicitly.
  for i, _ in enumerate(video_paths):
    session = video_paths[i].split("/")[-2]
    paths = video_paths[i], audio_paths[i], guide_cut_paths[i], all_data_paths[
        i]
    fnames = ["%s-%s" % (session, thing.split("/")[-1]) for thing in paths]
    yield {
        "video_path":
            generator_utils.maybe_download(tmp_dir, fnames[0], paths[0]),
        "audio_path":
            generator_utils.maybe_download(tmp_dir, fnames[1], paths[1]),
        "guide_cut_path":
            generator_utils.maybe_download(tmp_dir, fnames[2], paths[2]),
        "all_data_path":
            generator_utils.maybe_download(tmp_dir, fnames[3], paths[3]),
    }
