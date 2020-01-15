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

import tensorflow as tf

import os
import tempfile
import subprocess

import numpy as np
from scipy.io import wavfile


def standardize_audio_array(audio, audio_shape):

  # Incoming audio is in [-0.5, 0.5] and is of type float32

  pad_size = audio_shape[0] - len(audio)

  if pad_size > 0:
    audio = np.concatenate([audio, np.random.uniform(-0.5, 0.5, (pad_size,))])
  else:
    audio = audio[0:audio_shape[0]]

  audio = audio.tolist()

  return audio


def mp4_to_1d_array(mp4_path, audio_bitrate=44100):
  """Extract audio from MP4 and load as 1d array."""
  with tempfile.TemporaryDirectory() as tmpd:
    tmp_wav_path = os.path.join(tmpd, "mywav.wav")
    subprocess.check_output([
        "ffmpeg", "-loglevel", "quiet", "-i", mp4_path, "-f", "wav", "-ar",
        str(audio_bitrate), "-vn", tmp_wav_path
    ])
    audio_data = wavfile.read(tmp_wav_path)[1]
  audio_data = audio_data / np.iinfo(np.int16).max
  audio_data = audio_data.astype(np.float32)
  return audio_data
