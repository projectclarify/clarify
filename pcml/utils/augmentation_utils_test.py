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

"""Tests of augmentation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from pcml.utils import augmentation_utils


class TestVideoAugmentationUtils(tf.test.TestCase):

  def setUp(self):

    before_top = np.concatenate([np.ones((20,32,32,3))*0.8,
                                 np.zeros((20,32,32,3))+0.2], axis=1)
    before_bottom = np.concatenate([np.zeros((20,32,32,3))+0.3,
                                    np.ones((20,32,32,3))*0.65], axis=1)
    before = np.concatenate([before_top, before_bottom], axis=2)

    scale_min = 0.1
    scale_max = 1
    scale_steps = before.shape[0]

    for i in range(before.shape[0]):
      scale_factor = (scale_max - scale_min)
      scale_factor *= ((scale_steps-i)/(scale_steps-1))
      scale_factor += scale_min
      before[i] = before[i]*scale_factor

    self.test_video = before

  def test_random_temporal_subsample(self):
    
    sample = augmentation_utils.random_temporal_subsample(
      self.test_video, length=10, max_frame_skips=2
    )
    self.assertTrue(sample.shape[0] == 10)

  def test_e2e(self):

    augmented = augmentation_utils.augment_video(self.test_video)
    self.assertTrue(isinstance(augmented, np.ndarray))


class TestAudioAugmentationUtils(tf.test.TestCase):

  def setUp(self):

    x = np.array([i for i in range(360)])/10
    self.test_audio = np.sin(x)*0.5

  def test_e2e(self):

    aug_hparams = {
      "do_random_shift": True,
      "do_add_gaussian_noise": True,
      "shift_reduction": 0.05,
      "gaussian_snr": 10
    }

    augmented = augmentation_utils.augment_audio(self.test_audio,
                                                 **aug_hparams)


if __name__ == "__main__":
  tf.test.main()
