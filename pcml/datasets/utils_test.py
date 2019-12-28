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

"""Tests of utilities supporting data generation code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import json
import tempfile
import numpy as np
import tensorflow as tf

from pcml.datasets.test_utils import get_test_data_path
from pcml.datasets.utils import get_random_index_subinterval
from pcml.datasets.utils import get_temporal_subinterval
from pcml.datasets.utils import gen_dummy_schedule
from pcml.datasets.utils import gen_gif_encoded_from_field_spec
from pcml.datasets.utils import array2gif
#from pcml.datasets.mmimp import MultiModalImagingExampleSpec

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name  


class TestMultiProblemUtils(tf.test.TestCase):
  """
  def test_dummy_video_generator(self):
    spec = MultiModalImagingExampleSpec()
    encoded = array2gif(spec.fields["video"].mock_one(zeros=True).numpy())
    self.assertTrue(isinstance(encoded, bytes))
    self.assertTrue(len(encoded) > 0)
    encoded2 = gen_gif_encoded_from_field_spec(spec.fields["video"],
                                               zeros=True)
    self.assertEqual(encoded2, encoded)
  """

  def test_gen_dummy_schedule(self):
    
    num_problems = 10
    steps_per_problem = 10

    schedule = gen_dummy_schedule(
        num_problems=num_problems, steps_per_problem=steps_per_problem)

    assert schedule[0] == "step"
    assert len(schedule[1]) == num_problems
    assert len(schedule[2]) == num_problems
    assert len(schedule[2][0]) == num_problems


class TestTestUtils(tf.test.TestCase):
  """Tests of test utils."""

  def test_get_random_index_subinterval(self):

    samples = get_random_index_subinterval(5, 2, 100)
    self.assertTrue(np.amax(samples) < 5)
    self.assertTrue(np.amin(samples) >= 0)

  def test_get_temporal_subinterval(self):

    for duration, subinterval_length in [(10, 2), (10, 10), (10.0, 0.1)]:
      start, end = get_temporal_subinterval(duration, subinterval_length)
      self.assertTrue(((end - start) - subinterval_length) < 0.001)


if __name__ == "__main__":
  tf.test.main()
