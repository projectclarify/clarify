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

from pcml.utils import video_utils


class TestVideoUtils(tf.test.TestCase):

  def test_av_samplable_combined(self):

    l = 20
    al = 9999

    avs = video_utils.AVSamplable(video_length=l, audio_length=al)

    _ = avs._sample_frame_indices(sample_length=l)

    frame_indices = avs._sample_frame_indices(sample_length=int(l/2))
    self.assertEqual(len(frame_indices), int(l/2))

    audio_indices = avs._audio_given_frame_sample(frame_indices)
    audio_sample_len = len(audio_indices)
    self.assertEqual(audio_sample_len, 4999)

    # Assert raises ValueError
    with self.assertRaises(ValueError):
      _, _, _ = avs.sample_av_pair(num_frames=0)

    # For a variety of second video lengths
    #for scale_factor in [0.99, 1.01, 1.1, 2, 3.1]:
    # Actually decided not to enforce this.
    for scale_factor in [1]:

      vl2 = int(l*scale_factor)
      al2 = int(al*scale_factor)
      avs2 = video_utils.AVSamplable(video_length=vl2, audio_length=al2)

      for k in range(1, l - 5):

        f, a, m = avs.sample_av_pair(num_frames=k,
                                     max_frame_shift=0,
                                     max_frame_skip=0)

        self.assertEqual(len(f), k)

        for frame_shift in range(0, 3):

          for frame_skip in range(0, min(k-1, 3)):

            f_, a_, m_ = avs2.sample_av_pair(num_frames=k,
                                             max_frame_shift=frame_shift,
                                             max_frame_skip=frame_skip)

            self.assertEqual(len(f_), (len(f)))
            #self.assertEqual(len(a_), len(a))


if __name__ == "__main__":
  tf.test.main()
