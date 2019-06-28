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
import math
from PIL import Image

import numpy as np
import cv2


def mp4_to_frame_array(input_path):

  frames = []

  cap = cv2.VideoCapture(input_path)

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    tf.logging.error("Error opening video stream or file")

  i = 0

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
 
      frame = cv2.cvtColor(frame,
                           cv2.COLOR_BGR2RGB)

      frames.append(frame)
 
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
    # Break the loop
    else: 
      break
 
  # When everything done, release the video capture object
  cap.release()

  return np.asarray(frames)


def stream_mp4(input_path, downsample_size=(96, 96)):

  frames = []

  cap = cv2.VideoCapture(input_path)

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    tf.logging.error("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

      frame = cv2.cvtColor(frame,
                           cv2.COLOR_BGR2RGB)

      if isinstance(downsample_size, tuple):
        frame = Image.fromarray(frame).resize(size=downsample_size)

      yield np.asarray(frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else: 
      break

  # When everything done, release the video capture object
  cap.release()


def resize_video(input_frame_array, size):

  with tf.Session() as sess:
    img = tf.constant(input_frame_array)
    resize_op = tf.image.resize_images(
        img, [size, size],
        tf.image.ResizeMethod.AREA)
    resized = sess.run([resize_op])[0].tolist()
  return resized


def stream_mp4(input_path, downsample_size=(96, 96)):

  frames = []

  tf.logging.info(input_path)

  cap = cv2.VideoCapture(input_path)

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    tf.logging.error("Error opening video stream or file")

  i = 0

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      frame = cv2.cvtColor(frame,
                           cv2.COLOR_BGR2RGB)

      if isinstance(downsample_size, tuple):
        if len(downsample_size) != 2:
          msg = "If downsampling expected size of len 2, saw {}".format(
            downsample_size
          )
          raise ValueError(msg)

        frame = Image.fromarray(frame).resize(size=downsample_size)

      yield np.asarray(frame)

    # Break the loop
    else: 
      break

  # When everything done, release the video capture object
  cap.release()


class VideoFrame(object):

  def __init__(self, data, next_node=None, previous_node=None):
    self.data = data
    self.next_node = next_node
    self.previous_node = previous_node

  def set_next(self, node):
    self.next_node = node

  def set_previous(self, node):
    self.previous_node = node


class Video(object):

  def __init__(self):
    self.front = None
    self.tail = None
    self.length = 0

  def insert(self, data):
    new_node = VideoFrame(data=data)
    previous_tail = self.tail

    if previous_tail:
      previous_tail.next_node = new_node

    new_node.previous_node = previous_tail
    self.tail = new_node

    if not self.front:
      self.front = new_node
    
    self.length += 1

  def get_iterator(self):
    current = self.front
    while current is not None:
      yield current.data
      current = current.next_node

  def load_from_file(self, input_path, downsample_size=(96,96)):
    for frame in stream_mp4(input_path, downsample_size=downsample_size):
      self.insert(frame)


class AVSamplable(object):

  def __init__(self, video_length, audio_length):
    self.length = video_length
    self.audio_length = audio_length
    self.audio_steps_per_frame = audio_length/float(video_length)

  def _sample_frame_indices(self, sample_length):
    max_start_index = self.length - sample_length
    if max_start_index < 0:
      msg = "Sample length greater than video len, {}, {}".format(
        sample_length, self.length
      )
      raise ValueError(msg)
        
    def _get_frame_indices(start_index):
      return [start_index + i for i in range(sample_length)]

    # Uncommon case but sure if the sample size equals the video
    # size...
    if max_start_index == 0:
      msg = "Sampling is trivial because sample and video same len."
      tf.logging.info(msg)
      return _get_frame_indices(start_index=0)

    start_index = np.random.randint(0, max_start_index, 1)[0]
    return _get_frame_indices(start_index)

  def _audio_given_frame_sample(self, frame_sample):

    offset, num_frames = frame_sample[0], len(frame_sample)

    sampled_audio_length = int(self.audio_steps_per_frame*num_frames)

    audio_start_index = int(self.audio_steps_per_frame*offset)

    audio_end_index = audio_start_index + sampled_audio_length

    return [audio_start_index + i for i in range(sampled_audio_length)]

  def sample_av_pair(self, num_frames, max_frame_shift=0,
                     max_frame_skip=0):
    """

    Notes:
    * When max_frame_skip > 0 if a frame skip is not sampled then
      max_frame_skip frames are trimmed from the end of a sampled
      array of length num_frames + max_frame_skip.

    Args:
      num_frames(int): The number of frames that the resulting video
        sample should have.

    """
    
    tf.logging.debug("num_frames={}, max_shift={}, max_skip={}".format(
      num_frames, max_frame_shift, max_frame_skip
    ))

    if not isinstance(num_frames, int) or num_frames <= 0:
      msg = "Must sample num_frames >= 0, saw {}".format(
        num_frames
      )
      raise ValueError(msg)

    frame_shift = np.random.randint(0, 2*max_frame_shift + 1)

    frame_skip_size = np.random.randint(0, max_frame_skip + 1)

    meta = {
      "frame_skip_size": frame_skip_size,
      "frame_shift": frame_shift - max_frame_shift
    }

    sample_length = num_frames + max_frame_shift*2 + max_frame_skip

    # Get the raw index arrays
    frame_sample = self._sample_frame_indices(sample_length)
    audio_sample = self._audio_given_frame_sample(frame_sample)
    
    # Adjust audio indices for frame shift
    c = math.ceil(max_frame_shift/2.0)
    trim_audio = int(max_frame_shift*(self.audio_steps_per_frame))
    audio_sample_length = int(self.audio_steps_per_frame*num_frames)
    audio_sample = audio_sample[trim_audio+c:]
    audio_sample = audio_sample[:audio_sample_length]

    # Apply frame shift
    frame_sample = frame_sample[frame_shift:]
    frame_sample = frame_sample[:num_frames + frame_skip_size]

    # Sample frame indices to skip
    skip = np.random.choice(range(len(frame_sample)), frame_skip_size, replace=False)
    frame_sample = np.delete(frame_sample, skip)
    assert len(frame_sample) == num_frames
    
    meta["frame_sample_bounds"] = [frame_sample[0], frame_sample[-1]]
    meta["audio_sample_bounds"] = [audio_sample[0], audio_sample[-1]]

    return (np.asarray(frame_sample), np.asarray(audio_sample), meta)
