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

"""Additional distributed datagen and augmentation problem defs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import uuid
import tempfile
import numpy as np

# Has to be imported before registry
import pcml
from pcml.operations import cbt_datagen

from tensor2tensor.utils import registry

from pcml.utils import cbt_utils
from pcml.operations import extract

from pcml.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestCBTUtils(tf.test.TestCase):

  def setUp(self):

    self.project = TEST_CONFIG.get("project")
    self.instance = TEST_CONFIG.get("test_cbt_instance")
    self.tmpdir = tempfile.mkdtemp()
    self.table = "clarify-test-{}-cbt-utils".format(
      str(uuid.uuid4())[0:8]
    )

  def test_helper_models(self):
    """Currently these don't enforce types of the nested objects."""

    vm = cbt_utils.VideoMeta(video_length=1, audio_length=1,
                             shard_id=0, video_id=0)
    vm.as_dict()

    avcs = cbt_utils.AVCorrespondenceSample(video=np.array([1,2,3]),
                                  audio=np.array([1,2,3]),
                                  labels={"same_video": 1,
                                          "overlap": 0},
                                  meta={"video_source": vm,
                                        "audio_source": vm})

    vsm = cbt_utils.VideoShardMeta(num_videos=1,
                                   status="started",
                                   shard_id=0,
                                   num_shards=1)
    vsm.num_video = 10
    vsm.status = "finished"
    vsm.as_dict()

  def test_set_and_lookup_shard_meta(self):

    table_tag = "{}-meta".format(self.table)
    prefix = "train"

    selection = cbt_utils.RawVideoSelection(
      project=self.project,
      instance=self.instance,
      table=table_tag,
      prefix=prefix
    )

    sent_meta = cbt_utils.VideoShardMeta(
      shard_id=0,
      num_videos=1,
      status="finished",
      num_shards=1
    )

    selection.set_shard_meta(sent_meta)

    recv_meta = selection.lookup_shard_metadata()

    self.assertTrue("train_meta_0" in recv_meta)

    self.assertEqual(recv_meta["train_meta_0"].as_dict(),
                     sent_meta.as_dict())

  def test_generate_av_correspondence_examples(self):

    table_tag = "{}-pairs".format(self.table)
    prefix = "train"
    manifest_path = "gs://clarify-dev/test/extract/manifest.csv"
    frames_per_video = 15

    selection = cbt_utils.RawVideoSelection(
      project=self.project,
      instance=self.instance,
      table=table_tag,
      prefix=prefix
    )

    extract.extract_to_cbt(manifest_path=manifest_path,
                           shard_id=0, num_shards=1,
                           project=self.project,
                           instance=self.instance,
                           table=table_tag,
                           target_prefix=prefix,
                           tmp_dir=tempfile.mkdtemp())

    selection_meta = selection.lookup_shard_metadata()
    
    self.assertTrue(selection_meta["train_meta_0"].num_videos == 1)

    video_meta = selection._get_random_video_meta(selection_meta)

    generator = selection.sample_av_correspondence_examples(
        frames_per_video=frames_per_video,
        max_num_samples=1)

    sample = generator.__next__()

    self.assertTrue(isinstance(sample, dict))
    for key, value in sample.items():
      cond = isinstance(value, cbt_utils.AVCorrespondenceSample)
      self.assertTrue(cond)

    positive_same = sample["positive_same"]
    negative_same = sample["negative_same"]
    negative_different = sample["negative_different"]

    # The expected video and audio shapes
    video_shape = (frames_per_video, 96, 96, 3)

    def _verify(sample, same_video, overlap):
      self.assertEqual(sample.labels["same_video"], same_video)
      self.assertEqual(sample.labels["overlap"], overlap)
      self.assertEqual(type(sample.video), np.ndarray)
      self.assertEqual(type(sample.audio), np.ndarray)
      
      reshaped = np.reshape(sample.video, video_shape)
      flat = sample.video.flatten().tolist()
      self.assertTrue(isinstance(flat[0], int))

    _verify(positive_same, 1, 1)
    _verify(negative_same, 1, 0)
    _verify(negative_different, 0, 0)

  def test_tfexampleselection_e2e(self):

    table_tag = "{}-tfexe2e".format(self.table)
    prefix = "train_"
    video_shape = (4,16,16,3)
    audio_shape = (1234)
    mock_num_examples = 100

    selection = cbt_utils.TFExampleSelection(
      project=self.project,
      instance=self.instance,
      table=table_tag,
      prefix=prefix
    )

    # Check that the test table is empty
    self.assertTrue(not selection.rows_at_least(1))

    def _dummy_generator(n):
      for _ in range(n + 1):
        video = np.random.randint(0,255,video_shape).astype(np.uint8)
        audio = np.random.randint(0,255,audio_shape).astype(np.uint8)
        target_label = np.random.randint(0,2,(1)).astype(np.uint8)
        yield {
          "audio": audio.tolist(),
          "video": video.flatten().tolist(),
          "target": target_label.tolist()
        }

    """
    # Or, alternatively with a real video
    def generator():
      video = mp4_to_frame_array(local_test_video_path).astype(np.uint8)
      audio = mp4_to_1d_array(local_test_video_path).astype(np.uint8)
      target_label = np.random.randint(0,2,(1)).astype(np.uint8)
      yield {
        "audio": audio.tolist(),
        "video": video.flatten().tolist(),
        "target": target_label.tolist()
      }
    # If this is from VoxCeleb2 it will have the shape (236,224,224,3)
    # and can be visuaized in a notebook via
    # plt.imshow(np.reshape(recv_video, (236,224,224,3))[0])
    # following the iterate_tfexamples(selection) step below.
    # Of course sending and receiving a video with shape (4,16,16,3)
    # is much faster.
    """

    num_records_loaded = selection.random_load_from_generator(
      generator=_dummy_generator(mock_num_examples))

    self.assertEqual(num_records_loaded, mock_num_examples)

    # In astronomically rare cases this could flake but with an alphabet
    # of 26 and a prefix tag length of 4 the probability of having
    # more than 50 collisions is low... like < (50/(26^4))^50...
    # 9e-199 that's almost 1/(2*googles).
    self.assertTrue(selection.rows_at_least(0.5*mock_num_examples))

    example_iterator = selection.iterate_tfexamples()

    ex = example_iterator.__next__()

    recv_audio = ex.features.feature['audio'].int64_list.value
    recv_video = ex.features.feature['video'].int64_list.value
    recv_target = ex.features.feature['target'].int64_list.value

    _ = np.reshape(recv_audio, audio_shape)
    _ = np.reshape(recv_video, video_shape)
    _ = np.reshape(recv_target, (1))


  def test_e2e_via_problem(self):

    table_tag = "{}-prob".format(self.table)
    prefix = "train"
    manifest_path = "gs://clarify-dev/test/extract/manifest.csv"
    frames_per_video = 15
    source_table_tag = table_tag + "s"
    target_table_tag = table_tag + "t"

    source_selection = cbt_utils.RawVideoSelection(
      project=self.project,
      instance=self.instance,
      table=source_table_tag,
      prefix=prefix
    )

    extract.extract_to_cbt(manifest_path=manifest_path,
                           shard_id=0, num_shards=1,
                           project=self.project,
                           instance=self.instance,
                           table=source_table_tag,
                           target_prefix=prefix,
                           tmp_dir=tempfile.mkdtemp())

    test_problem = registry.problem("cbt_datagen_test_problem")

    example_generator = test_problem.sampling_generator(source_selection)

    target_selection = cbt_utils.TFExampleSelection(
      project=self.project,
      instance=self.instance,
      table=table_tag + "t",
      prefix=prefix)

    num_records_loaded = target_selection.random_load_from_generator(
      generator=example_generator)

    self.assertTrue(num_records_loaded > 0)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
