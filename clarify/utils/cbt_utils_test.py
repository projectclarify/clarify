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

import tensorflow as tf
import os
import uuid
import tempfile
import numpy as np

#from pcml.operations import cbt_datagen

from tensor2tensor.utils import registry

from clarify.utils import cbt_utils
#from pcml.operations import extract

from clarify.utils.cfg_utils import Config

TEST_CONFIG = Config()

from clarify.utils.cbt_utils import _lex_index


class TestCBTUtils(tf.test.TestCase):

  def setUp(self):

    self.project = TEST_CONFIG.get("project")
    self.instance = TEST_CONFIG.get("test_cbt_instance")
    self.tmpdir = tempfile.mkdtemp()
    self.table = "clarify-test-{}-cbt-utils".format(str(uuid.uuid4())[0:8])

  def test_helper_models(self):
    """Currently these don't enforce types of the nested objects."""

    vm = cbt_utils.VideoMeta(video_length=1,
                             audio_length=1,
                             shard_id=0,
                             video_id=0,
                             audio_block_size=1000)
    vm.as_dict()

    sampling_meta00 = {
        'frame_skip_size': 0,
        'frame_shift': 0,
        'frame_sample_bounds': [178, 192],
        'audio_sample_bounds': [315056, 341604]
    }

    abm = {
        "min_query_block": 1,
        "max_query_block": 3,
        "num_query_blocks": 3,
        "query_start": 100,
        "query_end": 200
    }

    avcs = cbt_utils.AVCorrespondenceSample(
        video=np.array([1, 2, 3]),
        audio=np.array([1, 2, 3]),
        labels={
            "same_video": 1,
            "overlap": 0
        },
        meta={
            "video_source": vm,
            "audio_source": vm,
            "video_sample_meta": sampling_meta00,
            "audio_sample_meta": sampling_meta00,
            "audio_keys": ["foo"],
            "frame_keys": ["foo", "foo"],
            "audio_block_meta": abm
        })

    serialized = avcs.serialize()

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

    selection = cbt_utils.RawVideoSelection(project=self.project,
                                            instance=self.instance,
                                            table=table_tag,
                                            prefix=prefix)

    sent_meta = cbt_utils.VideoShardMeta(shard_id=0,
                                         num_videos=1,
                                         status="finished",
                                         num_shards=1)

    selection.set_shard_meta(sent_meta)

    recv_meta = selection.lookup_shard_metadata()

    train_meta_key = "train_meta_{}".format(_lex_index(0)).encode()

    self.assertTrue(train_meta_key in recv_meta)

    self.assertEqual(recv_meta[train_meta_key].as_dict(), sent_meta.as_dict())


  """
  def test_generate_av_correspondence_examples(self):

    table_tag = "{}-pairs".format(self.table)
    prefix = "train"
    manifest_path = "gs://clarify-dev/test/extract/manifest.csv"
    frames_per_video = 15
    downsample_xy_dims = 64
    greyscale = True
    num_channels = 1

    selection = cbt_utils.RawVideoSelection(project=self.project,
                                            instance=self.instance,
                                            table=table_tag,
                                            prefix=prefix)

    extract.extract_to_cbt(manifest_path=manifest_path,
                           shard_id=0,
                           num_shards=1,
                           project=self.project,
                           instance=self.instance,
                           table=table_tag,
                           target_prefix=prefix,
                           tmp_dir=tempfile.mkdtemp(),
                           downsample_xy_dims=downsample_xy_dims,
                           greyscale=greyscale,
                           resample_every=2,
                           audio_block_size=1000)

    selection_meta = selection.lookup_shard_metadata()

    train_meta_key = "train_meta_{}".format(_lex_index(0)).encode()

    self.assertTrue(selection_meta[train_meta_key].num_videos == 1)

    video_meta = selection._get_random_video_meta(selection_meta)

    generator = selection.sample_av_correspondence_examples(
        frames_per_video=frames_per_video, max_num_samples=1)

    sample = generator.__next__()

    self.assertTrue(isinstance(sample, dict))
    for key, value in sample.items():
      cond = isinstance(value, cbt_utils.AVCorrespondenceSample)
      self.assertTrue(cond)

    positive_same = sample["positive_same"]
    negative_same = sample["negative_same"]
    #negative_different = sample["negative_different"]

    # The expected video and audio shapes
    video_shape = (frames_per_video, downsample_xy_dims, downsample_xy_dims,
                   num_channels)

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
    #_verify(negative_different, 0, 0)

    generator = selection.sample_av_correspondence_examples(
        frames_per_video=frames_per_video, max_num_samples=1, keys_only=True)

    sample = generator.__next__()

    serialized = sample["positive_same"].serialize()

  def test_tfexampleselection_e2e(self):

    table_tag = "{}-tfexe2e".format(self.table)
    prefix = "train_"
    video_shape = (4, 16, 16, 3)
    audio_shape = (1234)
    mock_num_examples = 100

    selection = cbt_utils.TFExampleSelection(project=self.project,
                                             instance=self.instance,
                                             table=table_tag,
                                             prefix=prefix)

    # Check that the test table is empty
    self.assertTrue(not selection.rows_at_least(1))

    def _dummy_generator(n):
      for _ in range(n + 1):
        video = np.random.randint(0, 255, video_shape).astype(np.uint8)
        audio = np.random.randint(0, 255, audio_shape).astype(np.uint8)
        target_label = np.random.randint(0, 2, (1)).astype(np.uint8)
        yield {
            "audio": audio.tolist(),
            "video": video.flatten().tolist(),
            "target": target_label.tolist()
        }

    num_records_loaded = selection.random_load_from_generator(
        generator=_dummy_generator(mock_num_examples))

    self.assertEqual(num_records_loaded, mock_num_examples)

    # In astronomically rare cases this could flake but with an alphabet
    # of 26 and a prefix tag length of 4 the probability of having
    # more than 50 collisions is low... like < (50/(26^4))^50...
    # 9e-199 that's almost 1/(2*googles).
    self.assertTrue(selection.rows_at_least(0.5 * mock_num_examples))

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

    source_selection = cbt_utils.RawVideoSelection(project=self.project,
                                                   instance=self.instance,
                                                   table=source_table_tag,
                                                   prefix=prefix)

    extract.extract_to_cbt(manifest_path=manifest_path,
                           shard_id=0,
                           num_shards=1,
                           project=self.project,
                           instance=self.instance,
                           table=source_table_tag,
                           target_prefix=prefix,
                           tmp_dir=tempfile.mkdtemp())

    test_problem = registry.problem("cbt_datagen_test_problem")

    example_generator = test_problem.sampling_generator(source_selection)

    target_selection = cbt_utils.TFExampleSelection(project=self.project,
                                                    instance=self.instance,
                                                    table=table_tag + "t",
                                                    prefix=prefix)

    num_records_loaded = target_selection.random_load_from_generator(
        generator=example_generator)

    self.assertTrue(num_records_loaded > 0)
  """

if __name__ == "__main__":
  tf.test.main()
