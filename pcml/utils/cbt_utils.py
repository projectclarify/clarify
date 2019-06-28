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
import numpy as np
import datetime
import json

from pcml.utils import video_utils

from google.cloud.bigtable import column_family as cbt_lib_column_family

from google.cloud import bigtable
from google.cloud.bigtable import row_filters

from tensor2tensor.data_generators.generator_utils import to_example

from collections import namedtuple


class BigTableSelection(object):

  def __init__(self, project, instance, table,
               column_families,
               prefix=None,
               sa_key_path=None,
               column_qualifier=None,
               column_family=None,
               *args, **kwargs):

    self.project = project    
    self.instance_name = instance
    self.prefix = prefix
    self.column_qualifier = column_qualifier

    # We require the full set of column families to be specified
    # whether we are making a selection of a specific column
    # family or not. This might change later.
    if not isinstance(column_families, list):
      raise ValueError("Expected type list for column_families.")
    self.column_families = column_families

    # When specifying a specific selection, if doing so, need to
    # specify the column_family (and it must be in column_families).
    if column_family:
      if not isinstance(column_family, str):
        raise ValueError("Expected type str for column_family.")
      if column_family not in column_families:
        raise ValueError("Saw column_family not in column_families.")
    self.column_family = column_family

    self.table_name = table
    
    self.materialize(sa_key_path=sa_key_path)

  def materialize(self, sa_key_path=None):

    if isinstance(sa_key_path, str):
      self.client = bigtable.Client.from_service_account_json(
        sa_key_path, admin=True)
    else:
      self.client = bigtable.Client(admin=True)

    self.instance = self.client.instance(self.instance_name)

    self.table = self.instance.table(self.table_name)

    if self.table.exists():
      return

    max_versions_rule = cbt_lib_column_family.MaxVersionsGCRule(1)
    cf = {key: max_versions_rule for key in self.column_families}
    self.table.create(column_families=cf)

  def get_basic_row_iterator(self):
    """Convenience function to obtain iterator, maybe using prefix."""

    table = self.table

    row_filter = None

    if isinstance(self.prefix, str):

      prefix = self.prefix

      if not prefix.endswith(".*"):
        prefix += ".*"

      row_filter = row_filters.RowKeyRegexFilter(
        regex=prefix
      )

    partial_rows = table.read_rows(filter_=row_filter)

    return partial_rows

  def rows_at_least(self, min_rows=1):
    """That there are at least `min_rows` in the table."""

    iterator = self.get_basic_row_iterator()

    i = 0

    for row in iterator:
      i += 1
      if i > min_rows:
        return True

    return i > min_rows

  def as_dict(self):
    return {"table_name": self.table_name,
            "instance_name": self.instance_name,
            "row_key_prefix": self.prefix,
            "column_families": self.column_families,
            "column_family": self.column_family,
            "column_qualifier": self.column_qualifier}


def _expect_type(obj, t):
  if not isinstance(obj, t):
    msg = "Expected numpy.ndarray, saw {}".format(type(obj))
    raise ValueError(msg)


class AVCorrespondenceSample(object):

  def __init__(self, video, audio, labels, meta):
    self.video = video
    self.audio = audio
    self.labels = labels
    self.meta = meta

  @property
  def video(self):
    return self._video

  @video.setter
  def video(self, x):
    _expect_type(x, np.ndarray)
    self._video = x

  @property
  def audio(self):
    return self._audio

  @audio.setter
  def audio(self, x):
    _expect_type(x, np.ndarray)
    self._audio = x

  @property
  def labels(self):
    return self._labels

  @labels.setter
  def labels(self, x):
    _expect_type(x, dict)
    assert "same_video" in x
    assert "overlap" in x
    for value in x.values():
      assert value in [0, 1]
    self._labels = x

  @property
  def meta(self):
    return self._meta

  @meta.setter
  def meta(self, x):
    _expect_type(x, dict)
    for key in ["video_source", "audio_source",
                "video_sample_meta",
                "audio_sample_meta"]:
      assert key in x
    assert len(x.keys()) == 4
    assert isinstance(x["video_source"], VideoMeta)
    assert isinstance(x["audio_source"], VideoMeta)
    self._meta = x

  def as_dict(self):
    return {
      "video": self.video,
      "audio": self.audio,
      "labels": self.labels,
      "meta": {
        "audio_source": self.meta["audio_source"].as_dict(),
        "video_source": self.meta["video_source"].as_dict(),
        "video_sample_meta": self.meta["video_sample_meta"],
        "audio_sample_meta": self.meta["audio_sample_meta"]
      }
    }


class VideoMeta(object):

  def __init__(self, video_length, audio_length, video_id, shard_id):
    self.video_length = video_length
    self.audio_length = audio_length
    self.video_id = video_id
    self.shard_id = shard_id

  @property
  def video_length(self):
    return self._video_length

  @video_length.setter
  def video_length(self, x):
    assert isinstance(x, int)
    assert x > 0
    self._video_length = x

  @property
  def audio_length(self):
    return self._audio_length

  @audio_length.setter
  def audio_length(self, x):
    assert isinstance(x, int)
    assert x > 0
    self._audio_length = x

  @property
  def video_id(self):
    return self._video_id

  @video_id.setter
  def video_id(self, x):
    self._video_id = x

  @property
  def shard_id(self):
    return self._shard_id

  @shard_id.setter
  def shard_id(self, x):
    self._shard_id = x

  def as_dict(self):
    return {"video_length": self.video_length,
            "audio_length": self.audio_length,
            "video_id": self.video_id,
            "shard_id": self.shard_id}


class VideoShardMeta(object):

  def __init__(self, num_videos, status, shard_id, num_shards):
    self.num_videos = num_videos
    self.status = status
    self.shard_id = shard_id
    self.num_shards = num_shards

  @property
  def num_videos(self):
    return self._num_videos

  @num_videos.setter
  def num_videos(self, x):
    _expect_type(x, int)
    self._num_videos = x

  @property
  def status(self):
    return self._status

  @status.setter
  def status(self, x):
    assert x in ["started", "finished"]
    self._status = x

  @property
  def shard_id(self):
    return self._shard_id

  @shard_id.setter
  def shard_id(self, x):
    _expect_type(x, int)
    self._shard_id = x

  @property
  def num_shards(self):
    return self._num_shards

  @num_shards.setter
  def num_shards(self, x):
    _expect_type(x, int)
    self._num_shards = x

  def as_dict(self):
    return {"num_videos": self.num_videos,
            "shard_id": self.shard_id,
            "status": self.status,
            "num_shards": self.num_shards}


def _validate_shard_meta_key(key):
  
  # Validating its basic structure not whether its encoded or not
  if isinstance(key, bytes):
    key = key.decode()

  key_array = key.split("_")
  if len(key_array) is not 2:
    expected_form = "{train|eval|test}_meta"
    raise ValueError("Meta row keys should have form {}, saw {}".format(
      expected_form, key
    ))
  prefix, meta = key_array
  expected_prefixes = ["train", "eval", "test"]
  if prefix not in expected_prefixes:
    raise ValueError("Unexpected prefix {}, not in {}".format(
      prefix, expected_prefixes
    ))

  assert meta == "meta"


def _compose_av_write(table, key, value, column_family, key_tag=None):
  """Write composition helper.

  Note: Casts numpy array values to uint8.

  """

  # If it's a dictionary, serialize it to a string
  if isinstance(value, dict):
    value = json.dumps(value).encode()
  elif isinstance(value, np.ndarray):
    value = bytes(value.astype(np.uint8).flatten().tolist())
  elif isinstance(value, list):
    value = bytes(value)
  elif not isinstance(value, bytes):
    msg = "Tried to write unrecognized type: {}".format(
      type(value)
    )
    raise ValueError(msg)

  # Compose key and obtain row
  if key_tag is not None:
    key = "{}_{}".format(key, key_tag)
  key = key.encode()

  row = table.row(key)

  # Compose write
  row.set_cell(column_family_id=column_family,
               column=column_family,
               value=value,
               timestamp=datetime.datetime.utcnow())

  return row


class RawVideoSelection(BigTableSelection):

  def __init__(self, *args, **kwargs):
    super(RawVideoSelection, self).__init__(
        # Defining these here instead of each time
        # the object is created makes this less
        # fragile.
        column_families=["audio",
                         "meta",
                         "video_frames"],
        *args, **kwargs)

  def set_shard_meta(self, shard_meta):
    if not isinstance(shard_meta, VideoShardMeta):
      msg = "Expected type VideoShardMeta, saw {}".format(type(shard_meta))
      raise ValueError(msg)

    #key = "{}_meta_{}".format(self.prefix,
    #                          shard_meta.shard_id).encode()
    key = "{}_meta".format(self.prefix).encode()
    _validate_shard_meta_key(key)

    row = self.table.row(key)
    row.set_cell(column_family_id="meta",
                 column="meta",
                 value=json.dumps(shard_meta.as_dict()),
                 timestamp=datetime.datetime.utcnow())
    self.table.mutate_rows([row])

  def lookup_shard_metadata(self, ignore_unfinished=False):
    if not isinstance(self.prefix, str):
      msg = ("Metadata lookup is expected after writing "
             "a shard of type train, eval, or test so "
             "once lookup_shard_metadata is called "
             "selection.prefix should be a non-trivial "
             "string.")
      raise ValueError(msg)

    metadata = {}

    key = "{}_meta".format(self.prefix).encode()

    row = self.table.read_row(key, None)

    if row is None:
      return metadata

    row_cells = row.cells["meta"]["meta".encode()]

    all_meta = [json.loads(thing.value.decode()) for thing in row_cells]

    for shard_meta in all_meta:

      if ignore_unfinished and shard_meta["status"] == "started":
        continue

      key = "{}_meta_{}".format(self.prefix, shard_meta["shard_id"])
      metadata[key] = VideoShardMeta(num_videos=shard_meta["num_videos"],
                                     status=shard_meta["status"],
                                     shard_id=shard_meta["shard_id"],
                                     num_shards=shard_meta["num_shards"])

    return metadata

  def _lookup_video_metadata(self, prefix, shard_id, video_id):
    key = "{}_{}_{}_{}".format(prefix, shard_id, video_id,
                               "meta").encode()
    msg = "looking up metadata for video with key {}, shard {}, video {}".format(
      key, shard_id, video_id
    )
    #tf.logging.info(msg)
    row = self.table.read_row(key)
    value = row.cells["meta"]["meta".encode()][0].value.decode()
    video_meta = json.loads(value)

    return VideoMeta(video_length=video_meta["video_length"],
                     audio_length=video_meta["audio_length"],
                     video_id=video_meta["video_id"],
                     shard_id=video_meta["shard_id"])

  def _get_random_video_meta(self, shard_meta):

    if not isinstance(shard_meta, dict):
      msg = "Expected meta dictionary, saw type {}.".format(
        type(shard_meta)
      )
      raise ValueError(msg)

    num_keys = len(list(shard_meta.keys()))
    sampled_shard_index = np.random.randint(0, num_keys)

    shard_meta_key = list(shard_meta.keys())[sampled_shard_index]
    sampled_shard_meta = shard_meta[shard_meta_key]

    videos_per_sampled_shard = sampled_shard_meta.num_videos
    sampled_video_index = np.random.randint(0, videos_per_sampled_shard)

    # Look up the length of the video
    return self._lookup_video_metadata(
        prefix=self.prefix,
        shard_id=sampled_shard_index,
        video_id=sampled_video_index)

  def write_av(self, frames, audio, shard_id, video_id):

    if not isinstance(frames, video_utils.Video):
      msg = "expected frames of type {}, saw {}.".format(
        video_utils.Video, type(frames)
      )
      raise ValueError(msg)

    meta = VideoMeta(video_length=frames.length,
                     audio_length=len(audio),
                     shard_id=shard_id,
                     video_id=video_id)

    key = "{}_{}_{}".format(self.prefix, shard_id, video_id)

    rows = []

    rows.append(_compose_av_write(table=self.table,
                                  key=key,
                                  value=meta.as_dict(),
                                  column_family="meta",
                                  key_tag="meta"))

    rows.append(_compose_av_write(table=self.table,
                                  key=key,
                                  value=audio,
                                  column_family="audio",
                                  key_tag="audio"))

    _ = self.table.mutate_rows(rows)
    rows = []

    frame_write_buffer_size = 32
    buffer_counter = 0

    frame_iterator = frames.get_iterator()
    for i, video_frame in enumerate(frame_iterator):

      video_frame = np.asarray(video_frame)

      tag = "frame_{}".format(i)
      rows.append(_compose_av_write(table=self.table,
                                    key=key,
                                    value=video_frame,
                                    column_family="video_frames",
                                    key_tag=tag))
      buffer_counter += 1

      if buffer_counter >= frame_write_buffer_size:
        _ = self.table.mutate_rows(rows)
        rows = []
        buffer_counter = 0

    if buffer_counter > 0:
      _ = self.table.mutate_rows(rows)

  def _frame_data_for_indices(self, indices, meta):

    assert isinstance(indices, np.ndarray)
    assert isinstance(meta, VideoMeta)

    frames = np.asarray([None for _ in indices])

    key_prefix = "{}_{}_{}".format(self.prefix, meta.shard_id, meta.video_id)

    for i in range(len(indices)):

      index = indices[i]

      frame_key = "{}_frame_{}".format(key_prefix, index)

      row = self.table.read_row(frame_key)

      if row is None:
        msg = "Frame data query with indices {} and meta {} got None.".format(
          indices, meta.as_dict()
        )
        raise ValueError(msg)

      frame_data = row.cells["video_frames"]["video_frames".encode()][0].value      
      frames[i] = np.asarray(list(frame_data), dtype=np.uint8)

    return np.asarray([np.asarray(thing) for thing in frames])

  def _audio_data_for_indices(self, indices, meta):

    assert isinstance(indices, np.ndarray)

    audio_key = "{}_{}_{}_audio".format(self.prefix,
                                        meta.shard_id,
                                        meta.video_id)

    def _logging_data():
      return {"key": audio_key,
              "meta": meta.as_dict(),
              "indices": indices}
    
    row = self.table.read_row(audio_key)
    if row is None:
      msg = "Audio data query got None, {}".format(_logging_data())
      raise ValueError(msg)

    value = row.cells["audio"]["audio".encode()][0].value
    all_audio_data = np.asarray(list(value), dtype=np.uint8)

    # For now we don't sample the audio with skips so this is sufficient
    ret = all_audio_data[indices[0]:indices[-1]]

    length = len(ret)
    if not isinstance(ret, np.ndarray) or length == 0:
      
      msg = "Wrong type or length: {}, {}; response len: {}; other: {}".format(
        type(ret), length, len(all_audio_data), _logging_data()
      )
      raise ValueError(msg)
      
    return ret

  def sample_av_correspondence_examples(self,
                                        frames_per_video,
                                        max_num_samples=None,
                                        max_frame_shift=0,
                                        max_frame_skip=0):

    all_shard_meta = self.lookup_shard_metadata(ignore_unfinished=True)
    
    # TODO: Provide more clear logging in the event there aren't any completed
    # shards.

    i = 0
    while True:

      v0 = self._get_random_video_meta(all_shard_meta)
      v1 = self._get_random_video_meta(all_shard_meta)

      def _sample(vlen, alen):
        avs = video_utils.AVSamplable(video_length=vlen,
                                      audio_length=alen)
        return avs.sample_av_pair(
          num_frames=frames_per_video,
          max_frame_shift=max_frame_shift,
          max_frame_skip=max_frame_skip)

      # Get indices for two samples from the first video
      # The first one frames and audio
      f00_, a00_, sampling_meta00 = _sample(v0.video_length, v0.audio_length)
      # and the second one only audio
      _, a01_, sampling_meta01 = _sample(v0.video_length, v0.audio_length)

      # Then look up the actual frame and audio data for those sampled indices
      # (from bigtable).
      f00 = self._frame_data_for_indices(f00_, meta=v0)
      a00 = self._audio_data_for_indices(a00_, meta=v0)
      a01 = self._audio_data_for_indices(a01_, meta=v0)

      # Then sample audio indices from the second video
      _, a10_, sampling_meta10 = _sample(v1.video_length, v1.audio_length)

      # And again look up the actual audio data from bigtable
      a10 = self._audio_data_for_indices(a10_, meta=v1)

      # Lastly store the sampled data in nice organized AVCorrespondenceSample
      # objects.
      positive_same = AVCorrespondenceSample(
        video=f00, audio=a00,
        labels={"same_video": 1, "overlap": 1},
        meta={"video_source": v0, "audio_source": v0,
              "video_sample_meta": sampling_meta00,
              "audio_sample_meta": sampling_meta00})

      negative_same = AVCorrespondenceSample(
        video=f00, audio=a01,
        labels={"same_video": 1, "overlap": 0},
        meta={"video_source": v0, "audio_source": v0,
              "video_sample_meta": sampling_meta00,
              "audio_sample_meta": sampling_meta01})

      negative_different = AVCorrespondenceSample(
        video=f00, audio=a10,
        labels={"same_video": 0, "overlap": 0},
        meta={"video_source": v0, "audio_source": v1,
              "video_sample_meta": sampling_meta00,
              "audio_sample_meta": sampling_meta10})

      yield {
        "positive_same": positive_same,
        "negative_same": negative_same,
        "negative_different": negative_different
      }

      i += 1

      if max_num_samples and max_num_samples <= i:
        break


class TFExampleSelection(BigTableSelection):
  def __init__(self, *args, **kwargs):  
    super(TFExampleSelection, self).__init__(
        # Defining these here instead of each time
        # the object is created makes this less
        # fragile.
        column_families=["tfexample"],
        column_qualifier="example",
        column_family="tfexample",
        *args, **kwargs)

  def random_load_from_generator(self,
                                 generator,
                                 prefix_tag_length=4,
                                 max_num_examples=-1,
                                 log_every=100):
    """Builds TFExample from dict, serializes, and writes to CBT."""

    prefix = self.prefix
    table = self.table

    for i, example_dict in enumerate(generator):

      if not isinstance(example_dict, dict):
        msg = "Expected generator to yield dict's, saw {}.".format(
          type(example_dict)
        )
        raise ValueError(msg)

      example = to_example(example_dict)
      example = example.SerializeToString()

      # Random target key
      target_key = random_key(prefix=prefix,
                              length=prefix_tag_length).encode()

      row = table.row(target_key)
      row.set_cell(column_family_id="tfexample",
                   column="example",
                   value=example)
                   # Don't set a timestamp so we set instead of
                   # append cell values.
                   #timestamp=datetime.datetime.utcnow())

      table.mutate_rows([row])

      if log_every > 0 and i % log_every == 0:
        tf.logging.info("Generated {} examples...".format(i))

      if max_num_examples > 0 and i >= max_num_examples:
        break

    tf.logging.info("Generated {} examples.".format(i))

    return i

  def iterate_tfexamples(self):

    row_iterator = self.get_basic_row_iterator()

    for row in row_iterator:

      ex = row.cells["tfexample"]["example".encode()][0].value

      parsed_example = tf.train.Example.FromString(ex)

      yield parsed_example


def random_key(prefix="raw_", length=4):

  a = "abcdefghijklmnopqrstuvwxyz"
  ind = list(np.random.randint(0, 26, length))
  key = "".join([a[i] for i in ind])
  key = (prefix + key)
  return key

