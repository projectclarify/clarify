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
import numpy as np
import datetime
import json
import math

from clarify.utils import video_utils

from google.cloud.bigtable import column_family as cbt_lib_column_family
from google.cloud import bigtable
from google.cloud.bigtable import row_filters

from tensor2tensor.data_generators.generator_utils import to_example

from collections import namedtuple

MAX_ALLOWABLE_FRAME_AUDIO_KEY_SUFFIX = 9999


class BigTableSelection(object):

  def __init__(self,
               project,
               instance,
               table,
               column_families,
               prefix=None,
               sa_key_path=None,
               column_qualifier=None,
               column_family=None,
               *args,
               **kwargs):

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
      self.client = bigtable.Client.from_service_account_json(sa_key_path,
                                                              admin=True)
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

      row_filter = row_filters.RowKeyRegexFilter(regex=prefix)

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
    return {
        "table_name": self.table_name,
        "instance_name": self.instance_name,
        "row_key_prefix": self.prefix,
        "column_families": self.column_families,
        "column_family": self.column_family,
        "column_qualifier": self.column_qualifier
    }


def _expect_type(obj, t):
  if not isinstance(obj, t):
    msg = "Expected numpy.ndarray, saw {}".format(type(obj))
    raise ValueError(msg)


def _maybe_decode_bytes(obj):
  if isinstance(obj, bytes):
    return obj.decode()
  return obj


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
    for key in [
        "video_source", "audio_source", "video_sample_meta",
        "audio_sample_meta", "audio_keys", "frame_keys", "audio_block_meta"
    ]:
      if not key in x:
        raise ValueError("Meta should contain key {}, saw {}".format(
            key, x.keys()))
    assert len(x.keys()) == 7
    assert isinstance(x["video_source"], VideoMeta)
    assert isinstance(x["audio_source"], VideoMeta)
    self._meta = x

  def as_dict(self):
    return {
        "video": self.video.tolist(),
        "audio": self.audio.tolist(),
        "labels": self.labels,
        "meta": {
            "audio_source": self.meta["audio_source"].as_dict(),
            "video_source": self.meta["video_source"].as_dict(),
            "video_sample_meta": self.meta["video_sample_meta"],
            "audio_sample_meta": self.meta["audio_sample_meta"],
            "audio_keys": self.meta["audio_keys"],
            "frame_keys": self.meta["frame_keys"],
            "audio_block_meta": self.meta["audio_block_meta"]
        }
    }

  def dump_keys(self):
    # For convenience of parsing in golang
    abm = self.meta["audio_block_meta"]
    return {
        "audioKeys": [
            _maybe_decode_bytes(key) for key in self.meta["audio_keys"]
        ],
        "frameKeys": [
            _maybe_decode_bytes(key) for key in self.meta["frame_keys"]
        ],
        "audioSampleBounds": [int(abm["query_start"]),
                              int(abm["query_end"])],
        "labels": self.labels
    }

  def serialize(self):
    d = self.dump_keys()
    return json.dumps(d)


class VideoMeta(object):

  def __init__(self,
               video_length,
               audio_length,
               video_id,
               shard_id,
               audio_block_size=256):
    self.video_length = video_length
    self.audio_length = audio_length
    self.video_id = video_id
    self.shard_id = shard_id
    self.audio_block_size = audio_block_size

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

  @property
  def audio_block_size(self):
    return self._audio_block_size

  @audio_block_size.setter
  def audio_block_size(self, x):
    assert isinstance(x, int)
    assert x > 0
    self._audio_block_size = x

  def as_dict(self):
    return {
        "video_length": self.video_length,
        "audio_length": self.audio_length,
        "video_id": self.video_id,
        "shard_id": self.shard_id,
        "audio_block_size": self.audio_block_size
    }


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
    return {
        "num_videos": self.num_videos,
        "shard_id": self.shard_id,
        "status": self.status,
        "num_shards": self.num_shards
    }


def _validate_shard_meta_key(key):

  # Validating its basic structure not whether its encoded or not
  if isinstance(key, bytes):
    key = key.decode()

  key_array = key.split("_")
  if len(key_array) is not 2:
    expected_form = "{train|eval|test}_meta"
    raise ValueError("Meta row keys should have form {}, saw {}".format(
        expected_form, key))
  prefix, meta = key_array
  expected_prefixes = ["train", "eval", "test"]
  if prefix not in expected_prefixes:
    raise ValueError("Unexpected prefix {}, not in {}".format(
        prefix, expected_prefixes))

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
    msg = "Tried to write unrecognized type: {}".format(type(value))
    raise ValueError(msg)

  # Compose key and obtain row
  #if key_tag is not None:
  #  key = "{}_{}".format(key, key_tag)
  #key = key.encode()

  row = table.row(key)

  # Compose write
  row.set_cell(column_family_id=column_family,
               column=column_family,
               value=value,
               timestamp=datetime.datetime(1970, 1, 1))
  #timestamp=datetime.datetime.utcnow())

  return row


def audio_blocks_for_indices(start, end, block_size):

  query_length = end - start

  min_query_block = int(math.floor(start / block_size))
  max_query_block = int(math.floor(end / block_size))
  num_query_blocks = max_query_block - min_query_block + 1

  query_start = start - min_query_block * block_size
  query_end = query_start + query_length

  return {
      "min_query_block": min_query_block,
      "max_query_block": max_query_block,
      "num_query_blocks": num_query_blocks,
      "query_start": query_start,
      "query_end": query_end
  }


def _lex_index(idx):
  max_allowable_suffix = MAX_ALLOWABLE_FRAME_AUDIO_KEY_SUFFIX
  tag_length = len(str(max_allowable_suffix))
  zero_str = "".join([str(0) for _ in range(tag_length)])
  str_idx = str(idx)
  diff_len = tag_length - len(str_idx)
  if diff_len > 0:
    str_idx = zero_str[0:diff_len] + str_idx
  letters = "abcdefghij"
  lexxed = ""
  for c in str_idx:
    l = letters[int(c)]
    lexxed += l
  return lexxed


def make_audio_key(table_prefix, shard_id, video_id, audio_block_id):

  max_allowable_suffix = MAX_ALLOWABLE_FRAME_AUDIO_KEY_SUFFIX

  key = "{}_{}_{}".format(table_prefix, shard_id, video_id)

  tag = "audio_{}".format(_lex_index(audio_block_id))

  return "{}_{}".format(key, tag).encode()


def make_frame_key(table_prefix, shard_id, video_id, frame_id):

  max_allowable_suffix = MAX_ALLOWABLE_FRAME_AUDIO_KEY_SUFFIX

  key = "{}_{}_{}".format(table_prefix, shard_id, video_id)

  tag = "frame_{}".format(_lex_index(frame_id))

  return "{}_{}".format(key, tag).encode()


def make_shard_meta_key(table_prefix, shard_id):
  """Construct the key for an individual shard's meta."""
  key = "{}_meta_{}".format(table_prefix, _lex_index(shard_id))
  return key.encode()


def make_shard_meta_common_prefix(table_prefix):
  """Construct the key for an individual shard's meta."""
  key = "{}_meta_.".format(table_prefix)
  return key


def make_shard_meta_first_key(table_prefix):
  """Construct the key for an individual shard's meta."""
  key = "{}_meta_{}".format(table_prefix, _lex_index(0))
  return key


def make_shard_meta_last_key(table_prefix, num_shards):
  key = "{}_meta_{}".format(table_prefix, _lex_index(num_shards))
  return key


def make_video_meta_key(table_prefix, shard_id, video_id):
  """Construct a key for an individual video's metadata."""
  key = "{}_{}_meta_{}".format(table_prefix, shard_id, _lex_index(video_id))
  return key.encode()


def make_video_meta_common_prefix(table_prefix, shard_id):
  """Construct a key for an individual video's metadata."""
  key = "{}_{}_meta_.".format(table_prefix, _lex_index(shard_id))
  return key


def make_video_meta_first_key(table_prefix, shard_id):
  """Construct a key for an individual video's metadata."""
  key = "{}_{}_meta_{}".format(table_prefix, shard_id, _lex_index(0))
  return key


def make_video_meta_last_key(table_prefix, shard_id, num_videos):
  """Construct a key for an individual video's metadata."""
  key = "{}_{}_meta_{}".format(table_prefix, shard_id, _lex_index(num_videos))
  return key


class RawVideoSelection(BigTableSelection):

  def __init__(self, *args, **kwargs):
    super(RawVideoSelection, self).__init__(
        # Defining these here instead of each time
        # the object is created makes this less
        # fragile.
        column_families=["audio", "meta", "video_frames"],
        *args,
        **kwargs)

  def set_shard_meta(self, shard_meta):
    if not isinstance(shard_meta, VideoShardMeta):
      msg = "Expected type VideoShardMeta, saw {}".format(type(shard_meta))
      raise ValueError(msg)

    #key = "{}_meta_{}".format(self.prefix,
    #                          shard_meta.shard_id).encode()
    #key = "{}_meta".format(self.prefix).encode()
    #_validate_shard_meta_key(key)

    key = make_shard_meta_key(table_prefix=self.prefix,
                              shard_id=shard_meta.shard_id)

    row = self.table.row(key)
    row.set_cell(column_family_id="meta",
                 column="meta",
                 value=json.dumps(shard_meta.as_dict()),
                 timestamp=datetime.datetime(1970, 1, 1))
    self.table.mutate_rows([row])

  def lookup_shard_metadata(self, num_shards=99999, ignore_unfinished=False):
    if not isinstance(self.prefix, str):
      msg = ("Metadata lookup is expected after writing "
             "a shard of type train, eval, or test so "
             "once lookup_shard_metadata is called "
             "selection.prefix should be a non-trivial "
             "string.")
      raise ValueError(msg)

    metadata = {}

    prefix = make_shard_meta_common_prefix(table_prefix=self.prefix)

    partial_rows = self.table.read_rows(
        start_key=make_shard_meta_first_key(self.prefix),
        end_key=make_shard_meta_last_key(self.prefix, num_shards))

    if partial_rows is None:
      return metadata

    for i, row in enumerate(partial_rows):

      cell = row.cells["meta"]["meta".encode()][0]
      shard_meta = json.loads(cell.value.decode())

      key = make_shard_meta_key(table_prefix=self.prefix,
                                shard_id=shard_meta["shard_id"])

      if ignore_unfinished and shard_meta["status"] == "started":
        continue

      metadata[key] = VideoShardMeta(num_videos=shard_meta["num_videos"],
                                     status=shard_meta["status"],
                                     shard_id=shard_meta["shard_id"],
                                     num_shards=shard_meta["num_shards"])

      # Hack: This shouldn't be necessary but doesn't cause any harm. There's
      # currently a bug where this loop iterates over the eval meta partial_rows
      # forever even after finding the one meta entry that is relevant. For
      # unknown reason.
      if i + 1 >= shard_meta["num_shards"]:
        break

    return metadata

  def _lookup_all_video_metadata(self, num_shards=1):

    all_shard_meta = self.lookup_shard_metadata(num_shards=num_shards,
                                                ignore_unfinished=True)

    all_video_meta = []

    # HACK: Currently the regex filter approach isn't yielding the correct list of videos.
    # But specifying the start and end key does. The only problem is that row keys are
    # sorted lexicographically. So for now a num_videos beyond what would be present is
    # used. And in the future we can move from using integers to some perhaps alphabet
    # letter alternative.
    num_videos = 99999999

    for shard_meta_key, shard_meta in all_shard_meta.items():

      shard_id = shard_meta._shard_id
      #num_videos = shard_meta._num_videos
      meta_prefix = make_video_meta_common_prefix(self.prefix, shard_id)
      start_key = make_video_meta_first_key(self.prefix, shard_id)
      end_key = make_video_meta_last_key(self.prefix, shard_id, num_videos)

      partial_rows = self.table.read_rows(start_key=start_key, end_key=end_key)

      for row in partial_rows:

        value = row.cells["meta"]["meta".encode()][0].value.decode()
        video_meta = json.loads(value)

        vm = VideoMeta(video_length=video_meta["video_length"],
                       audio_length=video_meta["audio_length"],
                       video_id=video_meta["video_id"],
                       shard_id=video_meta["shard_id"],
                       audio_block_size=video_meta["audio_block_size"])

        all_video_meta.append(vm)

    return all_video_meta

  def _lookup_video_metadata(self, prefix, shard_id, video_id):

    key = make_video_meta_key(table_prefix=prefix,
                              shard_id=shard_id,
                              video_id=video_id)

    msg = "looking up metadata for video with key {}, shard {}, video {}".format(
        key, shard_id, video_id)

    row = self.table.read_row(key)
    value = row.cells["meta"]["meta".encode()][0].value.decode()
    video_meta = json.loads(value)

    return VideoMeta(video_length=video_meta["video_length"],
                     audio_length=video_meta["audio_length"],
                     video_id=video_meta["video_id"],
                     shard_id=video_meta["shard_id"],
                     audio_block_size=video_meta["audio_block_size"])

  def _get_random_video_meta(self, shard_meta):

    if not isinstance(shard_meta, dict):
      msg = "Expected meta dictionary, saw type {}.".format(type(shard_meta))
      raise ValueError(msg)
    """
    
    Hack: This is to deal with the fact that cloud function extraction does not
    identify and re-try failures leaving gaps in the raw table and thus creating
    the opportunity for failures at least at this level (looking up meta at video
    and shard indices that have not been written).
    
    """
    while True:

      try:

        num_keys = len(list(shard_meta.keys()))
        sampled_shard_index = np.random.randint(0, num_keys)

        shard_meta_key = list(shard_meta.keys())[sampled_shard_index]
        sampled_shard_meta = shard_meta[shard_meta_key]

        videos_per_sampled_shard = sampled_shard_meta.num_videos
        sampled_video_index = np.random.randint(0, videos_per_sampled_shard)

        tf.logging.debug("sampled shard index: {}".format(sampled_shard_index))
        tf.logging.debug("sampled video index: {}".format(sampled_video_index))

        # Look up the length of the video
        meta = self._lookup_video_metadata(prefix=self.prefix,
                                           shard_id=sampled_shard_index,
                                           video_id=sampled_video_index)

        return meta

      except:
        tf.logging.info(
            "Failed fetching meta for video and shard, will retry: {},  {}".
            format(sampled_video_index, sampled_shard_index))

  def write_av(self, frames, audio, shard_id, video_id, audio_block_size=1000):

    if not isinstance(frames, video_utils.Video):
      msg = "expected frames of type {}, saw {}.".format(
          video_utils.Video, type(frames))
      raise ValueError(msg)

    meta = VideoMeta(video_length=frames.length,
                     audio_length=len(audio),
                     shard_id=shard_id,
                     video_id=video_id,
                     audio_block_size=audio_block_size)

    video_meta_key = make_video_meta_key(table_prefix=self.prefix,
                                         shard_id=shard_id,
                                         video_id=video_id)

    rows = []

    rows.append(
        _compose_av_write(table=self.table,
                          key=video_meta_key,
                          value=meta.as_dict(),
                          column_family="meta"))

    num_audio_blocks = math.ceil(meta.audio_length / audio_block_size)

    for i in range(num_audio_blocks):

      subset_start = i * audio_block_size
      subset_end = subset_start + audio_block_size

      audio_subset = audio[subset_start:subset_end]

      key = make_audio_key(table_prefix=self.prefix,
                           shard_id=shard_id,
                           video_id=video_id,
                           audio_block_id=i)

      rows.append(
          _compose_av_write(table=self.table,
                            key=key,
                            value=audio_subset,
                            column_family="audio"))

    _ = self.table.mutate_rows(rows)
    rows = []

    frame_write_buffer_size = 32
    buffer_counter = 0

    frame_iterator = frames.get_iterator()
    for i, video_frame in enumerate(frame_iterator):

      video_frame = np.asarray(video_frame)

      frame_key = make_frame_key(table_prefix=self.prefix,
                                 shard_id=shard_id,
                                 video_id=video_id,
                                 frame_id=i)

      rows.append(
          _compose_av_write(table=self.table,
                            key=frame_key,
                            value=video_frame,
                            column_family="video_frames"))
      buffer_counter += 1

      if buffer_counter >= frame_write_buffer_size:
        _ = self.table.mutate_rows(rows)
        rows = []
        buffer_counter = 0

    if buffer_counter > 0:
      _ = self.table.mutate_rows(rows)

  def _frame_keys_for_indices(self, indices, meta):

    assert isinstance(indices, np.ndarray)
    assert isinstance(meta, VideoMeta)

    frame_keys = []

    for i in range(len(indices)):

      index = indices[i]

      frame_key = make_frame_key(table_prefix=self.prefix,
                                 shard_id=meta.shard_id,
                                 video_id=meta.video_id,
                                 frame_id=index)

      frame_keys.append(frame_key)

    return frame_keys

  def _audio_keys(self, meta, indices):

    audio_block_meta = audio_blocks_for_indices(
        start=indices[0], end=indices[-1], block_size=meta.audio_block_size)

    keys = []

    for i in range(audio_block_meta["num_query_blocks"]):

      audio_key = make_audio_key(
          table_prefix=self.prefix,
          shard_id=meta.shard_id,
          video_id=meta.video_id,
          audio_block_id=audio_block_meta["min_query_block"] + i)

      keys.append(audio_key)

    return keys, audio_block_meta

  def _lookup_frame_data(self, frame_keys):

    frames = np.asarray([None for _ in frame_keys])

    for i, frame_key in enumerate(frame_keys):

      row = self.table.read_row(frame_key)

      if row is None:
        msg = "Frame data query for key {} got None.".format(frame_key)
        raise ValueError(msg)

      frame_data = row.cells["video_frames"]["video_frames".encode()][0].value
      frames[i] = np.asarray(list(frame_data), dtype=np.uint8)

    return np.asarray([np.asarray(thing) for thing in frames])

  def _lookup_audio_data(self, audio_keys, audio_block_meta):

    query_start = audio_block_meta["query_start"]
    query_end = audio_block_meta["query_end"]

    all_audio_data = np.array([])

    for audio_key in audio_keys:

      row = self.table.read_row(audio_key)

      if row is None:
        msg = "Audio data query got None, {}".format(audio_key)
        raise ValueError(msg)

      value = row.cells["audio"]["audio".encode()][0].value
      data = np.asarray(list(value), dtype=np.uint8)

      all_audio_data = np.concatenate([all_audio_data, data])

    ret = all_audio_data[query_start:query_end]

    length = len(ret)

    if not isinstance(ret, np.ndarray) or length == 0:

      msg = "Wrong type or length: {}, {}; response len: {}; other: {}".format(
          type(ret), length, len(all_audio_data), _logging_data())
      raise ValueError(msg)

    return ret

  def sample_av_correspondence_examples(self,
                                        frames_per_video,
                                        max_num_samples=None,
                                        max_frame_shift=0,
                                        max_frame_skip=0,
                                        keys_only=False):

    #make_video_meta_common_prefix(table_prefix, shard_id)

    #all_shard_meta = self.lookup_shard_metadata(ignore_unfinished=True)

    # TODO: Provide more clear logging in the event there aren't any completed
    # shards.

    all_video_meta = self._lookup_all_video_metadata()
    num_videos = len(all_video_meta)

    i = 0
    while True:

      #v0 = self._get_random_video_meta(all_shard_meta)
      #v1 = self._get_random_video_meta(all_shard_meta)
      v0i, v1i = np.random.randint(0, num_videos, 2)
      v0 = all_video_meta[v0i]
      v1 = all_video_meta[v1i]

      def _sample(vlen, alen):
        avs = video_utils.AVSamplable(video_length=vlen, audio_length=alen)
        return avs.sample_av_pair(num_frames=frames_per_video,
                                  max_frame_shift=max_frame_shift,
                                  max_frame_skip=max_frame_skip)

      # Get indices for two samples from the first video
      # The first one frames and audio
      f00_, a00_, sampling_meta00 = _sample(v0.video_length, v0.audio_length)
      # and the second one only audio
      _, a01_, sampling_meta01 = _sample(v0.video_length, v0.audio_length)

      # Then sample audio indices from the second video
      _, a10_, sampling_meta10 = _sample(v1.video_length, v1.audio_length)

      kf00 = self._frame_keys_for_indices(f00_, meta=v0)

      ka00, abm00 = self._audio_keys(meta=v0, indices=a00_)
      ka01, abm01 = self._audio_keys(meta=v0, indices=a01_)
      ka10, abm10 = self._audio_keys(meta=v1, indices=a10_)

      positive_same = AVCorrespondenceSample(
          video=np.array([]),
          audio=np.array([]),
          labels={
              "same_video": 1,
              "overlap": 1
          },
          meta={
              "video_source": v0,
              "audio_source": v0,
              "video_sample_meta": sampling_meta00,
              "audio_sample_meta": sampling_meta00,
              "audio_keys": ka00,
              "frame_keys": kf00,
              "audio_block_meta": abm00
          })

      negative_same = AVCorrespondenceSample(
          video=np.array([]),
          audio=np.array([]),
          labels={
              "same_video": 1,
              "overlap": 0
          },
          meta={
              "video_source": v0,
              "audio_source": v0,
              "video_sample_meta": sampling_meta00,
              "audio_sample_meta": sampling_meta01,
              "audio_keys": ka01,
              "frame_keys": kf00,
              "audio_block_meta": abm01
          })

      negative_different = AVCorrespondenceSample(
          video=np.array([]),
          audio=np.array([]),
          labels={
              "same_video": 0,
              "overlap": 0
          },
          meta={
              "video_source": v0,
              "audio_source": v1,
              "video_sample_meta": sampling_meta00,
              "audio_sample_meta": sampling_meta10,
              "audio_keys": ka10,
              "frame_keys": kf00,
              "audio_block_meta": abm10
          })

      example_set = {
          "positive_same": positive_same,
          "negative_same": negative_same,
          #"negative_different": negative_different
      }

      if keys_only:
        yield example_set
      else:

        # Then look up the actual frame and audio data for those sampled indices
        # (from bigtable).
        f00 = self._lookup_frame_data(kf00)

        a00 = self._lookup_audio_data(audio_keys=ka00, audio_block_meta=abm00)
        a01 = self._lookup_audio_data(audio_keys=ka01, audio_block_meta=abm01)
        a10 = self._lookup_audio_data(audio_keys=ka10, audio_block_meta=abm10)

        # Lastly store the sampled data in nice organized AVCorrespondenceSample
        # objects.

        example_set["positive_same"].video = f00
        example_set["positive_same"].audio = a00

        example_set["negative_same"].video = f00
        example_set["negative_same"].audio = a01

        #example_set["negative_different"].video = f00
        #example_set["negative_different"].audio = a10

        yield example_set

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
        *args,
        **kwargs)

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
            type(example_dict))
        raise ValueError(msg)

      example = to_example(example_dict)
      example = example.SerializeToString()

      # Random target key
      target_key = random_key(prefix=prefix, length=prefix_tag_length).encode()

      row = table.row(target_key)
      row.set_cell(column_family_id="tfexample",
                   column="example",
                   value=example,
                   timestamp=datetime.datetime(1970, 1, 1))
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
