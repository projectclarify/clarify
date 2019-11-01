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

"""Download FEC dataset.

The FEC dataset is at the top level a pair of CSV's that specify the URL of ~500k
images available from websites distributed around the internet. In the process of
obtaining each individually the download process needs to be tolerant to 404's as well
as to flag triplets as to whether any of the component images could not be obtained
(and if so throw out the whole triplet).

The top-level FEC CSV's also specify bounding boxes for each image where a face is
expected to be found. For now the download step is de-coupled from the downstream
step of verifying in the downloaded images that a face is present in the specified
region.

- 15% expansion beyond FEC bounding box (or to boundary of image), as a function of
  FEC bounding box dimensions (not total image dimensions)

- filtering of cropped regions for presence of face at 0.7 confidence for predictions
  produced by faced (https://github.com/iitzco/faced) which is based on YOLO and fine-
  tuned for face detection, seems to perform better than Haar cascades.

- after such filtering, a triplet only passes filter if all three cropped regions in
  the triplet have a face; only cropped images passing filter are written to disk; only
  the meta for triplets passing filter are written to nonfailed.json

- currently batch job stages out all of the original and cropped images along with
  the nonfailed meta.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import random
import numpy as np
import subprocess
import tempfile
import scipy
import requests
import json
import math
import datetime
import tempfile
import cv2
import urllib

from faced import FaceDetector

from tensor2tensor.data_generators import generator_utils

from pcml.utils.cmd_utils import run_and_output

import tensorflow as tf

from pcml.launcher.kube import Job
from pcml.launcher.kube import PCMLJob
from pcml.launcher.kube import gen_timestamped_uid
from pcml.launcher.kube import Resources


_FEC_ARCHIVE="https://storage.googleapis.com/public_release/FEC_dataset.zip"
_FEC_TEST_META_FILENAME="faceexp-comparison-data-test-public.csv"
_FEC_TRAIN_META_FILENAME="faceexp-comparison-data-train-public.csv"
_FEC_IMAGE_SIZE = 64
_SUCCESS_MESSAGE = "Successfully completed download and filtering of FEC dataset."


def _load_meta_line(line):
  def _load_meta_entry(entry_array):
    return {"url": entry_array[0][1:-1],
            "bounds": [float(entry_array[1]), float(entry_array[2]),
                       float(entry_array[3]), float(entry_array[4])]}
  a = _load_meta_entry(line[0:5])
  b = _load_meta_entry(line[5:10])
  c = _load_meta_entry(line[10:15])

  triplet_type = line[15]

  remainder = line[16:]
  ratings = {}
  num_ratings = int(len(remainder)/2)
  mean_rating = 0

  for i in range(num_ratings):
    rater_id = remainder[2*i]
    rating = remainder[2*i + 1]
    ratings[rater_id] = int(rating)
    mean_rating += int(rating)
 
  mean_rating = mean_rating / num_ratings

  mode = scipy.stats.mode(list(ratings.values()))[0][0]

  return {
    "a": a, "b": b, "c": c,
    "triplet_type": triplet_type,
    "mean_rating": mean_rating,
    "mode_rating": int(mode),
    "ratings": ratings
  }


def _load_meta(path):
  dat = []
  with open(path, "r") as f:
    for line in f:
      raw_line = line.strip().split(",")
      yield _load_meta_line(raw_line)


def _get_fec_meta(is_training=True, tmp_dir=None):
  """Download FEC meta files to directory."""

  if not tmp_dir:
    tmp_dir = tempfile.mktemp()

  test = os.path.join(tmp_dir, "FEC_dataset", _FEC_TEST_META_FILENAME)
  train = os.path.join(tmp_dir, "FEC_dataset", _FEC_TRAIN_META_FILENAME)

  if not tf.gfile.Exists(test) or not tf.gfile.Exists(train):

    tf.logging.info("Couldn't find FEC meta locally, obtaining...")
    err = "Problem in obtaining FEC dataset"
    filename = "FEC_dataset.zip"
    generator_utils.maybe_download(tmp_dir, filename, _FEC_ARCHIVE)
    os.chdir(tmp_dir)
    run_and_output(["unzip", filename])
    fec_dir = "FEC_dataset"

  for path in [test, train]:
    if not os.path.exists(path):
      raise Exception(err)

  meta_path = test
  if is_training:
    meta_path = train

  tf.logging.info("Finished maybe obtaining FEC meta.")

  with open(meta_path, "r") as f:
    for line in f:
      raw_line = line.strip().split(",")
      yield _load_meta_line(raw_line)


def _crop_to_fractional_xy_bbox(image, bbox, expand_rate=0.15):
  """Crop given relative position xy bounding box (x0,y0,x1,yl)."""

  assert len(bbox) == 4

  w, h, c = np.shape(image)

  x_start = int(bbox[2]*w)
  x_end = int(bbox[3]*w)

  y_start = int(bbox[0]*h)
  y_end = int(bbox[1]*h)

  # Expand as a function of cropped area not total image area
  x_expand = (x_end - x_start) * expand_rate / 2
  y_expand = (y_end - y_start) * expand_rate / 2

  x_start = max(int(x_start - x_expand), 0)
  x_end = min(int(x_end + x_expand), w)

  y_start = max(int(y_start - y_expand), 0)
  y_end = min(int(y_end + y_expand), h)

  # Make the result square if possible
  y_size = (y_end - y_start)
  x_size = (x_end - x_start)
  x_extra = 0
  y_extra = 0
  if y_size < x_size:
    y_extra = x_size - y_size
  elif x_size < y_size:
    x_extra = y_size - x_size
    
  x_extra_before = int(x_extra/2.0)
  x_extra_after = x_extra - x_extra_before
  y_extra_before = int(y_extra/2.0)
  y_extra_after = y_extra - y_extra_before

  x_start = max(int(x_start - x_extra_before), 0)
  y_start = max(int(y_start - y_extra_before), 0)

  x_end = min(int(x_end + x_extra_after), w)
  y_end = min(int(y_end + y_extra_after), h)

  return image[x_start:x_end, y_start:y_end]


def _read_image(image_path):
  img = cv2.imread(image_path)
  rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  
  return rgb_img


def _download_fec_data(tmp_dir, meta, target_shape):

  nonfailed = [None for _ in range(len(meta))]
  last_idx = 0

  detector = FaceDetector()
  detect_threshold = 0.7

  # For now just download all images and consider filtering for presence
  # of faces a secondary step
  for i, item in enumerate(meta):

    failed = False
    face_data = {}
    for case in ["a", "b", "c"]:
      url = item[case]["url"]
      # Use the URL as a filename to avoid collisions for
      # different images with the same filename
      filename = url.replace("/", "-")

      try:
        generator_utils.maybe_download(tmp_dir, filename, url)

        image_path = os.path.join(tmp_dir, filename)
        img = _read_image(image_path)

        bounds = item[case]["bounds"]
        cropped = _crop_to_fractional_xy_bbox(img, bounds)
        if cropped.shape[0] != cropped.shape[1]:
          failed = True

        #cropped = _normalize_dimensions(cropped, target_shape)

        face_data[case] = cropped

        # For now this should be fine. But alternatively could
        # hash the image content.
        # This being to give unique filename to which to write the
        # cropped image, given primary images may have multiple faces
        # within them thus we will be over-writing and mixing up faces
        # if we write different cropped regions of a primary image to the
        # same file.
        string_bounds = "-".join([str(thing) for thing in bounds])
        cropped_filename = "cropped@" + string_bounds + "#" + filename
        item[case]["cropped_filename"] = cropped_filename

        predictions = detector.predict(cropped, detect_threshold)
        has_face = len(predictions) > 0

        if not has_face:
          failed = True

      except:
        tf.logging.info("Exception case.")
        failed = True

    # End of for case in ["a", "b", "c"]
    if not failed:
      # If we have detected faces in all three cases let's build and write an
      # example.
      for case in ["a", "b", "c"]:
        out_path = os.path.join(tmp_dir, item[case]["cropped_filename"])
        cv2.imwrite(out_path, face_data[case])

    if not failed:
      nonfailed[last_idx] = item
      last_idx += 1

  nonfailed = nonfailed[:last_idx]

  nonfailed_file = os.path.join(tmp_dir, "nonfailed.json")
  with tf.gfile.Open(nonfailed_file, "w") as f:
    f.write(json.dumps(nonfailed))

  return nonfailed


def sharded_download_fec_data(tmp_dir, is_training, shard_id, num_shards):

  target_shape = (128, 128, 3)

  meta = [thing for thing in _get_fec_meta(is_training=is_training, tmp_dir=tmp_dir)]

  meta_len = len(meta)

  shard_step = math.floor(meta_len / num_shards)
  shard_start = shard_id * shard_step
  shard_end = shard_start + shard_step

  meta = meta[shard_start:min(shard_end, meta_len)]

  tf.logging.info("Processing shard with {} records of {} total".format(
    len(meta), meta_len
  ))

  return _download_fec_data(tmp_dir, meta, target_shape=target_shape)


class DownloadFec(PCMLJob):

  def __init__(self, output_bucket, is_training, *args, **kwargs):

    cmd = []

    cmd.append("python -m pcml.operations.download_fec")
    cmd.append("--output_bucket=%s " % output_bucket)
    cmd.append("--is_training=%s " % is_training)

    cmd = " ".join(cmd)

    command = ["/bin/sh", "-c"]
    command_args = [cmd]

    self.job_name_prefix = "download-fec"
    job_name = "%s-%s" % (self.job_name_prefix, gen_timestamped_uid())

    super(DownloadFec, self).__init__(
      job_name=job_name,
      command=command,
      command_args=command_args,
      namespace="kubeflow",
      num_local_ssd=1,
      resources=Resources(limits={"cpu": "750m", "memory": "4Gi"}),
      *args, **kwargs)


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  is_training = (FLAGS.is_training == 1)

  condition = "train"
  if not is_training:
    condition = "eval"

  tf.logging.info("Obtaining raw data for condition {}".format(condition))

  # TODO: need to check maybe transfer code above for downloading direct to GCS then
  # use it in fec.py
  output_dir = os.path.join(str(FLAGS.output_bucket),
                            condition, str(FLAGS.shard_id))

  #tmp_dir = tempfile.mktemp()
  tmp_dir = "/mnt/ssd0"
  tf.gfile.MakeDirs(tmp_dir)

  # TODO: download_fec_data needs modification to work in sharded form
  sharded_download_fec_data(tmp_dir=tmp_dir,
                            is_training=is_training,
                            shard_id=FLAGS.shard_id,
                            num_shards=FLAGS.num_shards)

  for filename in tf.gfile.ListDirectory(tmp_dir):
    
    # Might as well save all the data to avoid having to download it in the
    # future if something needs to change.
    #regex = r'cropped@[0-9a-z\.\-]*#http[0-9a-z\.\-]*:--'
    #hits = re.search(regex, filename)
    #if hits is not None:

    source_path = os.path.join(tmp_dir, filename)
    target_path = os.path.join(output_dir, filename)

    if not tf.gfile.Exists(target_path):
      valid_jpg = (filename.endswith("jpg") and filename.startswith("cropped"))
      if valid_jpg or filename.endswith("json"):
        tf.gfile.Copy(source_path, target_path, overwrite=True)

  tf.logging.info(_SUCCESS_MESSAGE)


if __name__ == "__main__":

  flags = tf.flags
  FLAGS = flags.FLAGS

  flags.DEFINE_string("output_bucket", None, "Bucket path to which to write data.")

  flags.DEFINE_integer("num_shards", 1, "Total num shards.")

  flags.DEFINE_integer("shard_id", 0, "Which shard.")

  flags.DEFINE_integer("is_training", None, "Integer specification of is_training.")
  
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
