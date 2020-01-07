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
"""A training experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pcml.launcher.kube import AttachedVolume
from pcml.launcher.kube import LocalSSD
from pcml.launcher.kube import Resources
from pcml.launcher.kube import Job
from pcml.launcher.util import generate_job_name


class Experiment(Job):
  pass


def main(argv):
  pass


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.app.run()
