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

"""Wrapper of trax.train enabling use with our Bazel setup.

When training in batch, we'd like to be able to initiate this via
a Bazel run command such as 
  `bazel run //clarify/research/train --config=image_fec/main.gin`
This is so that the model and other objects references from the gin
config file are available at runtime to the call to trax.train. I.e.
this wrapper can be configured with Bazel to depend on at least
everything on the Bazel path //clarify/research/... and thereby permit
training both with (1) any gin configs found on that path and (2)
references there within to anything on that path.

Alternative could vendor trax and patch a Bazel dependency onto
trax.trainer to be //clarify/research/...

"""

from absl import app

from trax.trainer import FLAGS
from trax.trainer import main

if __name__ == '__main__':
  app.run(main)
