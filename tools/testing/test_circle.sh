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

# On CircleCI, the checkout step places the code of a PR in
# /project. This will be mounted to the container path
# /home/jovyan/pcml.

# Tests are run in the latest runtime base container which should
# match internal users' development environment.

# The pcrun wrapper is used to first pip install dev-requirements.txt
# and to execute the subsequent command from /home/jovyan/pcml.
# Equivalently this could be a bash -c command that sources ~/.bashrc
# and writes out the pip install command.

docker run -it gcr.io/clarify/runtime-base:v0.1.0-b5f1 \
  -v /project:/home/jovyan/pcml \
  bash -c "source ~/.bashrc; ls; pwd"

#bash -c "source ~/.bashrc; pip install -r dev-requirements.txt --user; sh tools/testing/test_local.sh"
