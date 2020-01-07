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

# On CircleCI, the checkout step places the code of a PR in ~/project. 
# This will be mounted to the container path /home/jovyan/pcml.

# HACK
docker run -it gcr.io/clarify/runtime-base:v0.1.0-b5f1 \
  /bin/bash -c 'source ~/.bashrc; rm -rf /home/jovyan/pcml; cd /home/jovyan; git clone https://github.com/projectclarify/pcml.git; cd pcml; git checkout ${CIRCLE_SHA1}; pip install -r dev-requirements.txt --user; sh tools/testing/test_local.sh'

# Future
#docker run -it gcr.io/clarify/runtime-base:test-latest \
#  pcrun --pip_install=true \
#  --commit=${CIRCLE_SHA1} \
#  --cmd="sh tools/testing/test_local.sh"
