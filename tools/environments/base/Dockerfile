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

FROM gcr.io/kubeflow-images-public/tensorflow-1.14.0-notebook-cpu:v0.7.0

USER root

ADD ./ /usr/local/src/pcml

RUN sh /usr/local/src/pcml/tools/environments/base/install_base.sh
RUN sh /usr/local/src/pcml/tools/environments/base/install_tf.sh
RUN sh /usr/local/src/pcml/tools/environments/base/install_bazel.sh
RUN sh /usr/local/src/pcml/tools/environments/base/install_golang.sh
RUN sh /usr/local/src/pcml/tools/environments/base/install_jax.sh

RUN rm -rf /home/jovyan/.cache
RUN rm -rf /home/root/.cache

RUN sudo usermod -aG sudo jovyan
RUN echo "jovyan:Docker!" | chpasswd
RUN chown -R jovyan:root /home/jovyan
RUN chown -R jovyan:root /usr/local/src

RUN ln -s /usr/local/src/pcml /home/jovyan/pcml

RUN ln -s /usr/local/src/pcml/tools/fyre/fyre.py /usr/local/bin/fyre && chmod u+x /usr/local/bin/fyre

RUN ln -s /usr/local/src/pcml/tools/environments/base/setup_workspace.sh /usr/local/bin/setup-workspace && chmod u+x /usr/local/bin/setup-workspace

user jovyan

WORKDIR /home/jovyan/pcml
RUN /usr/local/bin/setup-workspace
