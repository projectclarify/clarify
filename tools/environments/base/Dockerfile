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

WORKDIR /home/jovyan
RUN git clone https://github.com/projectclarify/pcml.git

RUN sudo usermod -aG sudo jovyan
RUN echo "jovyan:Docker!" | chpasswd

WORKDIR /home/jovyan/pcml
RUN sh tools/environments/base/install_base.sh
RUN sh tools/environments/base/install_tf.sh
RUN sh tools/environments/base/install_bazel.sh
RUN sh tools/environments/base/install_jax.sh

RUN echo "export PATH=$PATH:$HOME/bin" >> $HOME/.bashrc

RUN rm -rf /home/jovyan/.cache
RUN chown -R jovyan:root /home/jovyan

user jovyan

WORKDIR /home/jovyan/pcml
RUN pip install -r dev-requirements.txt --user

#RUN /home/jovyan/bin/bazel build //pcml/utils/...

RUN ln -s /home/jovyan/pcml/tools/environments/pcrun.py /home/jovyan/bin/pcrun && chmod u+x /home/jovyan/bin/pcrun
