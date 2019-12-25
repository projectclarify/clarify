FROM gcr.io/kubeflow-images-public/tensorflow-1.6.0-notebook-cpu:v20180607-476e150e

USER root

RUN pip uninstall -y tensor2tensor tensorflow
RUN pip install --upgrade pip

RUN sudo usermod -aG sudo jovyan
RUN echo "jovyan:Docker!" | chpasswd

RUN sudo apt-get update
RUN sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip ffmpeg wget curl

ENV BAZEL_VERSION=0.29.1
RUN cd $HOME && wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

RUN chmod +x $HOME/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN cd $HOME && ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
RUN chmod ugo+x $HOME/bin/bazel && cp $HOME/bin/bazel /usr/local/bin/

USER jovyan

RUN pip install --upgrade tensorflow==1.13.1 seaborn --user
RUN pip install moviepy
