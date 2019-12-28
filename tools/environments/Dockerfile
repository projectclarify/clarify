FROM gcr.io/kubeflow-images-public/tensorflow-1.14.0-notebook-cpu:v0.7.0

USER root

ADD ./tools /build/tools

RUN pip uninstall -y tensor2tensor tensorflow
RUN pip install --upgrade pip

RUN sudo usermod -aG sudo jovyan
RUN echo "jovyan:Docker!" | chpasswd

RUN sh /build/tools/environments/install_base.sh
RUN sh /build/tools/environments/deb_install_bazel.sh
RUN sh /build/tools/environments/install_tf_serving.sh

RUN rm -rf /home/jovyan/.cache
RUN chown jovyan:root /home/jovyan

user jovyan
