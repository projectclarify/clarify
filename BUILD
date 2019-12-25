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

load("@io_bazel_rules_docker//python:image.bzl", "py_image")
load("@io_bazel_rules_docker//container:container.bzl", "container_push")
load("@io_bazel_rules_docker//container:container.bzl", "container_image")

# TODO: Build a py3 whl.
sh_binary(
    name = "install",
    srcs = ["tools/install.sh"],
    data = [
        "MANIFEST.in",
        "setup.py",
    ],
)

# TODO: Script that wraps :install followed by :push_trainer (implicitly
# :trainer_image) if building containers inside containers is an interest/
# priority. If we want to do so will need to rectify py2/py3 issue.
container_image(
    name = "trainer_image",
    base = "@base//image",
    #files = gloib(["build/**"]),
    data_path = "/"
)


container_push(
   name = "push_trainer",
   image = ":trainer_image",
   format = "Docker",
   registry = "gcr.io",
   repository = "clarify/trainer",
   tag = "dev"
)

container_image(
    name = "basic_alpine",
    base = "@alpine_linux_amd64//image",
    cmd = ["Hello World!"],
    entrypoint = ["echo"],
)


