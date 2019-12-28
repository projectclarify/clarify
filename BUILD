load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "setup",
    srcs = ["setup.py"],
    deps = [requirement("setuptools")],
)

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

container_image(
    name = "trainer_image",
    base = "@trainer_base//image",
    files = [],
    cmd = ["ls"]
)

container_push(
   name = "push_trainer",
   image = ":trainer_image",
   format = "Docker",
   registry = "gcr.io",
   repository = "clarify/trainer",
   tag = "dev"
)
