load("@my_deps//:requirements.bzl", "requirement")
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

# gazelle:prefix github.com/projectclarify/pcml/sampler
gazelle(name = "gazelle")

filegroup(
    name = "dev-requirements.txt",
    srcs = ["dev-requirements.txt"]
)

py_library(
    name = "setup",
    srcs = ["setup.py"],
    deps = [requirement("setuptools")],
)

load("@io_bazel_rules_docker//python:image.bzl", "py_image")
load("@io_bazel_rules_docker//container:container.bzl", "container_push")
load("@io_bazel_rules_docker//container:container.bzl", "container_image")

# Another rule creating a target //pcml:build, e.g. that is then
# included via srcs = ["//pcml:build"].

pkg_tar(
    name = "all_files_tar",
    strip_prefix = "/",
    package_dir = "/home/jovyan/",
    srcs = [],
    mode = "0644",
)

container_image(
    name = "runtime",
    base = "@runtime_base//image",
    data_path = "/home/jovyan",
    tars = [":all_files_tar"],
    cmd = ["pwd"],
)

container_push(
    name = "push_runtime",
    image = ":runtime",
    format = "Docker",
    registry = "gcr.io",
    repository = "clarify/runtime",
    tag = "{BUILD_USER}-{BUILD_TIMESTAMP}",
)
