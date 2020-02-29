load("@my_deps//:requirements.bzl", "requirement")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@io_bazel_rules_docker//container:container.bzl", "container_push")
load("@io_bazel_rules_docker//container:container.bzl", "container_image")
load("@bazel_gazelle//:def.bzl", "gazelle")

# gazelle:prefix github.com/projectclarify/pcml/sampler
gazelle(name = "gazelle")

filegroup(
    name = "dev-requirements.txt",
    srcs = [
        "dev-requirements.txt",
    ],
)

py_library(
    name = "setup",
    srcs = ["setup.py"],
    deps = [requirement("setuptools")],
)

# Must either be run in an environment where the right dev-requirements.txt has
# already been installed or install a packaged one before running clarify-train.
# Because rules_python does not package dependencies by fully recursing through
# the python dependency tree, only includes the ones explicitly specified.
# Alternatively can just go in and add dependencies to the deps of :train to have
# them included without the need to re-build the base container or pip install -r.
pkg_tar(
    name = "clarify_pkg_tar",
    strip_prefix = "/bazel-out/k8-fastbuild/bin",
    package_dir = "/usr/local/src/clarify-pkg/",
    srcs = glob([
        "bazel-out/k8-fastbuild/bin/clarify/research/**/*",
    ]),
    mode = "0755",
    symlinks = {
        "/clarify/bin/train": "/usr/local/src/clarify-pkg/clarify/research/train",
    },
)

pkg_tar(
    name = "clarify_configs_pkg_tar",
    strip_prefix = "/bazel-out/k8-fastbuild/bin/configs",
    package_dir = "/usr/local/src/clarify-configs",
    srcs = glob([
        "configs/**/*",
    ]),
    mode = "0755",
    symlinks = {
        "/clarify/configs": "/usr/local/src/clarify-configs/configs",
    },
)

container_image(
    name = "runtime",
    base = "@runtime_base//image",
    tars = [
        ":clarify_pkg_tar",
        ":clarify_configs_pkg_tar",
    ],
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
