package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "cloudrun_utils",
    srcs = ["cloudrun_utils.py"],
    deps = [
        requirement("python-dateutil"),
        requirement("tensorflow"),
        requirement("pyyaml"),
    ],
)

py_library(
    name = "deployment_utils",
    srcs = ["deployment_utils.py"],
    deps = [
        "//pcml/utils:cmd_utils",
        "//pcml/utils:fs_utils",
        requirement("google-cloud-pubsub"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "publish",
    srcs = ["publish.py"],
    deps = [
        requirement("tensorflow"),
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        requirement("tensorflow"),
    ],
)
