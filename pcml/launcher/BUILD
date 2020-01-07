package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/launcher:util",
        "//pcml/operations:eval",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "kube",
    srcs = ["kube.py"],
    deps = [
        "//pcml/launcher:util",
        "//pcml/utils:fs_utils",
        requirement("kubernetes"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "kube_test",
    srcs = ["kube_test.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/launcher:util",
        requirement("kubernetes"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//pcml/utils:cmd_utils",
        requirement("tensorflow"),
        requirement("pyyaml"),
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        "//pcml/launcher:util",
        requirement("tensorflow"),
    ],
)
