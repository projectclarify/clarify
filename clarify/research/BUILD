package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "//clarify/research/image_fec:__init__",
        "//clarify/research/gmcl:__init__",
        "//clarify/research/eeg_ssl:__init__",
        "//clarify/research/vid_fec:__init__",
        requirement("trax"),
        requirement("absl-py"),
    ],
)
