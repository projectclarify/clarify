package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "image_fec",
    srcs = ["image_fec.py"],
    deps = [
        requirement("trax"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "image_fec_test",
    srcs = ["image_fec_test.py"],
    deps = [
        "//pcml/models:image_fec",
        requirement("trax"),
        requirement("absl-py"),
    ],
)
