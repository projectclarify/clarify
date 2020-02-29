package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":models",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        requirement("trax"),
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        "//clarify/research/image_fec:models",
        requirement("trax"),
        requirement("absl-py"),
    ],
)
