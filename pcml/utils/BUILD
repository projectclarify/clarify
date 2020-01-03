package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "audio_utils",
    srcs = ["audio_utils.py"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "audio_utils_test",
    srcs = ["audio_utils_test.py"],
    deps = [
        "//pcml/utils:audio_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "augmentation_utils",
    srcs = ["augmentation_utils.py"],
    deps = [
        requirement("Pillow"),
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "augmentation_utils_test",
    srcs = ["augmentation_utils_test.py"],
    deps = [
        "//pcml/utils:augmentation_utils",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "cbt_utils",
    srcs = ["cbt_utils.py"],
    deps = [
        "//pcml/utils:video_utils",
        requirement("google-cloud-bigtable"),
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "cbt_utils_test",
    srcs = ["cbt_utils_test.py"],
    deps = [
        "//pcml/operations:cbt_datagen",
        "//pcml/operations:extract",
        "//pcml/utils:cbt_utils",
        "//pcml/utils:cfg_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "cfg_utils",
    srcs = ["cfg_utils.py"],
    deps = [
        "//pcml/utils:fs_utils",
        "//pcml:test_config",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "cfg_utils_test",
    srcs = ["cfg_utils_test.py"],
    deps = [
        "//pcml/utils:cfg_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "cmd_utils",
    srcs = ["cmd_utils.py"],
    deps = [requirement("tensorflow")],
)

py_test(
    name = "cmd_utils_test",
    srcs = ["cmd_utils_test.py"],
    deps = [
        "//pcml/utils:cmd_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "dev_utils",
    srcs = ["dev_utils.py"],
    deps = [
        "//pcml/utils:cfg_utils",
        requirement("grpcio"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
        requirement("tensorflow-serving-api"),
    ],
)

py_test(
    name = "dev_utils_test",
    srcs = ["dev_utils_test.py"],
    deps = [
        "//pcml/utils:dev_utils",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "fs_utils",
    srcs = ["fs_utils.py"],
    deps = [requirement("tensorflow")],
)

py_test(
    name = "fs_utils_test",
    srcs = ["fs_utils_test.py"],
    deps = [
        "//pcml/utils:fs_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "video_utils",
    srcs = ["video_utils.py"],
    deps = [
        requirement("Pillow"),
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "video_utils_test",
    srcs = ["video_utils_test.py"],
    deps = [
        "//pcml/utils:video_utils",
        requirement("tensorflow"),
    ],
)
