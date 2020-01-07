package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_test(
    name = "cbt_datagen_test",
    srcs = ["cbt_datagen_test.py"],
    deps = [
        "//pcml/launcher:kube_test",
        "//pcml/operations:cbt_datagen",
        "//pcml/operations:extract",
        "//pcml/utils:cbt_utils",
        "//pcml/utils:cfg_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "cbt_datagen",
    srcs = ["cbt_datagen.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/operations:extract",
        "//pcml/utils:cbt_utils",
        "//pcml/utils:cmd_utils",
        "//pcml/utils:fs_utils",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "download_fec",
    srcs = ["download_fec.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/utils:cmd_utils",
        requirement("opencv-python"),
        requirement("faced"),
        requirement("numpy"),
        requirement("requests"),
        requirement("scipy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "download_fec_test",
    srcs = ["download_fec_test.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/launcher:kube_test",
        "//pcml/operations:download_fec",
        "//pcml/utils:cfg_utils",
        requirement("kubernetes"),
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/utils:cmd_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "eval_test",
    srcs = ["eval_test.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/operations:eval",
        requirement("kubernetes"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "extract",
    srcs = ["extract.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/launcher:util",
        "//pcml/utils:audio_utils",
        "//pcml/utils:cbt_utils",
        "//pcml/utils:fs_utils",
        "//pcml/utils:video_utils",
        requirement("numpy"),
        requirement("scipy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "extract_test",
    srcs = ["extract_test.py"],
    deps = [
        "//pcml/launcher:kube_test",
        "//pcml/operations:extract",
        "//pcml/utils:cbt_utils",
        "//pcml/utils:cfg_utils",
        requirement("google-cloud-bigtable"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "glob2manifest",
    srcs = ["glob2manifest.py"],
    deps = [
        "//pcml/launcher:kube",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "glob2manifest_test",
    srcs = ["glob2manifest_test.py"],
    deps = [
        "//pcml/launcher:kube",
        "//pcml/launcher:kube_test",
        "//pcml/operations:glob2manifest",
        requirement("kubernetes"),
        requirement("tensorflow"),
    ],
)
