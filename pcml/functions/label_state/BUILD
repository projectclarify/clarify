package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "deploy",
    srcs = ["deploy.py"],
    deps = [
        "//pcml/functions/utils:deployment_utils",
        "//pcml/utils:fs_utils",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "deploy_test",
    srcs = ["deploy_test.py"],
    deps = [
        "//pcml/functions/label_state:deploy",
        "//pcml/utils:cfg_utils",
    ],
)

py_library(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "//pcml/functions/utils:deployment_utils",
        "//pcml/utils:fs_utils",
        "//pcml/models:dev",
        "//pcml/models:modality_correspondence",
        requirement("tensorflow"),
        requirement("numpy"),
        requirement("sklearn"),
        requirement("google-cloud-firestore"),
    ],
)

py_test(
    name = "main_test",
    srcs = ["main_test.py"],
    deps = [
        "//pcml/functions/label_state:main",
        requirement("mock"),
        requirement("numpy"),
    ],
)
