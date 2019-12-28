package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "dev",
    srcs = ["dev.py"],
    deps = [
        "//pcml/datasets:vox_celeb_cbt",
        "//pcml/models:modality_correspondence",
        "//pcml/models:modality_correspondence_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "image_fec",
    srcs = ["image_fec.py"],
    deps = [
        "//pcml/models:dev",
        "//pcml/models:modality_correspondence",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "image_fec_test",
    srcs = ["image_fec_test.py"],
    deps = [
        "//pcml/models:percep_similarity_emb",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "modality_correspondence",
    srcs = ["modality_correspondence.py"],
    deps = [
        "//pcml/models:modality_correspondence_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "modality_correspondence_test",
    srcs = ["modality_correspondence_test.py"],
    deps = [
        "//pcml/launcher:experiment",
        "//pcml/launcher:kube_test",
        "//pcml/utils:cfg_utils",
        "//pcml/utils:dev_utils",
        "//pcml/utils:fs_utils",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "modality_correspondence_utils",
    srcs = ["modality_correspondence_utils.py"],
    deps = [
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "modality_correspondence_utils_test",
    srcs = ["modality_correspondence_utils_test.py"],
    deps = [
        "//pcml/utils:cfg_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "percep_similarity_emb",
    srcs = ["percep_similarity_emb.py"],
    deps = [
        "//pcml/models:dev",
        "//pcml/models:modality_correspondence",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "percep_similarity_emb_test",
    srcs = ["percep_similarity_emb_test.py"],
    deps = [
        "//pcml/models:percep_similarity_emb",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)
