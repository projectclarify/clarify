package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "affectnet",
    srcs = ["affectnet.py"],
    deps = [
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "affectnet_test",
    srcs = ["affectnet_test.py"],
    deps = [
        "//pcml/datasets:affectnet",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        "//pcml/datasets:image_aug",
        "//pcml/utils:cbt_utils",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "celeba",
    srcs = ["celeba.py"],
    deps = [
        "//pcml/operations:tfrecord2bigtable",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "celeba_test",
    srcs = ["celeba_test.py"],
    deps = [
        "//pcml/datasets:celeba",
        "//pcml/datasets:test_utils",
        "//pcml/utils:dev_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "deap",
    srcs = ["deap.py"],
    deps = [
        "//pcml/datasets:utils",
        requirement("mne"),
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "deap_test",
    srcs = ["deap_test.py"],
    deps = [
        "//pcml/datasets:deap",
        "//pcml/datasets:test_utils",
        "//pcml/utils:dev_utils",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "example_utils",
    srcs = ["example_utils.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "example_utils_test",
    srcs = ["example_utils_test.py"],
    deps = [
        "//pcml/datasets:example_utils",
        "//pcml/datasets:utils",
        "//pcml/utils:dev_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "fec",
    srcs = ["fec.py"],
    deps = [
        "//pcml/datasets:base",
        "//pcml/utils:cbt_utils",
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "fec_test",
    srcs = ["fec_test.py"],
    deps = [
        "//pcml/datasets:fec",
        "//pcml/utils:cfg_utils",
        "//pcml/utils:dev_utils",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "image_aug",
    srcs = ["image_aug.py"],
    deps = [requirement("tensorflow")],
)

py_library(
    name = "mahnob_hci",
    srcs = ["mahnob_hci.py"],
    deps = [
        "//pcml/datasets:utils",
        requirement("mne"),
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "mahnob_hci_test",
    srcs = ["mahnob_hci_test.py"],
    deps = [
        "//pcml/datasets:mahnob_hci",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        "//pcml/datasets:test_utils",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        requirement("Pillow"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        "//pcml/datasets:test_utils",
        "//pcml/datasets:utils",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "vggface2",
    srcs = ["vggface2.py"],
    deps = [
        "//pcml/datasets:base",
        "//pcml/datasets:example_utils",
        "//pcml/datasets:fec",
        "//pcml/datasets:image_aug",
        "//pcml/datasets:utils",
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "vggface2_test",
    srcs = ["vggface2_test.py"],
    deps = [
        "//pcml/datasets:vggface2",
        "//pcml/utils:cfg_utils",
        "//pcml/utils:dev_utils",
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_library(
    name = "vox_celeb_cbt",
    srcs = ["vox_celeb_cbt.py"],
    deps = [
        "//pcml/utils:audio_utils",
        "//pcml/utils:augmentation_utils",
        "//pcml/utils:cbt_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "vox_celeb_cbt_test",
    srcs = ["vox_celeb_cbt_test.py"],
    deps = [
        "//pcml/datasets:vox_celeb_cbt",
        "//pcml/launcher:kube_test",
        "//pcml/operations:cbt_datagen",
        "//pcml/operations:extract",
        "//pcml/utils:cbt_utils",
        "//pcml/utils:cfg_utils",
        requirement("numpy"),
        requirement("tensor2tensor"),
        requirement("tensorflow"),
    ],
)
