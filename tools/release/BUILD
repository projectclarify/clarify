
load("@my_deps//:requirements.bzl", "requirement")

# Require command line flag confirmation before actually sending a
# release? TODO
config_setting(
    name = "confirm_release",
    values = {"confirm_release": "false"},
)

py_binary(
    name = "release_all",
    main = "release.py",
    srcs = ["release.py"],
    deps = [

    ],
)

py_library(
    name = "release",
    srcs = ["release.py"],
    deps = [

    ],
)

py_test(
    name = "release_test",
    srcs = ["release_test.py"],
    deps = [
        "//tools/release:release",
    ],
)
