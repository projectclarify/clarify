load("@pcml_deps//:requirements.bzl", "requirement")

py_library(
  name = "cmd_utils",
  srcs = ["cmd_utils.py"],
  deps = [
      requirement("tensorflow")
  ]
)

py_test(
  name = "cmd_utils_test",
  srcs = ["cmd_utils_test.py"],
  deps = [":cmd_utils"]
)

py_library(
    name = "fs_utils",
    srcs = ["fs_utils.py"],
    deps = [
        requirement("tensorflow"),
    ]
)

py_test(
  name = "fs_utils_test",
  srcs = ["fs_utils_test.py"],
  deps = [":fs_utils"]
)

