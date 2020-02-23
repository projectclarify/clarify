
package(default_visibility = ["//visibility:public"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = [
    ],
)

py_library(
    name = "check_membership",
    srcs = ["check_membership.py"],
    deps = [
        ":common",
    ],
)

py_test(
    name = "check_membership_test",
    srcs = ["check_membership_test.py"],
    deps = [
      ":check_membership"
    ],
)

py_binary(
    name = "deploy",
    srcs = ["deploy.py"],
    deps = [
        #":foolib",
    ],
)