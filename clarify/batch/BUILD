package(default_visibility = ["//visibility:public"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "fyre",
    srcs = ["fyre.py"],
    deps = [
        "//clarify/batch:job",
        "//clarify/utils:cmd_utils",
        requirement("kubernetes"),
    ],
)

py_test(
    name = "fyre_test",
    srcs = ["fyre_test.py"],
    deps = [
        "//clarify/batch:fyre",
        "//clarify/batch:jobs",
    ],
)

py_library(
    name = "job",
    srcs = ["job.py"],
    deps = [
        requirement("kubernetes"),
    ],
)

py_test(
    name = "job_test",
    srcs = ["job_test.py"],
    deps = [
        "//clarify/batch:job",
    ],
)

py_library(
    name = "jobs",
    srcs = ["jobs.py"],
    deps = [
        "//clarify/batch:job",
    ],
)

py_test(
    name = "jobs_test",
    srcs = ["jobs_test.py"],
    deps = [
        "//clarify/batch:jobs",
    ],
)
