workspace(name = "clarify")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/google/protobuf",
    tag = "v3.6.1.3",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Download the rules_docker repository at release v0.13.0
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "df13123c44b4a4ff2c2f337b906763879d94871d16411bf82dcfeba892b58607",
    strip_prefix = "rules_docker-0.13.0",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.13.0/rules_docker-v0.13.0.tar.gz"],
)

load(
    "@io_bazel_rules_docker//toolchains/docker:toolchain.bzl",
    docker_toolchain_configure = "toolchain_configure",
)

docker_toolchain_configure(
    name = "docker_config",
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)

container_pull(
    name = "runtime_base",
    registry = "gcr.io",
    repository = "clarify/runtime-base",

    # Derived from workspace base, has dev-requirements and tf 1.15, pending
    # build of leaner image.
    digest = "sha256:689b51bfd966a6b6e490b3b16a02303380f45ec452d9c3cfa92d4ff31a0a412b",

    # Older workspace base
    #digest = "sha256:a53a953e6451e3f776697292c2f721faecb50af6a50ad8325d614297e8091114"
)

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)

#git_repository(
#  name="rules_python",
#  commit="f641e9522d1812394d8b9de1c557daaaef1f8a44",
#  remote = "https://github.com/bazelbuild/rules_python"
#)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Only needed if using the packaging rules.
load("@rules_python//python:pip.bzl", "pip_repositories")

pip_repositories()

load("@rules_python//python:pip.bzl", "pip3_import")

# Create a central repo that knows about the dependencies needed for
# requirements.txt.
pip3_import(
    # or pip3_import
    name = "my_deps",
    requirements = "//:dev-requirements.txt",
)

# Load the central repo's install function from its `//:requirements.bzl` file,
# and call it.
load("@my_deps//:requirements.bzl", "pip_install")

pip_install()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/rules_go/releases/download/v0.20.3/rules_go-v0.20.3.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.20.3/rules_go-v0.20.3.tar.gz",
    ],
    sha256 = "e88471aea3a3a4f19ec1310a55ba94772d087e9ce46e41ae38ecebe17935de7b",
)

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies", "go_register_toolchains")

go_rules_dependencies()

go_register_toolchains()

http_archive(
    name = "bazel_gazelle",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.1/bazel-gazelle-v0.19.1.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.19.1/bazel-gazelle-v0.19.1.tar.gz",
    ],
    sha256 = "86c6d481b3f7aedc1d60c1c211c6f76da282ae197c3b3160f54bd3a8f847896f",
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()
