
## Release automation

The plan to to cut releases upon changes to VERSION as part of the prow postsubmit workflow which so far will include releases on PyPI and GitHub. Individually these will be able to be triggered with `*_release.py`  or both upon change to VERSION by way of Bazel e.g. `bazel run //tools/release:release_all`.
