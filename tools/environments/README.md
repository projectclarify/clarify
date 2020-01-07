
# Build runtime base

For now we're using the same environment for users' development environments on Jupyter lab, tests on our CI systems, and various batch jobs including training and non-training jobs. This has two parts, the "runtime base" and the "runtime". The build of the former is what is implemented here and can be triggered locally via the following:

```bash

python tools/environments/build.py \
  --build_mode=local

```

Here the `--and_push=true` argument can be used to push the resulting image to the repository indicated by the tag (see below).

A build on Google Container Builder can be triggered via the following (where the resulting image is always pushed):

```bash

python tools/environments/build.py \
  --build_mode=gcb

```

By default, both of the above produce an image with an ID of the form `gcr.io/[project=clarify]/runtime-base:[version]-[tag]` where version is set in `build.py` and corresponds to releases (i.e. on GitHub, PyPI). If you wish to push the base runtime to a different GCP/gcr.io project you can override this via the `--project` argument.

When building a container on Circle for use there immediately after building with a static image ID is more convenient, this can be provided via the `--static_image_id` argument.

## TODO and notes

- A cleaner way to do this would be with Bazel or even easier to put the Dockerfile in the root of the repo and just run docker build . where that Dockerfile references tools/environments appropriately. This script used to have logic to template Dockerfiles and thus made sense to do in python and may in the future benefit in this way so sticking with this.
  - Could use Bazel to automatically re-build when runtime_base build files change.
- For now image tags are randomly salted but would be better off tagged uniquely, e.g. with a timestamp for when the were generated plus some salt.
