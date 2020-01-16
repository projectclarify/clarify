## Testing and PRs

The nature of our project requires a test strategy that spans various languages, systems, and user roles.

### Languages:

The front-end platform is primarily TypeScript with tests and depend supporting functions written in Python, see https://github.com/projectclarify/clarify/tree/master/client.
A subset of the codebase concerned with high-throughput sampling of examples is written in Golang, https://github.com/projectclarify/clarify/tree/master/exp/sampler
The rest involving tensor2tensor datasets, models, means for launching training jobs on Kubernetes, and supporting utilities are written in Python, https://github.com/projectclarify/clarify/tree/master/clarify

### Roles:

For all user roles, CircleCI tests will run automatically without authentication meaning the tests that run there will be those that do not require it - i.e. local tests, linting, and coverage analysis. Provided these pass and upon review by a project admin, assigning an “/ok-to-test” to the PR will trigger prow tests on Kubernetes that require authentication (e.g. to Cloud BigTable and to create additional processes in the gke cluster testing namespace). 

Project Clarify GitHub project owners - circle and prow tests run automatically
Non-owners (including reviewers and approvers) - circle tests run automatically, prow tests require /ok-to-test from project owner.

### Local testing:

During local development in order to run tests that interact with cloud resources users will need to authenticate using a service account credential. For non-admin project members, these will be provided in the root of a user’s workspace directory. These credentials can be activated using the `gcloud auth activate-service account` command.

Upon authenticating, tests can be run using either Bazel or language-specific test methods but given tests of PRs will be performed using Bazel every PR should pass `bazel test //...` from the project root before being submitted. Some may find development easier by running tests directly which is fine but Bazel is what we use to ensure test reproducibility. Non project members will need to authenticate to a GCP project with credentials that permit tests to be run.

Tests that do not require authentication, i.e. those first-round tests that run on CircleCI, can be run via `sh tools/testing/test_local.sh`.

It is possible that tests will pass on your local but not on our remote test systems. To debug this you can run these tests locally (in the same containers as are run remotely) on a machine with Docker, e.g.

```bash

python tools/environments/build.py --build_mode=local --static_image_id=test-container --container_type=workspace

docker run -it test-container /bin/bash -c "source ~/.bashrc; cd /build; sh tools/testing/test_local.sh"

``

### Remote testing:

When a PR is submitted, CircleCI tests are always run including operations that don’t require special credentials as well as linting and test coverage analysis. If these pass a project owner reviews the contents of the PR and if no security or operational problems would be created by running the tests on prow applies an /ok-to-test label permitting prow tests to be run.
