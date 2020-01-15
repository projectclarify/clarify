# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#bazel build //clarify/batch/...
#bazel build //clarify/datasets/...
#bazel build //clarify/models/...
#bazel build //clarify/research/...
#bazel build //clarify/serve/...
#bazel build //clarify/tune/...
#bazel build //clarify/utils/...

bazel build //clarify/...

# Fails because batch jobs time out. Also probably not going to give
# circle jobs.create permissions.
# bazel test //clarify/batch/... --test_output=all

# Have not yet done conversion from old codebase to new Job object
# but both the batch and cbt operations in here would fail via
# circle without auth.
# bazel test //clarify/datasets/... --test_output=all

bazel test //clarify/models/... --test_output=all

bazel test //clarify/research/... --test_output=all

# Haven't yet converted this part of the codebase over, may change
# significantly with change to Trax.
# bazel test //clarify/serve/... --test_output=all

bazel test //clarify/tune/... --test_output=all

bazel test //clarify/utils:audio_utils_test --test_output=all
bazel test //clarify/utils:augmentation_utils_test --test_output=all
bazel test //clarify/utils:cmd_utils_test --test_output=all
bazel test //clarify/utils:fs_utils_test --test_output=all
bazel test //clarify/utils:video_utils_test --test_output=all
bazel test //clarify/utils:cfg_utils_test --test_output=all
bazel test //clarify/utils:dev_utils_test --test_output=all

# Requires BigTable permissions
#bazel test //pcml/utils:cbt_utils_test
