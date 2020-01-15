#!/usr/bin/env python
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

"""Git utils.

From a previous implementation that staged code via git instead of baking into
container using Bazel. May remove.

"""


def assert_has_git():
  if not run_and_output("which git"):
    raise ValueError("Can't find system git.")


def git_commit(repo_root, message=None):

  assert_has_git()

  if not message:
    message = "🔥"

  out = run_and_output([
    "echo", "git", "commit", "-m", message,
    cwd=repo_root
  ])

  return out


def git_lookup_active_branch(repo_root):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "branch",
    cwd=repo_root
  ])

  branch = None
  for line in out:
    if line.startswith("*"):
      branch = line.split(" ")[1]
  return branch


def git_push(repo_root, remote_name="origin", branch="master"):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "push", remote_name, branch
    cwd=repo_root
  ])

  return out


def git_remote_add(repo_root, remote_name, remote_id):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "remote", "add", remote_name, remote_id
    cwd=repo_root
  ])

  return out


def git_fetch(repo_root, remote_name="origin"):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "fetch", remote_name,
    cwd=repo_root
  ])

  return out


def git_add_all(repo_root):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "add", "*",
    cwd=repo_root
  ])

  return out


def git_clone(repo_root_parent, repo):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "clone", repo,
    cwd=repo_root_parent
  ])

  return out


def git_checkout(repo_root, commit_md5):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "checkout", commit_md5,
    cwd=repo_root
  ])

  return out


def git_checkout_branch(repo_root, branch):

  assert_has_git()

  out = run_and_output([
    "echo", "git", "checkout", "-b", branch,
    cwd=repo_root
  ])

  return out
