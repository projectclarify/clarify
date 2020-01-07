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

"""Batch run wrapper."""

import subprocess
import sys

def run_and_output(command, cwd=None, env=None):
  
  process = subprocess.Popen(
    command, cwd=cwd, env=env,
    stdout=subprocess.PIPE
  )

  output = []

  for line in process.stdout:
    line = line.decode("utf-8")
    sys.stdout.write(line)
    output.append(line)

  return output


def main(checkout=None, pip_install=False, cmd=None, 
         pcml_root="/home/jovyan/pcml"):

  if checkout:
    run_and_output(
      command=["git", "checkout", checkout],
      cwd=pcml_root
    )

  if pip_install:
    run_and_output(
      command=["pip", "install", "-r", "dev-requirements.txt",
               "--user"],
      cwd=pcml_root
    )

  if cmd:
    run_and_output(
      command=cmd.split(" "),
      cwd=pcml_root
    )

  

if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser(description='PCML run wrapper.')

  parser.add_argument('--checkout', type=str, default=None,
                      required=False,
                      help='A PCML commit to check out.')

  parser.add_argument('--pip_install', type=bool, default=False,
                      help='Whether to pip install PCML deps.')

  parser.add_argument('--cmd', type=str, default=None,
                      required=False,
                      help='A command to run.')

  parser.add_argument('--pcml_root', type=str, 
                      default="/home/jovyan/pcml",
                      help='Path to root of pcml repository.')

  args = parser.parse_args()

  main(
    checkout=args.checkout,
    pip_install=args.pip_install,
    cmd=args.cmd,
    pcml_root=args.pcml_root
  )
