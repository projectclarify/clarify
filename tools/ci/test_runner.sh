#!/bin/bash

set -v  # print commands as they're executed

# Instead of exiting on any failure with "set -e", we'll call set_status after
# each command and exit $STATUS at the end.
STATUS=0
function set_status() {
    local last_status=$?
    if [[ $last_status -ne 0 ]]
    then
      echo "<<<<<<FAILED>>>>>> Exit code: $last_status"
    fi
    STATUS=$(($last_status || $STATUS))
}

# Check env vars set
echo "${TF_VERSION:?}" && \
echo "${TF_LATEST:?}" && \
echo "${TRAVIS_PYTHON_VERSION:?}"
set_status
if [[ $STATUS -ne 0 ]]
then
  exit $STATUS
fi

set_status

pylint pcml
set_status

# Run tests
# Ignores:
# Tested separately:
#   * registry_testp
#   * trainer_lib_test
#   * visualization_test
#   * trainer_model_based_test
#   * allen_brain_test
#   * models/research
# algorithmic_math_test: flaky
pytest
  #--ignore=tensor2tensor/utils/registry_test.py \
set_status

#pytest tensor2tensor/utils/registry_test.py
#set_status

exit $STATUS