#!/bin/bash

DIR=${1:-.}
LINES=$(yapf -d -r "${DIR}" | wc -l | tr -s ' ')
exit ${LINES}

