#!/bin/bash
#
# Run pylint in the accurate local conda env

source /scratch-shared/meteoswiss/scratch/sadamov/mambaforge/bin/activate aldernet

set -o errexit

VERBOSE=${VERBOSE:-false}

cd "$(dirname "${0}")"

paths=(
    src/aldernet
    tests/test_aldernet
)
for path in "${paths[@]}"; do
    ${VERBOSE} && echo "pylint \"${path}\""
    pylint "${path}" || exit
done
