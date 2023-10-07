#!/bin/bash

SUFFIX='no_seed_change_stable'
JOBS=8
MACHINE='io52'
# OUTPUT_DIR="/export/${MACHINE}/data/rpilgri1"
OUTPUT_DIR="./"

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=$JOBS \
    --coeff-distribution=sparse_gaussian \
    "${OUTPUT_DIR}/results_small_${SUFFIX}"

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=$JOBS \
    --coeff-distribution=sparse_gaussian \
    "${OUTPUT_DIR}/results_large_${SUFFIX}"

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=$JOBS \
    --coeff-distribution=sparse_const \
    "${OUTPUT_DIR}/results_small_${SUFFIX}_const"

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=$JOBS \
    --coeff-distribution=sparse_const \
    "${OUTPUT_DIR}/results_large_${SUFFIX}_const"