#!/usr/bin/env bash
# P3 training chain (CPU). Times are wall-clock budgets per model.
set -u
cd "$(dirname "$0")/.."
run() { echo "=== $*"; uv run python spike2/p3_model.py "$@" 2>&1 | grep -E "^saved|Traceback|Error" ; }
run bi 25 a_bi
run stream 20 a_stream
for k in 0 1 2; do run bi 8 b_bi_f$k --init cache/s2r2/p3_a_bi.pt --rwth-fold $k --lr 5e-4; done
for k in 0 1 2; do run bi 5 c_bi_f$k --init cache/s2r2/p3_b_bi_f$k.pt --rwth-fold $k --rwth-calib --lr 3e-4; done
echo "=== all done"
