#!/usr/bin/env bash
# P3 chain, part 2: continue the bidirectional model (total ~55 min), then RWTH variants from it.
set -u
cd "$(dirname "$0")/.."
run() { echo "=== $*"; uv run python spike2/p3_model.py "$@" 2>&1 | grep -E "^saved|Traceback|Error" ; }
run bi 30 a_bi2 --init cache/s2r2/p3_a_bi.pt --lr 5e-4
for k in 0 1 2; do run bi 8 b_bi_f$k --init cache/s2r2/p3_a_bi2.pt --rwth-fold $k --lr 5e-4; done
for k in 0 1 2; do run bi 5 c_bi_f$k --init cache/s2r2/p3_b_bi_f$k.pt --rwth-fold $k --rwth-calib --lr 3e-4; done
echo "=== all done"
