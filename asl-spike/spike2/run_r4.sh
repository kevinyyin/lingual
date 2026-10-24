#!/usr/bin/env bash
# R4 ablation: 5 variants x 3 seeds, sequential (CPU)
set -u
cd "$(dirname "$0")/.."
for v in baseline aug wlasl popsign combined; do
  for s in 0 1 2; do
    if [ -f results2/r4_${v}_s${s}.json ]; then echo "skip $v $s"; continue; fi
    uv run python spike2/r4_train.py $v $s 2>&1 | grep -E "^$v|Traceback|Error"
  done
done
echo "=== r4 done"
