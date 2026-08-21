#!/usr/bin/env bash
# V6.1 rerun with the corrected prospective arm.
#
# The original inner loop used SGD and adapted the sibling by 0.000%,
# so the penalty was a zero-shot loss and the arm was applying the
# explicit-family-sharing objective under the prospective name. Adam at
# the task learning rate now reduces support loss 39% in 8 steps and
# 61% in 16.
#
# inner = 8 (adaptation is real), outer = 2 (cost). Every prospective
# cell before this commit is invalid and was deleted.
cd "$(dirname "$0")/.."
for w in 0 1 2; do
  out="artifacts/v6/prospective/world_${w}/lifecycle"
  [ -f "$out/summary.json" ] && continue
  python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
    --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
    --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
    --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle \
    --arm prospective --prospective-steps 2 --prospective-inner-steps 8 \
    --output "$out" >> tools/v6_fixed.log 2>&1
  echo "world $w exit=$?" >> tools/v6_fixed.log
done
echo V6_FIXED_DONE >> tools/v6_fixed.log
