#!/usr/bin/env bash
# V6.2: REPRESENTATION ALLOCATION.
#
# Not an H33 repair. V6.1 showed that where plasticity is allowed
# determines where recurrence gets stored: a frozen basis gives M ~ 7
# with dead prospective gradients, a fully plastic one gives M ~ 2-5
# with live gradients. That is V1/V2's continuous-manifold-versus-
# explicit-atoms result one level up, and it predicts a frontier rather
# than a nuisance.
#
# `--freeze-slots N` freezes the first N of 12 basis operators at task 8
# and leaves the rest trainable, so free capacity = 12 - N.
cd "$(dirname "$0")/.."
for frozen in 12 11 10 9 6; do
  free=$((12 - frozen))
  for arm in ordinary replay prospective; do
    for w in 0 1 2; do
      out="artifacts/v6_alloc/free${free}/${arm}/world_${w}/lifecycle"
      [ -f "$out/summary.json" ] && continue
      python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
        --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
        --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
        --operator-slots 12 --freeze-basis-at 8 --freeze-slots "$frozen" \
        --sleeps 16 24 32 48 64 --lifecycle --arm "$arm" \
        --prospective-steps 16 --prospective-inner-steps 16 \
        --output "$out" >> tools/v6_alloc.log 2>&1
      echo "free=$free $arm world $w exit=$?" >> tools/v6_alloc.log
    done
  done
done
echo V6_ALLOC_DONE >> tools/v6_alloc.log
