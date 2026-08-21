#!/usr/bin/env bash
# V6.1 pressure sweep on a CALIBRATED axis.
#
# Measured: one task of ordinary training moves the shared parameters by
# 2.51. The prospective hook moves them by 0.34 at 2 outer steps (13% of
# a task), 1.31 at 8 (52%), 4.99 at 32 (199%). So outer steps is a real
# pressure knob, unlike the weight, which Adam's scale invariance makes
# inert.
#
# H35 registers a non-monotonic optimum, so the grid spans an order of
# magnitude around parity with task training.
cd "$(dirname "$0")/.."
for outer in 2 8 32; do
  for w in 0 1 2; do
    out="artifacts/v6_pressure_fixed/o${outer}/world_${w}/lifecycle"
    [ -f "$out/summary.json" ] && continue
    python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
      --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
      --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
      --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle \
      --arm prospective --prospective-steps "$outer" \
      --prospective-inner-steps 8 \
      --output "$out" >> tools/v6_calibrated.log 2>&1
    echo "outer=$outer world $w exit=$?" >> tools/v6_calibrated.log
  done
done
echo V6_CALIBRATED_DONE >> tools/v6_calibrated.log
