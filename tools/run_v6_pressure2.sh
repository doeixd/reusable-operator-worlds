#!/usr/bin/env bash
# Vary the pressure by STEPS, since Adam's scale invariance makes the
# weight inert (measured: 100x weight, no additional effect).
cd "$(dirname "$0")/.."
for steps in 16 64; do
  for w in 0 1 2; do
    out="artifacts/v6_pressure/s${steps}/world_${w}/lifecycle"
    [ -f "$out/summary.json" ] && continue
    python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
      --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
      --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
      --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle \
      --arm prospective --prospective-steps "$steps" --prospective-weight 1.0 \
      --output "$out" >> tools/v6_pressure2.log 2>&1
    echo "steps=$steps world $w exit=$?" >> tools/v6_pressure2.log
  done
done
echo V6_PRESSURE2_DONE >> tools/v6_pressure2.log
