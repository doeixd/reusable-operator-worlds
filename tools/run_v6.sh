#!/usr/bin/env bash
# V6.1: four arms x three worlds, no basis freeze.
cd "$(dirname "$0")/.."
for w in 0 1 2; do
  for arm in ordinary replay prospective supervised; do
    out="artifacts/v6/${arm}/world_${w}/lifecycle"
    [ -f "$out/summary.json" ] && continue
    python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
      --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
      --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
      --operator-slots 12 --sleeps 16 24 32 48 64 \
      --lifecycle --arm "$arm" --prospective-steps 4 --prospective-weight 1.0 \
      --output "$out" >> tools/v6.log 2>&1
    echo "world $w $arm exit=$?" >> tools/v6.log
  done
done
echo V6_DONE >> tools/v6.log
