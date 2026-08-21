#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for arm in ordinary replay prospective supervised; do
  out="artifacts/v6_smoke/${arm}/world_0/lifecycle"
  [ -f "$out/summary.json" ] && continue
  python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
    --model prospective --world-seed 0 --r-meta 1.0 --meta-families 4 \
    --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
    --freeze-basis-at 8 --operator-slots 12 --sleeps 16 24 32 48 64 \
    --lifecycle --arm "$arm" --output "$out" >> tools/v6_smoke.log 2>&1
  echo "$arm exit=$?" >> tools/v6_smoke.log
done
echo V6_SMOKE_DONE >> tools/v6_smoke.log
