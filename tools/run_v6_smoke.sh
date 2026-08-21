#!/usr/bin/env bash
# V6.1 smoke: four arms, NO basis freeze.
#
# V5's protocol froze the shared basis at task 8 so that new structure
# had nowhere to go but task-local innovations, which is what made an
# explicit promoter necessary. V6 asks whether prospective pressure can
# SHAPE the shared representation, so a frozen basis makes every
# prospective gradient dead on arrival -- measured: all four arms
# returned bit-identical lifetimes. The V6 control is therefore
# `ordinary` WITHOUT the freeze, not V5's numbers.
cd "$(dirname "$0")/.."
for arm in ordinary replay prospective supervised; do
  out="artifacts/v6_smoke_nofreeze/${arm}/world_0/lifecycle"
  [ -f "$out/summary.json" ] && continue
  python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
    --model prospective --world-seed 0 --r-meta 1.0 --meta-families 4 \
    --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
    --operator-slots 12 --sleeps 16 24 32 48 64 \
    --lifecycle --arm "$arm" --output "$out" >> tools/v6_smoke2.log 2>&1
  echo "$arm exit=$?" >> tools/v6_smoke2.log
done
echo V6_SMOKE2_DONE >> tools/v6_smoke2.log
