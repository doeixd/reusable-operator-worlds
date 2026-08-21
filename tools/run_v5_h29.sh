#!/usr/bin/env bash
# H29 fresh runs: lifetimes that persist the P_0 snapshot.
cd "$(dirname "$0")/.."
for w in 600 601 602 603 604 605; do
  out="artifacts/v5_h29/r100/world_${w}/lifecycle"
  [ -f "$out/summary.json" ] && continue
  python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
    --model lifecycle --world-seed "$w" --r-meta 1.0 --meta-families 4 \
    --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
    --freeze-basis-at 8 --operator-slots 12 --sleeps 16 24 32 48 64 \
    --lifecycle --output "$out" >> tools/v5_h29.log 2>&1
  echo "world $w done" >> tools/v5_h29.log
done
echo H29_DONE >> tools/v5_h29.log
