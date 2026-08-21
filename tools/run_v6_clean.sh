#!/usr/bin/env bash
# V6.1, first valid run. Every review-55 fix applied:
#   * siblings drawn from held-out tasks (no future-label leak)
#   * replay matched to ordinary acquisition learning rates
#   * inner adaptation uses Adam and demonstrably adapts
#   * protocol knobs fingerprinted; resume refuses a mismatched arm
# Serialized: one lifetime at a time, per the Windows constraint.
cd "$(dirname "$0")/.."
for w in 0 1 2; do
  for arm in ordinary replay prospective supervised; do
    out="artifacts/v6_clean/${arm}/world_${w}/lifecycle"
    [ -f "$out/summary.json" ] && continue
    python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
      --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
      --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
      --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle --arm "$arm" \
      --prospective-steps 8 --prospective-inner-steps 8 \
      --output "$out" >> tools/v6_clean.log 2>&1
    echo "world $w $arm exit=$?" >> tools/v6_clean.log
  done
done
echo V6_CLEAN_DONE >> tools/v6_clean.log
