#!/usr/bin/env bash
# H35's discriminating cell: is Phi negative because the pressure is
# PAST the optimum, or because it is below it?
#
# At 52% of one task's training movement Phi_related = -8.58 (0/3
# worlds) and the harm is largest at the smallest support, which is the
# signature of over-alignment rather than of too little pressure.
#
# Over-alignment predicts: as pressure falls, Phi rises toward zero and
# never turns positive.
# Too-little-pressure predicts: Phi turns positive somewhere below.
#
# 1 outer step is ~6% of a task's movement, 2 is ~13%, against the 8
# already run at ~52%.
cd "$(dirname "$0")/.."
for outer in 1 2; do
  for w in 0 1 2; do
    out="artifacts/v6_lowp/o${outer}/world_${w}/lifecycle"
    [ -f "$out/summary.json" ] && continue
    python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
      --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
      --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
      --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle \
      --arm prospective --prospective-steps "$outer" \
      --prospective-inner-steps 8 \
      --output "$out" >> tools/v6_lowp.log 2>&1
    echo "outer=$outer world $w exit=$?" >> tools/v6_lowp.log
  done
done
echo V6_LOWP_DONE >> tools/v6_lowp.log
