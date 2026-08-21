#!/usr/bin/env bash
# V6.1 pressure sweep. At weight 1.0 the prospective arm differed from
# ordinary by 54 nats in 191,907 (0.03%), which is too small an
# intervention for a null to mean anything. H35 registers that the
# pressure has an optimum; this looks for the range where it bites.
cd "$(dirname "$0")/.."
for wgt in 10 100; do
  for w in 0 1 2; do
    out="artifacts/v6_pressure/w${wgt}/world_${w}/lifecycle"
    [ -f "$out/summary.json" ] && continue
    python -m row.experiments.mixed_lifetime --config configs/v5_h72.yaml \
      --model prospective --world-seed "$w" --r-meta 1.0 --meta-families 4 \
      --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8 \
      --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle \
      --arm prospective --prospective-steps 4 --prospective-weight "$wgt" \
      --output "$out" >> tools/v6_pressure.log 2>&1
    echo "w=$wgt world $w exit=$?" >> tools/v6_pressure.log
  done
done
echo V6_PRESSURE_DONE >> tools/v6_pressure.log
