"""H20a: is a schema over abstractions ever worth its bits?

Exogenous atoms, no PROMOTE in the loop. The family operators are
supplied directly, so a failure here means "higher-order factorization
does not pay", not "the upstream birth mechanism moved with r_meta".
H20b asks the separate question of whether a learner reaches this
region; a pass requires H20a (V5 spec, D16).

The comparison is the registered ambition ladder at MATCHED BEHAVIOURAL
BUDGET, which is the constitutional rule this project adopted after
V4.2 passed a sharing gate at full precision and failed it at equal
bits:

    COMPRESS    store each A_i privately, coded to the distortion budget
    FACTORIZE   store S once, plus per-member arguments and leftovers,
                every piece coded to the SAME budget

The schema is fitted on a calibration set and then FROZEN. Refitting S
at every M would make D*(S) and s_bar_schema functions of M, and M*
would be predicted from quantities that depend on what it predicts —
the circularity review 47 caught in the original protocol. So:

    calibrate on M_0 members, freeze S
    on each UNSEEN member, s_i = D*(A_i) - [D(alpha_i) + D*(E_i)]
    predict  M* = D*(S) / mean(s_i)          <- H25, registered at 15%
    then add unseen members one at a time and find where FACTORIZE wins

Everything is teacher-side and offline: no lifetimes, no learner.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from row.config import load_config
from row.experiments.audit_meta_recurrence import (
    fit_schema,
    rate_distortion_bits,
    residual_effect,
)
from row.meta_world import MetaFamilySpec, family_operators
from row.world import _rng

LN2 = math.log(2.0)
# A schema argument is a handful of reals; they are coded at the same
# depth the operators are, so the comparison never smuggles in a cheaper
# code for the side the hypothesis favours.
ARGUMENT_BITS = 8.0


def matrix_bits(matrix: np.ndarray, bits_per_scalar: float) -> float:
    return float(matrix.size) * bits_per_scalar


def effect_bits(effect: np.ndarray, probe: np.ndarray, shape, budget: float,
                alpha: float, V: np.ndarray, b: np.ndarray) -> float:
    """Rate needed to code a functional effect to the budget.

    The effect is carried by a U-shaped matrix against shared hidden
    features, so coding the effect is coding that matrix.
    """

    hidden = np.tanh(probe @ V.T + b)
    # Least-squares recovery of the matrix that produces this effect.
    target = effect.reshape(probe.shape[0], -1) / alpha
    recovered, *_ = np.linalg.lstsq(hidden, target, rcond=None)
    operator = replace(_PROTO, U=recovered.T, V=V, b=b, alpha=alpha)
    depth = rate_distortion_bits(operator, probe, budget)
    return matrix_bits(recovered, depth), depth


_PROTO = None  # bound in main once a Primitive instance exists


def main() -> None:
    global _PROTO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--r-meta", type=float, nargs="+", default=[0.0, 0.9, 1.0])
    parser.add_argument("--families", type=int, default=12,
                        help="must exceed the calibration set so members are unseen")
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--calibration", type=int, default=4, help="M_0")
    parser.add_argument("--probe", type=int, default=256)
    parser.add_argument("--distortion", type=float, default=1e-4)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_schema_economics.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for r_meta in args.r_meta:
        for world in args.worlds:
            spec = MetaFamilySpec(
                families=args.families,
                tasks_per_family=args.tasks_per_family,
                r_meta=r_meta,
                subspace_rank=args.subspace_rank,
            )
            world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
            operators = family_operators(world_config, spec)
            _PROTO = operators[0]
            probe = _rng(world, 82).normal(size=(args.probe, world_config.state_dim))
            V, b, alpha = operators[0].V, operators[0].b, operators[0].alpha

            # --- COMPRESS: each atom coded privately to the budget ---
            private_depths = [
                rate_distortion_bits(o, probe, args.distortion) for o in operators
            ]
            private_bits = [
                matrix_bits(o.U, d) for o, d in zip(operators, private_depths)
            ]

            # --- FACTORIZE: schema fitted on the calibration set, FROZEN ---
            effects = np.stack([residual_effect(o, probe) for o in operators])
            centre, basis = fit_schema(effects[: args.calibration], args.subspace_rank)

            # Cost of the schema itself: the centre plus K basis
            # directions, each coded as an operator at the same budget.
            schema_bits = 0.0
            for vector in (centre, *basis):
                bits, _ = effect_bits(
                    vector, probe, None, args.distortion, alpha, V, b)
                schema_bits += bits

            member_bits, savings = [], []
            for index in range(args.calibration, args.families):
                centred = effects[index] - centre
                coefficients = centred @ basis.T
                leftover = centred - coefficients @ basis
                leftover_error = float(np.mean(leftover ** 2))
                if leftover_error <= args.distortion:
                    # The schema already meets the budget; no leftover
                    # needs storing, which is the case FACTORIZE wants.
                    leftover_bits = 0.0
                else:
                    leftover_bits, _ = effect_bits(
                        leftover, probe, None, args.distortion, alpha, V, b)
                cost = ARGUMENT_BITS * len(coefficients) + leftover_bits
                member_bits.append(cost)
                savings.append(private_bits[index] - cost)

            s_bar = float(np.mean(savings)) if savings else 0.0
            predicted = schema_bits / s_bar if s_bar > 0 else float("inf")
            # Observed crossing: smallest M whose cumulative saving pays
            # for the schema. Members are added ONE AT A TIME, S fixed.
            cumulative, observed = 0.0, None
            for count, saving in enumerate(savings, start=1):
                cumulative += saving
                if cumulative > schema_bits and observed is None:
                    observed = count
            rows.append({
                "r_meta": r_meta, "world": world,
                "schema_bits": schema_bits,
                "mean_private_bits": float(np.mean(private_bits)),
                "mean_member_bits": float(np.mean(member_bits)) if member_bits else None,
                "s_bar_schema": s_bar,
                "predicted_M": predicted,
                "observed_M": observed,
                "unseen_members": len(savings),
            })

    print("H20a SCHEMA ECONOMICS — exogenous atoms, frozen schema, matched budget")
    print(f"  F={args.families}  M_0={args.calibration}  K={args.subspace_rank}"
          f"  probe={args.probe}  distortion={args.distortion:g}\n")
    print(f"  {'r_meta':>7} {'D*(S)':>9} {'private/atom':>13} {'member':>9} "
          f"{'s_bar':>9} {'M* pred':>9} {'M* obs':>8}")
    summary = {}
    for r_meta in args.r_meta:
        cells = [row for row in rows if row["r_meta"] == r_meta]
        schema = float(np.mean([c["schema_bits"] for c in cells]))
        private = float(np.mean([c["mean_private_bits"] for c in cells]))
        member = float(np.mean([c["mean_member_bits"] for c in cells]))
        s_bar = float(np.mean([c["s_bar_schema"] for c in cells]))
        predicted = float(np.mean([c["predicted_M"] for c in cells]))
        seen = [c["observed_M"] for c in cells if c["observed_M"] is not None]
        observed = float(np.mean(seen)) if seen else None
        summary[r_meta] = {
            "schema_bits": schema, "private_bits": private, "member_bits": member,
            "s_bar_schema": s_bar, "predicted_M": predicted,
            "observed_M": observed, "worlds_with_crossing": len(seen),
            "worlds": len(cells),
        }
        shown = "none" if observed is None else f"{observed:.1f}"
        print(f"  {r_meta:>7.2f} {schema:>9,.0f} {private:>13,.0f} {member:>9,.0f} "
              f"{s_bar:>9,.0f} {predicted:>9.1f} {shown:>8}")

    print("\n  H25 — does the frozen-schema prediction hold?")
    for r_meta, row in summary.items():
        if row["observed_M"] is None:
            print(f"    r_meta={r_meta:.2f}: no crossing within "
                  f"{args.families - args.calibration} unseen members "
                  f"(predicted {row['predicted_M']:.1f}) — "
                  f"{'consistent' if row['predicted_M'] > args.families - args.calibration else 'MISS'}")
            continue
        error = abs(row["observed_M"] - row["predicted_M"]) / row["predicted_M"]
        print(f"    r_meta={r_meta:.2f}: observed {row['observed_M']:.1f} vs "
              f"predicted {row['predicted_M']:.1f}, error {error:.1%} "
              f"{'PASS' if error <= 0.15 else 'MISS (registered 15%)'}"
              f"  [{row['worlds_with_crossing']}/{row['worlds']} worlds]")

    print("\n  G2 — does FACTORIZE beat matched-budget COMPRESS at all?")
    for r_meta, row in summary.items():
        pays = row["member_bits"] < row["private_bits"]
        print(f"    r_meta={r_meta:.2f}: {row['member_bits']:,.0f} bits/member vs "
              f"{row['private_bits']:,.0f} private  "
              f"{'FACTORIZE cheaper' if pays else 'COMPRESS cheaper'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"summary": {str(k): v for k, v in summary.items()}, "cells": rows},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
