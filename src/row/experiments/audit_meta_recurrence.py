"""H20 world-validity and balance gates, teacher-side and offline.

Run BEFORE any learner touches the meta-recurrence world. Two questions,
and the spec makes both preconditions rather than diagnostics:

VALIDITY — does the knob install the structure it claims? Measured as
functional shared-subspace capture, NOT pairwise correlation. With
A_1 = B[1,0] and A_2 = B[0,1] the two operators are coordinates of the
same two-dimensional family — maximally related in the sense H20 is
about — and their behavioral correlation is zero, so a correlation gate
would void a perfectly valid generator (review 47).

    R_meta = 1 - sum_f |A_f - S_hat(alpha_f)|^2 / sum_f |A_f - A_bar|^2

fitted on one probe set and evaluated on a DISJOINT one, and the
reported number is the leave-one-family-out version:

    R_LOO  = 1 - |A_held - S_hat(alpha_held)|^2 / |A_held - A_bar|^2

with S_hat fitted on the other families only. R_meta large while R_LOO
is small means the subspace is being memorized, not shared.

BALANCE — does the knob change anything OTHER than relatedness? V5.1
failed its registered proportionality because residual rank moved cost
and utility together, and the same confound one level up would move the
phase boundary for a reason that has nothing to do with schemas. So the
three generator-side quantities must hold within 10% across the sweep:
per-abstraction D* (its rate-distortion cost), per-use saving, and
behavioral contribution. Promotion rate is NOT among them — it is a
learner response, and gating on it would discard the finding that a
cheaper lower-level representation already absorbed the regularity.

Everything here is teacher-side and cheap: no lifetimes, no learner.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from row.config import load_config
from row.meta_world import MetaFamilySpec, family_operators
from row.world import _rng, _spectral_normalize

LN2 = math.log(2.0)
NULL_DRAWS = 100
NULL_PERCENTILE = 95


def residual_effect(operator, probe: np.ndarray) -> np.ndarray:
    """The operator's own contribution, flattened.

    Not the output: the output passes through a tanh shared with every
    other primitive, and comparing outputs would credit the family with
    the squashing every operator does.
    """

    hidden = np.tanh(probe @ operator.V.T + operator.b)
    return (operator.alpha * (hidden @ operator.U.T)).ravel()


def fit_schema(effects: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre plus a rank-K functional subspace, by SVD.

    `C_hat` is the family mean and is FITTED, never supplied: the
    generator has no fixed centre (V5 spec, H20). At r_meta = 0 it
    degenerates to the mean of unrelated operators and buys nothing,
    which is exactly what the structureless control checks.
    """

    centre = effects.mean(axis=0)
    centred = effects - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centre, vt[:rank]


def capture(effects: np.ndarray, centre: np.ndarray, basis: np.ndarray) -> float:
    centred = effects - centre
    projected = centred @ basis.T @ basis
    residual = float(np.sum((centred - projected) ** 2))
    total = float(np.sum(centred ** 2))
    return 1.0 - residual / total if total > 0 else 0.0


def loo_capture(effects: np.ndarray, rank: int) -> float:
    """Fit on the other families, score the held-out one."""

    scores = []
    count = len(effects)
    if count <= rank + 1:
        return float("nan")
    for index in range(count):
        others = np.delete(effects, index, axis=0)
        centre, basis = fit_schema(others, rank)
        held = effects[index] - centre
        coefficients = held @ basis.T
        residual = float(np.sum((held - coefficients @ basis) ** 2))
        total = float(np.sum(held ** 2))
        scores.append(1.0 - residual / total if total > 0 else 0.0)
    return float(np.mean(scores))


def isotropic_null(config, spec, probe: np.ndarray, rank: int) -> float:
    """95th percentile of LOO capture on matched INDEPENDENT operators.

    A rank-K fit to any M operators returns a positive R^2 by dimension
    counting alone; without this the existence gate passes on noise,
    which is the V4.2 error.
    """

    d, r = config.state_dim, config.teacher_rank
    scores = []
    for draw in range(NULL_DRAWS):
        generator = _rng(config.seed, 70, draw)
        V = _spectral_normalize(generator.normal(size=(r, d)))
        b = generator.normal(scale=0.2, size=r)
        hidden = np.tanh(probe @ V.T + b)
        effects = np.stack([
            (config.alpha * (hidden @ _spectral_normalize(
                generator.normal(size=(d, r))).T)).ravel()
            for _ in range(spec.families)
        ])
        scores.append(loo_capture(effects, rank))
    return float(np.nanpercentile(scores, NULL_PERCENTILE))


def rate_distortion_bits(operator, probe: np.ndarray, budget: float) -> float:
    """Bits/scalar whose functional error meets the budget, INTERPOLATED.

    Charged against the operator's own contribution, so the tolerance is
    contribution-relative rather than a fraction of total output scale.

    Returned as a real number rather than the first integer depth that
    clears the budget. Integer depths make this measure jump by whole
    bits, and averaged over a handful of families that granularity alone
    produced a 16.4% spread across r_meta — a balance-gate failure that
    was an artifact of the instrument, not a property of the generator.
    Quantization error falls ~4x per added bit, so interpolating log2
    error against depth recovers a continuous rate.
    """

    reference = residual_effect(operator, probe)
    scale = float(np.abs(operator.U).max())
    depths, errors = [], []
    for bits in (1, 2, 3, 4, 5, 6, 7, 8):
        levels = max(2, 2 ** bits)
        step = 2 * scale / (levels - 1)
        quantized = replace(operator, U=np.round(operator.U / step) * step)
        error = float(np.mean((residual_effect(quantized, probe) - reference) ** 2))
        depths.append(float(bits))
        errors.append(max(error, 1e-30))
        if error <= budget:
            if len(depths) == 1:
                return float(bits)
            # Interpolate in (bits, log error) between the last depth
            # that missed and this one, which met it.
            x0, x1 = depths[-2], depths[-1]
            y0, y1 = math.log(errors[-2]), math.log(errors[-1])
            target = math.log(budget)
            if y0 == y1:
                return float(bits)
            return x0 + (x1 - x0) * (y0 - target) / (y0 - y1)
    return 8.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--r-meta", type=float, nargs="+",
                        default=[0.0, 0.5, 0.9, 1.0])
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--probe", type=int, default=512)
    parser.add_argument("--distortion", type=float, default=1e-4,
                        help="functional MSE budget for the D* reading")
    parser.add_argument("--balance-tolerance", type=float, default=0.10)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_meta_recurrence_gates.json"))
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
            fit_probe = _rng(world, 80).normal(
                size=(args.probe, world_config.state_dim))
            # Disjoint evaluation probe: an in-sample fit would report
            # capture that a held-out input never sees.
            eval_probe = _rng(world, 81).normal(
                size=(args.probe, world_config.state_dim))

            fit_effects = np.stack([residual_effect(o, fit_probe) for o in operators])
            eval_effects = np.stack([residual_effect(o, eval_probe) for o in operators])
            centre, basis = fit_schema(fit_effects, args.subspace_rank)
            # Re-express the fitted subspace on the evaluation probe.
            eval_centre, eval_basis = fit_schema(eval_effects, args.subspace_rank)
            rows.append({
                "r_meta": r_meta,
                "world": world,
                "r_in_sample": capture(eval_effects, eval_centre, eval_basis),
                "r_loo": loo_capture(eval_effects, args.subspace_rank),
                "null_loo": isotropic_null(
                    world_config, spec, eval_probe, args.subspace_rank),
                "contribution": [
                    float(np.mean(e ** 2)) for e in eval_effects
                ],
                "bits": [
                    rate_distortion_bits(o, eval_probe, args.distortion)
                    for o in operators
                ],
                "norm": [float(np.linalg.norm(o.U)) for o in operators],
            })

    print("H20 META-RECURRENCE GATES (teacher-side, no learner)")
    print(f"  F={args.families}  m={args.tasks_per_family}  K={args.subspace_rank}"
          f"  probe={args.probe}  distortion budget={args.distortion:g}\n")
    print(f"  {'r_meta':>7} {'R_LOO':>8} {'null':>8} {'R_in':>8} "
          f"{'contribution':>13} {'D* bits':>9} {'|U|':>7}")
    summary = {}
    for r_meta in args.r_meta:
        cells = [row for row in rows if row["r_meta"] == r_meta]
        loo = float(np.mean([c["r_loo"] for c in cells]))
        null = float(np.mean([c["null_loo"] for c in cells]))
        insample = float(np.mean([c["r_in_sample"] for c in cells]))
        contribution = float(np.mean([np.mean(c["contribution"]) for c in cells]))
        bits = float(np.mean([np.mean(c["bits"]) for c in cells]))
        norm = float(np.mean([np.mean(c["norm"]) for c in cells]))
        summary[r_meta] = {
            "r_loo": loo, "null": null, "r_in_sample": insample,
            "contribution": contribution, "bits": bits, "norm": norm,
            "worlds": len(cells),
        }
        print(f"  {r_meta:>7.2f} {loo:>8.3f} {null:>8.3f} {insample:>8.3f} "
              f"{contribution:>13.5f} {bits:>9.2f} {norm:>7.3f}")

    print("\n  VALIDITY")
    lowest = min(summary)
    at_zero = summary[lowest]
    null_ok = at_zero["r_loo"] <= at_zero["null"]
    print(f"    r_meta={lowest:.2f} within the isotropic null      "
          f"{'PASS' if null_ok else 'FAIL'}  "
          f"(R_LOO {at_zero['r_loo']:.3f} vs null {at_zero['null']:.3f})")
    order = [summary[r]["r_loo"] for r in sorted(summary)]
    monotone = all(b >= a - 1e-9 for a, b in zip(order, order[1:]))
    print(f"    R_LOO monotone in r_meta                  "
          f"{'PASS' if monotone else 'FAIL'}  "
          f"({', '.join(f'{v:.3f}' for v in order)})")
    highest = max(summary)
    exists = summary[highest]["r_loo"] > summary[highest]["null"]
    print(f"    r_meta={highest:.2f} exceeds the null            "
          f"{'PASS' if exists else 'FAIL'}  "
          f"(R_LOO {summary[highest]['r_loo']:.3f} vs "
          f"null {summary[highest]['null']:.3f})")

    print(f"\n  BALANCE GATES (each within {args.balance_tolerance:.0%} across the sweep)")
    balance_ok = True
    for name in ("contribution", "bits", "norm"):
        values = [summary[r][name] for r in sorted(summary)]
        spread = (max(values) - min(values)) / max(np.mean(values), 1e-12)
        ok = spread <= args.balance_tolerance
        balance_ok &= ok
        print(f"    {name:<13} spread {spread:6.1%}  "
              f"{'PASS' if ok else 'FAIL'}   "
              f"({', '.join(f'{v:.4g}' for v in values)})")
    print("    promotion rate is deliberately NOT gated: it is a learner")
    print("    response and is reported as an H20b outcome instead.")

    verdict = null_ok and monotone and exists and balance_ok
    print(f"\n  GATES: {'PASS — the sweep is scoreable' if verdict else 'FAIL — unscoreable, redesign before any FACTORIZE oracle'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"summary": {str(k): v for k, v in summary.items()}, "cells": rows,
         "verdict": "PASS" if verdict else "FAIL"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
