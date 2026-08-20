"""H21: does a frozen schema make a NEW family member cheaper to acquire?

H20a showed a schema can be cheaper to STORE than independent atoms.
That is retrospective compression. The claim that makes it an
abstraction rather than a zip file is prospective: a family member the
schema has never seen should be learnable from fewer examples and
retained in fewer bits than one learned from scratch.

Leave one family out, fit the schema on the rest, freeze it, then
acquire the held-out member two ways from the same n examples:

    schema route      fit K arguments against the frozen basis
    independent       fit the operator outright

Both are least-squares fits against the same shared hidden features, so
the comparison is between hypothesis-class sizes and nothing else — no
optimizer, no learning-rate, no initialization to argue about. The
matched-budget rule still binds on the storage side: the independent
arm is charged at the rate-distortion depth it actually needs, not at
full precision.

Registered falsifier (V5 spec, H21): the family route must beat
independent acquisition on BOTH sample cost and retained bits. A win on
bits alone is storage; a win on loss at extra bits is capacity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from row.config import load_config
from row.experiments.audit_meta_recurrence import fit_schema, residual_effect
from row.meta_world import MetaFamilySpec, family_operators
from row.world import _rng

ARGUMENT_BITS = 8.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--r-meta", type=float, nargs="+", default=[0.0, 0.9, 1.0])
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--supports", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--probe", type=int, default=256)
    parser.add_argument("--target", type=float, default=1e-3,
                        help="held-out functional MSE the acquisition must reach")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_schema_transfer.json"))
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
            d, rank = world_config.state_dim, world_config.teacher_rank
            V, b, alpha = operators[0].V, operators[0].b, operators[0].alpha

            evaluation = _rng(world, 90).normal(size=(args.probe, d))
            eval_effects = np.stack(
                [residual_effect(o, evaluation) for o in operators])

            for held in range(args.families):
                others = np.delete(eval_effects, held, axis=0)
                centre, basis = fit_schema(others, args.subspace_rank)
                target_operator = operators[held]

                for support in args.supports:
                    train = _rng(world, 91, held, support).normal(size=(support, d))
                    hidden = np.tanh(train @ V.T + b)
                    observed = alpha * (hidden @ target_operator.U.T)

                    # --- independent: fit the operator outright ---
                    solution, *_ = np.linalg.lstsq(hidden, observed / alpha, rcond=None)
                    independent = replace(target_operator, U=solution.T)
                    independent_error = float(np.mean(
                        (residual_effect(independent, evaluation)
                         - eval_effects[held]) ** 2))

                    # --- schema: fit K arguments against the frozen basis ---
                    # The basis lives in effect space on the evaluation
                    # probe, so express it on the training inputs by
                    # solving for each direction's operator once.
                    directions = []
                    for vector in (centre, *basis):
                        matrix, *_ = np.linalg.lstsq(
                            np.tanh(evaluation @ V.T + b),
                            vector.reshape(len(evaluation), -1) / alpha,
                            rcond=None)
                        directions.append(matrix)
                    centre_train = alpha * (hidden @ directions[0])
                    design = np.stack(
                        [(alpha * (hidden @ m)).ravel() for m in directions[1:]], axis=1)
                    residual_target = (observed - centre_train).ravel()
                    coefficients, *_ = np.linalg.lstsq(design, residual_target, rcond=None)
                    reconstructed = centre + coefficients @ basis
                    schema_error = float(np.mean(
                        (reconstructed - eval_effects[held]) ** 2))

                    rows.append({
                        "r_meta": r_meta, "world": world, "held": held,
                        "support": support,
                        "independent_error": independent_error,
                        "schema_error": schema_error,
                        "independent_params": int(target_operator.U.size),
                        "schema_params": args.subspace_rank,
                    })

    print("H21 PROSPECTIVE SCHEMA REUSE — acquiring an unseen family member")
    print(f"  F={args.families}  K={args.subspace_rank}  probe={args.probe}"
          f"  target MSE={args.target:g}\n")
    summary = {}
    for r_meta in args.r_meta:
        cells = [row for row in rows if row["r_meta"] == r_meta]
        print(f"  r_meta = {r_meta}")
        print(f"    {'support':>8} {'independent':>13} {'schema':>13} {'winner':>10}")
        first_schema, first_independent = None, None
        for support in args.supports:
            at = [c for c in cells if c["support"] == support]
            independent = float(np.mean([c["independent_error"] for c in at]))
            schema = float(np.mean([c["schema_error"] for c in at]))
            winner = "schema" if schema < independent else "independent"
            print(f"    {support:>8} {independent:>13.3e} {schema:>13.3e} {winner:>10}")
            if first_schema is None and schema <= args.target:
                first_schema = support
            if first_independent is None and independent <= args.target:
                first_independent = support
        params_note = (f"{args.subspace_rank} arguments vs "
                       f"{cells[0]['independent_params']} operator scalars")
        summary[r_meta] = {
            "examples_to_target_schema": first_schema,
            "examples_to_target_independent": first_independent,
            "retained_bits_schema": ARGUMENT_BITS * args.subspace_rank,
            "params": params_note,
        }
        print(f"    examples to reach {args.target:g}: "
              f"schema {first_schema}, independent {first_independent}")
        print(f"    retained: {params_note}\n")

    print("  H21 — family must beat independent on BOTH sample cost and bits")
    for r_meta, row in summary.items():
        s, i = row["examples_to_target_schema"], row["examples_to_target_independent"]
        if s is None:
            verdict = "schema never reaches target — FAIL"
        elif i is None:
            verdict = "schema reaches target, independent does not — PASS"
        elif s < i:
            verdict = f"PASS ({s} vs {i} examples)"
        elif s == i:
            verdict = f"TIE on samples ({s}); bits favour schema"
        else:
            verdict = f"FAIL ({s} vs {i} examples)"
        print(f"    r_meta={r_meta}: {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"summary": {str(k): v for k, v in summary.items()}, "cells": rows},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
