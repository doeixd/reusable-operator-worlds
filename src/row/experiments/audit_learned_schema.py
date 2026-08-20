"""H20b G2: does a schema pay over the LEARNED library, at matched bits?

H20a asked whether a schema economy exists at all, using the teacher's
family operators. It could fit a schema directly to parameters because
those operators share hidden features by construction, so mixing `U`
matrices mixes functions. Learned abstractions have no shared frame: two
that compute the same function can have unrelated 198-vectors, and
fitting a schema to those vectors would be fitting coordinates. That is
the gauge-freedom trap the spec forbids.

So this scorer splits the two spaces:

    FIT in EFFECT space      an abstraction is identified with what it
                             computes on a probe set, which is invariant
                             to how it is parameterized
    CHARGE in PARAMETER space  bits are paid for what must be STORED,
                             which is the 198 scalars, not the effect

An abstraction applies `u_s . tanh(v_s z + b_s)` at each of three steps,
so its functional signature is computable from the flat vector alone --
no world, no learner, no teacher.

The ladder is deliberately conservative toward FACTORIZE, because this
project has a standing lesson (V4.2) about sharing claims scored with
the convenient accounting:

    COMPRESS    every atom coded privately to the distortion budget
    FACTORIZE   the schema, coded to the same budget, plus for each
                member EITHER its arguments (if the frozen schema
                reproduces it within budget) OR its full private cost
                (if it does not)

No leftover code is invented. A member the schema fails to cover is
charged full price, so FACTORIZE is never flattered by a cheap residual
term that nothing in the codebase actually stores.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

LN2 = math.log(2.0)
ARGUMENT_BITS = 8.0
DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)


def split(flat: np.ndarray, d: int, rank: int, steps: int):
    u_size, v_size, b_size = steps * d * rank, steps * rank * d, steps * rank
    if flat.size != u_size + v_size + b_size:
        raise ValueError(
            f"abstraction has {flat.size} scalars, expected "
            f"{u_size + v_size + b_size} for d={d} rank={rank} steps={steps}"
        )
    u = flat[:u_size].reshape(steps, d, rank)
    v = flat[u_size:u_size + v_size].reshape(steps, rank, d)
    b = flat[u_size + v_size:].reshape(steps, rank)
    return u, v, b


def effect(flat: np.ndarray, probe: np.ndarray, d: int, rank: int, steps: int):
    """What the abstraction COMPUTES, concatenated over its steps."""

    u, v, b = split(flat, d, rank, steps)
    pieces = []
    for step in range(steps):
        hidden = np.tanh(probe @ v[step].T + b[step])
        pieces.append(hidden @ u[step].T)
    return np.concatenate(pieces, axis=1).ravel()


def quantize(flat: np.ndarray, bits: int) -> np.ndarray:
    levels = max(2, 2 ** bits)
    scale = float(np.abs(flat).max())
    if scale == 0:
        return flat.copy()
    step = 2 * scale / (levels - 1)
    return np.round(flat / step) * step


def private_bits(flat, probe, d, rank, steps, budget) -> tuple[float, float]:
    """Bits/scalar to store this atom so its FUNCTION survives."""

    reference = effect(flat, probe, d, rank, steps)
    previous = None
    for bits in DEPTHS:
        error = float(np.mean(
            (effect(quantize(flat, bits), probe, d, rank, steps) - reference) ** 2
        ))
        if error <= budget:
            if previous is None:
                return float(bits) * flat.size, float(bits)
            x0, y0 = previous
            y1 = math.log(max(error, 1e-30))
            target = math.log(budget)
            depth = bits if y0 == y1 else x0 + (bits - x0) * (y0 - target) / (y0 - y1)
            return float(depth) * flat.size, float(depth)
        previous = (float(bits), math.log(max(error, 1e-30)))
    return 8.0 * flat.size, 8.0


def fit_schema(effects: np.ndarray, rank: int):
    centre = effects.mean(axis=0)
    _, _, vt = np.linalg.svd(effects - centre, full_matrices=False)
    return centre, vt[:rank]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_h20b"))
    parser.add_argument("--conditions", nargs="+", default=["r0", "r100"])
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=2)
    parser.add_argument("--task-steps", type=int, default=3)
    parser.add_argument("--schema-rank", type=int, default=2)
    parser.add_argument("--calibration", type=int, default=3, help="M_0")
    parser.add_argument("--probe", type=int, default=256)
    parser.add_argument("--distortion", type=float, default=1e-4)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_learned_schema.json"))
    args = parser.parse_args()

    d, rank, steps = args.state_dim, args.residual_rank, args.task_steps
    rows = []
    for condition in args.conditions:
        for world in args.worlds:
            path = args.root / condition / f"world_{world}" / "lifecycle"
            if not (path / "model.pt").exists():
                continue
            state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
            keys = sorted(
                (k for k in state if k.startswith("abstractions.")),
                key=lambda k: int(k.split(".")[1]),
            )
            atoms = [state[k].detach().cpu().numpy().astype(np.float64) for k in keys]
            if len(atoms) <= args.calibration + 1:
                rows.append({"condition": condition, "world": world,
                             "reason": f"only {len(atoms)} atoms, need > "
                                       f"{args.calibration + 1}"})
                continue
            probe = np.random.default_rng(1000 + world).normal(size=(args.probe, d))

            costs = [private_bits(a, probe, d, rank, steps, args.distortion)
                     for a in atoms]
            atom_bits = [c[0] for c in costs]
            effects = np.stack([effect(a, probe, d, rank, steps) for a in atoms])

            centre, basis = fit_schema(effects[: args.calibration], args.schema_rank)
            # The schema is realized as (K + 1) objects of the same shape
            # an abstraction has, coded at the same budget. Using the
            # calibration atoms' own depth is the honest charge: the
            # schema is no cheaper per scalar than what it replaces.
            schema_depth = float(np.mean([c[1] for c in costs[: args.calibration]]))
            schema_bits = (args.schema_rank + 1) * atoms[0].size * schema_depth

            covered, member_bits, savings = 0, [], []
            for index in range(args.calibration, len(atoms)):
                centred = effects[index] - centre
                coefficients = centred @ basis.T
                residual = float(np.mean((centred - coefficients @ basis) ** 2))
                if residual <= args.distortion:
                    covered += 1
                    cost = ARGUMENT_BITS * args.schema_rank
                else:
                    # Not covered: charged full private price. No
                    # leftover code is invented for it.
                    cost = atom_bits[index]
                member_bits.append(cost)
                savings.append(atom_bits[index] - cost)

            unseen = len(atoms) - args.calibration
            s_bar = float(np.mean(savings)) if savings else 0.0
            rows.append({
                "condition": condition, "world": world,
                "atoms": len(atoms), "unseen": unseen,
                "covered": covered,
                "mean_private_bits": float(np.mean(atom_bits)),
                "mean_member_bits": float(np.mean(member_bits)),
                "schema_bits": schema_bits,
                "s_bar_schema": s_bar,
                "compress_total": float(sum(atom_bits[args.calibration:])),
                "factorize_total": schema_bits + float(sum(member_bits)),
            })

    print("H20b G2 — schema over the LEARNED library, matched budget")
    print("  fit in effect space (gauge-free), charge in parameter space")
    print(f"  M_0={args.calibration}  K={args.schema_rank}  probe={args.probe}"
          f"  distortion={args.distortion:g}\n")
    print(f"  {'condition':<10} {'world':>5} {'M':>3} {'covered':>9} "
          f"{'COMPRESS':>10} {'FACTORIZE':>10} {'winner':>10}")
    summary = {}
    for condition in args.conditions:
        cells = [r for r in rows if r["condition"] == condition and "reason" not in r]
        for row in [r for r in rows if r["condition"] == condition]:
            if "reason" in row:
                print(f"  {condition:<10} {row['world']:>5}   unscoreable: {row['reason']}")
                continue
            winner = ("FACTORIZE" if row["factorize_total"] < row["compress_total"]
                      else "COMPRESS")
            print(f"  {condition:<10} {row['world']:>5} {row['atoms']:>3} "
                  f"{row['covered']:>4}/{row['unseen']:<4} "
                  f"{row['compress_total']:>10,.0f} {row['factorize_total']:>10,.0f} "
                  f"{winner:>10}")
        if cells:
            wins = sum(1 for c in cells
                       if c["factorize_total"] < c["compress_total"])
            summary[condition] = {
                "worlds": len(cells),
                "factorize_wins": wins,
                "mean_covered_fraction": float(np.mean(
                    [c["covered"] / max(c["unseen"], 1) for c in cells])),
                "mean_compress": float(np.mean([c["compress_total"] for c in cells])),
                "mean_factorize": float(np.mean([c["factorize_total"] for c in cells])),
            }

    print("\n  G2 VERDICT (FACTORIZE must beat matched-budget COMPRESS)")
    for condition, row in summary.items():
        print(f"    {condition:<8} FACTORIZE wins {row['factorize_wins']}/"
              f"{row['worlds']} worlds; schema covers "
              f"{row['mean_covered_fraction']:.0%} of unseen atoms; "
              f"{row['mean_factorize']:,.0f} vs {row['mean_compress']:,.0f} bits")
    if not summary:
        print("    no scoreable cells")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cells": rows}, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
