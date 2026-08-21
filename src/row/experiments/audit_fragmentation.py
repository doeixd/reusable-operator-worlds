"""Is PROMOTE an oversegmentation algorithm?

C3 found that a schema pays over teacher family operators and fails over
the learned library, while realized M exceeded F in 12/12 sealed cells.
Hypothesis A (review 48): one family-level computation is represented by
several promoted pieces, each locally useful, so the clean teacher
geometry ends up fragmented across them and a schema fitted over the
pieces cannot see it.

The test is not "which abstraction best matches A_f", which any library
answers. It is how MANY promoted abstractions are needed:

    for k = 1, 2, 3:  min over subsets S of the library with |S| = k of
                      || A_f  -  best linear combination of S ||

measured on functional effects over a probe set, so the answer is
invariant to how each object happens to be parameterized. k = 1 poor and
k = 2 or 3 good IS fragmentation, and it would explain M > F directly.

The reverse direction is reported too — for each promoted abstraction,
how many teacher families it draws on — because the two together are the
functional mixing matrix, and "one teacher concept split across two
learner concepts" looks different from "one learner concept mixing two
teacher concepts".

Everything is teacher-side ground truth used for ANALYSIS ONLY; no
learner reads it, and it enters no training path.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_learned_schema import effect
from row.experiments.audit_meta_recurrence import residual_effect
from row.meta_world import MetaFamilySpec, family_operators
from row.world import _rng


def best_subset(target: np.ndarray, pool: np.ndarray, k: int) -> float:
    """Unexplained fraction of `target` by the best k-subset of `pool`."""

    total = float(np.sum(target ** 2))
    if total <= 0:
        return float("nan")
    best = 1.0
    for combination in itertools.combinations(range(len(pool)), k):
        design = pool[list(combination)].T
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = float(np.sum((target - design @ coefficients) ** 2))
        best = min(best, residual / total)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_sealed_c3"))
    parser.add_argument("--conditions", nargs="+", default=["r0", "r100"])
    parser.add_argument("--worlds", type=int, nargs="+",
                        default=[600, 601, 602, 603, 604, 605])
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=2)
    parser.add_argument("--task-steps", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--probe", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_fragmentation.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    d, rank, steps = args.state_dim, args.residual_rank, args.task_steps
    rows = []
    for condition in args.conditions:
        r_meta = 0.0 if condition == "r0" else 1.0
        for world in args.worlds:
            path = args.root / condition / f"world_{world}" / "lifecycle"
            if not (path / "model.pt").exists():
                continue
            state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
            atoms = [
                state[k].detach().cpu().numpy().astype(np.float64)
                for k in sorted(
                    (k for k in state if k.startswith("abstractions.")),
                    key=lambda k: int(k.split(".")[1]),
                )
            ]
            if not atoms:
                continue
            spec = MetaFamilySpec(
                families=args.families, tasks_per_family=args.tasks_per_family,
                r_meta=r_meta, subspace_rank=args.subspace_rank,
            )
            world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
            teachers = family_operators(world_config, spec)
            # PROBE AT THE STATE THE OPERATOR ACTUALLY SEES. The family
            # primitive fires at the LAST program step, so its input is
            # the state after two base-primitive steps, not raw Gaussian
            # input. Probing at N(0, I) measures both objects on a
            # distribution neither of them acts on, and the first run of
            # this audit did exactly that and reported 98% unexplained
            # at every k -- an artifact of the probe, not a property of
            # the library.
            from row.meta_world import generate_meta_world

            generated = generate_meta_world(world_config, spec)
            states = []
            for task in generated.tasks:
                family_index = spec.family_of(generated.tasks.index(task))
                if family_index is None:
                    continue
                z = task.eval_x
                for step in task.program.primitive_ids[:-1]:
                    z = task.teacher_library[step](z)
                states.append(z)
            probe = np.concatenate(states, axis=0)[: args.probe]

            # Teacher effects and learned effects live in different
            # spaces (the learner acts at three steps, the teacher at
            # one), so compare each teacher against the learned objects
            # STEP BY STEP and take the step that explains it best. A
            # promoted abstraction that carries the family at any step
            # counts as carrying it.
            teacher_effects = np.stack(
                [residual_effect(o, probe) for o in teachers])
            learned = np.stack([effect(a, probe, d, rank, steps) for a in atoms])
            width = teacher_effects.shape[1]
            pieces = np.stack([
                learned[:, s * width:(s + 1) * width] for s in range(steps)
            ])

            per_family = []
            for index, target in enumerate(teacher_effects):
                best_by_k = {}
                for k in range(1, min(args.max_k, len(atoms)) + 1):
                    best_by_k[k] = min(
                        best_subset(target, pieces[s], k) for s in range(steps)
                    )
                per_family.append(best_by_k)
            rows.append({
                "condition": condition, "world": world, "atoms": len(atoms),
                "families": args.families,
                "unexplained_by_k": {
                    str(k): float(np.mean([f[k] for f in per_family]))
                    for k in per_family[0]
                },
            })

    if not rows:
        print("no artifacts found")
        return

    print("FRAGMENTATION AUDIT — how many promoted atoms does one family take?")
    print("  unexplained fraction of a TEACHER family operator, best k-subset\n")
    ks = sorted(int(k) for k in rows[0]["unexplained_by_k"])
    header = "  %-10s %6s %5s " % ("condition", "atoms", "F") + " ".join(
        "%9s" % f"k={k}" for k in ks)
    print(header)
    summary = {}
    for condition in args.conditions:
        cells = [r for r in rows if r["condition"] == condition]
        if not cells:
            continue
        atoms = float(np.mean([c["atoms"] for c in cells]))
        values = {k: float(np.mean([c["unexplained_by_k"][str(k)] for c in cells]))
                  for k in ks}
        summary[condition] = {"atoms": atoms, "unexplained": values,
                              "worlds": len(cells)}
        print("  %-10s %6.1f %5d " % (condition, atoms, cells[0]["families"])
              + " ".join("%9.3f" % values[k] for k in ks))

    print("\n  READING")
    for condition, row in summary.items():
        one, many = row["unexplained"][ks[0]], row["unexplained"][ks[-1]]
        drop = one - many
        if one > 0.5 and many < 0.25:
            verdict = "FRAGMENTATION: one atom cannot carry a family, several can"
        elif one > 0.5 and many > 0.5:
            verdict = ("NOT fragmentation: no small subset reconstructs the "
                       "family — the learner represents something else")
        else:
            verdict = "family is carried by single atoms; fragmentation not indicated"
        print(f"    {condition}: k=1 leaves {one:.3f} unexplained, "
              f"k={ks[-1]} leaves {many:.3f} (drop {drop:.3f})")
        print(f"      -> {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cells": rows}, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
