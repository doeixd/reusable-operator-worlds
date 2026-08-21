"""Is the learner's basis WORSE, or merely DIFFERENT?

The fragmentation audit ruled out oversegmentation: no small subset of
promoted atoms reconstructs a teacher family operator. That leaves the
possibility (review 48, Hypothesis B) that PROMOTE found a different but
equally economical decomposition, and that asking it to recover the
teacher's is asking the wrong question.

The right question is not

    A_f^teacher ~= A_i^learned ?

but whether the two representations describe the SAME lifetime at
comparable total cost:

    J_rep(L) = D*(L) + D*(per-task remainder | L)   at matched behaviour

Both libraries are scored the same way. For every family task, the
target is the innovation that task actually needs at the step where the
family fires, taken from the world rather than from either library. Each
library gets to pick its best object for that task and code whatever it
cannot express. Every piece is coded to the same distortion budget, so
neither side can buy accuracy with precision the other is denied.

Three outcomes, registered in advance (review 48):

    B1  learner cheaper        PROMOTE found a better basis than the
                               teacher's, and teacher recovery was never
                               the right objective
    B2  roughly tied           present-task MDL UNDERDETERMINES the
                               representation, which is what would make
                               abstraction fertility a real quantity
    B3  teacher-aligned cheaper  PROMOTE leaves global compression on
                               the table, pointing at a myopic objective

A tie is not a null here. It is the outcome that would motivate the
whole fertility programme.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_learned_schema import effect, private_bits
from row.experiments.audit_meta_recurrence import residual_effect
from row.meta_world import MetaFamilySpec, family_operators, generate_meta_world

ATOM_SCALARS = 198


def coding_depth(error: float, budget: float, floor: float = 1.0) -> float:
    """Bits/scalar to drive a residual of this size under the budget.

    Quantization error falls ~4x per bit, so the depth needed scales as
    half the log2 of the ratio. Returns 0 when the residual is already
    inside the budget: nothing has to be stored.
    """

    if error <= budget:
        return 0.0
    return max(floor, 0.5 * math.log2(error / budget))


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
    parser.add_argument("--probe", type=int, default=128)
    parser.add_argument("--distortion", type=float, default=1e-4)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_library_economy.json"))
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
            generated = generate_meta_world(world_config, spec)

            # Per family task: the state at the step where the family
            # fires, and the innovation the task actually needs there.
            # Both come from the WORLD, so neither library is scored
            # against the other's notion of the target.
            targets, states = [], []
            for index, task in enumerate(generated.tasks):
                family = spec.family_of(index)
                if family is None:
                    continue
                z = task.eval_x[: args.probe]
                for step in task.program.primitive_ids[:-1]:
                    z = task.teacher_library[step](z)
                states.append(z)
                targets.append((family, residual_effect(teachers[family], z)))

            # --- library bits, both sides at the same budget ---
            learner_library = sum(
                private_bits(a, states[0], d, rank, steps, args.distortion)[0]
                for a in atoms
            )
            teacher_library = 0.0
            for operator in teachers:
                reference = residual_effect(operator, states[0])
                scale = float(np.abs(operator.U).max())
                depth = 8.0
                for bits in (1, 2, 3, 4, 5, 6, 7, 8):
                    step_size = 2 * scale / (max(2, 2 ** bits) - 1)
                    quantized = replace(
                        operator, U=np.round(operator.U / step_size) * step_size)
                    if float(np.mean(
                        (residual_effect(quantized, states[0]) - reference) ** 2
                    )) <= args.distortion:
                        depth = float(bits)
                        break
                teacher_library += operator.U.size * depth

            # --- per-task remainder under each library ---
            reference_bits = math.log2(max(len(atoms), 1))
            teacher_reference_bits = math.log2(max(len(teachers), 1))
            learner_task, teacher_task = 0.0, 0.0
            for (family, target), z in zip(targets, states):
                # Learner: best atom at any step, then code the leftover.
                pool = []
                for atom in atoms:
                    full = effect(atom, z, d, rank, steps)
                    width = target.size
                    for s in range(steps):
                        pool.append(full[s * width:(s + 1) * width])
                best = min(
                    float(np.mean((target - p * (target @ p) / max(p @ p, 1e-30)) ** 2))
                    for p in pool
                )
                learner_task += reference_bits + ATOM_SCALARS * coding_depth(
                    best, args.distortion)
                # Teacher-aligned: the family's own operator.
                own = residual_effect(teachers[family], z)
                residual = float(np.mean((target - own) ** 2))
                teacher_task += teacher_reference_bits + ATOM_SCALARS * coding_depth(
                    residual, args.distortion)

            rows.append({
                "condition": condition, "world": world,
                "atoms": len(atoms), "families": len(teachers),
                "learner_total": learner_library + learner_task,
                "teacher_total": teacher_library + teacher_task,
                "learner_library": learner_library, "teacher_library": teacher_library,
                "learner_task": learner_task, "teacher_task": teacher_task,
                "tasks": len(targets),
            })

    if not rows:
        print("no artifacts found")
        return

    print("WHOLE-LIBRARY ECONOMY — learner basis vs teacher-aligned, matched budget")
    print(f"  distortion {args.distortion:g}, per-piece coding, targets from the world\n")
    print(f"  {'cond':<6} {'atoms':>5} {'F':>3} {'learner tot':>12} "
          f"{'teacher tot':>12} {'ratio':>7} {'winner':>9}")
    summary = {}
    for condition in args.conditions:
        cells = [r for r in rows if r["condition"] == condition]
        if not cells:
            continue
        for row in cells:
            ratio = row["learner_total"] / max(row["teacher_total"], 1e-9)
            print(f"  {condition:<6} {row['atoms']:>5} {row['families']:>3} "
                  f"{row['learner_total']:>12,.0f} {row['teacher_total']:>12,.0f} "
                  f"{ratio:>7.2f} "
                  f"{'learner' if ratio < 1 else 'teacher':>9}")
        learner = float(np.mean([c["learner_total"] for c in cells]))
        teacher = float(np.mean([c["teacher_total"] for c in cells]))
        summary[condition] = {
            "learner_total": learner, "teacher_total": teacher,
            "ratio": learner / max(teacher, 1e-9),
            "learner_wins": sum(1 for c in cells
                                if c["learner_total"] < c["teacher_total"]),
            "worlds": len(cells),
        }

    print("\n  VERDICT against the registered outcomes")
    for condition, row in summary.items():
        ratio = row["ratio"]
        if ratio < 0.9:
            verdict = "B1 — learner basis is CHEAPER; teacher recovery was never the objective"
        elif ratio <= 1.1:
            verdict = "B2 — roughly TIED; present-task MDL underdetermines the representation"
        else:
            verdict = "B3 — teacher-aligned cheaper; PROMOTE leaves compression on the table"
        print(f"    {condition}: ratio {ratio:.2f} "
              f"({row['learner_wins']}/{row['worlds']} worlds to the learner)")
        print(f"      -> {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cells": rows}, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
