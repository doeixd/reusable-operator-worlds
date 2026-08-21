"""Did prospective pressure crush the coordinates that identify a member?

V6.1 found Phi_prospective = -8.58, harm concentrated on RELATED futures
and largest at the smallest support. Two mechanisms predict that pattern
and Phi alone cannot separate them (review 56):

    OVER-ALIGNMENT   the representation collapsed family members toward
                     a shared mean, erasing the coordinates that
                     distinguish them
    CONDITIONING     adaptation simply became harder to optimize,
                     without any collapse

This module measures the difference directly, on frozen artifacts.

    DISCRIMINATION   D = d_between / d_within, where d_between is the
                     mean squared functional distance between family
                     members and d_within is the variation of one member
                     across its own inputs. Over-alignment predicts
                     D falls under prospective pressure.

    SENSITIVITY      ||df / dc_task||, how much the model's output moves
                     when the task code moves. This is the few-shot
                     variable: adaptation identifies a new member BY
                     moving the code, so a representation that has
                     become less responsive to it is literally harder to
                     specialize, however good its shared prior.

Sensitivity is the sharper test. A strong prior that ignores its
argument channel is exactly "made adaptation unnecessary rather than
effective", which is the failure review 56 predicts the outer objective
would drift toward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_effective_operator import load_learner, rollout
from row.meta_world import MetaFamilySpec, generate_meta_world

ARMS = ("ordinary", "replay", "prospective", "supervised")


def member_effects(model, tasks, probe: torch.Tensor, step: int) -> np.ndarray:
    """Each member's functional output on a COMMON input set."""

    out = []
    for task in tasks:
        with torch.no_grad():
            out.append(model(probe, task.task_id).cpu().numpy().ravel())
    return np.stack(out)


def code_sensitivity(model, task_id: str, probe: torch.Tensor,
                     epsilon: float = 0.05, draws: int = 8,
                     seed: int = 0) -> float:
    """||df/dc|| estimated by finite differences on the task code.

    The task code is the channel adaptation actually moves, so this is
    the responsiveness few-shot learning depends on.
    """

    code = model.task_codes[task_id]
    base = code.detach().clone()
    with torch.no_grad():
        reference = model(probe, task_id).clone()
    generator = torch.Generator().manual_seed(seed)
    deltas = []
    for _ in range(draws):
        direction = torch.randn(base.shape, generator=generator)
        direction = direction / direction.norm().clamp_min(1e-12)
        with torch.no_grad():
            code.copy_(base + epsilon * direction)
            moved = model(probe, task_id)
            deltas.append(float((moved - reference).norm()) / epsilon)
            code.copy_(base)
    return float(np.mean(deltas))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v6_clean"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--probe", type=int, default=64)
    parser.add_argument("--members", type=int, default=6,
                        help="members per family compared")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v6_identifiability.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for world in args.worlds:
        spec = MetaFamilySpec(
            families=args.families, tasks_per_family=args.tasks_per_family,
            r_meta=1.0, subspace_rank=args.subspace_rank,
        )
        world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
        generated = generate_meta_world(world_config, spec)
        for arm in args.arms:
            path = args.root / arm / f"world_{world}" / "lifecycle"
            if not (path / "model.pt").exists():
                continue
            model = load_learner(config, path, args.slots, kind="prospective")
            probe = torch.tensor(
                generated.tasks[0].eval_x[: args.probe], dtype=torch.float32)

            between, within, sensitivity = [], [], []
            for family in range(spec.families):
                start = spec.family_onset + family * spec.tasks_per_family
                members = [
                    generated.tasks[i]
                    for i in range(start, start + args.members)
                    if generated.tasks[i].task_id in model.task_codes
                ]
                if len(members) < 2:
                    continue
                effects = member_effects(model, members, probe,
                                         model.task_steps - 1)
                centre = effects.mean(axis=0)
                # BETWEEN: how far apart family members are functionally.
                between.append(float(np.mean(
                    [np.sum((e - centre) ** 2) for e in effects])))
                # WITHIN: how much one member varies across inputs, the
                # natural scale to normalize by.
                within.append(float(np.mean(
                    [np.var(e) * e.size for e in effects])))
                sensitivity.extend(
                    code_sensitivity(model, m.task_id, probe, seed=world)
                    for m in members)
            if not between:
                continue
            rows.append({
                "world": world, "arm": arm,
                "d_between": float(np.mean(between)),
                "d_within": float(np.mean(within)),
                "discrimination": float(np.mean(between) / max(np.mean(within), 1e-12)),
                "code_sensitivity": float(np.mean(sensitivity)),
            })

    if not rows:
        print("no artifacts scored")
        return

    print("V6 ARGUMENT IDENTIFIABILITY — did the member coordinates survive?\n")
    print(f"  {'arm':<12} {'discrimination':>15} {'||df/dc||':>11} {'worlds':>7}")
    summary = {}
    for arm in args.arms:
        cells = [r for r in rows if r["arm"] == arm]
        if not cells:
            continue
        discrimination = float(np.mean([c["discrimination"] for c in cells]))
        sensitivity = float(np.mean([c["code_sensitivity"] for c in cells]))
        summary[arm] = {"discrimination": discrimination,
                        "code_sensitivity": sensitivity,
                        "worlds": len(cells)}
        print(f"  {arm:<12} {discrimination:>15.4f} {sensitivity:>11.4f} "
              f"{len(cells):>7}")

    if "ordinary" in summary and "prospective" in summary:
        base, arm = summary["ordinary"], summary["prospective"]
        print("\n  MECHANISM")
        d_drop = (base["discrimination"] - arm["discrimination"]) / base["discrimination"]
        s_drop = (base["code_sensitivity"] - arm["code_sensitivity"]) / base["code_sensitivity"]
        print(f"    discrimination  {base['discrimination']:.4f} -> "
              f"{arm['discrimination']:.4f}  ({-d_drop:+.1%})")
        print(f"    ||df/dc||       {base['code_sensitivity']:.4f} -> "
              f"{arm['code_sensitivity']:.4f}  ({-s_drop:+.1%})")
        if s_drop > 0.1:
            print("    -> OVER-ALIGNMENT supported: the representation became")
            print("       materially LESS responsive to the task code, which is")
            print("       the channel few-shot adaptation moves. A strong prior")
            print("       that ignores its argument is 'adaptation made")
            print("       unnecessary rather than effective' (review 56).")
        elif d_drop > 0.1:
            print("    -> partial support: members are less separated but the")
            print("       argument channel is intact; collapse without loss of")
            print("       responsiveness.")
        else:
            print("    -> NOT supported. Neither discrimination nor code")
            print("       sensitivity fell materially, so the harm to Phi is")
            print("       more likely a CONDITIONING effect than a collapse.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cells": rows},
                                      indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
