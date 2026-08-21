"""V6's primary endpoint: does a representation make future learning cheap?

    Phi(R) = E_{T'~F}[ C_adapt(T' | R_baseline) - C_adapt(T' | R) ]

Phi is the pass. `R_effective` is a secondary, mechanistic endpoint,
because V4 and V5 each punished this project once for treating geometry
as the result (review 52, failure mode 4). A prospective learner may
find a representation unlike the teacher's and better for future
learning; that would be a success, not a miss.

Four guards are built in rather than left to the analyst:

  * THE SAME OPTIMIZER ADAPTS EVERY ARM. Both representations are frozen
    and adapted by one standardized routine with identical steps and
    learning rate, so an advantage cannot be an optimizer that learned
    better curvature (failure mode 3).
  * SUPPORT AND QUERY ARE DISJOINT. Adaptation sees support only; the
    reported cost is query loss (failure mode 1).
  * THE EVALUATION SIBLING IS RESERVED. The runner never offers the last
    member of a family as a prospective sibling, so the task Phi is
    measured on never entered any prospective gradient.
  * RELATED AND UNRELATED FUTURES ARE BOTH MEASURED. Phi_specific =
    Phi_related - Phi_unrelated separates fertility from generic
    plasticity (failure mode 2, H31).

Only task-local parameters move during adaptation. The shared
representation is what is being judged, so it is frozen throughout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_effective_operator import load_learner
from row.meta_world import MetaFamilySpec, generate_meta_world

ARMS = ("ordinary", "replay", "prospective", "supervised")


def adapt_cost(model, task, steps: int, support: int, inner_lr: float,
               sigma: float = 0.1) -> float:
    """Query loss after adapting ONLY task-local parameters.

    The standardized routine. Every arm gets exactly this, from a frozen
    shared representation, so the number reflects the representation and
    not the optimizer that produced it.
    """

    probe_id = f"__eval_{task.task_id}"
    if probe_id in model.task_codes:
        model.forget_task(probe_id)
    model.begin_task(probe_id)
    code = model.task_codes[probe_id]
    residual = model.task_residuals[probe_id]
    for parameter in model.shared_parameters():
        parameter.requires_grad_(False)

    support_x = torch.tensor(task.train_x[:support], dtype=torch.float32)
    support_y = torch.tensor(task.train_y[:support], dtype=torch.float32)
    query_x = torch.tensor(task.eval_x, dtype=torch.float32)
    query_y = torch.tensor(task.eval_y, dtype=torch.float32)

    # Adam at the project's task learning rate, not plain SGD. The
    # first version used SGD(lr=0.05) for 8 steps and moved the loss by
    # nothing at all -- gradient norms are ~1e-3 here, so the adaptation
    # curve was flat in support size, which is a broken instrument
    # rather than an absence of fertility.
    optimizer = torch.optim.Adam([code, residual], lr=inner_lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.mean((model(support_x, probe_id) - support_y) ** 2)
        loss.backward(inputs=[code, residual])
        optimizer.step()
    with torch.no_grad():
        query = float(torch.mean((model(query_x, probe_id) - query_y) ** 2))
    model.forget_task(probe_id)
    # Reported in the project's currency: Gaussian nats per target
    # scalar at the standing sigma.
    return query / (2 * sigma * sigma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v6"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--support", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="a CURVE, not a point: a single generous support "
                             "size saturates and every arm reports the same "
                             "number, which is a measurement failure rather "
                             "than an absence of fertility")
    parser.add_argument("--inner-lr", type=float, default=0.05,
                        help="task learning rate, matching the lifetime's")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v6_fertility.json"))
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

        # RELATED futures: family members GENERATED BUT NEVER TRAINED ON.
        # Using an in-lifetime task here is what made the first version
        # of this scorer useless -- such a task already sits at 0.006
        # support loss, so adaptation has nothing to do and every arm
        # reports the same saturated number.
        # An unseen FAMILY, from the same shared subspace: the future
        # that actually has headroom. Members of seen families are
        # nearly free for every arm and cannot discriminate.
        related = list(getattr(generated, "novel_family_tasks", ()))
        # UNRELATED futures: pre-onset tasks belong to no family. They
        # ARE in the lifetime, which biases against finding specificity
        # (an unrelated task the arm trained on is unfairly cheap), so
        # H31 is conservative in the direction that matters.
        unrelated = [generated.tasks[i] for i in range(min(4, spec.family_onset))]
        if not related:
            continue

        for arm in args.arms:
            path = args.root / arm / f"world_{world}" / "lifecycle"
            if not (path / "model.pt").exists():
                continue
            model = load_learner(config, path, args.slots, kind="prospective")
            rows.append({
                "world": world, "arm": arm,
                "related": {
                    str(k): [adapt_cost(model, t, args.steps, k, args.inner_lr)
                             for t in related]
                    for k in args.support
                },
                "unrelated": {
                    str(k): [adapt_cost(model, t, args.steps, k, args.inner_lr)
                             for t in unrelated]
                    for k in args.support
                },
            })

    if not rows:
        print("no artifacts scored")
        return

    print("V6 FERTILITY — adaptation cost on futures the arm never trained on")
    print(f"  standardized adaptation: {args.steps} steps, {args.support} support "
          f"examples, lr {args.inner_lr}, shared parameters FROZEN\n")
    print("  related-future adaptation cost by support size")
    print(f"  {'arm':<12} " + " ".join(f"{'k='+str(k):>9}" for k in args.support)
          + f" {'worlds':>7}")
    summary = {}
    for arm in args.arms:
        cells = [r for r in rows if r["arm"] == arm]
        if not cells:
            continue
        related = {str(k): float(np.mean([np.mean(c["related"][str(k)])
                                          for c in cells]))
                   for k in args.support}
        unrelated = {str(k): float(np.mean([np.mean(c["unrelated"][str(k)])
                                            for c in cells]))
                     for k in args.support}
        summary[arm] = {"related": related, "unrelated": unrelated,
                        "worlds": len(cells)}
        print(f"  {arm:<12} " + " ".join(f"{related[str(k)]:>9,.2f}"
                                          for k in args.support)
              + f" {len(cells):>7}")

    if "ordinary" not in summary:
        print("\n  no ordinary arm: Phi is defined against it, so nothing to report")
        return
    base = summary["ordinary"]
    print("\n  Phi against the ordinary arm (positive = cheaper future learning)")
    print(f"  {'arm':<12} {'Phi related':>12} {'Phi unrelated':>14} "
          f"{'Phi specific':>13}")
    for arm, row in summary.items():
        if arm == "ordinary":
            continue
        # Score Phi at the SMALLEST support size, where representation
        # quality actually decides the outcome; larger k saturates.
        k = str(min(args.support))
        phi_r = base["related"][k] - row["related"][k]
        phi_u = base["unrelated"][k] - row["unrelated"][k]
        row["phi_related"], row["phi_unrelated"] = phi_r, phi_u
        row["phi_specific"] = phi_r - phi_u
        per_world = []
        for cell in [c for c in rows if c["arm"] == arm]:
            reference = next(c for c in rows if c["arm"] == "ordinary"
                             and c["world"] == cell["world"])
            per_world.append(float(np.mean(reference["related"][k]))
                             - float(np.mean(cell["related"][k])))
        row["phi_per_world"] = per_world
        row["phi_sd"] = float(np.std(per_world))
        row["worlds_positive"] = int(sum(v > 0 for v in per_world))
        print(f"  {arm:<12} {phi_r:>12,.1f} {phi_u:>14,.1f} {phi_r - phi_u:>13,.1f}")

    print("\n  H30 (fertility exists): prospective Phi_related > 0")
    print("  H31 (structurally specific): Phi_specific >= half of Phi_related")
    if "prospective" in summary:
        row = summary["prospective"]
        # A mean > 0 is not a pass at n = 3. Require every world to
        # agree in sign AND the mean to exceed the spread, because the
        # first three-world reading had mean +0.043 with sd 0.162 and
        # signs 2/3 -- a "PASS" that was one world's noise.
        h30 = row["phi_related"] > 0 and row.get("worlds_positive", 0) == row["worlds"]             and row["phi_related"] > row.get("phi_sd", float("inf"))
        h31 = row["phi_specific"] >= 0.5 * row["phi_related"] if h30 else False
        print(f"    H30 {'PASS' if h30 else 'FAIL'}  "
              f"Phi_related = {row['phi_related']:+.3f} "
              f"sd {row['phi_sd']:.3f} "
              f"positive {row['worlds_positive']}/{row['worlds']} worlds")
        print(f"    H31 {'PASS' if h31 else 'FAIL'}  "
              f"Phi_specific = {row['phi_specific']:+.3f}")
        if "replay" in summary and h30:
            replay = summary["replay"]["phi_related"]
            print(f"    replay Phi_related = {replay:,.1f} — if comparable, "
                  f"continual multitask learning was enough")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cells": rows},
                                      indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
