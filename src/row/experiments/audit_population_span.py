"""Is the family structure present at the POPULATION level?

Review 49's global-rotation idea, made well posed. A global orthogonal
rotation applied to a set of effect vectors leaves leave-one-out
subspace capture unchanged — capture is a property of the point set, so
rotating every point cannot expose structure that is not there. What a
"global change of basis" can express, and what local adapters cannot, is
this: the teacher's atoms may be MIXTURES of the learner's, as in

    B_1 = A_1 + A_2,   B_2 = A_1 - A_2

where no adapter between any single B_i and any single A_f succeeds and
yet the two sets span the same space.

So the decisive question is a span question:

    does A_f^teacher lie in span{ I_tau }, the learner's effective
    task-conditioned innovations?

Answered by least squares, at three levels of generosity:

    full span     all innovations, unrestricted coefficients. If the
                  teacher operator is not in here it is not recoverable
                  by ANY linear reparameterization of the population,
                  cheap or otherwise, and the global-rotation hypothesis
                  is dead.
    top-k span    the k leading principal directions of the innovation
                  set. A cheap global reparameterization is one that
                  keeps few directions, so this is the priced version.
    single atom   the k = 1 baseline already reported by the
                  fragmentation audit, for continuity.

The full-span figure is an UPPER BOUND on what any global linear
refactor could achieve. That is what makes this the decisive test rather
than another similarity measurement: it bounds the hypothesis instead of
sampling it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_effective_operator import (
    effective_innovation,
    load_learner,
    rollout,
)
from row.experiments.audit_meta_recurrence import residual_effect
from row.meta_world import MetaFamilySpec, family_operators, generate_meta_world


def unexplained(target: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of `target` outside the row space of `basis`."""

    total = float(target @ target)
    if total <= 0:
        return float("nan")
    coefficients, *_ = np.linalg.lstsq(basis.T, target, rcond=None)
    residual = target - basis.T @ coefficients
    return float(residual @ residual) / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_h29"))
    parser.add_argument("--worlds", type=int, nargs="+",
                        default=[600, 601, 602, 603, 604, 605])
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--probe", type=int, default=64)
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_population_span.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for world in args.worlds:
        path = args.root / "r100" / f"world_{world}" / "lifecycle"
        if not (path / "model.pt").exists():
            continue
        spec = MetaFamilySpec(
            families=args.families, tasks_per_family=args.tasks_per_family,
            r_meta=1.0, subspace_rank=args.subspace_rank,
        )
        world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
        generated = generate_meta_world(world_config, spec)
        teachers = family_operators(world_config, spec)
        model = load_learner(config, path, args.slots)
        step = model.task_steps - 1

        known = [t for t in generated.tasks if t.task_id in model.task_codes]
        family_tasks = [
            task for index, task in enumerate(generated.tasks)
            if spec.family_of(index) is not None and task in known
        ][: args.max_tasks]
        if len(family_tasks) < 4:
            continue

        mean_code = torch.stack(
            [model.task_codes[t.task_id].detach() for t in known]
        ).mean(dim=0).view(model.task_steps, model.operator_slots)
        mean_route = torch.softmax(mean_code, dim=-1)

        # ONE COMMON STATE SET for every innovation and every target.
        # The first version built each innovation on its own task's
        # states and then used those mutually unaligned vectors as a
        # shared regression basis, so a "span" was fitted across
        # incomparable coordinates (review 55). A span question is only
        # meaningful when all vectors live in the same space.
        pooled = []
        for task in family_tasks:
            start = torch.tensor(task.eval_x[: args.probe], dtype=torch.float32)
            with torch.no_grad():
                pooled.append(rollout(model, task.task_id, start, step))
        common = torch.cat(pooled, dim=0)
        if len(common) > args.probe:
            pick = torch.randperm(
                len(common), generator=torch.Generator().manual_seed(world)
            )[: args.probe]
            common = common[pick]
        common_np = common.cpu().numpy().astype(np.float64)

        innovations, targets = [], []
        for task in family_tasks:
            innovations.append(
                effective_innovation(model, task.task_id, common, step,
                                     mean_route))
            index = generated.tasks.index(task)
            targets.append(
                residual_effect(teachers[spec.family_of(index)], common_np))

        pool = np.stack(innovations).astype(np.float64)
        centred = pool - pool.mean(axis=0)
        _, _, directions = np.linalg.svd(centred, full_matrices=False)

        full, by_k, single = [], {k: [] for k in args.top_k}, []
        for target in targets:
            full.append(unexplained(target, pool))
            for k in args.top_k:
                by_k[k].append(unexplained(target, directions[:k]))
            single.append(min(unexplained(target, pool[i:i + 1])
                              for i in range(len(pool))))
        rows.append({
            "world": world, "innovations": len(pool), "targets": len(targets),
            "full_span": float(np.mean(full)),
            "top_k": {str(k): float(np.mean(v)) for k, v in by_k.items()},
            "single": float(np.mean(single)),
        })

    if not rows:
        print("no artifacts scored")
        return

    print("POPULATION SPAN — is the teacher family inside the learner's span?")
    print("  unexplained fraction of a teacher family operator\n")
    ks = args.top_k
    print(f"  {'world':>6} {'innov':>6} {'single':>8} "
          + " ".join(f"{'top-'+str(k):>8}" for k in ks) + f" {'full':>8}")
    for row in rows:
        print(f"  {row['world']:>6} {row['innovations']:>6} {row['single']:>8.3f} "
              + " ".join(f"{row['top_k'][str(k)]:>8.3f}" for k in ks)
              + f" {row['full_span']:>8.3f}")

    full = float(np.mean([r["full_span"] for r in rows]))
    single = float(np.mean([r["single"] for r in rows]))
    tops = {k: float(np.mean([r["top_k"][str(k)] for r in rows])) for k in ks}
    print(f"\n  mean single-innovation unexplained  {single:.3f}")
    for k in ks:
        print(f"  mean top-{k} span unexplained        {tops[k]:.3f}")
    print(f"  mean FULL span unexplained          {full:.3f}   <- upper bound "
          f"on any global linear refactor")

    print("\n  VERDICT")
    if full < 0.25:
        cheap = min(tops.items(), key=lambda kv: kv[1])
        print("    The teacher family IS inside the learner's population span.")
        print("    The information is present and the coordinates hide it, so a")
        print("    global reparameterization is the right remedy.")
        print(f"    Cheapest tested: top-{cheap[0]} leaves {cheap[1]:.3f}.")
    elif full < single - 0.2:
        print("    Partly recoverable from the population but not from any single")
        print("    object: the structure is distributed across the innovation set.")
    else:
        print("    NOT in the span. No global linear reparameterization of this")
        print("    population recovers the teacher family, cheap or otherwise, so")
        print("    the global-rotation hypothesis is dead and what remains is that")
        print("    the WAKE OBJECTIVE never encoded the structure.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"rows": rows, "mean_full_span": full, "mean_single": single,
         "mean_top_k": {str(k): v for k, v in tops.items()}}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
