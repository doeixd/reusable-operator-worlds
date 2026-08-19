"""V4R RETAIN oracle: is a dormant abstraction worth carrying?

Three online dormancy designs failed to instantiate retention value, and
the cause was not the world. V3's PROMOTE fires in structureless
controls, creating roughly three abstractions where no family exists, so
an online learner simply manufactures a replacement after the gap and
deletion is never costly. A retention experiment cannot be run on top of
that.

This oracle removes the confound without touching the frozen V3
substrate: the library is FROZEN after the gap. The question becomes a
clean counterfactual over tasks that arrive after the return —

    is referencing a pre-gap abstraction better than referencing none?

paired across two arms that are byte-identical up to the gap:

    RETURNS    the regime resumes; a rational lifecycle RETAINS
    PERMANENT  the regime never resumes; it eventually RETIRES

A pass requires BOTH: material value in the returning arm and no value
in the permanent one. A single arm proves nothing here, for the same
reason it proved nothing in `audit_obsolescence` — value that appears in
both arms is not retention, it is an abstraction that happens to be
generically useful.

Scoring `V_retain = value_to_post_gap_tasks - carry_cost`, where the
carry cost is one residual's worth of description held to the end of the
lifetime.

REGISTERED PREREQUISITE (V4R §2.1): no online retention policy may be
implemented until this oracle shows a CROSSOVER in gap length — short
gaps favor retain, long gaps favor delete-and-relearn. A positive result
at one gap is an existence proof, not a policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_factorization import _load
from row.experiments.learned_lifetime import _tensor
from row.metrics import gaussian_nll
from row.task_group_world import TaskGroupSpec

LN2 = math.log(2.0)
BITS = 8


def audit(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int,
          gap_start: int, tasks_total: int) -> dict:
    model, world = _load(config, path, world_seed, spec, slots)
    lifecycle = json.loads((path / "summary.json").read_text(encoding="utf-8"))["lifecycle"]
    references = {k: int(v) for k, v in lifecycle["task_reference"].items()}
    pre_gap = [
        record["abstraction_id"]
        for record in lifecycle["lineage"]
        if record["born_at_task"] < gap_start
    ]
    order = {task.task_id: index for index, task in enumerate(world.tasks)}
    tasks = {task.task_id: task for task in world.tasks}
    post = [t for t in references if order.get(t, 0) >= gap_start]
    if not pre_gap or not post:
        return {
            "world_seed": world_seed,
            "scoreable": False,
            "reason": "no pre-gap abstraction" if not pre_gap else "no post-gap tasks",
        }
    for task_id, reference in references.items():
        model.task_reference[task_id] = reference
        model.retired.add(task_id)

    best_id, best_value = None, None
    with torch.no_grad():
        for candidate in pre_gap:
            total = 0.0
            for task_id in post:
                task = tasks[task_id]
                current = model.task_reference.get(task_id)
                model.task_reference.pop(task_id, None)
                without = gaussian_nll(
                    model(_tensor(task.eval_x), task_id).cpu().numpy(), task.eval_y, 0.1
                )
                model.task_reference[task_id] = candidate
                with_ref = gaussian_nll(
                    model(_tensor(task.eval_x), task_id).cpu().numpy(), task.eval_y, 0.1
                )
                if current is None:
                    model.task_reference.pop(task_id, None)
                else:
                    model.task_reference[task_id] = current
                total += without - with_ref
            if best_value is None or total > best_value:
                best_id, best_value = candidate, total

    width = model.residual_u_size + model.residual_v_size + model.residual_b_size
    carry = LN2 * BITS * width
    return {
        "world_seed": world_seed,
        "scoreable": True,
        "post_gap_tasks": len(post),
        "pre_gap_abstractions": pre_gap,
        "best_abstraction": best_id,
        "value_nats": best_value,
        "carry_cost_nats": carry,
        "v_retain_nats": best_value - carry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v4r_retain"))
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--gap-start", type=int, default=32)
    parser.add_argument("--gap-ends", type=int, nargs="+", default=[40, 48, 64])
    parser.add_argument("--onset", type=int, default=8)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("reports/v4r_retention.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    results = []
    print("V4R RETAIN ORACLE   (library frozen after the gap)")
    print("  pass needs value in RETURNS *and* none in PERMANENT")

    def run(arm: str, gap_end: int):
        spec = TaskGroupSpec(
            groups=1, eta=0.9, future_tasks=8, family_onset=args.onset,
            new_primitive_families=True,
            dormancy=(args.gap_start, gap_end),
            dormancy_returns=(arm == "returns"),
        )
        rows = []
        for world in args.worlds:
            path = args.root / f"{arm}_g{gap_end}" / f"world_{world}" / "lifecycle"
            if not (path / "summary.json").exists():
                continue
            row = audit(config, path, world, spec, args.slots, args.gap_start, args.tasks)
            row.update({"arm": arm, "gap_end": gap_end,
                        "gap_length": gap_end - args.gap_start})
            rows.append(row)
            results.append(row)
        return [r for r in rows if r["scoreable"]]

    print("\n  %-10s %6s %8s %10s %12s %10s" %
          ("arm", "gap", "scored", "mean V", "positive", "median V"))
    summary = {}
    for gap_end in args.gap_ends:
        for arm in ("returns", "permanent"):
            scored = run(arm, gap_end)
            if not scored:
                continue
            values = [r["v_retain_nats"] for r in scored]
            summary[(arm, gap_end - args.gap_start)] = values
            print("  %-10s %6d %8d %10.0f %12s %10.0f" %
                  (arm, gap_end - args.gap_start, len(scored), float(np.mean(values)),
                   "%d/%d" % (sum(v > 0 for v in values), len(values)),
                   float(np.median(values))))

    print("\n  CROSSOVER (V4R §2.1 prerequisite for any online policy):")
    for (arm, gap), values in sorted(summary.items()):
        if arm != "returns":
            continue
        mean = float(np.mean(values))
        print("    gap %2d: mean V_retain %+8.0f -> %s"
              % (gap, mean, "RETAIN" if mean > 0 else "DELETE"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
