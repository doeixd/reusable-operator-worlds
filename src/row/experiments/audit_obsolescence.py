"""V4.1 gate for OBSOLESCENCE: is any abstraction worth less than its bits?

The behavioral-cover oracle (`audit_lifecycle_oracle`) asks whether the
library contains REDUNDANT members — two abstractions where one can serve
both dependent sets. On the plain structured testbed the answer is no,
and that gate correctly fails there.

This module asks the other V4.1 question, the one §2.2's dormancy pair
was built for: is any abstraction OBSOLETE — still the unique server of
its dependents, but no longer worth carrying to the end of the lifetime?
The two are independent. An abstraction can be irreplaceable and still
not worth its bits, and that is precisely the case a usage counter
cannot see.

WHY THE ACCOUNTING IS SUBTLE. Under `J = L_preq + lambda*D_T`, deleting
an abstraction late is nearly free by construction: prequential loss for
tasks already scored is already paid and cannot be revised, so retiring
an abstraction at the final consolidation point removes its bits from
`D_T` and charges nothing. That would make "delete everything at the
end" the trivial optimum and the gate meaningless.

Two terms stop that, and both must be priced or the gate is vacuous:

  * OLD-TASK RETENTION. Deleting an abstraction strands its dependents,
    whose held-out predictions degrade. The spec scores this as an
    endpoint with a non-inferiority margin, so it is charged here at its
    full held-out Gaussian cost.
  * OCCUPANCY (`kappa`). Retaining an abstraction from `t_d` to `T`
    costs `kappa * bits * (T - t_d)`, which is what makes deletion TIMED
    rather than a final-state cleanup. At `kappa = 0` any gain reported
    here is compression, not lifecycle management, and is labelled so.

WHAT A PASS MEANS. The oracle scans each abstraction's deletion time
independently over the consolidation points plus infinity, takes the
best, and reports the total. The dormancy pair then supplies the
discrimination the scalar cannot: the PERMANENT arm should show a
positive optimum at a finite `t_d`, and the RETURNS arm — byte-identical
through `b` — should show its optimum at infinity. A world where both
arms want deletion is not testing real options, it is testing whether
bits are expensive.
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
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import gaussian_nll
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory

LN2 = math.log(2.0)
BITS = 8
SLEEPS = (24, 32, 48, 64)


def _load(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int):
    factory = TaskGroupWorldFactory(list(CANONICAL_PROFILE), spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        world = factory.generate(replace(config.world, seed=world_seed, reuse_rho=1.0))
    finally:
        learned_lifetime.World = original
    local = replace(
        config,
        shared_residual_model=replace(config.shared_residual_model, operator_slots=slots),
    )
    model = _build_model(local, "lifecycle")
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for key in state if key.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(
            torch.nn.Parameter(state[f"abstractions.{index}"].clone(), requires_grad=False)
        )
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    return model, world


@torch.no_grad()
def audit(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int,
          kappa: float, tasks_total: int = 64) -> dict:
    model, world = _load(config, path, world_seed, spec, slots)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    lifecycle = summary.get("lifecycle")
    if lifecycle is None or "task_reference" not in lifecycle:
        raise ValueError(
            f"{path} has no persisted reference table: run the `lifecycle` model"
        )
    references = {k: int(v) for k, v in lifecycle["task_reference"].items()}
    if not references:
        raise ValueError(f"{path} records a library with no dependents")
    for task_id, reference in references.items():
        model.task_reference[task_id] = reference
        model.retired.add(task_id)

    tasks = {task.task_id: task for task in world.tasks}
    residual_bits = BITS * (
        model.residual_u_size + model.residual_v_size + model.residual_b_size
    )
    order = {task.task_id: index for index, task in enumerate(world.tasks)}

    rows = []
    for abstraction in range(len(model.abstractions)):
        dependents = [t for t, r in references.items() if r == abstraction]
        if not dependents:
            continue
        # Held-out cost of stranding this abstraction's dependents. Charged
        # in full: the pessimistic direction, so a pass really passes.
        strand_cost = 0.0
        for task_id in dependents:
            task = tasks.get(task_id)
            if task is None:
                continue
            before = gaussian_nll(
                model(_tensor(task.eval_x), task_id).cpu().numpy(), task.eval_y, 0.1
            )
            model.task_reference.pop(task_id)
            after = gaussian_nll(
                model(_tensor(task.eval_x), task_id).cpu().numpy(), task.eval_y, 0.1
            )
            model.task_reference[task_id] = abstraction
            strand_cost += after - before

        last_use = max((order.get(t, 0) for t in dependents), default=0)
        best = {"t_d": None, "gain_nats": 0.0}
        for t_d in SLEEPS:
            # A deletion before the abstraction's last dependent arrives
            # would change the lifetime, not merely its endpoint, and the
            # frozen artifact cannot price that. Only post-use times are
            # admissible for this static audit.
            if t_d <= last_use:
                continue
            gain = (
                LN2 * residual_bits
                + kappa * residual_bits * max(0, tasks_total - t_d)
                - strand_cost
            )
            if gain > best["gain_nats"]:
                best = {"t_d": t_d, "gain_nats": gain}
        rows.append(
            {
                "abstraction": abstraction,
                "dependents": len(dependents),
                "last_use_task": last_use,
                "strand_cost_nats": strand_cost,
                "bits_value_nats": LN2 * residual_bits,
                "best_t_d": best["t_d"],
                "gain_nats": best["gain_nats"],
            }
        )
    total = sum(r["gain_nats"] for r in rows)
    return {
        "world_seed": world_seed,
        "kappa": kappa,
        "library_size": len(model.abstractions),
        "abstractions": rows,
        "deletions_that_pay": sum(1 for r in rows if r["gain_nats"] > 0),
        "total_gain_nats": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arm", default="lifecycle_on")
    parser.add_argument("--onset", type=int, default=16)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--dormancy", type=int, nargs=2, default=None)
    parser.add_argument("--dormancy-permanent", action="store_true")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.0, 1e-3])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=2, eta=0.9, future_tasks=8, family_onset=args.onset,
        new_primitive_families=True,
        dormancy=tuple(args.dormancy) if args.dormancy else None,
        dormancy_returns=not args.dormancy_permanent,
    )
    results = []
    print("V4.1 OBSOLESCENCE GATE  (timed deletion, not redundancy)")
    for kappa in args.kappas:
        rows = [
            audit(config, args.root / f"world_{w}" / args.arm, w, spec,
                  args.slots, kappa)
            for w in args.worlds
        ]
        results.extend(rows)
        print(f"  kappa = {kappa:g}")
        for row in rows:
            print(
                "    world %d: %d/%d deletions pay   total %+8.0f nats"
                % (row["world_seed"], row["deletions_that_pay"],
                   row["library_size"], row["total_gain_nats"])
            )
    paying = [r for r in results if r["deletions_that_pay"] > 0]
    print(
        "\n  Timed deletion pays in %d/%d cells." % (len(paying), len(results))
    )
    print(
        "  NOT A GATE VERDICT. A single arm cannot pass this gate: deleting at\n"
        "  the final consolidation point removes bits and, for tasks already\n"
        "  scored, charges nothing, so end-of-life cleanup always 'pays'. The\n"
        "  verdict requires the PAIRED §2.2 comparison -- run both arms and\n"
        "  compare `best_t_d` distributions. The gate passes only if the\n"
        "  PERMANENT arm retires the family abstraction materially earlier\n"
        "  than the byte-identical RETURNS arm. Measured 2026-08-19, the two\n"
        "  arms were indistinguishable (see PROGRESS.md), because the learner\n"
        "  promotes fresh abstractions after the gap instead of reusing the\n"
        "  dormant one, so no retention obligation exists to refuse."
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
