"""H29, corrected: measure the EFFECTIVE task-conditioned operator.

The first H29 measured shared-subspace capture over the learner's
private residual tensors and found 0.095. That licenses "the residual
OBJECTS carry little family geometry" and not "the learner lost the
family", because a task is solved as

    z <- parent(z, route_tau) + shared_innovation(z) + own_residual(z)

so two tasks drawing on the same teacher family can split the common
computation differently between the route and the residual and look
unrelated in the residuals alone. The residual tensor is then simply the
wrong computational unit to call the task's innovation (review 49).

This module takes the unit the reviewer specifies. At the step where the
family fires, on the on-trajectory state distribution:

    F_tau(z)   the learner's actual transformation for task tau, with
               its own route, its shared reference, and its residual
    F_0(z)     the same computation with TASK-SPECIFIC information
               nulled -- the mean route, no reference, no residual --
               rather than an arbitrary parameter baseline
    I_tau(z)   = F_tau(z) - F_0(z)

and runs the same leave-one-out capture instrument on {I_tau}. The fork
is registered in PREDICTIONS.md before this ran: R_effective high means
the information was distributed rather than lost and PROMOTE has been
promoting the wrong object; R_effective low keeps the coordinate and
global-rotation hypotheses alive.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_meta_recurrence import loo_capture
from row.experiments.audit_promotion_structure import isotropic_null
from row.experiments.learned_lifetime import _build_model
from row.meta_world import MetaFamilySpec, generate_meta_world


def load_learner(config, path: Path, slots: int, kind: str = "lifecycle"):
    local = replace(
        config,
        shared_residual_model=replace(
            config.shared_residual_model, operator_slots=slots),
    )
    model = _build_model(local, kind)
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    count = sum(1 for k in state if k.startswith("abstractions."))
    for index in range(count):
        model.abstractions.append(
            torch.nn.Parameter(state[f"abstractions.{index}"].clone(),
                               requires_grad=False)
        )
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    reference_table = summary.get("reference_table") or {}
    for task_id, reference in (reference_table.get("task_reference") or {}).items():
        model.task_reference[task_id] = int(reference)
    # RETIREMENT STATE. Without this, `forward` adds BOTH the promoted
    # abstraction and the private residual that retirement removed, for
    # every retired task -- and most tasks in these artifacts are
    # retired, so the reconstructed model computes something the trained
    # model never did (review 55).
    for task_id in reference_table.get("retired_task_ids") or []:
        model.retired.add(task_id)
    return model


@torch.no_grad()
def rollout(model, task_id: str, z, upto: int):
    """State after `upto` steps of the model's ACTUAL forward pass."""

    route, own_u, own_v, own_b = model._unpack(task_id)
    weights = torch.softmax(route, dim=-1)
    reference = model.task_reference.get(task_id)
    shared = (model._split_residual(model.abstractions[reference])
              if reference is not None else None)
    retired = task_id in model.retired
    for step in range(upto):
        candidates = torch.stack([o(z) for o in model.basis], dim=0)
        state = torch.sum(
            weights[step].view(model.operator_slots, 1, 1) * candidates, dim=0)
        if shared is not None:
            state = state + model._innovation(z, *shared, step)
        if not retired:
            state = state + model._innovation(z, own_u, own_v, own_b, step)
        z = state
    return z


@torch.no_grad()
def effective_innovation(model, task_id: str, z: torch.Tensor,
                         step: int, mean_route: torch.Tensor) -> np.ndarray:
    """F_tau(z) - F_0(z) at one step, on the given states."""

    route, own_u, own_v, own_b = model._unpack(task_id)
    own = torch.softmax(route, dim=-1)
    candidates = torch.stack([operator(z) for operator in model.basis], dim=0)

    def parent(weights):
        return torch.sum(
            weights[step].view(model.operator_slots, 1, 1) * candidates, dim=0)

    full = parent(own)
    reference = model.task_reference.get(task_id)
    if reference is not None:
        full = full + model._innovation(
            z, *model._split_residual(model.abstractions[reference]), step)
    if task_id not in model.retired:
        full = full + model._innovation(z, own_u, own_v, own_b, step)
    # F_0: task-specific information nulled. The mean route is the
    # population's default way of using the basis, so what remains in
    # the difference is what THIS task contributes.
    baseline = parent(mean_route)
    return (full - baseline).cpu().numpy().ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v5_h29"))
    parser.add_argument("--conditions", nargs="+", default=["r100"])
    parser.add_argument("--worlds", type=int, nargs="+",
                        default=[600, 601, 602, 603, 604, 605])
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--tasks-per-family", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--schema-rank", type=int, default=2)
    parser.add_argument("--probe", type=int, default=64)
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v5_effective_operator.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for condition in args.conditions:
        r_meta = 0.0 if condition == "r0" else 1.0
        for world in args.worlds:
            path = args.root / condition / f"world_{world}" / "lifecycle"
            if not (path / "model.pt").exists():
                continue
            spec = MetaFamilySpec(
                families=args.families, tasks_per_family=args.tasks_per_family,
                r_meta=r_meta, subspace_rank=args.subspace_rank,
            )
            world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
            generated = generate_meta_world(world_config, spec)
            model = load_learner(config, path, args.slots)
            step = model.task_steps - 1

            known = [t for t in generated.tasks if t.task_id in model.task_codes]
            family_tasks = [
                task for index, task in enumerate(generated.tasks)
                if spec.family_of(index) is not None and task in known
            ][: args.max_tasks]
            if len(family_tasks) < args.schema_rank + 2:
                continue

            # Mean route over the population: the default way this
            # learner uses its basis, so the difference isolates what
            # each task adds.
            # Task codes are stored FLAT (steps * slots); the softmax is
            # per step, as `forward` does it. Averaging and softmaxing
            # the flat vector normalizes across steps and yields a
            # single scalar per step, which is not a route at all.
            mean_code = torch.stack(
                [model.task_codes[t.task_id].detach() for t in known]
            ).mean(dim=0).view(model.task_steps, model.operator_slots)
            mean_route = torch.softmax(mean_code, dim=-1)

            # COMMON STATE SET. Every task's innovation must be
            # evaluated at the SAME inputs, or coordinate j of one
            # vector and coordinate j of another describe different
            # states and the SVD fits a subspace across incomparable
            # axes. The first version used each task's own eval_x and
            # the resulting capture was not interpretable (review 55).
            #
            # The shared states are the union of on-trajectory states,
            # subsampled: they are the states these operators actually
            # act on, unlike a fresh Gaussian probe.
            pooled = []
            for task in family_tasks:
                start = torch.tensor(task.eval_x[: args.probe],
                                     dtype=torch.float32)
                with torch.no_grad():
                    pooled.append(rollout(model, task.task_id, start, step))
            common = torch.cat(pooled, dim=0)
            if len(common) > args.probe:
                pick = torch.randperm(
                    len(common), generator=torch.Generator().manual_seed(world)
                )[: args.probe]
                common = common[pick]

            innovations = []
            for task in family_tasks:
                z = common
                innovations.append(
                    effective_innovation(model, task.task_id, z, step, mean_route))
            effects = np.stack(innovations).astype(np.float64)
            rows.append({
                "condition": condition, "world": world,
                "tasks": len(effects),
                "r_effective": loo_capture(effects, args.schema_rank),
                "null": isotropic_null(len(effects), effects.shape[1],
                                       args.schema_rank, world),
            })

    if not rows:
        print("no artifacts scored")
        return

    print("H29 CORRECTED — capture over the EFFECTIVE task-conditioned operator")
    print("  I_tau = F_tau - F_0, task-specific information nulled\n")
    print(f"  {'world':>6} {'tasks':>6} {'R_effective':>12} {'null':>7}")
    for row in rows:
        print(f"  {row['world']:>6} {row['tasks']:>6} "
              f"{row['r_effective']:>12.3f} {row['null']:>7.3f}")
    effective = float(np.mean([r["r_effective"] for r in rows]))
    null = float(np.mean([r["null"] for r in rows]))
    print(f"\n  mean R_effective  {effective:.3f}")
    print(f"  isotropic null    {null:.3f}")
    print(f"  R_residual (H29)  0.095   <- the same instrument on residuals alone")

    print("\n  REGISTERED FORK")
    if effective >= 0.5:
        print("    R_effective HIGH: the family information was never lost, only")
        print("    DISTRIBUTED across route and residual. PROMOTE is promoting the")
        print("    wrong object; abstraction boundaries should be found")
        print("    functionally rather than inherited from parameter boundaries.")
    elif effective > 0.095 * 1.5:
        print("    R_effective materially above the residual figure but still low:")
        print("    part of the family lives in the route and part in the residual,")
        print("    and neither unit alone shows it. The global-rotation test is the")
        print("    decisive one.")
    else:
        print("    R_effective no better than the residual figure: the structure is")
        print("    absent from the learner's task-conditioned computation too, and")
        print("    the coordinate and global-basis hypotheses are what remain.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"rows": rows, "mean_r_effective": effective, "null": null,
         "r_residual_reference": 0.095}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
