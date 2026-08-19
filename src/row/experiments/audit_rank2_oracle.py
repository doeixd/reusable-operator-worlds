"""Oracle for the promotion unit V3.1 actually specifies: a rank-2 abstraction.

The first oracle grew the library by whole rank-8 basis operators, 265
scalars apiece, and lost the two-part cell 0/3 because the new capacity cost
more than it bought. But V3.1's promoted abstraction is fixed at rank 2
(U tanh(V z + b), 2d + 1 = 66 scalars for two components at d = 16), which is
an order of magnitude cheaper, and promotion also RETIRES the task residual
the abstraction replaces.

This computes that bound directly on trained saturated-library artifacts:
group the post-onset task residuals by TRUE family (oracle knowledge, used
only here), average each family's rank-2 residual parameters per step into a
shared abstraction, hand every member task a reference to it instead of its
own copy, and price the result.

    D_before = shared + sum_tasks [routes + own rank-2 residual]
    D_after  = shared + families * steps * 66 scalars
               + sum_tasks [routes + log2(families + 1) reference bits]

The gate passes if D_after buys enough to beat D_before under
J = L + lambda*D at acceptable distortion. If even this oracle loses, no
promoter can win in this testbed and the world needs redesigning again.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import nmse
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory

LN2 = math.log(2.0)
BITS = 8


def audit(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int):
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
    model = _build_model(local, "shared_residual")
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    loss = float(summary["cumulative_prequential_gaussian_log_loss"])

    assignment = np.array(world.group_assignment[: len(world.tasks)])
    onset = spec.family_onset
    families = sorted(set(assignment.tolist()))

    with torch.no_grad():
        base = np.array(
            [
                nmse(model(_tensor(t.eval_x), t.task_id).cpu().numpy(), t.eval_y)
                for t in world.tasks
            ]
        )
        # One shared rank-2 abstraction per (family, step): the mean of its
        # member tasks' residual parameters.
        abstractions = {}
        for family in families:
            members = [
                world.tasks[i].task_id
                for i in range(onset, len(world.tasks))
                if assignment[i] == family
            ]
            if not members:
                continue
            stacked = torch.stack(
                [model.task_residuals[task_id].detach() for task_id in members]
            )
            abstractions[family] = stacked.mean(dim=0)

        promoted = copy.deepcopy(model)
        for index in range(onset, len(world.tasks)):
            family = int(assignment[index])
            if family in abstractions:
                promoted.task_residuals[world.tasks[index].task_id].copy_(
                    abstractions[family]
                )
        promoted.eval()
        after = np.array(
            [
                nmse(promoted(_tensor(t.eval_x), t.task_id).cpu().numpy(), t.eval_y)
                for t in world.tasks
            ]
        )

    tasks = len(world.tasks)
    promoted_tasks = tasks - onset
    residual_scalars = (
        model.residual_u_size + model.residual_v_size + model.residual_b_size
    )
    shared_bits = BITS * int(model.shared_parameter_count)
    route_bits = BITS * model.route_size * tasks

    before_bits = shared_bits + route_bits + BITS * residual_scalars * tasks
    reference_bits = math.ceil(math.log2(len(families) + 1))
    after_bits = (
        shared_bits
        + route_bits
        + BITS * residual_scalars * len(abstractions)  # the promoted objects
        + BITS * residual_scalars * onset  # pre-onset tasks keep their own
        + reference_bits * promoted_tasks
    )
    return {
        "world_seed": world_seed,
        "lifetime_loss": loss,
        "families": len(abstractions),
        "promoted_tasks": promoted_tasks,
        "before_bits": before_bits,
        "after_bits": after_bits,
        "bits_saved": before_bits - after_bits,
        "before_two_part": loss + LN2 * before_bits,
        "after_two_part": loss + LN2 * after_bits,
        "two_part_gain": LN2 * (before_bits - after_bits),
        "mean_nmse_before": float(np.mean(base)),
        "mean_nmse_after": float(np.mean(after)),
        "mean_nmse_increase": float(np.mean(after - base)),
        "promoted_task_nmse_increase": float(np.mean((after - base)[onset:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/v3_taskgroup/eta0.9_k6_onset16_frozen16")
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--onset", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_rank2_oracle.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=args.groups, eta=args.eta, future_tasks=8, family_onset=args.onset
    )
    rows = [
        audit(config, args.root / f"world_{w}" / "shared_residual", w, spec, args.slots)
        for w in args.worlds
    ]
    print("RANK-2 PROMOTION ORACLE (families known; the bound a promoter chases)")
    for row in rows:
        print(
            f"  world {row['world_seed']}: bits {row['before_bits']:,} -> {row['after_bits']:,} "
            f"(saved {row['bits_saved']:,} = {row['two_part_gain']:+.0f} nats) | "
            f"NMSE {row['mean_nmse_before']:.5f} -> {row['mean_nmse_after']:.5f} "
            f"(promoted tasks +{row['promoted_task_nmse_increase']:.5f})"
        )
    gains = [row["two_part_gain"] for row in rows]
    costs = [row["promoted_task_nmse_increase"] for row in rows]
    print(
        f"\n  mean two-part gain {np.mean(gains):+.0f} nats, positive "
        f"{sum(g > 0 for g in gains)}/{len(gains)} worlds; "
        f"mean promoted-task NMSE cost +{np.mean(costs):.5f}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
