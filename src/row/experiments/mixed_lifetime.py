"""Benchmark D lifetimes: run existing learners on mixed-recurrence worlds.

Reuses `learned_lifetime.run` wholesale (prequential protocol, replay,
tuned parameter groups, artifacts) by injecting a mixed-world factory in
place of `World` for the duration of the run. Profile provenance is
written to `rho_profile.json` beside the run's artifacts, following the
scrambled-ID pattern of keeping optional provenance outside `WorldConfig`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from row.config import load_config
from row.experiments import learned_lifetime
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory, teacher_group_clustering
from row.mixed_world import (
    CANONICAL_PROFILE,
    HIERARCHICAL_WEIGHTS,
    HierarchicalWorldFactory,
    MixedWorldFactory,
    per_primitive_recurrence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--model",
        choices=(
            "dense",
            "continuous",
            "shared_residual",
            "hypernetwork",
            "variational",
            "gated",
            "promoting",
        ),
        required=True,
    )
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument(
        "--profile",
        type=float,
        nargs="+",
        default=list(CANONICAL_PROFILE),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast-tuning", action="store_true", default=True)
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Benchmark E: global+family+task hierarchy instead of a rho profile",
    )
    # V3 promotion testbed (spec 2.1/2.3). eta = 0 reproduces the canonical
    # mixed world bit-exactly and is the structureless control.
    parser.add_argument("--task-group-eta", type=float, default=None)
    parser.add_argument("--task-groups", type=int, default=2)
    parser.add_argument(
        "--task-group-block-size",
        type=int,
        default=None,
        help="drifting-family control: redraw the family direction every N tasks",
    )
    parser.add_argument("--future-tasks", type=int, default=8)
    parser.add_argument(
        "--family-onset",
        type=int,
        default=0,
        help="delayed-family testbed: families appear only from this task index",
    )
    parser.add_argument(
        "--sleeps",
        type=int,
        nargs="*",
        default=[],
        help="task counts at which PROMOTE runs",
    )
    parser.add_argument("--promotion-epsilon", type=float, default=0.02)
    parser.add_argument(
        "--freeze-slots",
        type=int,
        default=None,
        help="oracle: freeze only the first N slots, letting the library grow",
    )
    parser.add_argument(
        "--freeze-basis-at",
        type=int,
        default=None,
        help="freeze the shared basis from this task index (clean promotion test)",
    )
    parser.add_argument(
        "--operator-slots",
        type=int,
        default=None,
        help="override shared basis capacity K (the V3 capacity sweep)",
    )
    parser.add_argument(
        "--resample-future",
        action="store_true",
        help="regime-change world: identical lifetime, fresh future directions",
    )
    args = parser.parse_args()

    if (args.output / "summary.json").exists():
        print("summary exists; skipping")
        return

    config = load_config(args.config)
    config = replace(
        config,
        world=replace(config.world, seed=args.world_seed, reuse_rho=1.0),
        evaluation=replace(
            config.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
        output_directory=args.output,
    )
    if args.operator_slots is not None:
        selected = {
            "dense": config.dense_model,
            "continuous": config.continuous_model,
            "hypernetwork": config.hypernetwork_model,
            "shared_residual": config.shared_residual_model,
            "variational": config.variational_model,
            "gated": config.gated_model,
            # PROMOTE reuses the frozen shared-residual configuration.
            "promoting": config.shared_residual_model,
        }[args.model]
        selected = replace(selected, operator_slots=args.operator_slots)
        field = (
            "shared_residual_model"
            if args.model == "promoting"
            else f"{args.model}_model"
        )
        config = replace(config, **{field: selected})

    task_group_spec = (
        TaskGroupSpec(
            groups=args.task_groups,
            eta=args.task_group_eta,
            block_size=args.task_group_block_size,
            future_tasks=args.future_tasks,
            resample_future=args.resample_future,
            family_onset=args.family_onset,
        )
        if args.task_group_eta is not None
        else None
    )
    if task_group_spec is not None:
        factory = TaskGroupWorldFactory(args.profile, task_group_spec)
    elif args.hierarchical:
        factory = HierarchicalWorldFactory(HIERARCHICAL_WEIGHTS)
    else:
        factory = MixedWorldFactory(args.profile)
    original_world = learned_lifetime.World
    learned_lifetime.World = factory  # type: ignore[assignment]
    try:
        summary = learned_lifetime.run(
            config,
            kind=args.model,
            freeze_shared_at=args.freeze_basis_at,
            freeze_slots=args.freeze_slots,
            sleeps=tuple(args.sleeps),
            promotion_epsilon=args.promotion_epsilon,
        )
    finally:
        learned_lifetime.World = original_world  # type: ignore[assignment]

    world = factory.generate(config.world)
    provenance: dict[str, object] = {
        "note": (
            "structured-recurrence world; the config.yaml reuse_rho field is "
            "a placeholder and this file is authoritative for the structure"
        ),
        "per_primitive_recurrence": per_primitive_recurrence(world),
    }
    if task_group_spec is not None:
        provenance["rho_profile"] = list(factory.profile)
        provenance["task_group_spec"] = task_group_spec.as_dict()
        # Ground truth for post-hoc scoring only; no learner reads this.
        provenance["group_assignment"] = list(world.group_assignment)
        provenance["teacher_group_clustering"] = teacher_group_clustering(world)
        label = f"task_group eta={task_group_spec.eta:g}"
    elif args.hierarchical:
        provenance["hierarchy_weights"] = list(factory.weights)
        label = f"hierarchy={list(factory.weights)}"
    else:
        provenance["rho_profile"] = list(factory.profile)
        label = f"profile={list(factory.profile)}"
    (args.output / "rho_profile.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{args.model} world={args.world_seed} {label}: "
        f"loss={summary['cumulative_prequential_gaussian_log_loss']:.1f}"
    )


if __name__ == "__main__":
    main()
