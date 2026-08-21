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

import torch
from dataclasses import replace
from pathlib import Path

from row.config import load_config
from row.experiments import learned_lifetime
from row.meta_world import MetaFamilySpec, MetaWorldFactory
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
            "lifecycle",
            "prospective",
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
        "--lifecycle",
        action="store_true",
        help="V4.1: re-home dependents and retire orphaned abstractions",
    )
    parser.add_argument("--lifecycle-kappa", type=float, default=0.0)
    parser.add_argument("--lifecycle-grace", type=int, default=8)
    parser.add_argument("--new-primitive-families", action="store_true")
    parser.add_argument(
        "--force-retire-at",
        type=int,
        default=None,
        help="delete the whole live library at this task (reacquisition probe)",
    )
    parser.add_argument(
        "--force-retire-one",
        action="store_true",
        help="delete only the most-depended-upon abstraction",
    )
    parser.add_argument(
        "--lifecycle-filter",
        action="store_true",
        help="refuse promotions worth less than their own carry cost",
    )
    parser.add_argument(
        "--dormancy",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="V4 real-options world: suspend the family primitive in [START, END)",
    )
    parser.add_argument(
        "--dormancy-permanent",
        action="store_true",
        help="the dormant regime never returns (the DELETE arm)",
    )
    parser.add_argument(
        "--arm",
        choices=("ordinary", "replay", "prospective", "supervised"),
        default="ordinary",
        help="V6.1 arm. ordinary = the V5 learner; replay = also train on a "
             "sibling's examples (does merely SEEING relatives suffice?); "
             "prospective = charge the cost of ADAPTING to a held-out sibling "
             "back to the shared representation; supervised = EXPLICIT-FAMILY-SHARING pressure, "
             "asking the shared representation to predict a sibling from "
             "its relative's route. NOT a substrate upper bound: it is a "
             "stronger and more specific demand than 'can the substrate "
             "represent family structure', so its failure would not "
             "license an architectural verdict (review 53)",
    )
    parser.add_argument("--prospective-weight", type=float, default=1.0)
    parser.add_argument("--prospective-steps", type=int, default=4,
                        help="OUTER steps: how hard the shared representation "
                             "is pushed per task")
    parser.add_argument("--prospective-inner-steps", type=int, default=16,
                        help="INNER steps: how far the sibling's task code is "
                             "adapted before its query loss is charged. These "
                             "were one knob, which conflated 'how much "
                             "pressure' with 'how much adaptation the pressure "
                             "is measured after'.")
    parser.add_argument("--prospective-support", type=int, default=8)
    parser.add_argument(
        "--r-meta",
        type=float,
        default=None,
        help="H20b: meta-recurrence world; how much of each family operator "
             "lies in a subspace shared with the other families",
    )
    parser.add_argument("--meta-families", type=int, default=4)
    parser.add_argument("--meta-tasks-per-family", type=int, default=16)
    parser.add_argument("--meta-subspace-rank", type=int, default=2)
    parser.add_argument(
        "--return-gain",
        type=float,
        default=1.0,
        help=(
            "V5 H19 s-arm (B1): scale the family primitive's contribution on "
            "RETURNING tasks only. Moves s_bar while leaving the promoted "
            "abstraction untouched. 1.0 reproduces the pre-B1 world exactly."
        ),
    )
    parser.add_argument(
        "--updates-per-example",
        type=int,
        default=None,
        help="compute-matched audit: give the comparator extra gradient steps",
    )
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
        # Verify the EXISTING artifact was produced by this protocol.
        # "summary.json exists" is not "the same experiment ran here".
        existing = args.output / "rho_profile.json"
        if existing.exists():
            recorded = json.loads(existing.read_text(encoding="utf-8"))
            stored = recorded.get("v6_arm", {}).get("arm")
            if stored is not None and stored != args.arm:
                raise SystemExit(
                    f"{args.output} holds arm '{stored}' but '{args.arm}' was "
                    "requested; refusing to treat it as a completed cell"
                )
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
    if args.updates_per_example is not None:
        selected = {
            "dense": config.dense_model,
            "continuous": config.continuous_model,
            "hypernetwork": config.hypernetwork_model,
            "shared_residual": config.shared_residual_model,
            "variational": config.variational_model,
            "gated": config.gated_model,
            "promoting": config.shared_residual_model,
            "lifecycle": config.shared_residual_model,
            "prospective": config.shared_residual_model,
        }[args.model]
        selected = replace(selected, updates_per_example=args.updates_per_example)
        field = (
            "shared_residual_model"
            if args.model in {"promoting", "lifecycle", "prospective"}
            else f"{args.model}_model"
        )
        config = replace(config, **{field: selected})
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
            "lifecycle": config.shared_residual_model,
            "prospective": config.shared_residual_model,
        }[args.model]
        selected = replace(selected, operator_slots=args.operator_slots)
        field = (
            "shared_residual_model"
            if args.model in {"promoting", "lifecycle", "prospective"}
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
            new_primitive_families=args.new_primitive_families,
            dormancy=tuple(args.dormancy) if args.dormancy else None,
            dormancy_returns=not args.dormancy_permanent,
            return_gain=args.return_gain,
        )
        if args.task_group_eta is not None
        else None
    )
    meta_spec = (
        MetaFamilySpec(
            families=args.meta_families,
            tasks_per_family=args.meta_tasks_per_family,
            r_meta=args.r_meta,
            subspace_rank=args.meta_subspace_rank,
            family_onset=args.family_onset,
        )
        if args.r_meta is not None
        else None
    )
    if meta_spec is not None:
        # The family layout fixes N; the config's own task count would
        # silently disagree with it.
        config = replace(
            config, world=replace(config.world, tasks=meta_spec.total_tasks)
        )

        # H20b: the learned-library arm. PROMOTE runs normally and the
        # realized library is an OUTCOME, never a validity condition --
        # a learner that collapses meta-structure into fewer atoms has
        # found an alternative solution, not broken the world (D17).
        factory = MetaWorldFactory(meta_spec)
    elif task_group_spec is not None:
        factory = TaskGroupWorldFactory(args.profile, task_group_spec)
    elif args.hierarchical:
        factory = HierarchicalWorldFactory(HIERARCHICAL_WEIGHTS)
    else:
        factory = MixedWorldFactory(args.profile)
    # ---- V6.1 arms -------------------------------------------------
    # Oracle knowledge enters in exactly ONE place: which task is offered
    # as a sibling. The learner never sees the teacher operator, the
    # family parameters, or the sibling's query set during adaptation.
    prospective_hook = None
    if args.arm != "ordinary":
        if meta_spec is None:
            raise SystemExit("V6 arms require --r-meta (the meta-recurrence world)")
        probe_world = factory.generate(config.world)

        # Siblings come from HELD-OUT tasks the lifetime never trains on.
        # The first version drew them from `world.tasks`, so the hook
        # trained shared parameters on a later task's support AND query
        # labels before that task legitimately arrived in the
        # prequential stream -- future-label leakage that voided every
        # arm-versus-arm comparison (review 55). The world already
        # generates `held_out_family_tasks` for exactly this purpose.
        held_out = list(getattr(probe_world, "held_out_family_tasks", ()))
        held_out_family = list(getattr(probe_world, "held_out_family_index", ()))

        def _sibling_of(world_task_index: int):
            """A held-out member of the CURRENT task's family, or None."""

            family = meta_spec.family_of(world_task_index)
            if family is None or not held_out:
                return None
            candidates = [
                task for task, owner in zip(held_out, held_out_family)
                if owner == family
            ]
            if not candidates:
                return None
            return candidates[world_task_index % len(candidates)]

        def prospective_hook(model, lifetime_index, world_task_index):
            sibling = _sibling_of(world_task_index)
            if sibling is None:
                return None
            support = args.prospective_support
            support_x = learned_lifetime._tensor(sibling.train_x[:support])
            support_y = learned_lifetime._tensor(sibling.train_y[:support])
            query_x = learned_lifetime._tensor(
                sibling.train_x[support:support * 2])
            query_y = learned_lifetime._tensor(
                sibling.train_y[support:support * 2])
            probe_id = f"__probe_{sibling.task_id}"

            if args.arm == "replay":
                # Does merely SEEING a relative produce the geometry?
                # Same examples, ordinary training, no adaptation loop.
                model.begin_task(probe_id)
                # MATCHED to ordinary acquisition: separate parameter
                # groups at the learning rates the lifetime itself uses
                # (route 0.05, residual 0.01, shared 0.003). The first
                # version put all three in one AdamW at 0.003, so replay
                # was not the control it claimed to be (review 55).
                settings = config.shared_residual_model
                optimizer = torch.optim.AdamW([
                    {"params": list(model.shared_parameters()),
                     "lr": settings.global_learning_rate},
                    {"params": [model.task_codes[probe_id]],
                     "lr": settings.task_learning_rate},
                    {"params": [model.task_residuals[probe_id]],
                     "lr": settings.residual_learning_rate},
                ])
                for _ in range(args.prospective_steps):
                    optimizer.zero_grad()
                    loss = torch.mean(
                        (model(support_x, probe_id) - support_y) ** 2)
                    loss.backward()
                    optimizer.step()
                value = float(loss.detach())
                model.forget_task(probe_id)
                return {"arm": "replay", "sibling": sibling.task_id,
                        "support_mse": value}

            if args.arm == "supervised":
                # Substrate upper bound: pull the sibling's route toward
                # the current task's directly. Not a candidate
                # architecture -- it answers only whether the substrate
                # CAN express the organization.
                current_id = probe_world.tasks[world_task_index].task_id
                if current_id not in model.task_codes:
                    return None
                # Ask the shared representation to make a sibling
                # predictable FROM ITS RELATIVE'S ROUTE, with no
                # adaptation at all. That is the strongest form of
                # "same-family structure is expressible here", and
                # unlike a distance between two task codes it actually
                # depends on the shared parameters -- the first version
                # penalized only task-local state, so its gradient into
                # the shared basis was identically zero and the arm
                # returned bit-identical lifetimes to ordinary.
                optimizer = torch.optim.AdamW(
                    model.shared_parameters(),
                    lr=config.shared_residual_model.global_learning_rate,
                )
                for _ in range(args.prospective_steps):
                    optimizer.zero_grad()
                    penalty = torch.mean(
                        (model(support_x, current_id) - support_y) ** 2)
                    (args.prospective_weight * penalty).backward()
                    optimizer.step()
                value = float(penalty.detach())
                model.forget_task(probe_id)
                return {"arm": "supervised", "sibling": sibling.task_id,
                        "code_distance": value}

            # prospective: charge the ADAPTATION cost to shared state.
            optimizer = torch.optim.AdamW(
                model.shared_parameters(),
                lr=config.shared_residual_model.global_learning_rate,
            )
            # Matched to the other arms: the same number of outer steps
            # on the shared parameters, so an arm cannot win by simply
            # taking more gradient.
            for _ in range(args.prospective_steps):
                optimizer.zero_grad()
                penalty = model.prospective_penalty(
                    probe_id, support_x, support_y, query_x, query_y,
                    steps=args.prospective_inner_steps,
                )
                (args.prospective_weight * penalty).backward()
                optimizer.step()
            value = float(penalty.detach())
            model.forget_task(probe_id)
            return {"arm": "prospective", "sibling": sibling.task_id,
                    "query_penalty": value}

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
            lifecycle_enabled=args.lifecycle,
            lifecycle_filter=args.lifecycle_filter,
            force_retire_at=args.force_retire_at,
            force_retire_one=args.force_retire_one,
            lifecycle_kappa=args.lifecycle_kappa,
            lifecycle_grace=args.lifecycle_grace,
            prospective_hook=prospective_hook,
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
    if meta_spec is not None:
        provenance["meta_family_spec"] = meta_spec.as_dict()
        # EVERY protocol knob. Resume only checks that summary.json
        # exists, so an artifact configured differently would otherwise
        # be silently accepted as a match (review 55).
        provenance["v6_arm"] = {
            "arm": args.arm,
            "prospective_weight": args.prospective_weight,
            "prospective_steps": args.prospective_steps,
            "prospective_inner_steps": args.prospective_inner_steps,
            "prospective_support": args.prospective_support,
            "freeze_basis_at": args.freeze_basis_at,
            "freeze_slots": args.freeze_slots,
            "operator_slots": args.operator_slots,
            "sleeps": list(args.sleeps),
            "lifecycle": args.lifecycle,
            "promotion_epsilon": args.promotion_epsilon,
        }
        label = f"meta r_meta={meta_spec.r_meta:g} F={meta_spec.families}"
    elif task_group_spec is not None:
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
