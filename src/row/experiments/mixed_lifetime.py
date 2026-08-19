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

    factory = (
        HierarchicalWorldFactory(HIERARCHICAL_WEIGHTS)
        if args.hierarchical
        else MixedWorldFactory(args.profile)
    )
    original_world = learned_lifetime.World
    learned_lifetime.World = factory  # type: ignore[assignment]
    try:
        summary = learned_lifetime.run(config, kind=args.model)
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
    if args.hierarchical:
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
