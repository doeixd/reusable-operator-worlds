"""V2 step 007: genuinely shorter and longer lifetimes at bracketing rho.

Condition met: the truncated-lifetime bridge analysis showed early
crossover movement (0.869 -> 0.822 between 16 and 32 tasks). This sweep
runs real 32- and 128-task lifetimes (not truncations) for Continuous and
Dense-C at the bracketing rho values, so the rho*(N) reading at short and
long N comes from genuinely re-run lifetimes with proportionally scaled
replay exposure.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from row.config import load_config
from row.experiments import learned_lifetime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--model", choices=("continuous", "dense"), required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--reuse-rho", type=float, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        print("summary exists; skipping")
        return
    config = load_config(args.config)
    config = replace(
        config,
        world=replace(
            config.world,
            seed=args.world_seed,
            reuse_rho=args.reuse_rho,
            tasks=args.tasks,
        ),
        evaluation=replace(
            config.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
        output_directory=args.output,
    )
    summary = learned_lifetime.run(config, kind=args.model)
    print(
        f"{args.model} rho={args.reuse_rho} world={args.world_seed} "
        f"tasks={args.tasks}: loss={summary['cumulative_prequential_gaussian_log_loss']:.1f}"
    )


if __name__ == "__main__":
    main()
