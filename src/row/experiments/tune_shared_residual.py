"""Tune rank-two shared-parent residual optimization at intermediate reuse."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config, run
from row.provenance import validate_artifact


RESIDUAL_LEARNING_RATES = (0.001, 0.005, 0.01)
RESIDUAL_PENALTIES = (0.001, 0.01, 0.1)
ESCAPE_HATCH_MAX_RATIO = 1.0


def _select(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if float(record["maximum_task_functional_ratio"])
        < ESCAPE_HATCH_MAX_RATIO
    ]
    if not eligible:
        raise ValueError("every shared-residual tuning run violates the escape-hatch guard")
    selected = min(eligible, key=lambda record: float(record["gaussian_log_loss"]))
    return {
        "criterion": (
            "lowest cumulative Gaussian log loss among rank-two runs whose "
            "maximum task residual-to-parent functional ratio is below 1.0"
        ),
        "escape_hatch_max_ratio": ESCAPE_HATCH_MAX_RATIO,
        "selected": selected,
        "records": sorted(
            records,
            key=lambda row: (
                float(row["residual_learning_rate"]),
                float(row["residual_penalty"]),
            ),
        ),
    }


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/shared_residual_tuning")
    )
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.75)
    parser.add_argument(
        "--residual-learning-rates",
        type=float,
        nargs="+",
        default=list(RESIDUAL_LEARNING_RATES),
    )
    parser.add_argument(
        "--residual-penalties",
        type=float,
        nargs="+",
        default=list(RESIDUAL_PENALTIES),
    )
    args = parser.parse_args()

    base = load_config(args.config)
    base = replace(
        base,
        world=replace(base.world, seed=args.world_seed, reuse_rho=args.rho),
        evaluation=replace(
            base.evaluation,
            lifetime_checkpoints=(),
            checkpoint_novel_tasks=1,
            extended_diagnostics=False,
        ),
    )
    records = []
    total = len(args.residual_learning_rates) * len(args.residual_penalties)
    completed = 0
    for residual_lr in args.residual_learning_rates:
        for penalty in args.residual_penalties:
            completed += 1
            output = (
                args.output
                / f"residual_lr_{_label(residual_lr)}"
                / f"penalty_{_label(penalty)}"
            )
            config = replace(
                base,
                shared_residual_model=replace(
                    base.shared_residual_model,
                    residual_learning_rate=residual_lr,
                    residual_penalty=penalty,
                ),
                output_directory=output,
            )
            summary_path = output / "summary.json"
            print(
                f"[{completed}/{total}] residual_lr={residual_lr:g} penalty={penalty:g}",
                flush=True,
            )
            if summary_path.exists():
                validate_artifact(
                    output,
                    resolved_learned_config(
                        config, "shared_residual", "forward"
                    ),
                    "shared_residual",
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run(config, kind="shared_residual")
                gc.collect()
            diagnostic = summary["residual_diagnostics"]
            records.append(
                {
                    "world_seed": args.world_seed,
                    "configured_rho": args.rho,
                    "residual_learning_rate": residual_lr,
                    "residual_penalty": penalty,
                    "gaussian_log_loss": summary[
                        "cumulative_prequential_gaussian_log_loss"
                    ],
                    "novel_32_shot_nmse": summary["novel_composition"][
                        "nmse_by_support"
                    ]["32"],
                    "mean_functional_ratio": diagnostic[
                        "mean_functional_residual_to_parent_update_ratio"
                    ],
                    "maximum_task_functional_ratio": diagnostic[
                        "maximum_task_functional_residual_to_parent_update_ratio"
                    ],
                    "task_state_scalar_count": summary["task_state_scalar_count"],
                }
            )

    report = _select(records)
    report["scope"] = (
        f"development world {args.world_seed}, configured rho {args.rho}; tuning only"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tuning.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
