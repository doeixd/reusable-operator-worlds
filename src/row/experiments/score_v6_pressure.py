"""Score H35's prospective-pressure curve on complete, validated artifacts.

The ordinary arm is pressure zero. Low-pressure prospective arms live under
``artifacts/v6_lowp/o{steps}``; the already-scored eight-step arm lives under
``artifacts/v6_clean/prospective``. Every representation is evaluated with the
same frozen-representation adaptor used by the corrected V6 fertility scorer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from row.config import load_config
from row.experiments.audit_effective_operator import load_learner
from row.experiments.score_v6_fertility import adapt_cost
from row.meta_world import MetaFamilySpec, generate_meta_world


def _artifact_path(args, pressure: int, world: int) -> Path:
    if pressure == 0:
        return args.baseline_root / f"world_{world}" / "lifecycle"
    if pressure == args.reference_pressure:
        return args.reference_root / f"world_{world}" / "lifecycle"
    return args.low_root / f"o{pressure}" / f"world_{world}" / "lifecycle"


def _validate_artifact(path: Path, pressure: int, world: int,
                       reference_pressure: int) -> dict[str, object]:
    required = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise SystemExit(f"incomplete H35 cell {path}: missing {missing}")

    provenance = json.loads((path / "rho_profile.json").read_text(encoding="utf-8"))
    protocol = provenance.get("v6_arm") or {}
    expected_arm = "ordinary" if pressure == 0 else "prospective"
    if protocol.get("arm") != expected_arm:
        raise SystemExit(
            f"protocol mismatch at {path}: arm={protocol.get('arm')!r}, "
            f"expected {expected_arm!r}"
        )
    if pressure != 0 and int(protocol.get("prospective_steps", -1)) != pressure:
        raise SystemExit(
            f"protocol mismatch at {path}: outer steps="
            f"{protocol.get('prospective_steps')!r}, expected {pressure}"
        )
    if pressure != 0 and int(protocol.get("prospective_inner_steps", -1)) != 8:
        raise SystemExit(f"H35 cell {path} did not use 8 inner steps")
    if protocol.get("freeze_basis_at") is not None or protocol.get("freeze_slots") is not None:
        raise SystemExit(f"H35 cell {path} is not the fully unfrozen regime")

    fingerprint = json.loads((path / "fingerprint.json").read_text(encoding="utf-8"))
    if int(fingerprint.get("world_seed", -1)) != world:
        raise SystemExit(f"world-seed mismatch at {path}")
    return {
        "path": str(path),
        "git_commit": fingerprint.get("git_commit"),
        "resolved_config_sha256": fingerprint.get("resolved_config_sha256"),
        "protocol": protocol,
        "reference_pressure": reference_pressure,
    }


def _mean_nested(cells: list[dict], key: str, metric: str,
                 support: int) -> float:
    return float(np.mean([
        np.mean([task[metric] for task in cell[key][str(support)]])
        for cell in cells
    ]))


def _paired_effect(baseline_cells: list[dict], pressure_cells: list[dict],
                   task_set: str, support: int) -> dict[str, object]:
    effects = []
    for cell in pressure_cells:
        base = next(reference for reference in baseline_cells
                    if reference["world"] == cell["world"])
        base_cost = np.mean([
            task["prequential"] for task in base[task_set][str(support)]
        ])
        cell_cost = np.mean([
            task["prequential"] for task in cell[task_set][str(support)]
        ])
        effects.append(float(base_cost - cell_cost))
    return {
        "mean": float(np.mean(effects)),
        "per_world": effects,
        "sd": float(np.std(effects)),
        "worlds_positive": int(sum(value > 0 for value in effects)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v5_h72.yaml"))
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("artifacts/v6_clean/ordinary"))
    parser.add_argument("--low-root", type=Path, default=Path("artifacts/v6_lowp"))
    parser.add_argument("--reference-root", type=Path,
                        default=Path("artifacts/v6_clean/prospective"))
    parser.add_argument("--reference-pressure", type=int, default=8)
    parser.add_argument("--pressures", type=int, nargs="+", default=[0, 1, 2, 8])
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--support", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--steps", type=int, default=40,
        help="adaptation steps; 40 matches the first valid V6 fertility result",
    )
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v6_h35_pressure.json"))
    args = parser.parse_args()

    pressures = list(dict.fromkeys(args.pressures))
    if not pressures or pressures[0] != 0:
        raise SystemExit("--pressures must start with the ordinary pressure-0 baseline")

    # Fail before loading or adapting any model when even one cell is missing or
    # mismatched. Scoring the complete baseline and only then discovering a
    # missing pressure cell wastes minutes and can leave a misleading partial
    # impression even though no report is written.
    paths: dict[tuple[int, int], Path] = {}
    sources: list[dict[str, object]] = []
    for world in args.worlds:
        for pressure in pressures:
            path = _artifact_path(args, pressure, world)
            paths[(world, pressure)] = path
            sources.append(_validate_artifact(
                path, pressure, world, args.reference_pressure))

    config = load_config(args.config)
    cells: list[dict[str, object]] = []
    for world in args.worlds:
        spec = MetaFamilySpec(families=4, tasks_per_family=16, r_meta=1.0,
                              subspace_rank=2)
        world_config = replace(config.world, seed=world, tasks=spec.total_tasks)
        generated = generate_meta_world(world_config, spec)
        task_sets = {
            "related": list(generated.novel_family_tasks),
            "unrelated": list(generated.unseen_unrelated_tasks),
            "within": list(generated.held_out_family_tasks),
        }
        if any(not tasks for tasks in task_sets.values()):
            raise SystemExit(f"world {world} has an empty H35 evaluation set")

        for pressure in pressures:
            path = paths[(world, pressure)]
            model = load_learner(config, path, args.slots, kind="prospective")
            scored: dict[str, object] = {
                "world": world,
                "pressure": pressure,
                "current_prequential_loss": json.loads(
                    (path / "summary.json").read_text(encoding="utf-8")
                )["cumulative_prequential_gaussian_log_loss"],
            }
            for name, tasks in task_sets.items():
                scored[name] = {
                    str(support): [
                        adapt_cost(model, task, args.steps, support, args.inner_lr)
                        for task in tasks
                    ]
                    for support in args.support
                }
            cells.append(scored)

    summary: dict[str, object] = {}
    baseline_cells = [cell for cell in cells if cell["pressure"] == 0]
    primary_support = min(args.support)
    for pressure in pressures:
        pressure_cells = [cell for cell in cells if cell["pressure"] == pressure]
        row: dict[str, object] = {
            "worlds": len(pressure_cells),
            "adaptation": {
                task_set: {
                    metric: {
                        str(support): _mean_nested(
                            pressure_cells, task_set, metric, support)
                        for support in args.support
                    }
                    for metric in ("prequential", "endpoint", "steps_to_target")
                }
                for task_set in ("related", "unrelated", "within")
            },
            # Compatibility aliases for the primary related-future curves.
            "related": {
                str(support): _mean_nested(
                    pressure_cells, "related", "prequential", support)
                for support in args.support
            },
            "endpoint": {
                str(support): _mean_nested(
                    pressure_cells, "related", "endpoint", support)
                for support in args.support
            },
            "current_prequential_loss": float(np.mean([
                cell["current_prequential_loss"] for cell in pressure_cells
            ])),
        }
        if pressure != 0:
            current_delta = []
            for cell in pressure_cells:
                base = next(reference for reference in baseline_cells
                            if reference["world"] == cell["world"])
                current_delta.append(float(
                    cell["current_prequential_loss"]
                    - base["current_prequential_loss"]
                ))
            effects = {
                task_set: _paired_effect(
                    baseline_cells, pressure_cells, task_set, primary_support)
                for task_set in ("related", "unrelated", "within")
            }
            row.update({
                "paired_effects": effects,
                "phi_related": effects["related"]["mean"],
                "phi_unrelated": effects["unrelated"]["mean"],
                "phi_within": effects["within"]["mean"],
                "phi_per_world": effects["related"]["per_world"],
                "phi_sd": effects["related"]["sd"],
                "worlds_positive": effects["related"]["worlds_positive"],
                "current_loss_delta_per_world": current_delta,
                "current_loss_delta_mean": float(np.mean(current_delta)),
                "current_loss_delta_sd": float(np.std(current_delta)),
                "current_loss_worlds_improved": int(sum(
                    value < 0 for value in current_delta
                )),
            })
        summary[str(pressure)] = row

    beneficial = [
        pressure for pressure in pressures if pressure != 0
        and summary[str(pressure)]["phi_related"] > 0
        and summary[str(pressure)]["worlds_positive"] == len(args.worlds)
        and summary[str(pressure)]["phi_related"] > summary[str(pressure)]["phi_sd"]
    ]
    lifetime_beneficial = [
        pressure for pressure in pressures if pressure != 0
        and summary[str(pressure)]["current_loss_delta_mean"] < 0
        and summary[str(pressure)]["current_loss_worlds_improved"] == len(args.worlds)
        and abs(summary[str(pressure)]["current_loss_delta_mean"])
        > summary[str(pressure)]["current_loss_delta_sd"]
    ]
    lifetime_harmful = [
        pressure for pressure in pressures if pressure != 0
        and summary[str(pressure)]["current_loss_delta_mean"] > 0
        and summary[str(pressure)]["current_loss_worlds_improved"] == 0
        and summary[str(pressure)]["current_loss_delta_mean"]
        > summary[str(pressure)]["current_loss_delta_sd"]
    ]
    phi_nonmonotonic = bool(
        beneficial
        and any(summary[str(later)]["phi_related"] < 0
                for later in pressures if later > min(beneficial))
    )
    lifetime_u_shape = bool(
        lifetime_beneficial
        and any(later in lifetime_harmful
                for later in pressures if later > min(lifetime_beneficial))
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "paired ordinary-minus-pressure cumulative adaptation-trajectory cost",
        "evaluation_protocol": {
            "worlds": args.worlds,
            "support": args.support,
            "adaptation_steps": args.steps,
            "inner_lr": args.inner_lr,
            "sigma": 0.1,
        },
        "primary_support": primary_support,
        "pressures": pressures,
        "beneficial_pressures": beneficial,
        "lifetime_beneficial_pressures": lifetime_beneficial,
        "lifetime_harmful_pressures": lifetime_harmful,
        "h35_phi_nonmonotonic_supported": phi_nonmonotonic,
        "h35_lifetime_u_shape_supported": lifetime_u_shape,
        "h35_nonmonotonic_supported": phi_nonmonotonic and lifetime_u_shape,
        "summary": summary,
        "sources": sources,
        "cells": cells,
    }

    # Commit the complete report atomically before presentation. A console
    # encoding failure must not discard a finished scientific computation.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)

    print("H35 PRESSURE CURVE - positive Phi means cheaper future acquisition")
    print(f"{'outer':>7} {'Phi k=1':>12} {'positive':>10} "
          f"{'current delta':>12} {'related k=1':>14}")
    for pressure in pressures:
        row = summary[str(pressure)]
        phi = 0.0 if pressure == 0 else row["phi_related"]
        positive = "baseline" if pressure == 0 else (
            f"{row['worlds_positive']}/{row['worlds']}")
        current = 0.0 if pressure == 0 else row["current_loss_delta_mean"]
        print(f"{pressure:>7} {phi:>12.3f} {positive:>10} "
              f"{current:>12.1f} {row['related'][str(primary_support)]:>14.3f}")
    print(f"H35 Phi optimum: "
          f"{'SUPPORTED' if phi_nonmonotonic else 'NOT SUPPORTED'}")
    print(f"H35 lifetime-cost U-shape: "
          f"{'SUPPORTED' if lifetime_u_shape else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
