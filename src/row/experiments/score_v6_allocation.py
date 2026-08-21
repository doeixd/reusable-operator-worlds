"""Score V6's plasticity-allocation matrix without treating it as an H30 rescue.

The intervention varies how many of twelve shared basis slots remain plastic
after task 8.  Its primary question is descriptive and structural: where does
recurring computation reside when optimization is allowed to move different
parts of the learner?  Fertility and schema economics are reported, but a
positive cell cannot retroactively rescue the separately closed H30/H35 test.

All 45 lifetime artifacts are validated before any checkpoint audit runs.
The scorer then reuses the repaired V6 fertility, effective-operator, and
learned-schema instruments with their protocols made explicit here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ARMS = ("ordinary", "replay", "prospective")


def artifact_path(root: Path, free: int, arm: str, world: int) -> Path:
    return root / f"free{free}" / arm / f"world_{world}" / "lifecycle"


def validate_artifact(path: Path, free: int, arm: str, world: int,
                      slots: int, outer_steps: int,
                      inner_steps: int) -> dict[str, object]:
    required = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise SystemExit(f"incomplete allocation cell {path}: missing {missing}")

    provenance = json.loads((path / "rho_profile.json").read_text(encoding="utf-8"))
    protocol = provenance.get("v6_arm") or {}
    expected = {
        "arm": arm,
        "freeze_basis_at": 8,
        "freeze_slots": slots - free,
        "operator_slots": slots,
        "prospective_steps": outer_steps,
        "prospective_inner_steps": inner_steps,
        "sleeps": [16, 24, 32, 48, 64],
        "lifecycle": True,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"allocation protocol mismatch at {path}: {mismatches}")

    fingerprint = json.loads((path / "fingerprint.json").read_text(encoding="utf-8"))
    if int(fingerprint.get("world_seed", -1)) != world:
        raise SystemExit(f"world-seed mismatch at {path}")
    return {
        "path": str(path),
        "git_commit": fingerprint.get("git_commit"),
        "resolved_config_sha256": fingerprint.get("resolved_config_sha256"),
        "protocol": protocol,
    }


def run_report(module: str, arguments: list[str], output: Path) -> dict:
    command = [sys.executable, "-m", module, *arguments, "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"{module} failed with exit {result.returncode}:\n{result.stderr[-1200:]}"
        )
    if not output.exists():
        raise SystemExit(f"{module} exited cleanly without writing {output}")
    return json.loads(output.read_text(encoding="utf-8"))


def basic_cell(path: Path, free: int, arm: str, world: int) -> dict[str, object]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    promotion = summary["promotion"]
    diagnostics = summary["residual_diagnostics"]
    return {
        "free_slots": free,
        "arm": arm,
        "world": world,
        "library_size": promotion["library_size"],
        "tasks_reusing_library": promotion["tasks_reusing_library"],
        "task_state_scalars": summary["task_state_scalar_count"],
        "shared_parameter_count": summary["shared_parameter_count"],
        "current_prequential_loss": summary[
            "cumulative_prequential_gaussian_log_loss"
        ],
        "mean_residual_to_parent_ratio": diagnostics[
            "mean_functional_residual_to_parent_update_ratio"
        ],
    }


def mean(values) -> float:
    return float(np.mean(list(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v6_alloc"))
    parser.add_argument("--free", type=int, nargs="+", default=[0, 1, 2, 3, 6])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--outer-steps", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=16)
    parser.add_argument("--adaptation-steps", type=int, default=40)
    parser.add_argument("--support", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--reuse-components-from", type=Path,
        help="reuse expensive component audits from a prior complete report; "
             "the current 45-cell artifact grid is still revalidated",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v6_allocation.json"))
    args = parser.parse_args()

    frees = list(dict.fromkeys(args.free))
    arms = list(dict.fromkeys(args.arms))
    worlds = list(dict.fromkeys(args.worlds))
    if not frees or not arms or not worlds:
        raise SystemExit("allocation grid dimensions must be non-empty")
    if "ordinary" not in arms:
        raise SystemExit("ordinary is required because Phi is paired against it")

    sources = []
    cells = []
    for free in frees:
        if not 0 <= free <= args.slots:
            raise SystemExit(f"invalid free-slot count {free}")
        for arm in arms:
            for world in worlds:
                path = artifact_path(args.root, free, arm, world)
                sources.append(validate_artifact(
                    path, free, arm, world, args.slots,
                    args.outer_steps, args.inner_steps,
                ))
                cells.append(basic_cell(path, free, arm, world))

    components: dict[str, object] = {}
    if args.reuse_components_from:
        prior = json.loads(
            args.reuse_components_from.read_text(encoding="utf-8")
        )
        expected_protocol = {
            "free_slots": frees,
            "arms": arms,
            "worlds": worlds,
            "operator_slots": args.slots,
            "freeze_basis_at": 8,
            "prospective_outer_steps": args.outer_steps,
            "prospective_inner_steps": args.inner_steps,
            "adaptation_steps": args.adaptation_steps,
            "support": args.support,
        }
        if prior.get("protocol") != expected_protocol:
            raise SystemExit(
                "component-cache protocol mismatch: "
                f"{prior.get('protocol')} != {expected_protocol}"
            )
        components = prior["components"]
        if set(components) != {str(free) for free in frees}:
            raise SystemExit("component cache does not cover every free-slot setting")
    else:
        with tempfile.TemporaryDirectory(prefix="row_v6_allocation_") as temporary:
            temporary_root = Path(temporary)
            for free in frees:
                root = args.root / f"free{free}"
                common = ["--root", str(root), "--arms", *arms,
                          "--worlds", *map(str, worlds)]
                fertility = run_report(
                    "row.experiments.score_v6_fertility",
                    [*common, "--steps", str(args.adaptation_steps),
                     "--support", *map(str, args.support)],
                    temporary_root / f"free{free}_fertility.json",
                )
                schema = run_report(
                    "row.experiments.audit_learned_schema",
                    ["--root", str(root), "--conditions", *arms,
                     "--worlds", *map(str, worlds)],
                    temporary_root / f"free{free}_schema.json",
                )
                effective = {}
                for arm in arms:
                    effective[arm] = run_report(
                        "row.experiments.audit_effective_operator",
                        ["--root", str(root), "--conditions", arm,
                         "--worlds", *map(str, worlds),
                         "--config", "configs/v5_h72.yaml"],
                        temporary_root / f"free{free}_{arm}_effective.json",
                    )
                components[str(free)] = {
                    "fertility": fertility,
                    "schema": schema,
                    "effective": effective,
                }

    summary: dict[str, object] = {}
    for free in frees:
        free_components = components[str(free)]
        schema_cells = free_components["schema"]["cells"]
        free_summary = {}
        for arm in arms:
            selected = [
                cell for cell in cells
                if cell["free_slots"] == free and cell["arm"] == arm
            ]
            schema_selected = [
                cell for cell in schema_cells
                if cell["condition"] == arm and "reason" not in cell
            ]
            margins = [
                cell["compress_total"] - cell["factorize_total"]
                for cell in schema_selected
            ]
            fertility = free_components["fertility"]["summary"][arm]
            effective = free_components["effective"][arm]
            row = {
                "worlds": len(selected),
                "library_size_mean": mean(c["library_size"] for c in selected),
                "library_size_per_world": [c["library_size"] for c in selected],
                "tasks_reusing_mean": mean(
                    c["tasks_reusing_library"] for c in selected
                ),
                "task_state_scalars_mean": mean(
                    c["task_state_scalars"] for c in selected
                ),
                "current_prequential_loss_mean": mean(
                    c["current_prequential_loss"] for c in selected
                ),
                "current_prequential_loss_per_world": [
                    c["current_prequential_loss"] for c in selected
                ],
                "mean_residual_to_parent_ratio": mean(
                    c["mean_residual_to_parent_ratio"] for c in selected
                ),
                "r_effective": effective["mean_r_effective"],
                "r_effective_per_world": [
                    cell["r_effective"] for cell in effective["rows"]
                ],
                "r_effective_null": effective["null"],
                "related_adaptation": fertility["related"],
                "unrelated_adaptation": fertility["unrelated"],
                "schema_scoreable_worlds": len(schema_selected),
                "factorize_wins": int(sum(margin > 0 for margin in margins)),
                "factorize_margin_mean": mean(margins) if margins else None,
                "factorize_margin_per_world": margins,
            }
            for name in (
                "phi_related", "phi_unrelated", "phi_specific",
                "phi_per_world", "phi_sd", "worlds_positive",
            ):
                if name in fertility:
                    row[name] = fertility[name]
            free_summary[arm] = row
        summary[str(free)] = free_summary

    for free in frees:
        ordinary = summary[str(free)]["ordinary"]
        for arm in arms:
            row = summary[str(free)][arm]
            row["current_loss_delta_vs_ordinary_per_world"] = [
                value - reference
                for value, reference in zip(
                    row["current_prequential_loss_per_world"],
                    ordinary["current_prequential_loss_per_world"],
                )
            ]
            row["current_loss_delta_vs_ordinary_mean"] = mean(
                row["current_loss_delta_vs_ordinary_per_world"]
            )
            row["r_effective_delta_vs_ordinary"] = (
                row["r_effective"] - ordinary["r_effective"]
            )

    trend_fields = (
        "library_size_mean",
        "r_effective",
        "current_loss_delta_vs_ordinary_mean",
        "task_state_scalars_mean",
        "mean_residual_to_parent_ratio",
        "phi_related",
        "factorize_margin_mean",
    )
    allocation_trends = {}
    for arm in arms:
        arm_trends = {"free_slots": frees}
        for field in trend_fields:
            points = [
                (free, summary[str(free)][arm].get(field)) for free in frees
            ]
            available = [(free, value) for free, value in points
                         if value is not None]
            values = [value for _, value in available]
            trend = {
                "values": values,
                "free_slots": [free for free, _ in available],
            }
            if len(available) >= 3 and len(set(values)) > 1:
                statistic, pvalue = spearmanr(
                    [free for free, _ in available], values
                )
                trend.update({
                    "spearman": float(statistic),
                    "pvalue_descriptive": float(pvalue),
                })
            arm_trends[field] = trend
        library_means = arm_trends["library_size_mean"]["values"]
        arm_trends["library_size_monotone_nonincreasing"] = all(
            later <= earlier
            for earlier, later in zip(library_means, library_means[1:])
        )
        arm_trends["free0_minus_free6_library_size"] = (
            library_means[0] - library_means[-1]
        )
        allocation_trends[arm] = arm_trends

    reliable_fertility = []
    for free in frees:
        row = summary[str(free)].get("prospective")
        if not row:
            continue
        if (
            row.get("phi_related", 0) > 0
            and row.get("worlds_positive", 0) == len(worlds)
            and row["phi_related"] > row.get("phi_sd", float("inf"))
        ):
            reliable_fertility.append(free)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_scope": (
            "plasticity allocation; descriptive and independent of the closed "
            "H30/H35 prospective-pressure claim"
        ),
        "protocol": {
            "free_slots": frees,
            "arms": arms,
            "worlds": worlds,
            "operator_slots": args.slots,
            "freeze_basis_at": 8,
            "prospective_outer_steps": args.outer_steps,
            "prospective_inner_steps": args.inner_steps,
            "adaptation_steps": args.adaptation_steps,
            "support": args.support,
        },
        "summary": summary,
        "allocation_trends": allocation_trends,
        "reliable_prospective_fertility_settings": reliable_fertility,
        "sources": sources,
        "cells": cells,
        "components": components,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)

    print("V6 ALLOCATION MATRIX - descriptive, not an H30 rescue")
    print("free arm          M  R_eff      Phi  F wins current loss")
    for free in frees:
        for arm in arms:
            row = summary[str(free)][arm]
            phi = row.get("phi_related")
            phi_text = "baseline" if phi is None else f"{phi:+.3f}"
            print(
                f"{free:>4} {arm:<11} {row['library_size_mean']:>4.1f} "
                f"{row['r_effective']:>6.3f} {phi_text:>9} "
                f"{row['factorize_wins']:>2}/{row['schema_scoreable_worlds']:<2} "
                f"{row['current_prequential_loss_mean']:>12.1f}"
            )
    print(f"reliable prospective-fertility settings: {reliable_fertility or 'none'}")


if __name__ == "__main__":
    main()
