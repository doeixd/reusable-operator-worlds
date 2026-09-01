"""Correct the invalid H-LBFGS arm under G5R_DIAGNOSIS_CORRECTION.md."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from row.config import load_config
from row.experiments.audit_rotated_g5r_diagnosis import (
    CELL_THRESHOLD,
    LBFGS_LR,
    LBFGS_STEPS,
    PRIMITIVES,
    QUERY_EXAMPLES,
    TRAIN_EXAMPLES,
    WORLDS,
    WORLDS_REQUIRED,
    arm_summary,
    assign_slots,
    examples,
    git_commit,
    require_clean_code,
    stage_b_cell,
    write,
)
from row.rotated_world import generate_rotated_world

FROZEN_PLAN = "G5R_DIAGNOSIS_CORRECTION.md"
FROZEN_PLAN_COMMIT = "c4e557d"
ORIGINAL_REPORT = "reports/rotated_g5r_diagnosis.json"
ORIGINAL_INSTRUMENT_COMMIT = "3b3820a"
ARM = "H-LBFGS"


def correction_protocol() -> dict:
    return {
        "worlds": list(WORLDS),
        "primitives": PRIMITIVES,
        "train_examples": TRAIN_EXAMPLES,
        "query_examples": QUERY_EXAMPLES,
        "arm": ARM,
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_steps": LBFGS_STEPS,
        "line_search": "strong_wolfe",
        "tolerance_grad": 1e-9,
        "tolerance_change": 1e-12,
        "objective": "data_mse",
        "weight_penalty": 0.0,
        "cell_threshold": CELL_THRESHOLD,
        "cells_per_world": 5,
        "worlds_required": WORLDS_REQUIRED,
        "original_report": ORIGINAL_REPORT,
        "original_instrument_commit": ORIGINAL_INSTRUMENT_COMMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rotated_g5r_diagnosis_lbfgs_correction.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("preregistration check failed")
    require_clean_code(args.output)

    expected_protocol = correction_protocol()
    if args.output.exists():
        out = json.loads(args.output.read_text(encoding="utf-8"))
        if out.get("protocol") != expected_protocol:
            raise SystemExit("existing correction report has a different protocol")
        if out.get("git_commit") != git_commit():
            raise SystemExit("existing correction report came from a different commit")
    else:
        out = {
            "frozen_plan": FROZEN_PLAN,
            "frozen_plan_commit": FROZEN_PLAN_COMMIT,
            "git_commit": git_commit(),
            "protocol": expected_protocol,
            "cells": {},
        }

    config = load_config(args.config)
    for world_seed in WORLDS:
        world = generate_rotated_world(replace(config.world, seed=world_seed))
        assignment = assign_slots(
            world.library,
            config.world.state_dim,
            config.discrete_model.seed,
            config.discrete_model.operator_slots,
        )
        for primitive_index, teacher in enumerate(world.library):
            key = f"w{world_seed}_p{primitive_index}_{ARM}"
            if key in out["cells"]:
                continue
            slot = assignment[primitive_index]
            seed = config.discrete_model.seed + 997 * slot
            data = examples(world_seed, primitive_index, teacher)
            result = stage_b_cell(teacher, seed, data, ARM)
            result.update(
                {
                    "world": world_seed,
                    "primitive": primitive_index,
                    "slot": slot,
                    "arm": ARM,
                }
            )
            out["cells"][key] = result
            write(out, args.output)
            print(
                f"[corrected {ARM} w{world_seed} p{primitive_index}] query "
                f"{result['final_query_nmse']:.5f} -> "
                f"{'PASS' if result['passes'] else 'FAIL'}",
                flush=True,
            )

    if len(out["cells"]) != len(WORLDS) * PRIMITIVES:
        raise RuntimeError("correction report does not contain exactly 18 cells")
    out["summary"] = arm_summary(out["cells"], ARM)
    out["classification"] = (
        "H_LBFGS_CORRECTED_PASS"
        if out["summary"]["passes"]
        else "H_LBFGS_CORRECTED_FAIL"
    )
    out["primary_diagnosis_unchanged"] = True
    out["complete"] = True
    write(out, args.output)
    print(out["classification"])


if __name__ == "__main__":
    main()
