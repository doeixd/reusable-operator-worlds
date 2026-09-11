"""Independent J1c recomputation from durable cells and saved models.

Re-scores every saved stage-3 model on its regenerated world with an
independent NMSE, verifies the library-transfer chain from the saved stage
models themselves (not only from the runner's recorded hashes), and recomputes
the ordered classification of `J1C_LENGTH_CURRICULUM_PLAN.md`.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments.audit_rotated_g5r_interference import world_config
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import restore_model
from row.rotated_world import generate_rotated_world

REPORT = Path("reports/j1c_curriculum.json")
SO1 = Path("reports/so1_budget_bracket_r2.json")
CELLS = Path("artifacts/j1c_curriculum/cells")
WORLDS = (0, 1, 2)
THRESHOLD = 0.05
STAGES = {1: (60, 1), 2: (64, 2), 3: (None, 3)}


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stage_world(world_seed: int, stage: int):
    base = load_config("configs/v1.yaml")
    tasks, length = STAGES[stage]
    if stage == 3:
        cfg = world_config(base, world_seed)
        world = generate_rotated_world(cfg.world)
    else:
        cfg = replace(base, world=CurriculumWorldConfig.from_world(
            replace(base.world, seed=world_seed), program_length=length, tasks=tasks, stage=f"length-{length}"))
        world = generate_curriculum_world(cfg.world)
    return replace(cfg, discrete_model=replace(cfg.discrete_model, task_steps=length)), world


def library_sha(model) -> str:
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()


def main() -> int:
    torch.set_num_threads(1)
    report = json.loads(REPORT.read_text())
    baseline = {w: json.loads(SO1.read_text())["cells"]["L_b2_g131072"][str(w)]["terminal_median"] for w in WORLDS}
    problems, medians, harness = [], {}, True
    for arm in ("STAGED", "RESET"):
        for w in WORLDS:
            directory = CELLS / f"{arm}_w{w}"
            stored = json.loads((directory / "result.json").read_text())
            result = stored["result"]
            if sha_json(result) != stored["result_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{arm}_w{w}: record hash/commit mismatch")
            # Library transfer, recomputed from the saved stage models themselves.
            shas = {}
            for stage in (1, 2, 3):
                cfg, world = stage_world(w, stage)
                model = restore_model(directory / f"stage{stage}", cfg, world, build_fast)
                shas[stage] = library_sha(model)
                if stage == 3:
                    model.eval()
                    errors = []
                    with torch.no_grad():
                        for task in world.tasks:
                            y = np.asarray(task.eval_y, dtype=np.float64)
                            p = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id).numpy().astype(np.float64)
                            errors.append(np.mean((y - p) ** 2) / np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))
                    median = float(np.median(errors))
                    medians[(arm, w)] = median
                    if abs(median - result["terminal_median"]) > 1e-6:
                        problems.append(f"{arm}_w{w}: re-scored {median} vs recorded {result['terminal_median']}")
                if shas[stage] != result["stages"][str(stage)]["library_sha256"]:
                    problems.append(f"{arm}_w{w}: stage {stage} saved model differs from recorded library")
            for stage in (2, 3):
                carried = result["stages"][str(stage)]["library_sha256_at_start"]
                if arm == "RESET" and stage == 3:
                    harness &= carried is None
                else:
                    harness &= carried == shas[stage - 1]
            for stage in result["stages"].values():
                harness &= (math.isfinite(stage["terminal_median"]) and stage["shared_relative_change"] > 0
                            and stage["code_relative_change"] > 0)
            # Task ids must not be shared between stages (codes would be reused).
            ids = [set(result["stages"][str(s)]["final_per_task"]) for s in (1, 2, 3)]
            harness &= not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    staged = {w: medians[("STAGED", w)] for w in WORLDS}
    reset = {w: medians[("RESET", w)] for w in WORLDS}
    passing = [w for w in WORLDS if staged[w] <= THRESHOLD]
    if not harness:
        label = "HARNESS_FAILED"
    elif len(passing) >= 2 and all(staged[w] < reset[w] and staged[w] < baseline[w] for w in passing):
        label = "J1C_ACQUIRES"
    elif sum(staged[w] <= 0.5 * min(reset[w], baseline[w]) for w in WORLDS) >= 2:
        label = "J1C_IMPROVES"
    else:
        label = "J1C_FAILS"
    if report.get("classification") != label or report.get("complete") is not True:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "harness": harness, "staged": staged, "reset": reset,
                      "baseline": baseline, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
