"""Independent J1c-R recomputation from durable cells and saved models."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import restore_model

REPORT = Path("reports/j1cr_replication.json")
J1C = Path("reports/j1c_curriculum.json")
CELLS = Path("artifacts/j1cr_replication/cells")
WORLDS = (0, 1, 2)
THRESHOLD = 0.05
MODEL_SEED = 3001


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def library_sha(model) -> str:
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()


def main() -> int:
    torch.set_num_threads(1)
    report = json.loads(REPORT.read_text())
    j1c = {w: json.loads(J1C.read_text())["cells"][f"STAGED_w{w}"]["terminal_median"] for w in WORLDS}
    problems, medians, harness = [], {}, True
    for arm in ("STAGED-R", "NON-STAGED-R"):
        for w in WORLDS:
            directory = CELLS / f"{arm}_w{w}"
            stored = json.loads((directory / "result.json").read_text())
            result = stored["result"]
            if sha_json(result) != stored["result_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{arm}_w{w}: record hash/commit mismatch")
            harness &= result["model_seed"] == MODEL_SEED and stored["stamp"]["model_seed"] == MODEL_SEED
            shas = {}
            for stage in sorted(int(s) for s in result["stages"]):
                cfg, world, _, _ = stage_setup(w, stage, MODEL_SEED)
                model = restore_model(directory / f"stage{stage}", cfg, world, build_fast)
                shas[stage] = library_sha(model)
                if shas[stage] != result["stages"][str(stage)]["library_sha256"]:
                    problems.append(f"{arm}_w{w}: stage {stage} saved model differs from record")
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
            if arm == "STAGED-R":
                for stage in (2, 3):
                    harness &= result["stages"][str(stage)]["library_sha256_at_start"] == shas[stage - 1]
                ids = [set(result["stages"][s]["final_per_task"]) for s in ("1", "2", "3")]
                harness &= not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            for stage in result["stages"].values():
                harness &= (math.isfinite(stage["terminal_median"]) and stage["shared_relative_change"] > 0
                            and stage["code_relative_change"] > 0)
    staged = {w: medians[("STAGED-R", w)] for w in WORLDS}
    control = {w: medians[("NON-STAGED-R", w)] for w in WORLDS}
    passing = [w for w in WORLDS if staged[w] <= THRESHOLD]
    if not harness:
        label = "HARNESS_FAILED"
    elif len(passing) >= 2 and all(staged[w] < control[w] for w in passing):
        label = "REPLICATES"
    elif not passing:
        label = "FAILS_TO_REPLICATE"
    else:
        label = "PARTIAL"
    if report.get("classification") != label or report.get("complete") is not True:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "harness": harness, "staged_seed3001": staged,
                      "non_staged_seed3001": control, "staged_seed5000_j1c": j1c,
                      "ratio_to_j1c": {w: staged[w] / j1c[w] for w in WORLDS}, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
