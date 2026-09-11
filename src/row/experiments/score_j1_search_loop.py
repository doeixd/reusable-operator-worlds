"""Independent J1 recomputation from durable cells and saved models.

Re-reads every durable record (hash-checked), re-scores each saved terminal
model on the regenerated world's query data with an independent NMSE, checks
the equivalence records against SO1's oracle cell, and recomputes the ordered
classification of `J1_SEARCH_IN_THE_LOOP_PLAN.md` without the runner's
decision function. Fails unless the report agrees.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_rotated_g5r_interference import world_config
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import restore_model
from row.rotated_world import generate_rotated_world

REPORT = Path("reports/j1_search_loop.json")
SO1 = Path("reports/so1_budget_bracket_r2.json")
CELLS = Path("artifacts/j1_search_loop/cells")
THRESHOLD = 0.05


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    torch.set_num_threads(1)
    report = json.loads(REPORT.read_text())
    so1 = json.loads(SO1.read_text())["cells"]
    problems, medians, gate = [], {}, True
    for w in (0, 1, 2):
        stored = json.loads((CELLS / f"EQ_w{w}" / "result.json").read_text())
        reference = so1["O_b2_g131072"][str(w)]["checkpoints"]["8192"]["per_task"]
        gate &= stored["result"]["checkpoint_per_task"]["8192"] == reference
        gate &= sha_json(stored["result"]) == stored["result_sha256"]
    for arm in ("J1", "SHAM"):
        for w in (0, 1, 2):
            directory = CELLS / f"{arm}_w{w}"
            stored = json.loads((directory / "result.json").read_text())
            if sha_json(stored["result"]) != stored["result_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{arm}_w{w}: record hash/commit mismatch")
            for name, sha in stored["artifact_sha256"].items():
                if sha_file(directory / name) != sha:
                    problems.append(f"{arm}_w{w}: artifact {name} hash mismatch")
            cfg = world_config(load_config("configs/v1.yaml"), w)
            world = generate_rotated_world(cfg.world)
            model = restore_model(directory, cfg, world, build_fast)
            model.eval()
            errors = []
            with torch.no_grad():
                for task in world.tasks:
                    route = stored["result"]["routes"][task.task_id]
                    if torch.argmax(model.task_codes[task.task_id], dim=-1).tolist() != route:
                        problems.append(f"{arm}_w{w}: saved route differs for {task.task_id}")
                    y = np.asarray(task.eval_y, dtype=np.float64)
                    p = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id).numpy().astype(np.float64)
                    errors.append(np.mean((y - p) ** 2) / np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))
            median = float(np.median(errors))
            if abs(median - stored["result"]["terminal_median"]) > 1e-6:
                problems.append(f"{arm}_w{w}: re-scored median {median} vs recorded {stored['result']['terminal_median']}")
            medians[(arm, w)] = median
            r = stored["result"]
            gate &= r["finite"] and r["pinned_one_hot"] and r["shared_relative_change"] > 0 and r["reload_exact"] \
                and all(x["argmin_ok"] for x in r["rounds"])
    learned = {w: so1["L_b2_g131072"][str(w)]["terminal_median"] for w in (0, 1, 2)}
    j1 = {w: medians[("J1", w)] for w in (0, 1, 2)}
    sham = {w: medians[("SHAM", w)] for w in (0, 1, 2)}
    passing = [w for w in (0, 1, 2) if j1[w] <= THRESHOLD]
    if not gate:
        label = "HARNESS_FAILED"
    elif len(passing) >= 2 and all(j1[w] < sham[w] and j1[w] < learned[w] for w in passing):
        label = "J1_ACQUIRES"
    elif sum(j1[w] <= 0.5 * min(sham[w], learned[w]) for w in (0, 1, 2)) >= 2:
        label = "J1_IMPROVES"
    else:
        label = "J1_FAILS"
    if report.get("classification") != label or report.get("complete") is not True:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "gate": gate, "j1": j1, "sham": sham, "learned": learned,
                      "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
