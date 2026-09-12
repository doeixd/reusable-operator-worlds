"""J1c-R: does staged formation replicate at a second initialization?

Frozen in `J1CR_REPLICATION_PLAN.md` (9d62cc7). Identical to J1c
(`audit_j1c_curriculum`, reused unchanged) except the learner's initialization
seed, 3001 instead of 5000, plus a matched NON-STAGED arm at the same seed
(65,536 updates on the canonical world, SO1 stream 103) because SO1's reused
baseline was seed 5000.

Restartable: one durable hashed record per (arm, world); relaunch resumes;
timestamped run.log and atomic status.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import (
    STAGES, THRESHOLD, WORLDS, run_arm, stage_setup, train_stage,
)
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, save_model, writer_lock,
)

PLAN = Path("J1CR_REPLICATION_PLAN.md")
J1C_PLAN = Path("J1C_LENGTH_CURRICULUM_PLAN.md")
J1C_REPORT = Path("reports/j1c_curriculum.json")
OUTPUT = Path("reports/j1cr_replication.json")
ROOT = Path("artifacts/j1cr_replication")
PROTOCOL_ID = "J1cR-replication-v1"
MODEL_SEED = 3001
TOTAL_UPDATES = sum(u for _, _, u, _, _ in STAGES)  # 65,536


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "replicates": J1C_PLAN.as_posix(),
            "model_seed": MODEL_SEED, "worlds": list(WORLDS), "threshold": THRESHOLD,
            "stages": [{"length": L, "tasks": n, "updates": u} for L, n, u, _, _ in STAGES],
            "non_staged_updates": TOTAL_UPDATES,
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, J1C_PLAN, J1C_REPORT)},
            "environment": environment()}


def run_non_staged(world_seed: int, artifact: Path | None = None, scale: int = 1) -> dict:
    """The matched control: the canonical world alone, same total budget and seed."""
    torch.set_num_threads(1)
    cfg, world, _, stream = stage_setup(world_seed, 3, MODEL_SEED)
    model, result = train_stage(cfg, world, max(1, TOTAL_UPDATES // scale), stream)
    if artifact is not None:
        save_model(artifact / "stage3", model, cfg)
    return {"arm": "NON-STAGED-R", "world": world_seed, "model_seed": MODEL_SEED,
            "stages": {"3": result}, "transfers": [], "survival": [],
            "terminal_median": result["terminal_median"], "persisting_pairings": None}


def classify(cells: dict) -> str:
    expected = {f"{a}_w{w}" for a in ("STAGED-R", "NON-STAGED-R") for w in WORLDS}
    if not expected <= set(cells):
        return "HARNESS_FAILED"
    for name, cell in cells.items():
        if cell["model_seed"] != MODEL_SEED:
            return "HARNESS_FAILED"
        for stage in cell["stages"].values():
            if not (math.isfinite(stage["terminal_median"]) and stage["shared_relative_change"] > 0
                    and stage["code_relative_change"] > 0):
                return "HARNESS_FAILED"
        if name.startswith("STAGED-R"):
            for k in ("2", "3"):
                if cell["stages"][k]["library_sha256_at_start"] != cell["stages"][str(int(k) - 1)]["library_sha256"]:
                    return "HARNESS_FAILED"
            ids = [set(cell["stages"][s]["final_per_task"]) for s in ("1", "2", "3")]
            if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
                return "HARNESS_FAILED"
    staged = {w: cells[f"STAGED-R_w{w}"]["terminal_median"] for w in WORLDS}
    control = {w: cells[f"NON-STAGED-R_w{w}"]["terminal_median"] for w in WORLDS}
    passing = [w for w in WORLDS if staged[w] <= THRESHOLD]
    if len(passing) >= 2 and all(staged[w] < control[w] for w in passing):
        return "REPLICATES"
    if len(passing) == 1 and staged[passing[0]] < control[passing[0]]:
        return "PARTIAL"
    if not passing:
        return "FAILS_TO_REPLICATE"
    return "PARTIAL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="world 1, budgets divided by 64")
    args = parser.parse_args()
    if args.dry_run:
        staged = run_arm("STAGED", 1, scale=64, model_seed=MODEL_SEED)
        control = run_non_staged(1, scale=64)
        print("staged", {k: round(v["terminal_median"], 4) for k, v in staged["stages"].items()},
              "seed", staged["model_seed"], "persist", staged["persisting_pairings"])
        print("non-staged", round(control["terminal_median"], 4), "seed", control["model_seed"])
        return 0

    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    require_clean_code(OUTPUT)
    expected = protocol()
    sha = fingerprint(expected)
    j1c = json.loads(J1C_REPORT.read_text())["cells"]
    run_log, status_path = ROOT / "run.log", ROOT / "status.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    with writer_lock(ROOT / "launcher.lock"):
        if OUTPUT.exists():
            out = json.loads(OUTPUT.read_text())
            if out.get("protocol_sha256") != sha or out.get("git_commit") != git_commit():
                raise SystemExit("existing report has a different protocol or commit; preserve it first")
            if out.get("complete"):
                return 0
            log_line(run_log, f"RESUME at {git_commit()}")
        else:
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": git_commit(), "protocol": expected,
                   "protocol_sha256": sha, "started_utc": now(), "cells": {},
                   "j1c_seed5000_medians": {w: j1c[f"STAGED_w{w}"]["terminal_median"] for w in map(str, WORLDS)},
                   "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID} at {git_commit()} pid {os.getpid()} seed {MODEL_SEED}")
        atomic_json(OUTPUT, out)
        jobs = [(arm, w) for arm in ("STAGED-R", "NON-STAGED-R") for w in (1, 2, 0)]
        try:
            for done, (arm, w) in enumerate(jobs):
                name = f"{arm}_w{w}"
                cell_dir = ROOT / "cells" / name
                path = cell_dir / "result.json"
                stamp = {"arm": arm, "world": w, "model_seed": MODEL_SEED, "git_commit": git_commit(),
                         "protocol_sha256": sha}
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": git_commit(),
                                          "cells_done": done, "cells_total": len(jobs), "current": name,
                                          "updated_utc": now()})
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["result_sha256"] != fingerprint(stored["result"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    result = stored["result"]
                    log_line(run_log, f"[{name}] reused validated durable cell")
                else:
                    log_line(run_log, f"[{name}] start")
                    started = time.perf_counter()
                    if arm == "STAGED-R":
                        result = run_arm("STAGED", w, artifact=cell_dir, model_seed=MODEL_SEED)
                        result["arm"] = arm
                    else:
                        result = run_non_staged(w, artifact=cell_dir)
                    result["seconds"] = round(time.perf_counter() - started, 1)
                    stages = [s for s in ("1", "2", "3") if s in result["stages"]]
                    atomic_json(path, {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                                       "artifact_sha256": {f"stage{s}/model.pt": digest(cell_dir / f"stage{s}" / "model.pt")
                                                           for s in stages}, "finished_utc": now()})
                out["cells"][name] = result
                atomic_json(OUTPUT, out)
                log_line(run_log, f"[{name}] saved: stages "
                                  f"{ {k: round(v['terminal_median'], 4) for k, v in result['stages'].items()} } "
                                  f"below_0.05 {result['stages']['3']['below_0.05']}/64 "
                                  f"persist {result['persisting_pairings']} ari3 "
                                  f"{[round(a, 2) for a in result['stages']['3']['ari_by_position']]}")
            out["classification"] = classify(out["cells"])
            out["complete"] = True
            out["finished_utc"] = now()
            atomic_json(OUTPUT, out)
            atomic_json(status_path, {"state": "complete", "classification": out["classification"],
                                      "cells_done": len(jobs), "cells_total": len(jobs), "updated_utc": now()})
            log_line(run_log, f"COMPLETE classification {out['classification']}")
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
