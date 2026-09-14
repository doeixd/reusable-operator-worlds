"""J2A: is the staged library a reusable vocabulary? (export audit)

Frozen in `J2A_STAGED_LIBRARY_AUDIT_PLAN.md` (8913ed4). No training: the 12
existing stage-3 libraries (staged at two seeds, non-staged, reset) are
reloaded frozen and asked to execute 64 UNSEEN teacher programs under
support-only exhaustive route search, alongside the SO1R instrument on their
own trained tasks.

Restartable: one durable hashed record per library; relaunch resumes;
timestamped run.log and atomic status.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary, optimize_route, unflatten
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, restore_model, writer_lock,
)
from row.metrics import nmse
from row.world import Program, _rng

PLAN = Path("J2A_STAGED_LIBRARY_AUDIT_PLAN.md")
J1C_REPORT = Path("reports/j1c_curriculum.json")
J1CR_REPORT = Path("reports/j1cr_replication.json")
OUTPUT = Path("reports/j2a_staged_library.json")
ROOT = Path("artifacts/j2a_staged_library")
PROTOCOL_ID = "J2A-staged-library-audit-v1"
WORLDS = (0, 1, 2)
THRESHOLD = 0.05
HELD_OUT = 64
SEED = 1709
EXPORT_RATIO = 4.0
# name -> (cell directory template, model seed, report, report cell template)
SOURCES = {
    "STAGED5000": ("artifacts/j1c_curriculum/cells/STAGED_w{w}", 5000, J1C_REPORT, "STAGED_w{w}"),
    "STAGED3001": ("artifacts/j1cr_replication/cells/STAGED-R_w{w}", 3001, J1CR_REPORT, "STAGED-R_w{w}"),
    "NONSTAGED3001": ("artifacts/j1cr_replication/cells/NON-STAGED-R_w{w}", 3001, J1CR_REPORT, "NON-STAGED-R_w{w}"),
    "RESET5000": ("artifacts/j1c_curriculum/cells/RESET_w{w}", 5000, J1C_REPORT, "RESET_w{w}"),
}
STAGED = ("STAGED5000", "STAGED3001")


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "worlds": list(WORLDS),
            "sources": {k: {"cells": v[0], "model_seed": v[1], "report": v[2].as_posix()} for k, v in SOURCES.items()},
            "held_out_programs": HELD_OUT, "seed": SEED, "threshold": THRESHOLD, "export_ratio": EXPORT_RATIO,
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, J1C_REPORT, J1CR_REPORT)},
            "environment": environment()}


def held_out_tasks(cfg, world):
    """64 unseen length-3 programs, built with the world's own teacher library."""
    trained = {tuple(int(p) for p in t.program.primitive_ids) for t in world.tasks}
    unseen = [p for p in product(range(cfg.world.teacher_primitives), repeat=cfg.world.program_length)
              if p not in trained]
    order = _rng(cfg.world.seed, SEED).permutation(len(unseen))[:HELD_OUT]
    library = world.tasks[0].teacher_library
    tasks = []
    for index, choice in enumerate(order):
        program = Program(tuple(unseen[int(choice)]))
        support = _rng(cfg.world.seed, SEED, 1 + index).normal(
            size=(cfg.world.examples_per_task, cfg.world.state_dim))
        query = _rng(cfg.world.seed, SEED, 10_000 + index).normal(
            size=(cfg.world.evaluation_examples, cfg.world.state_dim))
        tasks.append({"program": tuple(program.primitive_ids), "train_x": support,
                      "train_y": program.execute(library, support), "eval_x": query,
                      "eval_y": program.execute(library, query)})
    if any(t["program"] in trained for t in tasks):
        raise ValueError("held-out program appears in the training set")
    return tasks


def enum_route(library, support_x, support_y):
    mse = library.all_route_support_mse(support_x, support_y)
    return tuple(unflatten(int(torch.argmin(mse)), library.slots, library.steps))


def query_error(library, route, x, y) -> float:
    with torch.no_grad():
        prediction = library.hard(torch.tensor(x, dtype=torch.float32), list(route))
    return float(nmse(prediction.cpu().numpy(), y))


def audit_library(name: str, world_seed: int, index: int) -> dict:
    torch.set_num_threads(1)
    template, model_seed, report_path, cell_template = SOURCES[name]
    directory = Path(template.format(w=world_seed))
    cfg, world, _, _ = stage_setup(world_seed, 3, model_seed)
    stored = json.loads((directory / "result.json").read_text())
    for artifact, sha in stored["artifact_sha256"].items():
        if digest(directory / artifact) != sha:
            raise ValueError(f"artifact hash mismatch: {directory / artifact}")
    recorded = stored["result"]["stages"]["3"]["final_per_task"]
    model = restore_model(directory / "stage3", cfg, world, build_fast)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()
    library = FrozenLibrary(model)
    routes = model.hard_routes()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, world_seed, 3, index]))
    trained, started = {}, time.perf_counter()
    for task in world.tasks:
        x = torch.tensor(task.train_x, dtype=torch.float32)
        y = torch.tensor(task.train_y, dtype=torch.float32)
        chosen = enum_route(library, x, y)
        opt, initial, final, code_change = optimize_route(library, x, y, 2000)
        random_route = tuple(int(v) for v in rng.integers(0, library.slots, size=library.steps))
        trained[task.task_id] = {
            "as_trained_route": list(routes[task.task_id]), "enum_route": list(chosen), "opt_route": list(opt),
            "as_trained": query_error(library, routes[task.task_id], task.eval_x, task.eval_y),
            "enum": query_error(library, chosen, task.eval_x, task.eval_y),
            "opt": query_error(library, opt, task.eval_x, task.eval_y),
            "random": query_error(library, random_route, task.eval_x, task.eval_y),
            "opt_support_drop": (initial - final) / initial, "opt_code_abs_sum": code_change,
        }
    held = {}
    for position, task in enumerate(held_out_tasks(cfg, world)):
        x = torch.tensor(task["train_x"], dtype=torch.float32)
        y = torch.tensor(task["train_y"], dtype=torch.float32)
        chosen = enum_route(library, x, y)
        random_route = tuple(int(v) for v in rng.integers(0, library.slots, size=library.steps))
        held[str(position)] = {"program": list(task["program"]), "enum_route": list(chosen),
                               "enum": query_error(library, chosen, task["eval_x"], task["eval_y"]),
                               "random": query_error(library, random_route, task["eval_x"], task["eval_y"])}
    after = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()
    if before != after:
        raise ValueError("library parameters changed during the audit")
    return {"name": name, "world": world_seed, "model_seed": model_seed, "library_sha256": before,
            "recorded_per_task": recorded, "trained": trained, "held_out": held,
            "seconds": round(time.perf_counter() - started, 1)}


def summarize(record: dict) -> dict:
    trained = list(record["trained"].values())
    held = list(record["held_out"].values())
    med = lambda values: float(np.median(values))
    trained_median = med([r["as_trained"] for r in trained])
    held_median = med([r["enum"] for r in held])
    return {
        "trained": {arm: med([r[arm] for r in trained]) for arm in ("as_trained", "enum", "opt", "random")},
        "held_out": {arm: med([r[arm] for r in held]) for arm in ("enum", "random")},
        "held_out_below_threshold": int(sum(r["enum"] <= THRESHOLD for r in held)),
        "export_ratio": held_median / trained_median,
        "g": med([math.log(r["opt"] / r["enum"]) for r in trained]),
        "as_trained_bitwise": all(r["as_trained"] == record["recorded_per_task"][t]
                                  for t, r in record["trained"].items()),
        "enum_equals_as_trained_route": float(np.mean([r["enum_route"] == r["as_trained_route"] for r in trained])),
        "finite": all(math.isfinite(v) for r in trained + held for k, v in r.items() if isinstance(v, float)),
    }


def classify(cells: dict) -> str:
    staged = [c for name, c in cells.items() if name.split("_w")[0] in STAGED]
    if len(cells) != len(SOURCES) * len(WORLDS) or len(staged) != 2 * len(WORLDS):
        return "HARNESS_FAILED"
    if any(not c["as_trained_bitwise"] or not c["finite"] for c in cells.values()):
        return "HARNESS_FAILED"
    if any(c["trained"]["random"] < c["trained"]["enum"] for c in staged):
        return "HARNESS_FAILED"
    exporting = sum(c["held_out"]["enum"] <= THRESHOLD and c["export_ratio"] <= EXPORT_RATIO for c in staged)
    if exporting >= 5:
        return "EXPORTS"
    if sum(c["held_out"]["enum"] <= THRESHOLD for c in staged) >= 3:
        return "EXPORTS_WEAKLY"
    return "DOES_NOT_EXPORT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="one staged library, world 1, few tasks")
    args = parser.parse_args()
    if args.dry_run:
        torch.set_num_threads(1)
        cfg, world, _, _ = stage_setup(1, 3, 5000)
        tasks = held_out_tasks(cfg, world)
        trained = {tuple(int(p) for p in t.program.primitive_ids) for t in world.tasks}
        print("held-out:", len(tasks), "programs, disjoint:", not any(t["program"] in trained for t in tasks),
              "example:", tasks[0]["program"])
        started = time.perf_counter()
        record = audit_library("STAGED5000", 1, 0)
        print(json.dumps(summarize(record), indent=1), f"\n{time.perf_counter() - started:.1f}s")
        return 0

    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    require_clean_code(OUTPUT)
    expected = protocol()
    sha = fingerprint(expected)
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
                   "protocol_sha256": sha, "started_utc": now(), "cells": {}, "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID} at {git_commit()} pid {os.getpid()}")
        atomic_json(OUTPUT, out)
        jobs = [(name, w, i) for i, name in enumerate(SOURCES) for w in WORLDS]
        try:
            for done, (name, w, index) in enumerate(jobs):
                key = f"{name}_w{w}"
                path = ROOT / "cells" / key / "result.json"
                stamp = {"name": name, "world": w, "git_commit": git_commit(), "protocol_sha256": sha}
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "cells_done": done,
                                          "cells_total": len(jobs), "current": key, "updated_utc": now()})
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["record_sha256"] != fingerprint(stored["record"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    record = stored["record"]
                    log_line(run_log, f"[{key}] reused validated durable cell")
                else:
                    log_line(run_log, f"[{key}] start")
                    record = audit_library(name, w, index)
                    atomic_json(path, {"stamp": stamp, "record": record, "record_sha256": fingerprint(record),
                                       "finished_utc": now()})
                summary = summarize(record)
                out["cells"][key] = summary
                atomic_json(OUTPUT, out)
                log_line(run_log, f"[{key}] trained {summary['trained']['as_trained']:.4f} held-out "
                                  f"{summary['held_out']['enum']:.4f} ({summary['held_out_below_threshold']}/64) "
                                  f"ratio {summary['export_ratio']:.2f} g {summary['g']:.3f} "
                                  f"random {summary['held_out']['random']:.3f} ({record['seconds']}s)")
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
