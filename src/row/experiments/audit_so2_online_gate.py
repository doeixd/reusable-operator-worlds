"""SO2: does staged formation survive the online lifetime? (B2)

Frozen in `SO2_ONLINE_GATE_PLAN.md` (6347243) and `SO2_AMENDMENT_1.md`
(77412e8). Per world: STAGED runs three consecutive online lifetimes (60
length-1, 64 length-2, 64 canonical length-3 tasks) through
`learned_lifetime.run` unchanged, carrying ONLY the shared library between
them; PLAIN runs one lifetime on the canonical tasks alone. The export margin
reuses G5R's construction verbatim (12 held-out programs, `SeedSequence([1500,
world])`, `adapt_cell` at `ADAPT_STEPS`, `scratch_model(cfg,
"rotated_discrete", 7717)`, natural log of geometric means), so the registered
0.75 threshold means what it meant there.

Restartable: one durable hashed record per (arm, world); relaunch resumes;
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_STEPS, scratch_model
from row.experiments.audit_e8_length import adapt_cell
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_j2a_staged_library import enum_route, held_out_tasks, query_error
from row.experiments.audit_rotated_g5 import held_out_programs
from row.experiments.audit_rotated_g5r import _final_nmse
from row.experiments.audit_rotated_g5r_interference import ANCHOR_TOLERANCE, _last_task_end_of_task, score
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.learned_lifetime import run
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, save_model, writer_lock,
)
from row.rotated_world import rotated_library
from row.support_split_world import _build_tasks

PLAN = Path("SO2_ONLINE_GATE_PLAN.md")
AMENDMENT = Path("SO2_AMENDMENT_1.md")
OUTPUT = Path("reports/so2_online_gate.json")
ROOT = Path("artifacts/so2_online_gate")
PROTOCOL_ID = "SO2-online-gate-v1"
KIND = "rotated_discrete_fast"
WORLDS = (0, 1, 2)
MODEL_SEED = 5000
THRESHOLD = 0.05
MARGIN = 0.75
HELD_OUT = 12  # G5R's count, with G5R's draw
MARGIN_SEED = 1500
STAGES = (1, 2, 3)
# Memory-bounded pool (AGENTS.md: 3-4 for slots=12 lifetimes). Cells are independent
# single-threaded computations from fixed seeds, so the pool changes scheduling only.
WORKERS = 3


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "amendment": AMENDMENT.as_posix(),
            "kind": KIND, "worlds": list(WORLDS), "model_seed": MODEL_SEED, "threshold": THRESHOLD,
            "margin": MARGIN, "held_out": HELD_OUT, "adapt_steps": ADAPT_STEPS,
            "margin_construction": "G5R verbatim: held_out_programs(SeedSequence([1500, world])), "
                                   "adapt_cell(ADAPT_STEPS), scratch_model(cfg, 'rotated_discrete', 7717), "
                                   "log(geo scratch) - log(geo trained)",
            "stages": "60 length-1, 64 length-2, 64 canonical length-3; library-only transfer",
            "terminal": "M_terminal is the TERMINAL model's median query NMSE (G5R Stage D `score`), with "
                        "end-of-task median reported beside it and a last-task anchor at ANCHOR_TOLERANCE",
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, AMENDMENT)},
            "environment": environment()}


def geo(values) -> float:
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values)))


def lifetime(cfg, world, output: Path, model=None, scale: int = 1):
    """One online lifetime through the unchanged runner; returns summary and model."""
    if scale != 1:
        cfg = replace(cfg, world=replace(cfg.world, tasks=max(2, cfg.world.tasks // scale),
                                         examples_per_task=max(4, cfg.world.examples_per_task // scale),
                                         evaluation_examples=max(4, cfg.world.evaluation_examples // scale)))
    summary = run(replace(cfg, output_directory=output), KIND, world=world, model=model, return_model=True)
    terminal = summary.pop("terminal_model")
    return summary, terminal, cfg  # the cfg actually run, so a scaled dry run reports its own sizes


def carry_library(previous, cfg):
    """A fresh model for the next stage's program length, previous library, fresh codes."""
    fresh = build_fast(cfg)
    fresh.library.load_state_dict(previous.library.state_dict())
    if library_sha(fresh) != library_sha(previous):
        raise RuntimeError("library did not transfer bitwise")
    if fresh.task_codes:
        raise RuntimeError("task codes carried across a stage boundary")
    return fresh


def export_margin(cfg, world, model, world_seed: int) -> dict:
    """G5R's clause A, recomputed on this artifact with G5R's own construction."""
    rng = np.random.default_rng(np.random.SeedSequence([MARGIN_SEED, world_seed]))
    programs = held_out_programs(cfg, world, rng, HELD_OUT)
    library = rotated_library(cfg.world)
    scratch = scratch_model(cfg, "rotated_discrete", 7717)
    trained_values, scratch_values, rows = [], [], []
    for index, program in enumerate(programs):
        task = _build_tasks(cfg.world, library, [program], [f"so2_{world_seed}_{index}"],
                            index_offset=95000 + index)[0]
        trained = adapt_cell(model, task, f"so2L_{world_seed}_{index}", cfg.world.program_length,
                            False, library, program, steps=ADAPT_STEPS)
        fresh = adapt_cell(scratch, task, f"so2S_{world_seed}_{index}", cfg.world.program_length,
                           True, library, program, steps=ADAPT_STEPS)
        # `adapt_cell` reports `query_nmse`; G5R reads exactly this key.
        trained_values.append(trained["query_nmse"])
        scratch_values.append(fresh["query_nmse"])
        rows.append({"program": list(program), "trained": trained["query_nmse"],
                     "scratch": fresh["query_nmse"]})
    trained_geo, scratch_geo = geo(trained_values), geo(scratch_values)
    return {"held_out": HELD_OUT, "trained_geo_nmse": trained_geo, "scratch_geo_nmse": scratch_geo,
            "margin": math.log(scratch_geo) - math.log(trained_geo), "rows": rows,
            "scratch_kind": "rotated_discrete (G5R's construction; the trained arm is the fast kind, "
                            "which agrees with it to ~1e-7)"}


def export_diagnostic(cfg, world, model) -> dict:
    """J2A's 64-program export test, reported beside the margin (not a threshold)."""
    model.eval()
    frozen = FrozenLibrary(model)
    values = []
    for task in held_out_tasks(cfg, world):
        route = enum_route(frozen, torch.tensor(task["train_x"], dtype=torch.float32),
                           torch.tensor(task["train_y"], dtype=torch.float32))
        values.append(query_error(frozen, route, task["eval_x"], task["eval_y"]))
    return {"programs": len(values), "median_nmse": float(np.median(values)),
            "below_threshold": int(sum(v <= THRESHOLD for v in values))}


def run_arm(arm: str, world_seed: int, artifact: Path | None = None, scale: int = 1) -> dict:
    torch.set_num_threads(1)
    stages, model, records = ([1, 2, 3] if arm == "STAGED" else [3]), None, {}
    started = time.perf_counter()
    for stage in stages:
        cfg, world, _, _ = stage_setup(world_seed, stage, MODEL_SEED)
        carried = None
        if model is not None:
            model = carry_library(model, cfg)
            carried = library_sha(model)
        output = (artifact or ROOT / "scratch") / f"stage{stage}"
        summary, model, ran = lifetime(cfg, world, output, model=model, scale=scale)
        # The plan's M_terminal is the TERMINAL model on every task the stage trained
        # (research program B2, "terminal versus end-of-task error for online arms"),
        # scored as G5R Stage D scores it. The summary's final_nmse is END-OF-TASK
        # error, measured while the library was still moving; it is reported beside.
        # Anchor: nothing trains after the last task, so its terminal and end-of-task
        # errors must agree (the terminal probe only fits its own code).
        trained = SimpleNamespace(tasks=[t for t in world.tasks if t.task_id in model.task_codes])
        terminal = score(model, trained)
        last_id, last_end, end_of_task_median = _last_task_end_of_task(output)
        records[str(stage)] = {
            "length": ran.discrete_model.task_steps, "tasks": ran.world.tasks,
            "novel_probe_available": summary.get("novel_composition", {}).get("available", True),
            "terminal_median": terminal["median"], "terminal_below_threshold": terminal["below_0.05"],
            "terminal_per_task": terminal["per_task"],
            "final_nmse_median": _final_nmse(summary), "end_of_task_median": end_of_task_median,
            "anchor_task_id": last_id, "anchor_abs_error": abs(terminal["per_task"][last_id] - last_end),
            "cumulative_prequential_gaussian_log_loss": summary.get("cumulative_prequential_gaussian_log_loss"),
            "online_examples": ran.world.tasks * ran.world.examples_per_task,
            "library_sha256": library_sha(model), "library_sha256_at_start": carried,
        }
        if artifact is not None:
            save_model(output, model, ran)
    cfg, world, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    result = {"arm": arm, "world": world_seed, "model_seed": MODEL_SEED, "stages": records,
              "terminal_median": records["3"]["terminal_median"],
              "end_of_task_median": records["3"]["end_of_task_median"],
              "prequential_total": sum(r["cumulative_prequential_gaussian_log_loss"] or 0.0
                                       for r in records.values()),
              "online_examples_total": sum(r["online_examples"] for r in records.values()),
              "seconds": round(time.perf_counter() - started, 1)}
    if scale == 1:
        result["margin"] = export_margin(cfg, world, model, world_seed)
        result["export_diagnostic"] = export_diagnostic(cfg, world, model)
    return result


def run_cell(arm: str, world_seed: int, stamp: dict, root: Path = ROOT, scale: int = 1) -> dict:
    """Pool worker: one cell, written durably by this worker alone, then returned."""
    cell_dir = root / "cells" / f"{arm}_w{world_seed}"
    result = run_arm(arm, world_seed, artifact=cell_dir, scale=scale)
    atomic_json(cell_dir / "result.json", {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                                           "finished_utc": now()})
    return result


def classify(cells: dict) -> str:
    expected = {f"{a}_w{w}" for a in ("STAGED", "PLAIN") for w in WORLDS}
    if not expected <= set(cells):
        return "HARNESS_FAILED"
    for name, cell in cells.items():
        if cell["model_seed"] != MODEL_SEED or not math.isfinite(cell["terminal_median"]):
            return "HARNESS_FAILED"
        if any(not s["anchor_abs_error"] <= ANCHOR_TOLERANCE for s in cell["stages"].values()):
            return "HARNESS_FAILED"
        if name.startswith("STAGED"):
            for stage in ("2", "3"):
                if cell["stages"][stage]["library_sha256_at_start"] != cell["stages"][str(int(stage) - 1)]["library_sha256"]:
                    return "HARNESS_FAILED"
        elif cell["stages"]["3"]["library_sha256_at_start"] is not None:
            return "HARNESS_FAILED"
    staged = {w: cells[f"STAGED_w{w}"] for w in WORLDS}
    acquires = sum(staged[w]["terminal_median"] <= THRESHOLD for w in WORLDS) >= 2
    margins = sum(staged[w]["margin"]["margin"] >= MARGIN for w in WORLDS) >= 2
    if acquires and margins:
        return "SO2_PASSES"
    if acquires:
        return "SO2_ACQUIRES_ONLY"
    return "SO2_FAILS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="world 1, reduced tasks/examples, no margin")
    parser.add_argument("--smoke", action="store_true",
                        help="restart-path test of this launcher: world 1, both arms, scale 16, no margin, "
                             "separate root, no classification; never a result")
    args = parser.parse_args()
    if args.dry_run:
        for arm in ("STAGED", "PLAIN"):
            started = time.perf_counter()
            r = run_arm(arm, 1, scale=16)
            print(arm, {k: round(v["final_nmse_median"], 4) for k, v in r["stages"].items()},
                  "carried", [r["stages"][s]["library_sha256_at_start"] is not None for s in sorted(r["stages"])],
                  f"{time.perf_counter() - started:.1f}s")
        return 0

    root, output, scale, workers = ROOT, OUTPUT, 1, WORKERS
    jobs = [(arm, w) for arm in ("STAGED", "PLAIN") for w in (1, 2, 0)]  # longest (STAGED) first
    if args.smoke:
        root, scale, workers = Path("artifacts/so2_online_gate_smoke"), 16, 2
        output, jobs = root / "report.json", [("STAGED", 1), ("PLAIN", 1)]
    else:
        for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
            if subprocess.run([sys.executable, tool]).returncode != 0:
                raise SystemExit(f"{tool} failed")
        require_clean_code(OUTPUT)
    expected = protocol() | ({"smoke": {"scale": scale, "jobs": jobs}} if args.smoke else {})
    sha = fingerprint(expected)
    run_log, status_path = root / "run.log", root / "status.json"
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "run.pid", {"pid": os.getpid(), "started_utc": now()})
    with writer_lock(root / "launcher.lock"):
        if output.exists():
            out = json.loads(output.read_text())
            if out.get("protocol_sha256") != sha or out.get("git_commit") != git_commit():
                raise SystemExit("existing report has a different protocol or commit; preserve it first")
            if out.get("complete"):
                return 0
            log_line(run_log, f"RESUME at {git_commit()}")
        else:
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": git_commit(), "protocol": expected,
                   "protocol_sha256": sha, "started_utc": now(), "cells": {}, "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID}{' SMOKE' if args.smoke else ''} at {git_commit()} "
                              f"pid {os.getpid()} workers {workers}")
        atomic_json(output, out)
        commit = git_commit()
        try:
            pending, started_at = {}, time.time()
            for arm, w in jobs:
                name = f"{arm}_w{w}"
                path = root / "cells" / name / "result.json"
                stamp = {"arm": arm, "world": w, "git_commit": commit, "protocol_sha256": sha}
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["result_sha256"] != fingerprint(stored["result"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    out["cells"][name] = stored["result"]
                    log_line(run_log, f"[{name}] reused validated durable cell")
                else:
                    pending[name] = (arm, w, stamp)
            atomic_json(output, out)
            fresh_total = len(pending)

            def write_status() -> None:
                finished = fresh_total - len(pending) - len(running)  # cells finished in this launch
                elapsed = time.time() - started_at
                eta = elapsed / finished * (len(pending) + len(running)) if finished else None
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": commit,
                                          "started_utc": out["started_utc"], "cells_done": len(out["cells"]),
                                          "cells_total": len(jobs), "running": sorted(running.values()),
                                          "eta_seconds": None if eta is None else round(eta), "updated_utc": now()})

            # One writer per cell: each worker writes only its own cell directory and
            # result.json; this parent alone writes the report, status and log.
            running = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for name, (arm, w, stamp) in list(pending.items()):
                    running[pool.submit(run_cell, arm, w, stamp, root, scale)] = name
                    log_line(run_log, f"[{name}] start")
                pending.clear()
                write_status()
                remaining = set(running)
                while remaining:
                    done_now, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                    for future in done_now:
                        name = running.pop(future)
                        result = future.result()  # a failed cell raises here and fails the launch
                        out["cells"][name] = result
                        atomic_json(output, out)
                        margin = (f"margin {result['margin']['margin']:+.2f} export "
                                  f"{result['export_diagnostic']['below_threshold']}/64 "
                                  if "margin" in result else "")
                        log_line(run_log, f"[{name}] saved: terminal {result['terminal_median']:.4f} end-of-task "
                                          f"{result['end_of_task_median']:.4f} {margin}prequential "
                                          f"{result['prequential_total']:.0f} ({result['seconds']}s)")
                        write_status()
            out["classification"] = "SMOKE (not a result)" if args.smoke else classify(out["cells"])
            out["complete"] = True
            out["finished_utc"] = now()
            atomic_json(output, out)
            atomic_json(status_path, {"state": "complete", "classification": out["classification"],
                                      "cells_done": len(jobs), "cells_total": len(jobs), "updated_utc": now()})
            log_line(run_log, f"COMPLETE classification {out['classification']}")
            atomic_json(root / "exit.json", {"git_commit": git_commit(), "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(root / "exit.json", {"git_commit": git_commit(), "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
