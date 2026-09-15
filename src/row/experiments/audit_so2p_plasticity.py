"""SO2-P: stage-3 shared-library plasticity (Tier 1, EXPLORATORY; `SO2P_PLASTICITY_TIER1_PLAN.md`).

One development world (1). Every arm continues stage 3 from SO2's saved STAGED_w1
stage-2 library exactly as SO2's runner did (`carry_library`, then
`learned_lifetime.run`), changing one registered config field. No verdict: the
registered triage rule decides only whether a Tier 2 plan is worth writing.

Restartable: one durable hashed record per arm; relaunch resumes; timestamped
run.log, atomic status.json, exit.json.
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import ANCHOR_TOLERANCE, _last_task_end_of_task, score
from row.experiments.audit_so2_online_gate import carry_library, export_diagnostic, lifetime
from row.experiments.census_so2_interference import slot_drift, spearman, task_summaries
from row.experiments.score_so2_online_gate import restore_stage
from row.experiments.so1_storage import atomic_json, digest, environment, fingerprint, log_line, now, save_model, writer_lock

PLAN = Path("SO2P_PLASTICITY_TIER1_PLAN.md")
SO2_CELL = Path("artifacts/so2_online_gate/cells/STAGED_w1")
OUTPUT = Path("reports/so2p_plasticity.json")
ROOT = Path("artifacts/so2p_plasticity")
PROTOCOL_ID = "SO2P-plasticity-tier1-v1"
WORLD = 1
MODEL_SEED = 5000
THRESHOLD = 0.05
TOL = 1e-6
WORKERS = 3

# arm -> (discrete_model field, value) ; BASE changes nothing.
ARMS = {
    "BASE": None,
    "LR_1/2": ("global_learning_rate", 0.0005),
    "LR_1/4": ("global_learning_rate", 0.00025),
    "LR_1/10": ("global_learning_rate", 0.0001),
    "REPLAY_2x": ("replay_examples_per_task", 8),
    "REPLAY_4x": ("replay_examples_per_task", 16),
}
LR_ORDER = ("BASE", "LR_1/2", "LR_1/4", "LR_1/10")


def slug(arm: str) -> str:
    return arm.replace("/", "_")


def so2_record() -> dict:
    return json.loads((SO2_CELL / "result.json").read_text())["result"]


def arm_config(cfg, arm: str):
    change = ARMS[arm]
    if change is None:
        return cfg
    return replace(cfg, discrete_model=replace(cfg.discrete_model, **{change[0]: change[1]}))


def changed_fields(base_cfg, arm_cfg) -> list[str]:
    """Top-level-section.field names whose values differ (one level of dataclass nesting)."""
    out = []
    for section in fields(base_cfg):
        a, b = getattr(base_cfg, section.name), getattr(arm_cfg, section.name)
        if a == b:
            continue
        if hasattr(a, "__dataclass_fields__"):
            out += [f"{section.name}.{f.name}" for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)]
        else:
            out.append(section.name)
    return out


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "world": WORLD, "model_seed": MODEL_SEED,
            "kind": "rotated_discrete_fast", "arms": {k: list(v) if v else None for k, v in ARMS.items()},
            "start_state": (SO2_CELL / "stage2").as_posix(), "threshold": THRESHOLD, "tolerance": TOL,
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, SO2_CELL / "result.json",
                                                             SO2_CELL / "stage2" / "model.pt")},
            "environment": environment()}


def load_stage2():
    record = so2_record()
    cfg2, world2, _, _ = stage_setup(WORLD, 2, MODEL_SEED)
    model2 = restore_stage(SO2_CELL / "stage2", cfg2, world2, record["stages"]["2"]["novel_probe_available"])
    if library_sha(model2) != record["stages"]["2"]["library_sha256"]:
        raise RuntimeError("saved SO2 stage-2 library does not match its record")
    return model2


def run_arm(arm: str, output: Path, scale: int = 1) -> dict:
    torch.set_num_threads(1)
    started = time.perf_counter()
    model2 = load_stage2()
    base_cfg, world, _, _ = stage_setup(WORLD, 3, MODEL_SEED)
    cfg = arm_config(base_cfg, arm)
    model = carry_library(model2, cfg)
    carried = library_sha(model)
    summary, model, ran = lifetime(cfg, world, output, model=model, scale=scale)
    trained = SimpleNamespace(tasks=[t for t in world.tasks if t.task_id in model.task_codes])
    terminal = score(model, trained)
    last_id, last_end, end_median = _last_task_end_of_task(output)
    rows = task_summaries(output)
    ids = [r["task_id"] for r in rows]
    end = [r["final_nmse"] for r in rows]
    term = [terminal["per_task"][i] for i in ids]
    ratio = [math.log10(max(t, 1e-12) / max(e, 1e-12)) for t, e in zip(term, end)]
    save_model(output, model, ran)
    compute = {k: v for k, v in summary.items()
               if isinstance(v, (int, float)) and any(s in k for s in ("gradient", "update", "example"))}
    return {
        "arm": arm, "change": list(ARMS[arm]) if ARMS[arm] else None, "scale": scale,
        "changed_fields": changed_fields(base_cfg, cfg),
        "library_sha256_at_start": carried, "library_sha256": library_sha(model),
        "tasks": len(ids),
        "terminal_median": terminal["median"], "terminal_below": terminal["below_0.05"],
        "terminal_per_task": terminal["per_task"],
        "end_of_task_median": end_median, "end_of_task_below": int(sum(e <= THRESHOLD for e in end)),
        "lost_threshold": int(sum(e <= THRESHOLD < t for t, e in zip(term, end))),
        "gained_threshold": int(sum(t <= THRESHOLD < e for t, e in zip(term, end))),
        "spearman_position_vs_log_ratio": spearman(list(range(len(ids))), ratio),
        "anchor_task_id": last_id, "anchor_abs_error": abs(terminal["per_task"][last_id] - last_end),
        "drift_from_stage2": slot_drift(model2, model, WORLD, cfg.world.state_dim),
        "prequential": summary.get("cumulative_prequential_gaussian_log_loss"),
        "summary_compute_fields": compute,
        "online_examples": ran.world.tasks * ran.world.examples_per_task,
        # The lifetime summary carries no gradient count; record the resolved training
        # knobs so each arm's example-gradient budget can be derived and disclosed.
        "training": {k: getattr(ran.discrete_model, k) for k in (
            "global_learning_rate", "task_learning_rate", "updates_per_example",
            "replay_examples_per_task", "replay_ratio", "weight_decay")},
        "export_diagnostic": export_diagnostic(cfg, world, model) if scale == 1 else None,
        "seconds": round(time.perf_counter() - started, 1),
    }


def gates(arms: dict, so2: dict) -> dict:
    base = arms["BASE"]
    rec3 = so2["stages"]["3"]
    g0 = (base["library_sha256"] == rec3["library_sha256"] and
          base["library_sha256_at_start"] == rec3["library_sha256_at_start"] and
          max(abs(base["terminal_per_task"][k] - v) for k, v in rec3["terminal_per_task"].items()) <= TOL)
    g1 = all(a["library_sha256"] != base["library_sha256"] and len(a["changed_fields"]) == 1 and
             a["changed_fields"][0] == f"discrete_model.{a['change'][0]}"
             for n, a in arms.items() if n != "BASE") and base["changed_fields"] == []
    g2 = all(a["anchor_abs_error"] <= ANCHOR_TOLERANCE for a in arms.values())
    return {"G0_reproduction": bool(g0), "G1_non_vacuity": bool(g1), "G2_anchor": bool(g2)}


def triage(arms: dict, gate: dict) -> str:
    """The frozen plan's exploratory triage rule. Not a verdict."""
    if set(arms) != set(ARMS) or not all(gate.values()):
        return "UNINFORMATIVE"
    base = arms["BASE"]
    bd = base["drift_from_stage2"]["median"]
    others = [a for n, a in arms.items() if n != "BASE"]
    if any(a["terminal_median"] <= THRESHOLD and a["terminal_below"] >= 32 and a["drift_from_stage2"]["median"] < bd
           and a["end_of_task_median"] <= 2 * base["end_of_task_median"] for a in others):
        return "LIVE"
    if any((a["terminal_median"] <= base["terminal_median"] / 2 or a["lost_threshold"] <= base["lost_threshold"] / 2)
           and a["drift_from_stage2"]["median"] < bd for a in others):
        return "PARTIAL"
    drifts = [arms[n]["drift_from_stage2"]["median"] for n in LR_ORDER]
    if all(x > y for x, y in zip(drifts, drifts[1:])):
        return "DISFAVOURED"
    return "UNINFORMATIVE"


def run_cell(arm: str, stamp: dict, root: Path, scale: int) -> dict:
    cell = root / "cells" / slug(arm)
    result = run_arm(arm, cell / "stage3", scale=scale)
    atomic_json(cell / "result.json", {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                                       "finished_utc": now()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="BASE only at full scale into a scratch root; checks G0; no durable record")
    parser.add_argument("--smoke", action="store_true",
                        help="restart-path test: BASE and LR_1/2 at scale 16, separate root; never a result")
    args = parser.parse_args()
    if args.dry_run:
        r = run_arm("BASE", ROOT.parent / "so2p_plasticity_dry" / "stage3")
        rec3 = so2_record()["stages"]["3"]
        worst = max(abs(r["terminal_per_task"][k] - v) for k, v in rec3["terminal_per_task"].items())
        print(f"BASE sha match {r['library_sha256'] == rec3['library_sha256']} worst per-task {worst:.2e} "
              f"anchor {r['anchor_abs_error']:.1e} drift {r['drift_from_stage2']['median']:.3f} "
              f"compute {r['summary_compute_fields']} ({r['seconds']}s)")
        return 0

    root, output, scale, arms = ROOT, OUTPUT, 1, list(ARMS)
    if args.smoke:
        root, scale, arms = Path("artifacts/so2p_plasticity_smoke"), 16, ["BASE", "LR_1/2"]
        output = root / "report.json"
    else:
        for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
            if subprocess.run([sys.executable, tool]).returncode != 0:
                raise SystemExit(f"{tool} failed")
        require_clean_code(OUTPUT)
    expected = protocol() | ({"smoke": {"scale": scale, "arms": arms}} if args.smoke else {})
    sha = fingerprint(expected)
    root.mkdir(parents=True, exist_ok=True)
    run_log, status_path = root / "run.log", root / "status.json"
    atomic_json(root / "run.pid", {"pid": os.getpid(), "started_utc": now()})
    commit = git_commit()
    with writer_lock(root / "launcher.lock"):
        if output.exists():
            out = json.loads(output.read_text())
            if out.get("protocol_sha256") != sha or out.get("git_commit") != commit:
                raise SystemExit("existing report has a different protocol or commit; preserve it first")
            if out.get("complete"):
                return 0
            log_line(run_log, f"RESUME at {commit}")
        else:
            out = {"status": "TIER 1 EXPLORATORY - not a verdict", "frozen_plan": PLAN.as_posix(),
                   "git_commit": commit, "protocol": expected, "protocol_sha256": sha,
                   "started_utc": now(), "arms": {}, "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID}{' SMOKE' if args.smoke else ''} at {commit} pid {os.getpid()}")
        try:
            pending = {}
            for arm in arms:
                path = root / "cells" / slug(arm) / "result.json"
                stamp = {"arm": arm, "git_commit": commit, "protocol_sha256": sha}
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["result_sha256"] != fingerprint(stored["result"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    out["arms"][arm] = stored["result"]
                    log_line(run_log, f"[{arm}] reused validated durable cell")
                else:
                    pending[arm] = stamp
            atomic_json(output, out)
            started_at, running = time.time(), {}

            def write_status() -> None:
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": commit,
                                          "started_utc": out["started_utc"], "cells_done": len(out["arms"]),
                                          "cells_total": len(arms), "submitted_unfinished": sorted(running.values()),
                                          "elapsed_seconds": round(time.time() - started_at),
                                          "updated_utc": now()})

            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                for arm, stamp in pending.items():
                    running[pool.submit(run_cell, arm, stamp, root, scale)] = arm
                    log_line(run_log, f"[{arm}] submitted")
                write_status()
                remaining = set(running)
                while remaining:
                    done_now, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                    for future in done_now:
                        arm = running.pop(future)
                        result = future.result()
                        out["arms"][arm] = result
                        atomic_json(output, out)
                        log_line(run_log, f"[{arm}] saved: terminal {result['terminal_median']:.4f} "
                                          f"({result['terminal_below']}/64) end-of-task {result['end_of_task_median']:.4f} "
                                          f"lost {result['lost_threshold']} drift {result['drift_from_stage2']['median']:.3f} "
                                          f"({result['seconds']}s)")
                        write_status()
            if args.smoke:
                out["triage"] = "SMOKE (not a result)"
            else:
                out["gates"] = gates(out["arms"], so2_record())
                out["triage"] = triage(out["arms"], out["gates"])
            out["complete"], out["finished_utc"] = True, now()
            atomic_json(output, out)
            atomic_json(status_path, {"state": "complete", "triage": out["triage"], "gates": out.get("gates"),
                                      "cells_done": len(arms), "cells_total": len(arms), "updated_utc": now()})
            log_line(run_log, f"COMPLETE gates {out.get('gates')} triage {out['triage']}")
            atomic_json(root / "exit.json", {"git_commit": commit, "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(root / "exit.json", {"git_commit": commit, "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
