"""SO1: acquisition dose-response bracket (BUDGET_LIMITED branch).

Frozen in `SO1_BUDGET_BRACKET_PLAN.md`. Two oracle-route offline IID dose
curves (batch 2 and batch 64) sharing five example-gradient levels, every cell
the unchanged Stage D `offline_cell` construction on the versioned
`rotated_discrete_fast` kind, dispatched through `row.pool`. Stage 2 (learned
routes at the lowest passing envelope per curve) runs only if an oracle cell
passes. Development worlds 0-2. Teacher identities enter only the cells
labelled oracle.

Resumable: every cell is written to the report as it completes under a
protocol fingerprint; a cell computed under a different fingerprint or git
commit is refused.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import sys
from dataclasses import replace
from pathlib import Path

import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code, write
from row.experiments.audit_rotated_g5r_interference import (
    THRESHOLD,
    _assignment,
    offline_cell,
    world_config,
    score,
)
from row.models import FastRotatedDiscreteLibraryLearner
from row.pool import PoolBudget, free_memory_bytes, measure_rss_bytes, run_pool
from row.rotated_world import generate_rotated_world
from row.experiments.so1_storage import (
    atomic_json, cell_stamp, digest, environment, fingerprint, load_cell,
    memory_snapshot, now, resolved, restore_model, save_model, world_digest, writer_lock,
)

WORLDS = (0, 1, 2)
WORLDS_REQUIRED = 2
BATCHES = (2, 64)
GRADIENT_LEVELS = (16384, 32768, 65536, 131072, 262144)
STAGE_D_REPORT = Path("reports/rotated_g5r_interference.json")
ANCHOR_TOLERANCE = 0.02
PERSISTENCE_LATER = 2
PROTOCOL_ID = "SO1-budget-bracket-v2-restart"
IMPLEMENTATION = FastRotatedDiscreteLibraryLearner.implementation
# Sampling-stream cell indices: 100-109 oracle grid, 110-111 stage 2.
ORACLE_CELL_INDEX = {(b, g): 100 + i * len(GRADIENT_LEVELS) + j
                     for i, b in enumerate(BATCHES) for j, g in enumerate(GRADIENT_LEVELS)}
STAGE2_CELL_INDEX = {2: 110, 64: 111}
ANCHORS = {(2, 16384): "C_lo", (64, 262144): "C_hi"}


def build_fast(config) -> FastRotatedDiscreteLibraryLearner:
    selected = config.discrete_model
    return FastRotatedDiscreteLibraryLearner(
        d=config.world.state_dim,
        operator_slots=selected.operator_slots,
        operator_rank=selected.operator_rank,
        task_steps=selected.task_steps,
        alpha=selected.operator_alpha_init,
        initial_temperature=selected.initial_temperature,
        final_temperature=selected.final_temperature,
        seed=selected.seed,
        learnable_alpha=selected.learnable_alpha,
        activation=selected.operator_activation,
    )


def checkpoints_for(updates: int) -> tuple[int, ...]:
    return tuple(sorted({updates // 8, updates // 4, updates // 2, 3 * updates // 4, updates}))


def cell_key(batch: int, gradients: int, oracle: bool) -> str:
    return f"{'O' if oracle else 'L'}_b{batch}_g{gradients}"


def persistence(trajectory: dict[str, float]) -> tuple[int | None, str]:
    points = sorted((int(k), float(v)) for k, v in trajectory.items())
    for index, (checkpoint, value) in enumerate(points):
        if value <= THRESHOLD:
            later = [v for _, v in points[index + 1:]]
            if len(later) < PERSISTENCE_LATER:
                return checkpoint, "crossed, persistence unobservable"
            if all(v <= THRESHOLD for v in later[:PERSISTENCE_LATER]):
                return checkpoint, "persistent"
            return checkpoint, "crossed, not persistent"
    return None, "not crossed"


def monotone(trajectory: dict[str, float]) -> bool:
    values = [v for _, v in sorted((int(k), v) for k, v in trajectory.items())]
    return all(b <= a for a, b in zip(values, values[1:]))


def run_job(job: dict) -> dict:
    """Each worker owns its durable cell; partial optimizers are never resumed."""
    if "artifact" not in job:
        return _run_job(job)
    path = Path(job["artifact"])
    with writer_lock(path / "writer.lock"):
        stamp = cell_stamp(job)
        if (path / "result.json").exists():
            return load_cell(path, stamp)
        manifest = {"stamp": stamp, "resolved": resolved(world_config(load_config(job["config"]), job["world"]))}
        manifest_path = path / "fingerprint.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"partial cell fingerprint mismatch: {path}")
        atomic_json(manifest_path, manifest)
        result = _run_job(job)
        atomic_json(path / "result.json", {
            "stamp": stamp, "complete": True, "finished_utc": now(),
            "result": result, "result_sha256": fingerprint(result),
            "artifact_sha256": {name: digest(path / name) for name in
                                ("model.pt", "model_state.json", "config.yaml", "fingerprint.json")},
        })
        return result


def _run_job(job: dict) -> dict:
    torch.set_num_threads(1)
    base = load_config(job["config"])
    cfg = world_config(base, job["world"])
    if "resolved_sha256" in job and fingerprint(resolved(cfg)) != job["resolved_sha256"]:
        raise ValueError("configuration changed between parent and worker")
    world = generate_rotated_world(cfg.world)
    started = time.perf_counter()
    result = offline_cell(
        cfg, world, _assignment(cfg, world),
        oracle=job["oracle"], updates=job["updates"], batch=job["batch"],
        cell_index=job.get("sampling_index", job["cell_index"]), checkpoints_requested=checkpoints_for(job["updates"]),
        build=build_fast,
        on_complete=(lambda model: save_model(job["artifact"], model, cfg)) if "artifact" in job else None,
        retain_per_task=True,
    )
    result = {k: v for k, v in result.items() if k != "final_per_task"} | {
        "final_per_task": result["final_per_task"],
        "seconds": round(time.perf_counter() - started, 1),
        "rss_bytes": measure_rss_bytes(),
        "implementation": IMPLEMENTATION,
        "cell_index": job["cell_index"],
        "sampling_index": job.get("sampling_index", job["cell_index"]),
        "world_sha256": world_digest(world),
        "started_from_seed": cfg.discrete_model.seed,
        "gradients": job["updates"] * job["batch"],
    }
    trajectory = {k: v["median"] for k, v in result["checkpoints"].items()}
    crossing, label = persistence(trajectory)
    result["first_crossing"] = crossing
    result["persistence"] = label
    result["monotone"] = monotone(trajectory)
    if not result["finite"] or result["shared_relative_change"] <= 0:
        raise ValueError("non-vacuity check failed: finite shared learning required")
    if job["oracle"] and not (result["pinned_routes_preserved"] and result["pinned_one_hot_at_1.0"]):
        raise ValueError("oracle pinning failed")
    if not job["oracle"] and result["code_relative_change"] <= 0:
        raise ValueError("learned routes did not move")
    if "artifact" in job:
        restored = restore_model(job["artifact"], cfg, world, build_fast)
        reloaded = score(restored, world)
        if reloaded["per_task"] != result["final_per_task"]:
            raise ValueError("terminal model reload changed predictions")
        result["reload_exact"] = True
    return result


def cell_passes(cells: dict, key: str) -> bool:
    return sum(bool(cells[key][str(w)]["passes"]) for w in WORLDS if str(w) in cells[key]) >= WORLDS_REQUIRED


def envelope(cells: dict, batch: int) -> dict:
    """Lowest passing gradient level on one curve; both persistence readings."""
    out = {"lowest_passing": None, "lowest_passing_excluding_unobservable": None}
    for g in GRADIENT_LEVELS:
        key = cell_key(batch, g, True)
        if key not in cells or len(cells[key]) < len(WORLDS):
            continue
        if cell_passes(cells, key):
            if out["lowest_passing"] is None:
                out["lowest_passing"] = g
            persistent = sum(
                cells[key][str(w)]["passes"] and cells[key][str(w)]["persistence"] == "persistent"
                for w in WORLDS
            ) >= WORLDS_REQUIRED
            if persistent and out["lowest_passing_excluding_unobservable"] is None:
                out["lowest_passing_excluding_unobservable"] = g
    return out


def paired_differences(cells: dict) -> dict:
    """Per gradient level, per world: median(B=2) - median(B=64)."""
    out = {}
    for g in GRADIENT_LEVELS:
        a, b = cell_key(2, g, True), cell_key(64, g, True)
        if a in cells and b in cells:
            out[str(g)] = {
                str(w): cells[a][str(w)]["terminal_median"] - cells[b][str(w)]["terminal_median"]
                for w in WORLDS if str(w) in cells[a] and str(w) in cells[b]
            }
    return out


def anchor_check(cells: dict, stage_d: dict) -> dict:
    out = {}
    for (batch, g), name in ANCHORS.items():
        key = cell_key(batch, g, True)
        rows = {}
        for w in WORLDS:
            mine = cells.get(key, {}).get(str(w))
            theirs = stage_d["cells"][name][str(w)]
            if mine is None:
                continue
            rows[str(w)] = {
                "so1": mine["terminal_median"], "stage_d": theirs["terminal_median"],
                "abs_error": abs(mine["terminal_median"] - theirs["terminal_median"]),
                "same_verdict": bool(mine["passes"]) == bool(theirs["passes"]),
                "passes": abs(mine["terminal_median"] - theirs["terminal_median"]) <= ANCHOR_TOLERANCE
                and bool(mine["passes"]) == bool(theirs["passes"]),
            }
        out[key] = {"stage_d_cell": name, "worlds": rows,
                    "passes": len(rows) == len(WORLDS) and all(r["passes"] for r in rows.values())}
    return out


def classify(oracle_any: bool, learned_any: bool | None) -> str:
    if not oracle_any:
        return "NO_ORACLE_CELL_PASSES"  # Track B stop rule 1: D_form below required strength
    if learned_any is None:
        return "ORACLE_PASSES_STAGE2_PENDING"
    if not learned_any:
        return "ORACLE_PASSES_LEARNED_FAILS"  # stop rule 2: writer cannot acquire it
    return "ORACLE_AND_LEARNED_PASS"  # SO2 licensed at this envelope


def protocol(config) -> dict:
    return {
        "id": PROTOCOL_ID, "frozen_plan": "SO1_BUDGET_BRACKET_PLAN.md",
        "implementation": IMPLEMENTATION, "worlds": list(WORLDS),
        "worlds_required": WORLDS_REQUIRED, "batches": list(BATCHES),
        "gradient_levels": list(GRADIENT_LEVELS), "threshold": THRESHOLD,
        "anchor_tolerance": ANCHOR_TOLERANCE, "persistence_later_checkpoints": PERSISTENCE_LATER,
        "checkpoints": "U/8, U/4, U/2, 3U/4, U",
        "sampling_stream": "SeedSequence([1702, world, oracle_cell_index]); stage 2 paired to oracle",
        "resolved_configs": {str(w): resolved(world_config(config, w)) for w in WORLDS},
        "input_sha256": {p.as_posix(): digest(p) for p in (
            Path("SO1_BUDGET_BRACKET_PLAN.md"), Path("SO1_RESTART_AMENDMENT.md"), STAGE_D_REPORT)},
        "environment": environment(),
        "global_lr": config.discrete_model.global_learning_rate,
        "task_lr": config.discrete_model.task_learning_rate,
        "weight_decay": config.discrete_model.weight_decay,
        "model_seed": config.discrete_model.seed, "slots": config.discrete_model.operator_slots,
        "temperature": [config.discrete_model.initial_temperature, config.discrete_model.final_temperature],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/so1_budget_bracket.json"))
    parser.add_argument("--measured-rss-mib", type=int, required=True,
                        help="resident size of one batch-64 fast cell, measured before launch")
    parser.add_argument("--hard-cap", type=int, default=None)
    parser.add_argument("--gate", type=Path, default=Path("artifacts/so1_restart/gate.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/so1_restart/cells"))
    parser.add_argument("--dry-run", action="store_true",
                        help="structural check: 16 updates per corner cell, world 0, no report")
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = load_config(args.config)

    if args.dry_run:
        for (batch, g) in ANCHORS:
            job = {"config": str(args.config), "world": 0, "oracle": True, "batch": batch,
                   "updates": 16, "cell_index": ORACLE_CELL_INDEX[(batch, g)]}
            r = run_job(job)
            print(f"dry-run b{batch}: median {r['terminal_median']:.4f} pinned {r['pinned_routes_preserved']} "
                  f"one-hot {r['pinned_one_hot_at_1.0']} shared_change {r['shared_relative_change']:.3e} "
                  f"{r['seconds']}s rss {r['rss_bytes'] / 2**20:.0f} MiB")
        job = {"config": str(args.config), "world": 0, "oracle": False, "batch": 2,
               "updates": 16, "cell_index": STAGE2_CELL_INDEX[2]}
        r = run_job(job)
        print(f"dry-run learned: code_change {r['code_relative_change']:.3e} finite {r['finite']}")
        return

    output_lock = Path("artifacts/so1_report_locks") / (fingerprint(str(args.output.resolve())) + ".lock")
    with writer_lock(output_lock), writer_lock(args.artifact_root.parent / "report.lock"):
        run_grid(args, base)


def run_grid(args, base):
    if subprocess.run([sys.executable, "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("preregistration check failed")
    if subprocess.run([sys.executable, "tools/check_invalid.py"]).returncode != 0:
        raise SystemExit("invalid-artifact check failed")
    require_clean_code(args.output)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if (gate.get("gate") != "PASS" or gate.get("git_commit") != git_commit()
            or gate.get("implementation") != IMPLEMENTATION
            or gate.get("protocol_sha256") != fingerprint(protocol(base))):
        raise SystemExit("current-commit fast SO1 pool gate required")
    if args.hard_cap != 2 or args.measured_rss_mib < gate["budget_mib"]:
        raise SystemExit("restart requires the calibrated two-worker budget")

    stage_d = json.loads(STAGE_D_REPORT.read_text(encoding="utf-8"))
    expected = protocol(base)
    manifest_path = args.artifact_root.parent / "run_manifest.json"
    manifest_identity = {"git_commit": git_commit(), "protocol_sha256": fingerprint(expected),
                         "output": str(args.output.resolve()), "artifact_root": str(args.artifact_root.resolve())}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if {k: manifest.get(k) for k in manifest_identity} != manifest_identity:
            raise ValueError("run manifest mismatch; preserve the old batch before restarting")
    else:
        manifest = manifest_identity | {"started_utc": now()}
        atomic_json(manifest_path, manifest)
    if args.output.exists():
        out = json.loads(args.output.read_text(encoding="utf-8"))
        if out.get("protocol") != expected:
            raise SystemExit("existing report has a different protocol fingerprint")
        if out.get("git_commit") != git_commit():
            raise SystemExit("existing report came from a different git commit")
    else:
        out = {"frozen_plan": "SO1_BUDGET_BRACKET_PLAN.md", "git_commit": git_commit(),
               "protocol": expected, "protocol_sha256": fingerprint(expected),
               "launch": {}, "cells": {}, "complete": False, "started_utc": manifest["started_utc"]}
    if out.get("protocol_sha256") != fingerprint(expected):
        raise SystemExit("report protocol hash mismatch")
    if out["started_utc"] != manifest["started_utc"]:
        raise ValueError("report/run-manifest start time mismatch")
    out["complete"] = False
    out.setdefault("launch_history", []).append(out["launch"])
    out["launch"] = {"free_memory_bytes_at_launch": free_memory_bytes(),
                     "measured_rss_bytes": args.measured_rss_mib * 2**20,
                     "pool_gate_commit": gate.get("git_commit"),
                     "gate_path": str(args.gate), "gate_sha256": digest(args.gate),
                     "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
                     "artifact_root": str(args.artifact_root)}
    atomic_json(args.output, out)

    def pending_jobs(oracle: bool, grid) -> list[dict]:
        jobs = []
        for (batch, g), cell_index in grid:
            key = cell_key(batch, g, oracle)
            for w in WORLDS:
                job = {"key": key, "config": str(args.config), "world": w, "oracle": oracle,
                       "batch": batch, "updates": g // batch, "cell_index": cell_index,
                       "sampling_index": ORACLE_CELL_INDEX[(batch, g)],
                       "git_commit": git_commit(), "protocol_sha256": out["protocol_sha256"],
                       "resolved_sha256": fingerprint(expected["resolved_configs"][str(w)]),
                       "artifact": str(args.artifact_root / key / f"world_{w}")}
                artifact = Path(job["artifact"])
                if (artifact / "result.json").exists():
                    result = load_cell(artifact, cell_stamp(job))
                    old = out["cells"].get(key, {}).get(str(w))
                    if old is not None and old != result:
                        raise ValueError("report and durable cell disagree")
                    out["cells"].setdefault(key, {})[str(w)] = result
                    atomic_json(args.output, out)
                elif str(w) in out["cells"].get(key, {}):
                    raise ValueError("report cell has no durable artifact")
                else:
                    jobs.append(job)
        return jobs

    budget = PoolBudget(measured_rss_bytes=args.measured_rss_mib * 2**20, hard_cap=args.hard_cap)

    def run_and_record(jobs: list[dict]) -> None:
        # Smallest budgets first so the anchor corners and cheap cells land early.
        jobs.sort(key=lambda j: (j["updates"] * j["batch"], j["batch"], j["world"]))
        def record(job, result):
            out["cells"].setdefault(job["key"], {})[str(job["world"])] = result
            atomic_json(args.output, out)
            print(f"[{job['key']} w{job['world']}] durable cell saved ({result['seconds']}s)", flush=True)
        def available():
            sample = memory_snapshot()
            return min(sample["physical_available"], sample["commit_available"])
        run_pool(run_job, jobs, budget, free_probe=available, on_result=record)

    run_and_record(pending_jobs(True, [(pair, ORACLE_CELL_INDEX[pair]) for pair in ANCHORS]))

    out["anchor"] = anchor_check(out["cells"], stage_d)
    if not all(a["passes"] for a in out["anchor"].values()):
        out["classification"] = "ANCHOR_FAILED_NOTHING_READ"
        out["finished_utc"] = now()
        atomic_json(args.output, out)
        raise SystemExit("ANCHOR FAILED: SO1 stopped; no dose curve or scientific verdict licensed")

    atomic_json(args.output, out)
    run_and_record(pending_jobs(True, ORACLE_CELL_INDEX.items()))

    out["envelope"] = {str(b): envelope(out["cells"], b) for b in BATCHES}
    out["paired_differences_b2_minus_b64"] = paired_differences(out["cells"])
    out["dose_monotonicity"] = {str(b): {str(w): monotone({str(g): out["cells"][cell_key(b, g, True)][str(w)]["terminal_median"]
                                                           for g in GRADIENT_LEVELS}) for w in WORLDS} for b in BATCHES}
    out["cell_passes"] = {k: cell_passes(out["cells"], k) for k in out["cells"]}
    oracle_any = any(out["envelope"][str(b)]["lowest_passing"] is not None for b in BATCHES)

    learned_any = None
    if oracle_any:
        grid = [((b, out["envelope"][str(b)]["lowest_passing"]), STAGE2_CELL_INDEX[b])
                for b in BATCHES if out["envelope"][str(b)]["lowest_passing"] is not None]
        run_and_record(pending_jobs(False, grid))
        out["cell_passes"] = {k: cell_passes(out["cells"], k) for k in out["cells"]}
        learned_any = any(out["cell_passes"][cell_key(b, g, False)] for (b, g), _ in grid)
    out["classification"] = classify(oracle_any, learned_any)
    out["complete"] = True
    out["finished_utc"] = now()
    atomic_json(args.output, out)
    print("envelope:", out["envelope"])
    print("classification:", out["classification"])


if __name__ == "__main__":
    main()
