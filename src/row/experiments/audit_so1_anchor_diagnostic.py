"""SO1 anchor diagnostic: implementation vs resampling vs trajectory sensitivity.

Frozen in `SO1_ANCHOR_DIAGNOSTIC_AMENDMENT.md` (7587a5a). Every cell is the
Stage D C_lo construction (oracle routes, batch 2, 8,192 updates, Stage D
`offline_cell` and checkpoint list); only the model kind, the sampling stream
and a float-level init perturbation vary. Development worlds 0-2.

Each worker owns one durable cell record; the parent owns the aggregate report.
Run from clean committed code; exits nonzero unless all 24 cells are accepted.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import (
    CHECKPOINTS, LIFETIME_BATCH, LIFETIME_UPDATES, _assignment, build_model,
    offline_cell, score, world_config,
)
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import (
    MemorySampler, atomic_json, canonical, digest, environment, fingerprint,
    load_cell, memory_snapshot, now, resolved, restore_model, save_model,
    world_digest, writer_lock,
)
from row.models import FastRotatedDiscreteLibraryLearner, RotatedDiscreteLibraryLearner
from row.pool import PoolBudget, measure_rss_bytes, run_pool
from row.rotated_world import generate_rotated_world

PLAN = Path("SO1_ANCHOR_DIAGNOSTIC_AMENDMENT.md")
STAGE_D_REPORT = Path("reports/rotated_g5r_interference.json")
SO1_REPORT = Path("reports/so1_budget_bracket.json")
OUTPUT = Path("reports/so1_anchor_diagnostic.json")
ROOT = Path("artifacts/so1_anchor_diagnostic")
PROTOCOL_ID = "SO1-anchor-diagnostic-v1"
WORLDS = (0, 1, 2)
EPS = 1e-7
PERTURB_SEED = 1703
TOLERANCE = 0.02
REPRO_TOLERANCE = 1e-6
MIB = 2**20
RESERVE = 4 * 2**30
HARD_CAP = 2
# name: (kind, stream, eps)
CELLS = {
    "R": ("sequential", 0, 0.0),
    "P": ("sequential", 0, EPS),
    "F0": ("fast", 0, 0.0),
    "S100": ("sequential", 100, 0.0),
    "F100": ("fast", 100, 0.0),
    "F120": ("fast", 120, 0.0),
    "F121": ("fast", 121, 0.0),
    "F122": ("fast", 122, 0.0),
}
FAST_STREAMS = ("F0", "F100", "F120", "F121", "F122")
BUILDERS = {"sequential": build_model, "fast": build_fast}
IMPLEMENTATIONS = {"sequential": "rotated_discrete_sequential",
                   "fast": FastRotatedDiscreteLibraryLearner.implementation}


def perturb_shared(model, eps: float, world: int) -> None:
    """Multiply every shared parameter elementwise by 1 + eps * z (z ~ N(0,1))."""
    if eps == 0.0:
        return
    generator = torch.Generator().manual_seed(PERTURB_SEED * 1000 + world)
    with torch.no_grad():
        for p in model.shared_parameters():
            z = torch.randn(p.shape, generator=generator, dtype=p.dtype)
            p.mul_(1.0 + eps * z)


def builder_for(kind: str, eps: float, world: int):
    base = BUILDERS[kind]
    def build(config):
        model = base(config)
        perturb_shared(model, eps, world)
        return model
    return build


def protocol(config) -> dict:
    return {
        "id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(),
        "cells": {k: {"kind": v[0], "stream": v[1], "eps": v[2]} for k, v in CELLS.items()},
        "implementations": IMPLEMENTATIONS, "worlds": list(WORLDS),
        "updates": LIFETIME_UPDATES, "batch": LIFETIME_BATCH, "checkpoints": list(CHECKPOINTS),
        "perturbation": f"p *= 1 + eps*z, z~N(0,1), torch.Generator seed {PERTURB_SEED}*1000+world, shared only",
        "sampling_stream": "SeedSequence([1702, world, stream])",
        "tolerance": TOLERANCE, "repro_tolerance": REPRO_TOLERANCE,
        "resolved_configs": {str(w): resolved(world_config(config, w)) for w in WORLDS},
        "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, STAGE_D_REPORT, SO1_REPORT)},
        "environment": environment(),
    }


def stamp(job: dict) -> dict:
    return {k: job[k] for k in ("key", "world", "kind", "stream", "eps", "updates",
                                "git_commit", "protocol_sha256")}


def run_job(job: dict) -> dict:
    if "artifact" not in job:
        return _run_job(job)
    path = Path(job["artifact"])
    with writer_lock(path / "writer.lock"):
        if (path / "result.json").exists():
            return load_cell(path, stamp(job))
        result = _run_job(job)
        atomic_json(path / "result.json", {
            "stamp": stamp(job), "complete": True, "finished_utc": now(),
            "result": result, "result_sha256": fingerprint(result),
            "artifact_sha256": {name: digest(path / name) for name in
                                ("model.pt", "model_state.json", "config.yaml")},
        })
        return result


def _run_job(job: dict) -> dict:
    torch.set_num_threads(1)
    cfg = world_config(load_config(job["config"]), job["world"])
    if "resolved_sha256" in job and fingerprint(resolved(cfg)) != job["resolved_sha256"]:
        raise ValueError("configuration changed between parent and worker")
    world = generate_rotated_world(cfg.world)
    build = builder_for(job["kind"], job["eps"], job["world"])
    reference = BUILDERS[job["kind"]](cfg)
    probe = build(cfg)
    moved = any(not torch.equal(a, b) for a, b in
                zip(reference.shared_parameters(), probe.shared_parameters()))
    if moved != (job["eps"] > 0):
        raise ValueError("perturbation non-vacuity failed: eps>0 must move, eps=0 must not")
    expected_cls = RotatedDiscreteLibraryLearner if job["kind"] == "sequential" else FastRotatedDiscreteLibraryLearner
    if type(probe) is not expected_cls:
        raise ValueError(f"wrong model class {type(probe).__name__} for kind {job['kind']}")
    started = time.perf_counter()
    result = offline_cell(
        cfg, world, _assignment(cfg, world), oracle=True, updates=job["updates"],
        batch=LIFETIME_BATCH, cell_index=job["stream"],
        checkpoints_requested=CHECKPOINTS, build=build,
        on_complete=(lambda model: save_model(job["artifact"], model, cfg)) if "artifact" in job else None,
        retain_per_task=True,
    )
    if not (result["finite"] and result["shared_relative_change"] > 0
            and result["pinned_routes_preserved"] and result["pinned_one_hot_at_1.0"]):
        raise ValueError("non-vacuity failed: finite, shared learning and pinning required")
    out = {
        "terminal_median": result["terminal_median"],
        "trajectory": {k: v["median"] for k, v in result["checkpoints"].items()},
        "final_per_task": result["final_per_task"],
        "shared_relative_change": result["shared_relative_change"],
        "pinned": True, "finite": True, "kind": job["kind"], "model_class": type(probe).__name__,
        "stream": job["stream"], "eps": job["eps"], "world_sha256": world_digest(world),
        "seconds": round(time.perf_counter() - started, 1), "rss_bytes": measure_rss_bytes(),
    }
    if "artifact" in job:
        restored = restore_model(job["artifact"], cfg, world, BUILDERS[job["kind"]])
        if score(restored, world)["per_task"] != result["final_per_task"]:
            raise ValueError("terminal model reload changed predictions")
        out["reload_exact"] = True
    return out


def quantities(cells: dict, stage_d: dict) -> dict:
    """Per-world registered quantities and the ordered classification."""
    per_world = {}
    for w in WORLDS:
        c = {k: cells[k][str(w)]["terminal_median"] for k in CELLS}
        sd = stage_d["cells"]["C_lo"][str(w)]["terminal_median"]
        fast = [c[k] for k in FAST_STREAMS]
        per_world[str(w)] = {
            "stage_d": sd, "cells": c,
            "d_repro": abs(c["R"] - sd), "d_impl0": abs(c["F0"] - sd),
            "d_impl100": abs(c["F100"] - c["S100"]), "d_chaos": abs(c["P"] - c["R"]),
            "spread": max(fast) - min(fast), "fast_sd": float(np.std(fast, ddof=1)),
            "miss": abs(c["F100"] - sd),
        }
        per_world[str(w)]["miss_within_spread"] = per_world[str(w)]["miss"] <= per_world[str(w)]["spread"]
    rows = per_world.values()
    if any(r["d_repro"] > REPRO_TOLERANCE for r in rows):
        label = "HARNESS_FAILED"
    elif all(r["d_impl0"] <= TOLERANCE and r["d_impl100"] <= TOLERANCE for r in rows):
        label = "IMPLEMENTATION_EQUIVALENT"
    elif all(r["d_chaos"] > TOLERANCE for r in rows
             if r["d_impl0"] > TOLERANCE or r["d_impl100"] > TOLERANCE):
        label = "TRAJECTORY_SENSITIVE"
    else:
        label = "IMPLEMENTATION_DIVERGES"
    return {"per_world": per_world, "classification": label}


def first_departure(trajectory: dict, reference: dict) -> int | None:
    for k in sorted(reference, key=int):
        if k in trajectory and abs(trajectory[k] - reference[k]) > TOLERANCE:
            return int(k)
    return None


def available() -> int:
    s = memory_snapshot()
    return min(s["physical_available"], s["commit_available"])


def host_precondition(budget_mib: int) -> dict:
    samples = [memory_snapshot()]
    for _ in range(6):
        time.sleep(10)
        samples.append(memory_snapshot())
    need = RESERVE + HARD_CAP * budget_mib * MIB
    record = {"utc": now(), "budget_mib": budget_mib, "samples": samples, "need": need,
              "passes": min(min(s["physical_available"], s["commit_available"]) for s in samples) > need
              and max(s["pagefile_used"] for s in samples) <= samples[0]["pagefile_used"]}
    atomic_json(ROOT / f"precondition_{time.time_ns()}.json", record)
    if not record["passes"]:
        raise SystemExit("host precondition failed; see durable precondition record")
    return record


def gate(config: str, pmeta: dict) -> dict:
    """Reduced-update serial-versus-pooled bitwise gate over every cell kind."""
    jobs = [{"key": k, "config": config, "world": w, "kind": v[0], "stream": v[1],
             "eps": v[2], "updates": 16} for k, v in CELLS.items() if k in ("R", "P", "F0") for w in (0, 1)]
    host_precondition(768)
    with MemorySampler() as sampler:
        serial = [run_job(j) for j in jobs]
    serial_memory = sampler.summary()
    budget_mib = max(768, math.ceil(1.5 * max(serial_memory["max_parent_rss"],
                                             serial_memory["max_parent_private"]) / MIB))
    host_precondition(budget_mib)
    logs = []
    with MemorySampler() as sampler:
        pooled = run_pool(run_job, jobs, PoolBudget(budget_mib * MIB, hard_cap=HARD_CAP),
                          free_probe=available, log=lambda m: (logs.append(m), print(m, flush=True)))
    pooled_memory = sampler.summary()
    strip = lambda r: {k: v for k, v in r.items() if k not in ("seconds", "rss_bytes")}
    identical = [canonical(strip(a)) == canonical(strip(b)) for a, b in zip(serial, pooled)]
    p_moves = [r["terminal_median"] for r, j in zip(serial, jobs) if j["key"] == "P"] != \
              [r["terminal_median"] for r, j in zip(serial, jobs) if j["key"] == "R"]
    ok = (all(identical) and p_moves and not pooled_memory["errors"]
          and pooled_memory["min_commit_available"] > RESERVE
          and pooled_memory["pagefile_growth"] <= 0
          and any(f"({HARD_CAP} running)" in line for line in logs))
    record = {"git_commit": git_commit(), "protocol_sha256": pmeta["sha"], "jobs": jobs,
              "bitwise_identical_cells": sum(identical), "expected_cells": len(jobs),
              "perturbation_changes_result": p_moves, "budget_mib": budget_mib,
              "serial_memory": serial_memory, "pooled_memory": pooled_memory,
              "gate": "PASS" if ok else "FAIL", "finished_utc": now()}
    atomic_json(ROOT / "gate.json", record)
    print(f"GATE {record['gate']}: {sum(identical)}/{len(jobs)} bitwise; P moves {p_moves}; "
          f"budget {budget_mib} MiB", flush=True)
    if not ok:
        raise SystemExit("diagnostic pool gate failed")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="16 updates, world 0, every cell kind; no durable output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = load_config(args.config)

    if args.dry_run:
        for key, (kind, stream, eps) in CELLS.items():
            r = run_job({"key": key, "config": args.config, "world": 0, "kind": kind,
                         "stream": stream, "eps": eps, "updates": 16})
            print(f"dry-run {key:5s} {r['model_class']:34s} median {r['terminal_median']:.8f} "
                  f"shared_change {r['shared_relative_change']:.3e} {r['seconds']}s", flush=True)
        return 0

    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    require_clean_code(OUTPUT)
    if OUTPUT.exists():
        raise SystemExit("diagnostic report already exists; preserve it rather than overwrite")
    expected = protocol(base)
    pmeta = {"sha": fingerprint(expected)}
    ROOT.mkdir(parents=True, exist_ok=True)
    with writer_lock(ROOT / "launcher.lock"):
        gate_record = gate(args.config, pmeta)
        host = host_precondition(gate_record["budget_mib"])
        stage_d = json.loads(STAGE_D_REPORT.read_text(encoding="utf-8"))
        out = {"frozen_plan": PLAN.as_posix(), "git_commit": git_commit(), "protocol": expected,
               "protocol_sha256": pmeta["sha"], "started_utc": now(), "host": host,
               "gate_sha256": digest(ROOT / "gate.json"), "cells": {}, "complete": False}
        atomic_json(OUTPUT, out)
        jobs = []
        for key, (kind, stream, eps) in CELLS.items():
            for w in WORLDS:
                jobs.append({"key": key, "config": args.config, "world": w, "kind": kind,
                             "stream": stream, "eps": eps, "updates": LIFETIME_UPDATES,
                             "git_commit": git_commit(), "protocol_sha256": pmeta["sha"],
                             "resolved_sha256": fingerprint(expected["resolved_configs"][str(w)]),
                             "artifact": str(ROOT / "cells" / key / f"world_{w}")})
        # Sequential cells first: they dominate wall time.
        jobs.sort(key=lambda j: (j["kind"] != "sequential", j["key"], j["world"]))

        def record(job, result):
            out["cells"].setdefault(job["key"], {})[str(job["world"])] = result
            atomic_json(OUTPUT, out)
            print(f"[{job['key']} w{job['world']}] median {result['terminal_median']:.4f} "
                  f"({result['seconds']}s)", flush=True)

        with MemorySampler() as sampler:
            run_pool(run_job, jobs, PoolBudget(gate_record["budget_mib"] * MIB, hard_cap=HARD_CAP),
                     free_probe=available, on_result=record)
        atomic_json(ROOT / "run_memory.json", {"summary": sampler.summary()})
        if sum(len(v) for v in out["cells"].values()) != len(CELLS) * len(WORLDS):
            raise SystemExit("missing cells")
        q = quantities(out["cells"], stage_d)
        for w in WORLDS:
            ref = {k: v["median"] for k, v in stage_d["cells"]["C_lo"][str(w)]["checkpoints"].items()}
            q["per_world"][str(w)]["first_departure_F0"] = first_departure(out["cells"]["F0"][str(w)]["trajectory"], ref)
            q["per_world"][str(w)]["first_departure_P"] = first_departure(out["cells"]["P"][str(w)]["trajectory"], ref)
        out.update(q)
        out["complete"] = True
        out["finished_utc"] = now()
        atomic_json(OUTPUT, out)
        print("classification:", out["classification"], flush=True)
        for w, r in out["per_world"].items():
            print(f"w{w}: repro {r['d_repro']:.2e} impl0 {r['d_impl0']:.4f} impl100 {r['d_impl100']:.4f} "
                  f"chaos {r['d_chaos']:.4f} spread {r['spread']:.4f} miss {r['miss']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
