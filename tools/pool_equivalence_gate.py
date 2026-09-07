"""CONCURRENCY_PLAN.md gate 1: serial versus pooled cells must be bitwise equal.

Runs the real Stage D `offline_cell` construction (rotated worlds 0-2, the
frozen learner, the registered SeedSequence streams) at a reduced update
count, once serially in this process and once through `row.pool.run_pool`,
and compares every number in the returned records. Writes
`reports/pool_equivalence_gate.json`. Exit 0 only on 100% bitwise agreement.

Run before the pool is used for a scientific batch of a new cell family; rerun
whenever `row/pool.py` or the cell construction changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_interference import (
    OFFLINE_CELLS,
    WORLDS,
    _assignment,
    offline_cell,
    world_config,
)
from row.pool import PoolBudget, free_memory_bytes, run_pool
from row.rotated_world import generate_rotated_world

UPDATES = 48
CHECKPOINTS = (0, 16, 48)
MEASURED_RSS_BYTES = 900 * 1024**2  # batch-64 rotated cell, measured 2026-09-05


def run_cell(job):
    name, seed = job
    torch.set_num_threads(1)
    base = load_config("configs/v1.yaml")
    cfg = world_config(base, seed)
    world = generate_rotated_world(cfg.world)
    oracle, _, batch, cell_index = OFFLINE_CELLS[name]
    return offline_cell(
        cfg, world, _assignment(cfg, world),
        oracle=oracle, updates=UPDATES, batch=batch, cell_index=cell_index,
        checkpoints_requested=CHECKPOINTS,
    )


def canonical(result) -> str:
    return json.dumps(result, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/pool_equivalence_gate.json"))
    parser.add_argument("--hard-cap", type=int, default=None)
    args = parser.parse_args()
    jobs = [(name, seed) for name in OFFLINE_CELLS for seed in WORLDS]

    t0 = time.perf_counter()
    serial = [run_cell(job) for job in jobs]
    t_serial = time.perf_counter() - t0

    budget = PoolBudget(measured_rss_bytes=MEASURED_RSS_BYTES, hard_cap=args.hard_cap)
    t0 = time.perf_counter()
    pooled = run_pool(run_cell, jobs, budget)
    t_pool = time.perf_counter() - t0

    mismatches = []
    for job, a, b in zip(jobs, serial, pooled):
        if canonical(a) != canonical(b):
            worst = max(abs(a["final_per_task"][k] - b["final_per_task"][k]) for k in a["final_per_task"])
            mismatches.append({"cell": f"{job[0]} w{job[1]}", "max_abs_final_nmse_diff": worst})
    out = {
        "git_commit": git_commit(),
        "jobs": len(jobs),
        "updates_per_cell": UPDATES,
        "checkpoints": list(CHECKPOINTS),
        "cap_at_launch": budget.cap(free_memory_bytes()),
        "measured_rss_bytes": MEASURED_RSS_BYTES,
        "serial_seconds": round(t_serial, 1),
        "pooled_seconds": round(t_pool, 1),
        "bitwise_identical_cells": len(jobs) - len(mismatches),
        "mismatches": mismatches,
        "gate": "PASS" if not mismatches else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps(out, indent=1))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
