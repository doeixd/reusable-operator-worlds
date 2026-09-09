"""Calibrate the repaired SO1 family, then launch its anchor-first grid.

Run from clean committed code. All operational files live under artifacts/;
the aggregate scientific report is the sole allowed untracked report.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1_budget_bracket import IMPLEMENTATION, protocol, run_job
from row.experiments.so1_storage import (
    MemorySampler, atomic_json, canonical, fingerprint, memory_snapshot, now, writer_lock,
)
from row.pool import PoolBudget, run_pool

ROOT = Path("artifacts/so1_restart")
MIB = 2**20
RESERVE = 4 * 2**30


def scientific(result):
    return {k: v for k, v in result.items() if k not in {"seconds", "rss_bytes"}}


def host_precondition(budget_mib):
    samples = [memory_snapshot()]
    for _ in range(6):
        time.sleep(10)
        samples.append(memory_snapshot())
    need = RESERVE + 2 * budget_mib * MIB
    tenants = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info["memory_info"]
            if info and info.rss > 200 * MIB:
                tenants.append({"pid": proc.pid, "name": proc.info["name"],
                                "rss": info.rss, "private": info.private})
        except psutil.NoSuchProcess:
            pass
    record = {"utc": now(), "budget_mib": budget_mib, "samples": samples, "tenants": tenants}
    record["passes"] = (min(min(s["physical_available"], s["commit_available"]) for s in samples) > need
                         and max(s["pagefile_used"] for s in samples) <= samples[0]["pagefile_used"])
    atomic_json(ROOT / f"precondition_{time.time_ns()}.json", record)
    if not record["passes"]:
        raise RuntimeError("host precondition failed; see durable precondition record")
    print(f"HOST PASS: physical and commit headroom > {need / MIB:.0f} MiB; page file stable", flush=True)
    return record


def calibrate(config):
    # Reduced-update implementation checks, never SO1 scientific cells.
    jobs = [{"config": config, "world": w, "oracle": oracle, "batch": batch,
             "updates": 16, "cell_index": index, "sampling_index": index}
            for oracle, batch, index in ((True, 2, 100), (True, 64, 105), (False, 64, 105))
            for w in (0, 1, 2)]
    host_precondition(768)
    with MemorySampler() as sampler:
        serial = [run_job(job) for job in jobs]
    serial_memory = sampler.summary()
    # Conservative floor plus 50% over observed interpreter/model peak.
    budget_mib = max(768, math.ceil(1.5 * max(serial_memory["max_parent_rss"],
                                             serial_memory["max_parent_private"]) / MIB))
    host_precondition(budget_mib)
    def available():
        s = memory_snapshot()
        return min(s["physical_available"], s["commit_available"])
    logs = []
    def log(message):
        logs.append(message)
        print(message, flush=True)
    with MemorySampler() as sampler:
        pooled = run_pool(run_job, jobs, PoolBudget(budget_mib * MIB, hard_cap=2),
                          free_probe=available, log=log)
    pooled_memory = sampler.summary()
    identical = [canonical(scientific(a)) == canonical(scientific(b)) for a, b in zip(serial, pooled)]
    memory_ok = (not serial_memory["errors"] and not pooled_memory["errors"]
                 and pooled_memory["min_physical_available"] > RESERVE
                 and pooled_memory["min_commit_available"] > RESERVE
                 and pooled_memory["pagefile_growth"] <= 0
                 and max(pooled_memory["max_worker_rss"], pooled_memory["max_worker_private"]) <= budget_mib * MIB
                 and any("(2 running)" in line for line in logs))
    gate = {"git_commit": git_commit(), "implementation": IMPLEMENTATION,
            "protocol_sha256": fingerprint(protocol(load_config(config))),
            "jobs": jobs, "serial": list(map(scientific, serial)), "pooled": list(map(scientific, pooled)),
            "bitwise_identical_cells": sum(identical), "expected_cells": len(jobs),
            "serial_memory": serial_memory, "pooled_memory": pooled_memory,
            "budget_mib": budget_mib, "hard_cap": 2, "memory_passes": memory_ok,
            "gate": "PASS" if all(identical) and memory_ok else "FAIL", "finished_utc": now()}
    atomic_json(ROOT / "gate_memory_samples.json", sampler.samples)
    atomic_json(ROOT / "gate.json", gate)
    print(f"GATE {gate['gate']}: {sum(identical)}/{len(jobs)} bitwise; budget {budget_mib} MiB", flush=True)
    if gate["gate"] != "PASS":
        raise RuntimeError("SO1 fast-family equivalence/memory gate failed")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--reuse-gate", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    require_clean_code(Path("reports/so1_budget_bracket.json"))
    with writer_lock(ROOT / "launcher.lock"):
        try:
            existing_path = Path("reports/so1_budget_bracket.json")
            if existing_path.exists():
                existing = json.loads(existing_path.read_text())
                if existing.get("complete") or existing.get("classification") == "ANCHOR_FAILED_NOTHING_READ":
                    # Do not overwrite the original gate/exit of a terminal run.
                    return subprocess.run([sys.executable, "-u", "-m", "row.experiments.score_so1_budget_bracket"]).returncode
            if args.reuse_gate:
                gate = json.loads((ROOT / "gate.json").read_text())
                if (gate["git_commit"] != git_commit() or gate["gate"] != "PASS"
                        or gate["protocol_sha256"] != fingerprint(protocol(load_config(args.config)))):
                    raise ValueError("stale or failed gate")
            else:
                gate = calibrate(args.config)
            if args.calibrate_only:
                return 0
            host = host_precondition(gate["budget_mib"])
            atomic_json(ROOT / "launch.json", {"git_commit": git_commit(), "started_utc": now(),
                                               "host": host, "gate": gate["gate"]})
            with MemorySampler() as sampler:
                process = subprocess.run([
                    sys.executable, "-u", "-m", "row.experiments.audit_so1_budget_bracket",
                    "--config", args.config, "--hard-cap", "2", "--measured-rss-mib", str(gate["budget_mib"]),
                ])
            atomic_json(ROOT / "run_memory.json", {"summary": sampler.summary(), "samples": sampler.samples})
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": process.returncode,
                                             "finished_utc": now()})
            print(f"SO1_EXIT={process.returncode}", flush=True)
            report_path = Path("reports/so1_budget_bracket.json")
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get("complete") or report.get("classification") == "ANCHOR_FAILED_NOTHING_READ":
                    checked = subprocess.run([sys.executable, "-u", "-m", "row.experiments.score_so1_budget_bracket"])
                    atomic_json(ROOT / "scorer_exit.json", {"exit_code": checked.returncode, "finished_utc": now()})
                    if checked.returncode:
                        return checked.returncode
            return process.returncode
        except Exception as error:
            atomic_json(ROOT / "launcher_error.json", {"git_commit": git_commit(), "error": repr(error), "utc": now()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
