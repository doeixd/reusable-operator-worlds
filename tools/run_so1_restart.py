"""Calibrate the fast SO1 family, then launch (or resume) the licensed SO1 relaunch.

Relaunching the same command resumes: validated durable cells are reused, a
current-commit PASS gate is reused, and unfinished cells restart from
initialization. The first attempt's paths (artifacts/so1_restart,
reports/so1_budget_bracket.json) are never written.

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

ROOT = Path("artifacts/so1_restart2")
REPORT = Path("reports/so1_budget_bracket_r2.json")
MIB = 2**20
# Operational memory settings (PI-authorized 2026-09-10 to be lowered on this
# host); recorded in every gate/launch/precondition record. They cannot change
# any number: the bitwise serial-vs-pooled gate must still pass.
OPS = {"reserve_bytes": 4 * 2**30, "budget_factor": 1.5, "pagefile_tolerance_bytes": 0}


SCORER = [sys.executable, "-u", "-m", "row.experiments.score_so1_budget_bracket",
          "--report", str(REPORT), "--exit-record", str(ROOT / "exit.json"),
          "--output", str(ROOT / "independent_score.json")]


def scientific(result):
    return {k: v for k, v in result.items() if k not in {"seconds", "rss_bytes"}}


def host_precondition(budget_mib):
    samples = [memory_snapshot()]
    for _ in range(6):
        time.sleep(10)
        samples.append(memory_snapshot())
    need = OPS["reserve_bytes"] + 2 * budget_mib * MIB
    tenants = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info["memory_info"]
            if info and info.rss > 200 * MIB:
                tenants.append({"pid": proc.pid, "name": proc.info["name"],
                                "rss": info.rss, "private": info.private})
        except psutil.NoSuchProcess:
            pass
    record = {"utc": now(), "budget_mib": budget_mib, "ops": dict(OPS), "samples": samples, "tenants": tenants}
    record["passes"] = (min(min(s["physical_available"], s["commit_available"]) for s in samples) > need
                         and max(s["pagefile_used"] for s in samples) <= samples[0]["pagefile_used"] + OPS["pagefile_tolerance_bytes"])
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
    budget_mib = max(768, math.ceil(OPS["budget_factor"] * max(serial_memory["max_parent_rss"],
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
        pooled = run_pool(run_job, jobs, PoolBudget(budget_mib * MIB, hard_cap=2, reserve_bytes=OPS["reserve_bytes"]),
                          free_probe=available, log=log)
    pooled_memory = sampler.summary()
    identical = [canonical(scientific(a)) == canonical(scientific(b)) for a, b in zip(serial, pooled)]
    memory_ok = (not serial_memory["errors"] and not pooled_memory["errors"]
                 and pooled_memory["min_physical_available"] > OPS["reserve_bytes"]
                 and pooled_memory["min_commit_available"] > OPS["reserve_bytes"]
                 and pooled_memory["pagefile_growth"] <= OPS["pagefile_tolerance_bytes"]
                 and max(pooled_memory["max_worker_rss"], pooled_memory["max_worker_private"]) <= budget_mib * MIB
                 and any("(2 running)" in line for line in logs))
    gate = {"git_commit": git_commit(), "implementation": IMPLEMENTATION,
            "protocol_sha256": fingerprint(protocol(load_config(config))),
            "jobs": jobs, "serial": list(map(scientific, serial)), "pooled": list(map(scientific, pooled)),
            "bitwise_identical_cells": sum(identical), "expected_cells": len(jobs),
            "serial_memory": serial_memory, "pooled_memory": pooled_memory,
            "budget_mib": budget_mib, "hard_cap": 2, "ops": dict(OPS), "memory_passes": memory_ok,
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
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--budget-factor", type=float, default=1.5)
    parser.add_argument("--pagefile-tolerance-mib", type=int, default=0)
    args = parser.parse_args()
    OPS.update(reserve_bytes=args.reserve_mib * MIB, budget_factor=args.budget_factor,
               pagefile_tolerance_bytes=args.pagefile_tolerance_mib * MIB)
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    require_clean_code(REPORT)
    with writer_lock(ROOT / "launcher.lock"):
        try:
            if REPORT.exists() and json.loads(REPORT.read_text()).get("complete"):
                # Do not overwrite the original gate/exit of a terminal run.
                return subprocess.run(SCORER).returncode
            gate_path = ROOT / "gate.json"
            if gate_path.exists() and not args.reuse_gate:
                old = json.loads(gate_path.read_text())
                args.reuse_gate = (old.get("gate") == "PASS" and old.get("git_commit") == git_commit()
                                   and old.get("protocol_sha256") == fingerprint(protocol(load_config(args.config))))
                print(f"existing gate {'reused' if args.reuse_gate else 'stale; recalibrating'}", flush=True)
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
            # One record per launch, so a resumed run keeps every launch's host evidence.
            atomic_json(ROOT / f"launch_{time.time_ns()}.json", {"git_commit": git_commit(), "started_utc": now(),
                                                                "host": host, "gate": gate["gate"]})
            with MemorySampler() as sampler:
                process = subprocess.run([
                    sys.executable, "-u", "-m", "row.experiments.audit_so1_budget_bracket",
                    "--config", args.config, "--hard-cap", "2", "--measured-rss-mib", str(gate["budget_mib"]),
                    "--reserve-mib", str(args.reserve_mib),
                ])
            atomic_json(ROOT / "run_memory.json", {"summary": sampler.summary(), "samples": sampler.samples})
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": process.returncode,
                                             "finished_utc": now()})
            print(f"SO1_EXIT={process.returncode}", flush=True)
            if REPORT.exists():
                report = json.loads(REPORT.read_text())
                if report.get("complete"):
                    checked = subprocess.run(SCORER)
                    atomic_json(ROOT / "scorer_exit.json", {"exit_code": checked.returncode, "finished_utc": now()})
                    if checked.returncode:
                        return checked.returncode
            return process.returncode
        except Exception as error:
            atomic_json(ROOT / "launcher_error.json", {"git_commit": git_commit(), "error": repr(error), "utc": now()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
