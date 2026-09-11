"""J0: does gradient route inference degrade with library quality? (CF2)

Frozen in `J0_LIBRARY_QUALITY_CENSUS_PLAN.md` (967ca78) and
`J0_AMENDMENT_1.md` (0e37ec8). Runs the unchanged SO1R instrument
(`audit_so1r_route_only.run_library`, imported, not re-implemented) on all 30
SO1 oracle libraries, then correlates each library's route-inference gap
g_L = median_t log(OPT_t / ENUM_t) with its quality q_L = log(median ORACLE).

Restartable: one durable hashed record per library; relaunch resumes;
timestamped run.log and atomic status.json in the run directory.
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

import numpy as np
from scipy.stats import spearmanr

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1r_route_only import run_library
from row.experiments.so1_storage import atomic_json, digest, environment, fingerprint, log_line, now, writer_lock

PLAN = Path("J0_LIBRARY_QUALITY_CENSUS_PLAN.md")
AMENDMENT = Path("J0_AMENDMENT_1.md")
SO1_REPORT = Path("reports/so1_budget_bracket_r2.json")
SO1R_CELLS = Path("artifacts/so1r_route_only/cells")
OUTPUT = Path("reports/j0_library_quality.json")
ROOT = Path("artifacts/j0_library_quality")
PROTOCOL_ID = "J0-library-quality-v1"
LEVELS = (16384, 32768, 65536, 131072, 262144)
LIBRARIES = tuple(f"O_b{b}_g{g}" for b in (2, 64) for g in LEVELS)  # RANDOM library_index 0-9
WORLDS = (0, 1, 2)
PERMUTATIONS = 10_000
PERM_SEED = 1705
RANDOM_FIELDS = {"random_route", "random"}


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "amendment": AMENDMENT.as_posix(),
            "libraries": list(LIBRARIES), "worlds": list(WORLDS), "permutations": PERMUTATIONS,
            "perm_seed": PERM_SEED, "instrument": "audit_so1r_route_only.run_library (559580d)",
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, AMENDMENT, SO1_REPORT)},
            "environment": environment()}


def summarize(record: dict) -> dict:
    rows = list(record["tasks"].values())
    med = {a: float(np.median([r[a] for r in rows])) for a in ("oracle", "enum", "opt", "random", "k0")}
    values = [r[a] for r in rows for a in ("oracle", "enum", "opt", "random", "k0")]
    return {
        "medians": med,
        "q": math.log(med["oracle"]),
        "g": float(np.median([math.log(r["opt"] / r["enum"]) for r in rows])),
        "oracle_bitwise": all(r["oracle"] == record["so1_final_per_task"][t] for t, r in record["tasks"].items()),
        "finite": all(math.isfinite(v) for v in values),
        "enum_equals_oracle_route": float(np.mean([r["enum_route"] == r["oracle_route"] for r in rows])),
        "opt_equals_oracle_route": float(np.mean([r["opt_route"] == r["oracle_route"] for r in rows])),
        "opt_median_support_drop": float(np.median([(r["opt_support_initial"] - r["opt_support_final"])
                                                    / r["opt_support_initial"] for r in rows])),
    }


def so1r_reproduction(name: str, record: dict) -> bool | None:
    """Bitwise per-task agreement with an existing SO1R record, RANDOM excluded (Amendment 1)."""
    path = SO1R_CELLS / name / "result.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())["record"]["tasks"]
    strip = lambda rows: {t: {k: v for k, v in r.items() if k not in RANDOM_FIELDS} for t, r in rows.items()}
    return strip(old) == strip(record["tasks"])


def statistics(cells: dict) -> dict:
    names = [f"{key}_w{w}" for key in LIBRARIES for w in WORLDS]
    q = np.array([cells[n]["q"] for n in names])
    g = np.array([cells[n]["g"] for n in names])
    rho = float(spearmanr(q, g).statistic)
    rng = np.random.default_rng(np.random.SeedSequence([PERM_SEED]))
    count = sum(float(spearmanr(q, rng.permutation(g)).statistic) >= rho for _ in range(PERMUTATIONS))
    within = {}
    for w in WORLDS:
        idx = [i for i, n in enumerate(names) if n.endswith(f"_w{w}")]
        within[str(w)] = float(spearmanr(q[idx], g[idx]).statistic)
    return {"pooled_rho": rho, "permutation_p": (count + 1) / (PERMUTATIONS + 1), "within_world_rho": within}


def classify(cells: dict, stats: dict, harness_ok: bool) -> str:
    random_below = sum(c["medians"]["random"] < c["medians"]["enum"] for c in cells.values())
    if not harness_ok or len(cells) != 30 or random_below > 3:
        return "HARNESS_FAILED"
    rho = stats["pooled_rho"]
    if rho >= 0.5 and stats["permutation_p"] < 0.05 and sum(r > 0 for r in stats["within_world_rho"].values()) >= 2:
        return "CF2_SUPPORTED"
    if rho <= -0.3:
        return "CF2_REVERSED"
    if abs(rho) < 0.3:
        return "CF2_FLAT"
    return "INCONCLUSIVE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="one library, 2 tasks, 20 OPT steps")
    args = parser.parse_args()
    if args.dry_run:
        started = time.perf_counter()
        record = run_library(LIBRARIES[0], 2, 0, tasks_limit=2, opt_steps=20)
        record["so1_final_per_task"] = {t: record["so1_final_per_task"][t] for t in record["tasks"]}
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
                log_line(run_log, "report already complete")
                return 0
            log_line(run_log, f"RESUME at {git_commit()}")
        else:
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": git_commit(), "protocol": expected,
                   "protocol_sha256": sha, "started_utc": now(), "cells": {}, "so1r_reproduction": {},
                   "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID} at {git_commit()} pid {os.getpid()}")
        atomic_json(OUTPUT, out)
        jobs = [(key, w, i) for i, key in enumerate(LIBRARIES) for w in WORLDS]
        try:
            for done, (key, w, i) in enumerate(jobs):
                name = f"{key}_w{w}"
                path = ROOT / "cells" / name / "result.json"
                stamp = {"key": key, "world": w, "library_index": i, "git_commit": git_commit(),
                         "protocol_sha256": sha}
                seconds = [c.get("seconds") for c in out["cells"].values() if c.get("seconds")]
                eta = (sum(seconds) / len(seconds)) * (len(jobs) - done) / 3600 if seconds else None
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": git_commit(),
                                          "cells_done": done, "cells_total": len(jobs), "current": name,
                                          "eta_hours": None if eta is None else round(eta, 2),
                                          "updated_utc": now()})
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["record_sha256"] != fingerprint(stored["record"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    record = stored["record"]
                    log_line(run_log, f"[{name}] reused validated durable cell")
                else:
                    started = time.perf_counter()
                    log_line(run_log, f"[{name}] start")
                    record = run_library(key, w, i)
                    record["seconds"] = round(time.perf_counter() - started, 1)
                    atomic_json(path, {"stamp": stamp, "record": record, "record_sha256": fingerprint(record),
                                       "finished_utc": now()})
                summary = summarize(record) | {"seconds": record.get("seconds")}
                out["cells"][name] = summary
                reproduced = so1r_reproduction(name, record)
                if reproduced is not None:
                    out["so1r_reproduction"][name] = reproduced
                atomic_json(OUTPUT, out)
                m = summary["medians"]
                log_line(run_log, f"[{name}] saved: q {summary['q']:.3f} g {summary['g']:.3f} oracle {m['oracle']:.4f} "
                                  f"enum {m['enum']:.4f} opt {m['opt']:.4f} random {m['random']:.4f}"
                                  f"{'' if reproduced is None else f' so1r_bitwise {reproduced}'} ({record.get('seconds')}s)")
            harness_ok = (all(c["oracle_bitwise"] and c["finite"] for c in out["cells"].values())
                          and len(out["so1r_reproduction"]) == 6 and all(out["so1r_reproduction"].values()))
            out["statistics"] = statistics(out["cells"])
            out["harness_ok"] = harness_ok
            out["classification"] = classify(out["cells"], out["statistics"], harness_ok)
            out["complete"] = True
            out["finished_utc"] = now()
            atomic_json(OUTPUT, out)
            atomic_json(status_path, {"state": "complete", "classification": out["classification"],
                                      "cells_done": len(jobs), "cells_total": len(jobs), "updated_utc": now()})
            log_line(run_log, f"COMPLETE classification {out['classification']} stats {out['statistics']}")
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
