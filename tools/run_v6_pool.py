"""V6 sweeps through a bounded process pool.

Memory, not cores, is the binding constraint on this host: a lifetime
pins one thread, so serial execution wasted about 94% of the machine,
while five concurrent `slots=12` promoting runs exhausted RAM and killed
113 of 120 cells. The cap here is 4, and it is a MEMORY cap — the
promotion clustering step holds the largest tensors, so lighter models
tolerate more.

A pool also fixes the other failure this project hit: two shell-loop
launcher instances over the same output paths raced and logged
completions for cells that had no artifacts. One pool over one job list
gives exactly one writer per cell by construction.

Resumable: a cell with `summary.json` is skipped, so this can take over
from a partially finished serial run without repeating work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(__file__).with_name("v6_pool.log")


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def valid_cell(job: dict) -> bool:
    out = ROOT / job["out"]
    required = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json")
    if any(not (out / name).exists() for name in required):
        return False
    try:
        protocol = json.loads(
            (out / "rho_profile.json").read_text(encoding="utf-8")
        )["v6_arm"]
        fingerprint = json.loads(
            (out / "fingerprint.json").read_text(encoding="utf-8")
        )
    except (KeyError, json.JSONDecodeError, OSError):
        return False
    return (
        int(fingerprint.get("world_seed", -1)) == job["world"]
        and protocol.get("arm") == job["arm"]
        and protocol.get("freeze_basis_at")
        == (8 if job.get("frozen") is not None else None)
        and protocol.get("freeze_slots") == job.get("frozen")
        and int(protocol.get("prospective_steps", -1)) == job["outer"]
        and int(protocol.get("prospective_inner_steps", -1)) == job["inner"]
        and protocol.get("operator_slots") == 12
        and protocol.get("sleeps") == [16, 24, 32, 48, 64]
        and protocol.get("lifecycle") is True
    )


def run_cell(job: dict) -> str:
    out = ROOT / job["out"]
    if valid_cell(job):
        return f"skip {job['out']}"
    if out.exists() and any(out.iterdir()):
        return f"MISMATCH {job['out']}"
    command = [
        sys.executable, "-m", "row.experiments.mixed_lifetime",
        "--config", "configs/v5_h72.yaml", "--model", "prospective",
        "--world-seed", str(job["world"]),
        "--r-meta", "1.0", "--meta-families", "4",
        "--meta-tasks-per-family", "16", "--meta-subspace-rank", "2",
        "--family-onset", "8", "--operator-slots", "12",
        "--sleeps", "16", "24", "32", "48", "64", "--lifecycle",
        "--arm", job["arm"],
        "--prospective-steps", str(job["outer"]),
        "--prospective-inner-steps", str(job["inner"]),
        "--output", str(out),
    ]
    if job.get("frozen") is not None:
        command += ["--freeze-basis-at", "8",
                    "--freeze-slots", str(job["frozen"])]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"FAIL {job['out']}: {result.stderr[-400:]}")
        return f"FAIL {job['out']}"
    return f"done {job['out']}"


def allocation_jobs() -> list[dict]:
    jobs = []
    for frozen in (12, 11, 10, 9, 6):
        free = 12 - frozen
        for arm in ("ordinary", "replay", "prospective"):
            for world in (0, 1, 2):
                jobs.append({
                    "out": f"artifacts/v6_alloc/free{free}/{arm}/world_{world}/lifecycle",
                    "arm": arm, "world": world, "frozen": frozen, "outer": 16,
                    "inner": 16,
                })
    return jobs


def lowpressure_jobs() -> list[dict]:
    return [
        {"out": f"artifacts/v6_lowp/o{outer}/world_{world}/lifecycle",
         "arm": "prospective", "world": world, "frozen": None, "outer": outer,
         "inner": 8}
        for outer in (1, 2)
        for world in (0, 1, 2)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=("allocation", "lowpressure", "both"),
                        default="both")
    parser.add_argument("--jobs", type=int, default=4,
                        help="MEMORY cap, not a core count: 4 for slots=12 "
                             "promoting runs, which is where RAM ran out at 5")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4 for slots=12 lifetimes")

    jobs = []
    if args.sweep in ("allocation", "both"):
        jobs += allocation_jobs()
    if args.sweep in ("lowpressure", "both"):
        jobs += lowpressure_jobs()

    pending = [j for j in jobs if not valid_cell(j)]
    log(f"START {len(pending)} pending of {len(jobs)} cells, pool={args.jobs}")
    print(f"{len(pending)} pending of {len(jobs)}; pool {args.jobs}")
    done = failed = skipped = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, jobs):
            if outcome.startswith(("FAIL", "MISMATCH")):
                failed += 1
                print(outcome)
            elif outcome.startswith("skip"):
                skipped += 1
            else:
                done += 1
            if (done + failed + skipped) % 5 == 0:
                log(f"progress done={done} failed={failed} skipped={skipped}")
    log(f"END done={done} failed={failed} skipped={skipped}")
    print(f"done={done} failed={failed} skipped={skipped}")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("V6_POOL_DONE\n" if failed == 0 else "V6_POOL_FAILED\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
