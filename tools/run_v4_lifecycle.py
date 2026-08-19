"""Detachable driver for V4.1 development lifetimes.

Runs the lineage-recording `lifecycle` learner on the frozen V3 promotion
testbed (V4 spec 2.1), structured and structureless arms. Resumable via
existing summary.json; logs to tools/v4_lifecycle.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tools" / "v4_lifecycle.log"
SLEEPS = ("24", "32", "48", "64")


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def run_cell(args: tuple[int, float]) -> str:
    world, eta = args
    condition = "structured" if eta > 0 else "control"
    out = ROOT / "artifacts" / "v4_dev" / condition / f"world_{world}" / "lifecycle"
    if (out / "summary.json").exists():
        return f"skip {out}"
    command = [
        sys.executable, "-m", "row.experiments.mixed_lifetime",
        "--config", "configs/v1.yaml", "--model", "lifecycle",
        "--world-seed", str(world),
        "--task-group-eta", str(eta), "--task-groups", "2",
        "--operator-slots", "6",
        "--family-onset", "16", "--freeze-basis-at", "16",
        "--sleeps", *SLEEPS,
        "--output", str(out),
    ]
    if eta > 0:
        command.append("--new-primitive-families")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"stderr {out}: {result.stderr[-600:]}")
        return f"FAIL rc={result.returncode} {out}"
    return f"done {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--etas", type=float, nargs="+", default=[0.9, 0.0])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    cells = [(w, eta) for eta in args.etas for w in args.worlds]
    log(f"V4.1 starting: {len(cells)} cells, jobs={args.jobs}")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, cells):
            log(outcome)
    log("V4_LIFECYCLE_COMPLETE")


if __name__ == "__main__":
    main()
