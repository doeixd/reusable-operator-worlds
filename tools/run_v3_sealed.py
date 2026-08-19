"""V3 sealed block: PROMOTE on seeds 300-329.

Executes exactly the configuration frozen in V3_CONFIRMATION_PLAN.md
(commit bcc8319), which was registered before these seeds were generated.
Structured arm supplies O1-O4 (promoting against the identical unpromoted
learner); control arm supplies O5. Resumable via existing summary.json;
logs to tools/v3_sealed.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tools" / "v3_sealed.log"
SLEEPS = ("24", "32", "48", "64")


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def run_cell(args: tuple[int, str, float]) -> str:
    world, model, eta = args
    condition = "structured" if eta > 0 else "control"
    out = ROOT / "artifacts" / "v3_sealed" / condition / f"world_{world}" / model
    if (out / "summary.json").exists():
        return f"skip {out}"
    command = [
        sys.executable, "-m", "row.experiments.mixed_lifetime",
        "--config", "configs/v1.yaml", "--model", model,
        "--world-seed", str(world),
        "--task-group-eta", str(eta), "--task-groups", "2",
        "--operator-slots", "6",
        "--family-onset", "16", "--freeze-basis-at", "16",
        "--output", str(out),
    ]
    if eta > 0:
        command.append("--new-primitive-families")
    if model == "promoting":
        command += ["--sleeps", *SLEEPS]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"stderr {out}: {result.stderr[-600:]}")
        return f"FAIL rc={result.returncode} {out}"
    return f"done {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(300, 330)))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    cells = []
    for world in args.worlds:
        cells.append((world, "promoting", 0.9))
        cells.append((world, "shared_residual", 0.9))
        cells.append((world, "promoting", 0.0))
    log(f"SEALED BLOCK starting: {len(cells)} cells, jobs={args.jobs}")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, cells):
            log(outcome)
    log("V3_SEALED_COMPLETE")


if __name__ == "__main__":
    main()
