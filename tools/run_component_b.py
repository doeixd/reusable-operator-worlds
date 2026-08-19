"""Standalone, detachable driver for V2 sealed Component B.

Canonical mixed-profile Benchmark D on sealed seeds 200-229: shared
residual, continuous, and dense, 90 lifetimes in a 4-process pool,
resumable, then done. Triggered by the development H9a verdict per the
frozen decision rule in V2_CONFIRMATION_PLAN.md. Logs to
tools/component_b.log.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tools" / "component_b.log"


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def run_cell(args: tuple[int, str]) -> str:
    world, model = args
    out = ROOT / "artifacts" / "v2_mixed_confirmatory" / f"world_{world}" / model
    if (out / "summary.json").exists():
        return f"skip {out}"
    result = subprocess.run(
        [sys.executable, "-m", "row.experiments.mixed_lifetime",
         "--config", "configs/v1.yaml", "--model", model,
         "--world-seed", str(world), "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    status = "done" if result.returncode == 0 else f"FAIL rc={result.returncode}"
    return f"{status} {out}"


def main() -> None:
    cells = [
        (world, model)
        for world in range(200, 230)
        for model in ("shared_residual", "continuous", "dense")
    ]
    pending = [
        c for c in cells
        if not (ROOT / "artifacts" / "v2_mixed_confirmatory" / f"world_{c[0]}"
                / c[1] / "summary.json").exists()
    ]
    log(f"starting: {len(pending)} pending of {len(cells)}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        for outcome in pool.map(run_cell, pending):
            log(outcome)
    log("COMPONENT_B_COMPLETE")


if __name__ == "__main__":
    main()
