"""Standalone, detachable driver for V2 sealed Component A.

Runs the remaining seeds-200-229 sweep cells in a 4-process pool,
resumable (skips cells with summary.json), then executes the sweep_rho
validation/assembly pass. Designed to be started detached (Start-Process)
so interactive-session teardown cannot kill it. Writes progress to
tools/component_a.log.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tools" / "component_a.log"
LABELS = {
    "0.0": "rho_0", "0.25": "rho_0p25", "0.5": "rho_0p5",
    "0.75": "rho_0p75", "0.9": "rho_0p9", "1.0": "rho_1",
}


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def run_cell(args: tuple[int, str, str]) -> str:
    world, rho, model = args
    out = ROOT / "artifacts" / "rho_confirmatory_v2" / LABELS[rho] / f"world_{world}" / model
    if (out / "summary.json").exists():
        return f"skip {out}"
    result = subprocess.run(
        [sys.executable, "-m", "row.experiments.learned_lifetime",
         "--config", "configs/v1.yaml", "--model", model,
         "--world-seed", str(world), "--reuse-rho", rho,
         "--fast-tuning", "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    status = "done" if result.returncode == 0 else f"FAIL rc={result.returncode}"
    return f"{status} {out}"


def main() -> None:
    cells = [
        (world, rho, model)
        for world in range(200, 230)
        for rho in LABELS
        for model in ("continuous", "dense")
    ]
    pending = [
        c for c in cells
        if not (ROOT / "artifacts" / "rho_confirmatory_v2" / LABELS[c[1]]
                / f"world_{c[0]}" / c[2] / "summary.json").exists()
    ]
    log(f"starting: {len(pending)} pending of {len(cells)}")
    with ProcessPoolExecutor(max_workers=4) as pool:
        for outcome in pool.map(run_cell, pending):
            log(outcome)
    log("prepopulate done; running sweep assembly")
    result = subprocess.run(
        [sys.executable, "-m", "row.experiments.sweep_rho",
         "--config", "configs/v1.yaml",
         "--output", "artifacts/rho_confirmatory_v2",
         "--worlds", *[str(w) for w in range(200, 230)]],
        cwd=ROOT, capture_output=True, text=True,
    )
    log(f"assembly rc={result.returncode}")
    log("COMPONENT_A_COMPLETE" if result.returncode == 0 else "ASSEMBLY_FAILED")


if __name__ == "__main__":
    main()
