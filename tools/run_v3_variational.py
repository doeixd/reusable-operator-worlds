"""Standalone, detachable driver for V3 checklist item 1.

Scores prediction P-2026-08-18-A: the variational wake learner against the
frozen fixed architectures on canonical mixed worlds. Development worlds
0-2 first (stage one), with an optional beta grid. Resumable via existing
summary.json, logs to tools/v3_variational.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tools" / "v3_variational.log"


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def _config_for_beta(beta: float) -> Path:
    """Write a config whose only change from canonical is description_beta."""

    if beta == 1.0:
        return ROOT / "configs" / "v1.yaml"
    raw = yaml.safe_load((ROOT / "configs" / "v1.yaml").read_text(encoding="utf-8"))
    raw["variational_model"]["description_beta"] = beta
    path = ROOT / "configs" / f"v3_variational_beta{beta:g}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def run_cell(args: tuple[int, float]) -> str:
    world, beta = args
    label = "canonical" if beta == 1.0 else f"beta{beta:g}"
    out = ROOT / "artifacts" / "v3_variational" / label / f"world_{world}" / "variational"
    if (out / "summary.json").exists():
        return f"skip {out}"
    config = _config_for_beta(beta)
    result = subprocess.run(
        [sys.executable, "-m", "row.experiments.mixed_lifetime",
         "--config", str(config), "--model", "variational",
         "--world-seed", str(world), "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    status = "done" if result.returncode == 0 else f"FAIL rc={result.returncode}"
    if result.returncode != 0:
        log(f"stderr {out}: {result.stderr[-800:]}")
    return f"{status} {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--betas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()

    cells = [(world, beta) for beta in args.betas for world in args.worlds]
    log(f"starting: {len(cells)} cells, jobs={args.jobs}")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, cells):
            log(outcome)
    log("V3_VARIATIONAL_COMPLETE")


if __name__ == "__main__":
    main()
