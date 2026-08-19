"""Detachable driver for the V3 promotion-testbed validity gate.

Runs the frozen shared-residual learner on task-group worlds across an eta
grid so P-2026-08-18-D can be scored on a TRAINED learner's residuals. eta
is tunable only until the gate passes, and every value tried is logged here
and in the artifact tree (V3 spec 2.6). Resumable; logs to
tools/v3_taskgroup.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tools" / "v3_taskgroup.log"
# PROMOTE runs after enough post-onset tasks have accumulated.
SLEEPS = (24, 32, 48, 64)


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def run_cell(args) -> str:
    world, eta, model, groups, uniform_rho, slots, onset, freeze, freeze_slots, newprim = args
    label = f"eta{eta:g}" if groups == 2 else f"eta{eta:g}_g{groups}"
    if uniform_rho is not None:
        label = f"{label}_rho{uniform_rho:g}"
    if slots is not None:
        label = f"{label}_k{slots}"
    if onset:
        label = f"{label}_onset{onset}"
    if freeze is not None:
        label = f"{label}_frozen{freeze}"
    if freeze_slots is not None:
        label = f"{label}_fs{freeze_slots}"
    if newprim:
        label = f"{label}_newprim"
    out = ROOT / "artifacts" / "v3_taskgroup" / label / f"world_{world}" / model
    if (out / "summary.json").exists():
        return f"skip {out}"
    result = subprocess.run(
        [sys.executable, "-m", "row.experiments.mixed_lifetime",
         "--config", "configs/v1.yaml", "--model", model,
         "--world-seed", str(world), "--task-group-eta", str(eta),
         "--task-groups", str(groups), "--output", str(out)]
        + (["--sleeps"] + [str(x) for x in SLEEPS] if model == "promoting" else [])
        + (["--profile"] + [str(uniform_rho)] * 6 if uniform_rho is not None else [])
        + (["--operator-slots", str(slots)] if slots is not None else [])
        + (["--family-onset", str(onset)] if onset else [])
        + (["--freeze-basis-at", str(freeze)] if freeze is not None else [])
        + (["--freeze-slots", str(freeze_slots)] if freeze_slots is not None else [])
        + (["--new-primitive-families"] if newprim else []),
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"stderr {out}: {result.stderr[-800:]}")
        return f"FAIL rc={result.returncode} {out}"
    return f"done {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.9])
    parser.add_argument("--model", default="shared_residual")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--uniform-rho", type=float, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--family-onset", type=int, default=0)
    parser.add_argument("--freeze-basis-at", type=int, default=None)
    parser.add_argument("--freeze-slots", type=int, default=None)
    parser.add_argument("--new-primitive-families", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    cells = [
        (
            world,
            eta,
            args.model,
            args.groups,
            args.uniform_rho,
            args.slots,
            args.family_onset,
            args.freeze_basis_at,
            args.freeze_slots,
            args.new_primitive_families,
        )
        for eta in args.etas
        for world in args.worlds
    ]
    log(f"starting: {len(cells)} cells, jobs={args.jobs}, etas={args.etas}")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, cells):
            log(outcome)
    log("V3_TASKGROUP_COMPLETE")


if __name__ == "__main__":
    main()
