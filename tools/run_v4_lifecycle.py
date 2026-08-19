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
# §2.2: family goes quiet at 32. Late enough for the abstraction to be
# established and reused, early enough that the remaining 32 tasks make
# obsolescence measurable in the final description.
DORMANCY = (32, 64)


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def run_cell(args: tuple[int, float, bool, str | None]) -> str:
    world, eta, lifecycle, dormancy = args
    condition = "structured" if eta > 0 else "control"
    if dormancy is not None:
        # The V4.1 DELETE opportunity is OBSOLESCENCE, which the plain
        # structured world does not contain -- its families stay live to
        # the end, so the exact oracle correctly found no room for timed
        # deletion there. §2.2's permanent arm is the world where an
        # abstraction stops being worth its bits; the returning arm is
        # its byte-identical refusal control.
        condition = f"dormancy_{dormancy}"
    arm = "lifecycle_on" if lifecycle else "lifecycle"
    out = ROOT / "artifacts" / "v4_dev" / condition / f"world_{world}" / arm
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
    if lifecycle:
        command.append("--lifecycle")
    if dormancy is not None:
        command += ["--dormancy", str(DORMANCY[0]), str(DORMANCY[1])]
        if dormancy == "permanent":
            command.append("--dormancy-permanent")
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
    parser.add_argument("--arms", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--dormancy",
        nargs="*",
        default=[],
        choices=["returns", "permanent"],
        help="run the §2.2 dormancy arms instead of the plain worlds",
    )
    args = parser.parse_args()

    variants = args.dormancy or [None]
    cells = [
        (w, eta, on, d)
        for d in variants
        for on in args.arms
        for eta in args.etas
        for w in args.worlds
    ]
    log(f"V4.1 starting: {len(cells)} cells, jobs={args.jobs}")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, cells):
            log(outcome)
    log("V4_LIFECYCLE_COMPLETE")


if __name__ == "__main__":
    main()
