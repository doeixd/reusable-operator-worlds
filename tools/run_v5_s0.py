"""V5 H19 s-arm (S0, "return-value gain") plus the slots=6 pairing.

S0 is the cut V5.1 was missing. Residual rank moved `D(A)` and `s_bar`
together, which is why the proportional reading failed; `eta` and
`new_primitive_families` (S1/S2) move the learned function as well as
its payoff. The return gain moves ONLY what the family is worth to a
returning task:

    y = f_base(x) + g * A_family(x),   g in {0.5, 1.0, 1.5}

applied strictly after the gap closes at task 40. The promoted
abstraction is born pre-gap and carries `requires_grad=False`, so the
carried object is bit-identical across g and `lambda * D*(A)` is not
merely "constant to 2%" but the SAME NUMBER. Only `s_bar` moves, and the
law then predicts H*(g) = lambda D*(A) / s_bar(g).

STAGE 1 (this driver's default) measures s_bar(g) on a two-point
horizon grid. It deliberately does not try to locate a crossing: §9
requires the H_R grid to bracket the predicted crossing by >= 4
returning tasks on both sides, and the crossing MOVES with g, so the
grid cannot be chosen before s_bar(g) is known. Stage 2 runs the
bracketing grid per gain.

The g = 1.0 cells double as the slots=6 pairing when `--slots 6` is
passed: the scored D-arm ran at slots=12 against a registered constant
of 6, and that pairing is an outstanding repair.

Resumable by design: a cell with `summary.json` is skipped, so an
interrupted batch is relaunched with the same command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(__file__).with_name("v5_s0.log")

# Gap closes at 40, so H_R = N - 40.
GAP = (32, 40)
SLEEPS = ("16", "24", "32")
FAMILY_ONSET = "8"
ETA = "0.9"


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")


def cell_root(gain: float, slots: int, total: int, arm: str,
              rank: int = 2, root: str = "v5_s0") -> Path:
    prefix = "" if rank == 2 else f"r{rank}_"
    tag = f"{prefix}g{int(round(gain * 100)):03d}_s{slots}_N{total}_{arm}"
    return ROOT / "artifacts" / root / tag


def config_for(rank: int, total: int) -> str:
    """Rank 2 is the default config family; 1 and 4 have their own."""

    return (f"configs/v5_h{total}.yaml" if rank == 2
            else f"configs/v5_r{rank}_h{total}.yaml")


def run_cell(job: tuple[float, int, int, str, int, int]) -> str:
    gain, slots, total, arm, world, rank = job
    out = cell_root(gain, slots, total, arm, rank) / f"world_{world}" / "lifecycle"
    if (out / "summary.json").exists():
        return f"skip {out}"
    command = [
        sys.executable, "-m", "row.experiments.mixed_lifetime",
        "--config", config_for(rank, total),
        "--model", "lifecycle",
        "--world-seed", str(world),
        "--task-group-eta", ETA, "--task-groups", "1",
        "--new-primitive-families",
        "--family-onset", FAMILY_ONSET, "--freeze-basis-at", "8",
        "--operator-slots", str(slots),
        "--sleeps", *SLEEPS,
        "--dormancy", str(GAP[0]), str(GAP[1]),
        "--return-gain", str(gain),
        "--lifecycle",
        "--output", str(out),
    ]
    if arm == "deleted":
        # The counterfactual is "this abstraction was not carried",
        # applied at the gap sleep so the arms are byte-identical up to
        # the moment of the intervention.
        command += ["--force-retire-at", "32", "--force-retire-one"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"stderr {out}: {result.stderr[-600:]}")
        return f"FAIL rc={result.returncode} {out}"
    return f"done {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--totals", type=int, nargs="+", default=[56, 72],
                        help="N per cell; H_R = N - 40")
    parser.add_argument("--slots", type=int, nargs="+", default=[12],
                        help="12 pairs with the scored D-arm; 6 is the registered constant")
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(500, 510)))
    parser.add_argument("--rank", type=int, default=2,
                        help="residual rank; selects the config family")
    parser.add_argument("--jobs", type=int, default=5,
                        help="memory, not cores, is the binding constraint here")
    args = parser.parse_args()

    jobs = [
        (gain, slots, total, arm, world, args.rank)
        for gain in args.gains
        for slots in args.slots
        for total in args.totals
        for arm in ("retained", "deleted")
        for world in args.worlds
    ]
    log(f"START {len(jobs)} cells rank={args.rank} gains={args.gains} "
        f"slots={args.slots} totals={args.totals} worlds={args.worlds[0]}.."
        f"{args.worlds[-1]} jobs={args.jobs}")
    print(f"{len(jobs)} cells, {args.jobs} at a time")
    done = failed = skipped = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for outcome in pool.map(run_cell, jobs):
            if outcome.startswith("FAIL"):
                failed += 1
                print(outcome)
            elif outcome.startswith("skip"):
                skipped += 1
            else:
                done += 1
            if (done + failed + skipped) % 10 == 0:
                log(f"progress done={done} failed={failed} skipped={skipped}")
    log(f"END done={done} failed={failed} skipped={skipped}")
    print(f"done={done} failed={failed} skipped={skipped}")


if __name__ == "__main__":
    main()
