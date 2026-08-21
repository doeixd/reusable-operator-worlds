"""H39 sealed confirmation launcher (H39_CONFIRMATION_PLAN.md).

Seeds 700-729, three paired arms, bounded pool of three, one writer per
cell. Refuses to run unless `tools/check_prereg.py` passes. Each cell
carries its complete intervention record; the runner refuses a mismatched
existing artifact. Exits nonzero if any cell fails.
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(700, 730))
COMMON = [
    sys.executable, "-m", "row.experiments.mixed_lifetime",
    "--config", "configs/v5_h72.yaml",
    "--r-meta", "1.0", "--meta-families", "4", "--meta-tasks-per-family", "16",
    "--meta-subspace-rank", "2", "--family-onset", "8", "--operator-slots", "12",
    "--sleeps", "16", "24", "32", "48", "64", "--lifecycle", "--arm", "ordinary",
    "--prospective-steps", "8", "--prospective-inner-steps", "8", "--snapshot-history",
]
ARMS = {
    "ordinary": ["--model", "prospective"],
    "m2k32": ["--model", "pslot", "--pslot-count", "2", "--slot-args", "32"],
    "g2k32": ["--model", "pslot", "--pslot-count", "2", "--slot-args", "32", "--freeze-matrices"],
}
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt")


def run(job: tuple[str, int]) -> tuple[str, int, int]:
    arm, seed = job
    out = ROOT / "artifacts" / "h39_confirmation" / arm / f"world_{seed}" / "lifecycle"
    log = ROOT / "tools" / "h39_confirmation_logs" / f"{arm}_world_{seed}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        code = subprocess.run(
            COMMON + ARMS[arm] + ["--world-seed", str(seed), "--output", str(out)],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
        handle.write(f"exit={code}\n")
    if any(not (out / f).exists() for f in REQUIRED):
        return arm, seed, 1
    return arm, seed, code


def main() -> int:
    prereg = subprocess.run([sys.executable, "tools/check_prereg.py"], cwd=ROOT)
    if prereg.returncode != 0:
        print("prereg check failed; refusing to open sealed seeds")
        return 2
    jobs = [(arm, seed) for seed in SEEDS for arm in ARMS]  # world-major: triples complete together
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, jobs))
    failed = [(a, s) for a, s, c in results if c != 0]
    print(f"{len(results) - len(failed)}/{len(results)} cells OK")
    if failed:
        print(f"FAILED cells: {failed}")
        return 1
    print("H39_CONFIRMATION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
