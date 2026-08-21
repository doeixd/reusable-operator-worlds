"""H39 world-0 pilot launcher (H39_PILOT_PLAN.md, Amendment 1).

Three cells, one writer each, bounded pool of three. Every cell carries
its complete intervention record; a mismatched existing artifact is
refused by the runner. Exits nonzero if any cell fails.
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = [
    sys.executable, "-m", "row.experiments.mixed_lifetime",
    "--config", "configs/v5_h72.yaml", "--world-seed", "0",
    "--r-meta", "1.0", "--meta-families", "4", "--meta-tasks-per-family", "16",
    "--meta-subspace-rank", "2", "--family-onset", "8", "--operator-slots", "12",
    "--sleeps", "16", "24", "32", "48", "64", "--lifecycle", "--arm", "ordinary",
    "--prospective-steps", "8", "--prospective-inner-steps", "8",
]
CELLS = {
    "ordinary_history": ["--model", "prospective", "--snapshot-history"],
    "factorized_grouped": ["--model", "factorized", "--schema-grouping", "oracle",
                           "--schema-dim", "2", "--snapshot-history"],
    "factorized_pooled": ["--model", "factorized", "--schema-grouping", "pooled",
                          "--schema-dim", "8", "--snapshot-history"],
}
REQUIRED = ("model.pt", "summary.json", "rho_profile.json", "fingerprint.json",
            "config.yaml", "history.pt")


def run(name: str) -> tuple[str, int]:
    out = ROOT / "artifacts" / "h39_pilot" / name / "world_0" / "lifecycle"
    log = ROOT / "tools" / f"h39_pilot_{name}.log"
    with log.open("a", encoding="utf-8") as handle:
        code = subprocess.run(
            COMMON + CELLS[name] + ["--output", str(out)],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
        handle.write(f"exit={code}\n")
    missing = [f for f in REQUIRED if not (out / f).exists()]
    if missing:
        return name, 1
    return name, code


def main() -> int:
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, list(CELLS)))
    failed = [n for n, c in results if c != 0]
    for name, code in results:
        print(f"{name}: exit={code}")
    if failed:
        print(f"FAILED cells: {failed}")
        return 1
    print("H39_PILOT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
