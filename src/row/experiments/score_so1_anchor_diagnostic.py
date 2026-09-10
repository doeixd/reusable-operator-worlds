"""Independent recomputation of the SO1 anchor diagnostic from durable cells.

Reads every cell's `result.json` (hash-verified), recomputes the registered
quantities and the ordered classification of
`SO1_ANCHOR_DIAGNOSTIC_AMENDMENT.md` without importing the runner's decision
code, and fails unless the report agrees exactly.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPORT = Path("reports/so1_anchor_diagnostic.json")
STAGE_D = Path("reports/rotated_g5r_interference.json")
ROOT = Path("artifacts/so1_anchor_diagnostic/cells")
KEYS = ("R", "P", "F0", "S100", "F100", "F120", "F121", "F122")
FAST = ("F0", "F100", "F120", "F121", "F122")
TOL, REPRO = 0.02, 1e-6


def canonical_sha(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def durable(key: str, world: int) -> float:
    path = ROOT / key / f"world_{world}"
    record = json.loads((path / "result.json").read_text())
    assert record["complete"] is True, path
    for name, sha in record["artifact_sha256"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == sha, path / name
    assert canonical_sha(record["result"]) == record["result_sha256"], path
    assert record["stamp"]["key"] == key and record["stamp"]["world"] == world
    return record["result"]["terminal_median"]


def classify(rows: list[dict]) -> str:
    if max(r["d_repro"] for r in rows) > REPRO:
        return "HARNESS_FAILED"
    missed = [r for r in rows if max(r["d_impl0"], r["d_impl100"]) > TOL]
    if not missed:
        return "IMPLEMENTATION_EQUIVALENT"
    if all(r["d_chaos"] > TOL for r in missed):
        return "TRAJECTORY_SENSITIVE"
    return "IMPLEMENTATION_DIVERGES"


def main() -> int:
    report = json.loads(REPORT.read_text())
    stage_d = json.loads(STAGE_D.read_text())
    rows, problems = [], []
    for w in (0, 1, 2):
        m = {k: durable(k, w) for k in KEYS}
        sd = stage_d["cells"]["C_lo"][str(w)]["terminal_median"]
        fast = [m[k] for k in FAST]
        row = {"d_repro": abs(m["R"] - sd), "d_impl0": abs(m["F0"] - sd),
               "d_impl100": abs(m["F100"] - m["S100"]), "d_chaos": abs(m["P"] - m["R"]),
               "spread": max(fast) - min(fast), "fast_sd": statistics.stdev(fast),
               "miss": abs(m["F100"] - sd)}
        rows.append(row)
        mine = report["per_world"][str(w)]
        for k, v in row.items():
            if abs(mine[k] - v) > 1e-12:
                problems.append(f"world {w} {k}: report {mine[k]} vs recomputed {v}")
    label = classify(rows)
    if report.get("complete") is not True:
        problems.append("report not complete")
    if report.get("classification") != label:
        problems.append(f"classification: report {report.get('classification')} vs {label}")
    print(json.dumps({"classification": label, "per_world": rows, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
