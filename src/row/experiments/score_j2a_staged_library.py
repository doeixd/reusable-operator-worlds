"""Independent J2A recomputation from durable per-library records."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from itertools import product
from pathlib import Path

REPORT = Path("reports/j2a_staged_library.json")
CELLS = Path("artifacts/j2a_staged_library/cells")
SOURCES = ("STAGED5000", "STAGED3001", "NONSTAGED3001", "RESET5000")
STAGED = ("STAGED5000", "STAGED3001")
WORLDS = (0, 1, 2)
THRESHOLD, EXPORT_RATIO = 0.05, 4.0


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> int:
    report = json.loads(REPORT.read_text())
    problems, summaries, harness = [], {}, True
    for name in SOURCES:
        for w in WORLDS:
            key = f"{name}_w{w}"
            stored = json.loads((CELLS / key / "result.json").read_text())
            record = stored["record"]
            if sha_json(record) != stored["record_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{key}: record hash/commit mismatch")
            trained = list(record["trained"].values())
            held = list(record["held_out"].values())
            if len(trained) != 64 or len(held) != 64:
                problems.append(f"{key}: expected 64 trained and 64 held-out rows")
            # Held-out programs must be unseen and distinct.
            programs = [tuple(r["program"]) for r in held]
            if len(set(programs)) != len(programs):
                problems.append(f"{key}: duplicate held-out programs")
            if any(p not in list(product(range(6), repeat=3)) for p in programs):
                problems.append(f"{key}: held-out program outside the program space")
            harness &= all(r["as_trained"] == record["recorded_per_task"][t] for t, r in record["trained"].items())
            harness &= all(math.isfinite(v) for r in trained + held for v in r.values() if isinstance(v, float))
            med = lambda values: statistics.median(values)
            summary = {
                "trained_as_trained": med([r["as_trained"] for r in trained]),
                "trained_enum": med([r["enum"] for r in trained]),
                "trained_opt": med([r["opt"] for r in trained]),
                "trained_random": med([r["random"] for r in trained]),
                "held_enum": med([r["enum"] for r in held]),
                "held_random": med([r["random"] for r in held]),
                "held_below": sum(r["enum"] <= THRESHOLD for r in held),
                "g": med([math.log(r["opt"] / r["enum"]) for r in trained]),
            }
            summary["export_ratio"] = summary["held_enum"] / summary["trained_as_trained"]
            summaries[key] = summary
            mine = report["cells"][key]
            if (abs(mine["export_ratio"] - summary["export_ratio"]) > 1e-12
                    or abs(mine["g"] - summary["g"]) > 1e-12
                    or mine["held_out_below_threshold"] != summary["held_below"]
                    or abs(mine["held_out"]["enum"] - summary["held_enum"]) > 1e-12
                    or abs(mine["trained"]["as_trained"] - summary["trained_as_trained"]) > 1e-12):
                problems.append(f"{key}: summary disagrees with recomputation")
    staged = [summaries[f"{n}_w{w}"] for n in STAGED for w in WORLDS]
    if any(s["trained_random"] < s["trained_enum"] for s in staged):
        harness = False
    if not harness or len(summaries) != len(SOURCES) * len(WORLDS):
        label = "HARNESS_FAILED"
    elif sum(s["held_enum"] <= THRESHOLD and s["export_ratio"] <= EXPORT_RATIO for s in staged) >= 5:
        label = "EXPORTS"
    elif sum(s["held_enum"] <= THRESHOLD for s in staged) >= 3:
        label = "EXPORTS_WEAKLY"
    else:
        label = "DOES_NOT_EXPORT"
    if report.get("classification") != label or report.get("complete") is not True:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "harness": harness,
                      "staged_held_out": {f"{n}_w{w}": summaries[f"{n}_w{w}"]["held_enum"] for n in STAGED for w in WORLDS},
                      "staged_export_ratio": {f"{n}_w{w}": summaries[f"{n}_w{w}"]["export_ratio"] for n in STAGED for w in WORLDS},
                      "controls_held_out": {f"{n}_w{w}": summaries[f"{n}_w{w}"]["held_enum"]
                                            for n in SOURCES if n not in STAGED for w in WORLDS},
                      "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
