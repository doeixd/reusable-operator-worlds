"""Independent SO1R recomputation from durable per-library records.

Re-reads every durable record (hash-checked), recomputes medians, eligibility,
non-vacuity and the ordered classification of `SO1R_ROUTE_ONLY_PLAN.md`
without the runner's decision functions, checks ORACLE against the SO1
report's per-task scores, and fails unless the report agrees exactly.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REPORT = Path("reports/so1r_route_only.json")
SO1 = Path("reports/so1_budget_bracket_r2.json")
CELLS = Path("artifacts/so1r_route_only/cells")
LIBRARIES = ("O_b2_g131072", "O_b64_g262144")
ARMS = ("oracle", "enum", "opt", "random", "k0")
THRESHOLD, DROP = 0.05, 0.10


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def main() -> int:
    report = json.loads(REPORT.read_text())
    so1 = json.loads(SO1.read_text())["cells"]
    problems, eligible, vacuous, summaries = [], [], False, {}
    for key in LIBRARIES:
        for w in (0, 1, 2):
            name = f"{key}_w{w}"
            stored = json.loads((CELLS / name / "result.json").read_text())
            record = stored["record"]
            if sha(record) != stored["record_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{name}: durable record hash/commit mismatch")
            rows = record["tasks"]
            if set(rows) != set(so1[key][str(w)]["final_per_task"]) or len(rows) != 64:
                problems.append(f"{name}: task set incomplete")
            med = {a: statistics.median(r[a] for r in rows.values()) for a in ARMS}
            bitwise = all(r["oracle"] == so1[key][str(w)]["final_per_task"][t] for t, r in rows.items())
            drop = statistics.median((r["opt_support_initial"] - r["opt_support_final"]) / r["opt_support_initial"]
                                     for r in rows.values())
            ok = (bitwise and all(r["enum_support_mse"] <= r["oracle_support_mse"] for r in rows.values())
                  and all(r["opt_code_abs_sum"] > 0 for r in rows.values()) and drop >= DROP
                  and med["k0"] != med["opt"])
            vacuous |= not ok
            is_eligible = med["oracle"] <= THRESHOLD
            summaries[name] = (med, is_eligible)
            if is_eligible:
                eligible.append(med)
            mine = report["cells"][name]
            if mine["medians"] != med or mine["eligible"] != is_eligible or mine["oracle_bitwise"] != bitwise:
                problems.append(f"{name}: report summary disagrees with recomputation")
    if vacuous or len(eligible) < 2 or any(m["random"] <= THRESHOLD for m in eligible):
        label = "HARNESS_FAILED"
    elif all(m["enum"] <= THRESHOLD and m["opt"] <= THRESHOLD for m in eligible):
        label = "ROUTES_RECOVERABLE"
    elif all(m["enum"] <= THRESHOLD for m in eligible):
        label = "SEARCH_ONLY"
    else:
        label = "NOT_IDENTIFIABLE"
    if report.get("complete") is not True:
        problems.append("report not complete")
    if report.get("classification") != label:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "eligible_libraries": len(eligible),
                      "medians": {k: v[0] for k, v in summaries.items()}, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
