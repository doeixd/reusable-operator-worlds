"""Independent J0 recomputation from durable per-library records.

Recomputes q_L, g_L, the harness checks (ORACLE bitwise vs SO1; SO1R bitwise
excluding RANDOM per Amendment 1), a numpy-only Spearman (average ranks), the
seeded permutation p-value and the ordered classification, without the
runner's decision functions or SciPy. Fails unless the report agrees.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPORT = Path("reports/j0_library_quality.json")
SO1 = Path("reports/so1_budget_bracket_r2.json")
CELLS = Path("artifacts/j0_library_quality/cells")
SO1R = Path("artifacts/so1r_route_only/cells")
LEVELS = (16384, 32768, 65536, 131072, 262144)
LIBRARIES = [f"O_b{b}_g{g}" for b in (2, 64) for g in LEVELS]


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tie = x == value
        r[tie] = r[tie].mean()
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / math.sqrt((ra @ ra) * (rb @ rb)))


def main() -> int:
    report = json.loads(REPORT.read_text())
    so1 = json.loads(SO1.read_text())["cells"]
    problems, q, g, names, harness, random_below = [], [], [], [], True, 0
    for key in LIBRARIES:
        for w in (0, 1, 2):
            name = f"{key}_w{w}"
            stored = json.loads((CELLS / name / "result.json").read_text())
            rec = stored["record"]
            if sha(rec) != stored["record_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{name}: record hash/commit mismatch")
            rows = rec["tasks"]
            if set(rows) != set(so1[key][str(w)]["final_per_task"]):
                problems.append(f"{name}: task set incomplete")
            harness &= all(r["oracle"] == so1[key][str(w)]["final_per_task"][t] for t, r in rows.items())
            harness &= all(math.isfinite(r[a]) for r in rows.values() for a in ("oracle", "enum", "opt", "random", "k0"))
            old = SO1R / name / "result.json"
            if old.exists():
                strip = lambda t: {k: {f: v for f, v in r.items() if f not in ("random_route", "random")} for k, r in t.items()}
                harness &= strip(json.loads(old.read_text())["record"]["tasks"]) == strip(rows)
            med = {a: float(np.median([r[a] for r in rows.values()])) for a in ("oracle", "enum", "random")}
            random_below += med["random"] < med["enum"]
            names.append(name)
            q.append(math.log(med["oracle"]))
            g.append(float(np.median([math.log(r["opt"] / r["enum"]) for r in rows.values()])))
            mine = report["cells"][name]
            if abs(mine["q"] - q[-1]) > 1e-12 or abs(mine["g"] - g[-1]) > 1e-12:
                problems.append(f"{name}: q/g disagree")
    rho = spearman(q, g)
    rng = np.random.default_rng(np.random.SeedSequence([1705]))
    count = sum(spearman(q, rng.permutation(np.asarray(g))) >= rho - 1e-12 for _ in range(10_000))
    p = (count + 1) / 10_001
    within = {str(w): spearman([q[i] for i, n in enumerate(names) if n.endswith(f"_w{w}")],
                               [g[i] for i, n in enumerate(names) if n.endswith(f"_w{w}")]) for w in (0, 1, 2)}
    if not harness or len(names) != 30 or random_below > 3:
        label = "HARNESS_FAILED"
    elif rho >= 0.5 and p < 0.05 and sum(v > 0 for v in within.values()) >= 2:
        label = "CF2_SUPPORTED"
    elif rho <= -0.3:
        label = "CF2_REVERSED"
    elif abs(rho) < 0.3:
        label = "CF2_FLAT"
    else:
        label = "INCONCLUSIVE"
    s = report["statistics"]
    if abs(s["pooled_rho"] - rho) > 1e-9 or abs(s["permutation_p"] - p) > 2 / 10_001:
        problems.append(f"statistics disagree: report {s['pooled_rho']}/{s['permutation_p']} vs {rho}/{p}")
    if report.get("classification") != label or report.get("complete") is not True:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "pooled_rho": rho, "permutation_p": p,
                      "within_world_rho": within, "harness_ok": harness, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
