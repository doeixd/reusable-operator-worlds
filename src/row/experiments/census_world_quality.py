"""World-quality census (Tier 0, descriptive; `WORLD_QUALITY_CENSUS_PLAN.md`).

Does a staged cell's stage-1/stage-2 prefix predict its stage-3 outcome? Reads
ONLY the committed report JSON of the three online runs (SO2, SO3, SO4), so it
is reproducible from the repository with no artifacts and no torch. Trains
nothing, opens no world, produces no verdict.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PLAN = Path("WORLD_QUALITY_CENSUS_PLAN.md")
OUTPUT = Path("reports/world_quality_census.json")
SO2_REPORT = Path("reports/so2_online_gate.json")
SO3_REPORT = Path("reports/so3_consolidation.json")
SO4_REPORT = Path("reports/so4_b2_retest.json")
THRESHOLD = 0.05
PERMUTATIONS = 2000
PERM_SEED = 2029
EXPECTED = {"SO2": 3, "SO3": 9, "SO4": 12}
EXPECTED_LABELS = {"SO2": "SO2_FAILS", "SO3": "SO3_FAILS", "SO4": "SO4_FAILS"}
PREDICTORS = ("stage1_terminal", "stage2_terminal", "stage2_end_of_task", "stage1_to_stage2_ratio")


def _ranks(values) -> np.ndarray:
    """Average ranks, so tied values share a rank.

    `argsort(argsort(x))` invents a strict order for ties and can therefore
    report a perfect correlation for a CONSTANT predictor; that is the
    can't-fail statistic this project forbids.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y) -> float:
    """Spearman correlation with tie-corrected ranks; nan when either side is constant."""
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _stage_pair(stage1: dict, stage2: dict, stage3_terminal: float, run: str, world: int, stream: int) -> dict:
    s1, s2 = float(stage1["terminal_median"]), float(stage2["terminal_median"])
    return {"run": run, "world": world, "stream": stream,
            "stage1_terminal": s1, "stage2_terminal": s2,
            "stage2_end_of_task": float(stage2["end_of_task_median"]),
            "stage1_to_stage2_ratio": s2 / s1 if s1 > 0 else float("nan"),
            "stage3_terminal": float(stage3_terminal),
            "passed": bool(float(stage3_terminal) <= THRESHOLD)}


def collect(so2: dict, so3: dict, so4: dict) -> list[dict]:
    """The 24 staged cells of the unchanged protocol, in run order."""
    rows = []
    for w in (0, 1, 2):
        stages = so2["cells"][f"STAGED_w{w}"]["stages"]
        rows.append(_stage_pair(stages["1"], stages["2"], stages["3"]["terminal_median"], "SO2", w, 0))
    for w in (3, 4, 5):
        for s in (0, 1, 2):
            stages = so3["prefixes"][f"w{w}_s{s}"]["stages"]
            rows.append(_stage_pair(stages["1"], stages["2"],
                                    so3["cells"][f"w{w}_s{s}_BASE"]["terminal_median"], "SO3", w, s))
    for w in (6, 7, 8, 9):
        for s in (0, 1, 2):
            stages = so4["prefixes"][f"w{w}_s{s}"]["stages"]
            rows.append(_stage_pair(stages["1"], stages["2"],
                                    so4["cells"][f"w{w}_s{s}"]["terminal_median"], "SO4", w, s))
    return rows


def guards(rows: list[dict], reports: dict) -> list[str]:
    problems = []
    for run, expected in EXPECTED.items():
        got = sum(r["run"] == run for r in rows)
        if got != expected:
            problems.append(f"{run}: {got} cells, expected {expected}")
        label = reports[run].get("classification")
        label = label.get("program") if isinstance(label, dict) else label
        if label != EXPECTED_LABELS[run]:
            problems.append(f"{run}: report classification {label!r}, expected {EXPECTED_LABELS[run]!r}")
    for r in rows:
        values = [r[k] for k in PREDICTORS] + [r["stage3_terminal"]]
        if not all(math.isfinite(v) for v in values):
            problems.append(f"{r['run']} w{r['world']} s{r['stream']}: non-finite value")
    return problems


def permutation_fraction(x, y, rho: float) -> float:
    """One-sided fraction of permuted correlations at or above the observed one."""
    rng = np.random.default_rng(np.random.SeedSequence([PERM_SEED]))
    y = np.asarray(y, dtype=float)
    count = sum(spearman(x, rng.permutation(y)) >= rho - 1e-12 for _ in range(PERMUTATIONS))
    return (count + 1) / (PERMUTATIONS + 1)


def separation(rows: list[dict], key: str) -> dict:
    failing = [r[key] for r in rows if not r["passed"]]
    passing = [r[key] for r in rows if r["passed"]]
    med_f = float(np.median(failing)) if failing else float("nan")
    med_p = float(np.median(passing)) if passing else float("nan")
    return {"failing_median": med_f, "passing_median": med_p,
            "ratio": med_f / med_p if passing and med_p > 0 else float("nan"),
            "failing_cells": len(failing), "passing_cells": len(passing)}


def analyse(rows: list[dict]) -> dict:
    out = {"cells": len(rows), "passing": sum(r["passed"] for r in rows), "predictors": {}}
    outcome = [r["stage3_terminal"] for r in rows]
    for key in PREDICTORS:
        values = [r[key] for r in rows]
        rho = spearman(values, outcome)
        out["predictors"][key] = {
            "spearman_pooled": rho,
            "permutation_fraction": permutation_fraction(values, outcome, rho),
            "spearman_by_run": {run: spearman([r[key] for r in rows if r["run"] == run],
                                              [r["stage3_terminal"] for r in rows if r["run"] == run])
                                for run in EXPECTED if sum(r["run"] == run for r in rows) >= 3},
            "separation": separation(rows, key),
        }
    mixed_worlds = sorted({(r["run"], r["world"]) for r in rows
                           if len({x["passed"] for x in rows if (x["run"], x["world"]) == (r["run"], r["world"])}) > 1})
    mixed = [r for r in rows if (r["run"], r["world"]) in mixed_worlds]
    out["mixed_outcome_worlds"] = [f"{run}_w{w}" for run, w in mixed_worlds]
    out["mixed_cells"] = len(mixed)
    if len(mixed) >= 4:
        out["mixed_only"] = {key: {"spearman": spearman([r[key] for r in mixed],
                                                        [r["stage3_terminal"] for r in mixed]),
                                   "separation": separation(mixed, key)} for key in PREDICTORS}
    return out


def triage(analysis: dict) -> str:
    """The plan's registered triage rule, read on stage-2 terminal error."""
    p = analysis["predictors"]["stage2_terminal"]
    rho, frac, ratio = p["spearman_pooled"], p["permutation_fraction"], p["separation"]["ratio"]
    separated = math.isfinite(ratio) and ratio >= 2.0
    if rho >= 0.5 and frac <= 0.05 and separated:
        return "PREFIX-PREDICTIVE"
    if (rho <= 0.2 or frac > 0.2) and not separated:
        return "STAGE3-LOCALIZED"
    return "MIXED"


def main() -> int:
    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    reports = {"SO2": json.loads(SO2_REPORT.read_text()), "SO3": json.loads(SO3_REPORT.read_text()),
               "SO4": json.loads(SO4_REPORT.read_text())}
    rows = collect(reports["SO2"], reports["SO3"], reports["SO4"])
    problems = guards(rows, reports)
    if problems:
        print(json.dumps({"problems": problems}, indent=1))
        return 1
    analysis = analyse(rows)
    analysis["triage"] = triage(analysis)
    payload = {"plan": PLAN.as_posix(), "status": "TIER 0 DESCRIPTIVE - not a verdict",
               "sources": {k: str(v) for k, v in (("SO2", SO2_REPORT), ("SO3", SO3_REPORT), ("SO4", SO4_REPORT))},
               "source_commits": {k: reports[k].get("git_commit") for k in reports},
               "threshold": THRESHOLD, "permutations": PERMUTATIONS, "permutation_seed": PERM_SEED,
               "rows": rows, "analysis": analysis}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(OUTPUT)
    for key, value in analysis["predictors"].items():
        s = value["separation"]
        print(f"{key:24s} rho {value['spearman_pooled']:+.3f} perm {value['permutation_fraction']:.4f} "
              f"fail/pass medians {s['failing_median']:.4f}/{s['passing_median']:.4f} ratio {s['ratio']:.2f} "
              f"by run {({k: round(v, 2) for k, v in value['spearman_by_run'].items()})}")
    print(f"cells {analysis['cells']} passing {analysis['passing']}; mixed worlds "
          f"{analysis['mixed_outcome_worlds']} ({analysis['mixed_cells']} cells)")
    print("TRIAGE:", analysis["triage"])
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
