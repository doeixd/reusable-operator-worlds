"""Route-margin normalization census (Tier 0; `ROUTE_MARGIN_NORMALIZATION_PLAN.md`).

Does the geometry census's `route_margin` result survive being made comparable?

Test A: within-world concordance over the 7 three-stream worlds - does the
worst-outcome stream hold the lowest margin? Never compares across worlds, so the
300x scale spread cannot produce it. Null: Binomial(7, 1/3).

Test B: `self_margin`, a library-intrinsic identifiability measure computed on
targets the library generates itself, so it carries no teacher task set and is
comparable across worlds.

Frozen stage-2 models and committed reports only. No training, no new world, no
verdict.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.census_library_geometry import load_stage2
from row.experiments.census_world_quality import spearman
from row.experiments.so1_storage import atomic_json, digest, now

PLAN = Path("ROUTE_MARGIN_NORMALIZATION_PLAN.md")
OUTPUT = Path("reports/route_margin_normalization.json")
GEOMETRY_CENSUS = Path("reports/library_geometry_census.json")
SO2_REPORT = Path("reports/so2_online_gate.json")
SO3_REPORT = Path("reports/so3_consolidation.json")
SO4_REPORT = Path("reports/so4_b2_retest.json")
PROBE_STATES = 64
PROBE_SEED = 2031
ROUTE_SEED = 2037
ROUTES = 16
PERMUTATIONS = 2000
PERM_SEED = 2029
THRESHOLD = 0.05
RECOVERY_MIN = 15  # of ROUTES, per library
SEEDS = {"SO2": 5000, "SO3": 6000, "SO4": 7000}
THREE_STREAM_WORLDS = (("SO3", 3), ("SO3", 4), ("SO3", 5),
                       ("SO4", 6), ("SO4", 7), ("SO4", 8), ("SO4", 9))


def probe(world_seed: int, stream: int, d: int) -> torch.Tensor:
    values = np.random.default_rng(np.random.SeedSequence([PROBE_SEED, world_seed, stream])).normal(
        size=(PROBE_STATES, d))
    return torch.tensor(values, dtype=torch.float32)


def self_margin(model, world_seed: int, stream: int, d: int) -> dict:
    """Identifiability of the library's OWN routes: the second-best route's support MSE,
    scaled by target variance. The generating route must be the argmin (recovery gate)."""
    frozen = FrozenLibrary(model)
    x = probe(world_seed, stream, d)
    rng = np.random.default_rng(np.random.SeedSequence([ROUTE_SEED, world_seed, stream]))
    margins, recovered = [], 0
    with torch.no_grad():
        for _ in range(ROUTES):
            route = [int(v) for v in rng.integers(0, frozen.slots, size=frozen.steps)]
            y = frozen.hard(x, route)
            mse = frozen.all_route_support_mse(x, y)
            order = torch.argsort(mse)
            if unflatten(int(order[0]), frozen.slots, frozen.steps) == route:
                recovered += 1
            variance = float(torch.mean((y - y.mean(dim=0, keepdim=True)) ** 2))
            margins.append(float(mse[order[1]]) / max(variance, 1e-12))
    return {"self_margin": float(np.median(margins)), "recovered_routes": recovered,
            "self_margin_min": float(min(margins)), "self_margin_max": float(max(margins))}


def collect(geometry_cells: list[dict], reports: dict) -> tuple[list[dict], list[str]]:
    out, problems = [], []
    for cell in geometry_cells:
        run, w, s = cell["run"], cell["world"], cell["stream"]
        if run == "SO2":
            record = reports["SO2"]["cells"][f"STAGED_w{w}"]["stages"]["2"]
        else:
            record = reports[run]["prefixes"][f"w{w}_s{s}"]["stages"]["2"]
        cfg, _, model, _ = load_stage2(run, w, s, record, SEEDS[run])
        if library_sha(model) != record["library_sha256"]:
            problems.append(f"{run} w{w} s{s}: stage-2 library hash differs from its run record")
        measures = self_margin(model, w, s, cfg.world.state_dim)
        out.append({"run": run, "world": w, "stream": s,
                    "route_margin": cell["route_margin"], "stage2_terminal": cell["stage2_terminal"],
                    "stage3_terminal": cell["stage3_terminal"], "passed": cell["passed"], **measures})
    return out, problems


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def concordance(cells: list[dict], key: str) -> dict:
    """Test A: does the worst-outcome stream hold the LOWEST `key` in each world?"""
    per_world, ties = {}, []
    for run, w in THREE_STREAM_WORLDS:
        trio = [c for c in cells if c["run"] == run and c["world"] == w]
        if len(trio) != 3:
            return {"available": False, "reason": f"{run} w{w} has {len(trio)} streams"}
        worst = max(trio, key=lambda c: c["stage3_terminal"])
        values = [c[key] for c in trio]
        if len(set(values)) < 3:
            ties.append(f"{run}_w{w}")
        per_world[f"{run}_w{w}"] = {"worst_stream": worst["stream"], "worst_value": worst[key],
                                    "min_value": min(values), "concordant": worst[key] == min(values)}
    k = sum(v["concordant"] for v in per_world.values())
    n = len(per_world)
    return {"available": True, "k": k, "n": n, "expected": n / 3.0,
            "binomial_upper_tail": binomial_upper_tail(k, n, 1 / 3), "ties": ties, "per_world": per_world}


def permutation_fraction(x, y, rho: float) -> float:
    if not math.isfinite(rho):
        return float("nan")
    rng = np.random.default_rng(np.random.SeedSequence([PERM_SEED]))
    y = np.asarray(y, dtype=float)
    sign = 1.0 if rho >= 0 else -1.0
    count = sum(1 for _ in range(PERMUTATIONS)
                if sign * spearman(x, rng.permutation(y)) >= sign * rho - 1e-12)
    return (count + 1) / (PERMUTATIONS + 1)


def separation(cells: list[dict], key: str) -> dict:
    failing = [c[key] for c in cells if not c["passed"]]
    passing = [c[key] for c in cells if c["passed"]]
    med_f = float(np.median(failing)) if failing else float("nan")
    med_p = float(np.median(passing)) if passing else float("nan")
    return {"failing_median": med_f, "passing_median": med_p,
            "ratio": med_f / med_p if passing and med_p > 0 else float("nan")}


def within_world_z(cells: list[dict], key: str) -> list[tuple[float, float]]:
    """(z-scored key, stage-3) over the three-stream worlds only."""
    out = []
    for run, w in THREE_STREAM_WORLDS:
        trio = [c for c in cells if c["run"] == run and c["world"] == w]
        values = np.array([c[key] for c in trio], dtype=float)
        sd = values.std()
        if sd == 0:
            continue
        for c, z in zip(trio, (values - values.mean()) / sd):
            out.append((float(z), c["stage3_terminal"]))
    return out


def analyse(cells: list[dict]) -> dict:
    outcome = [c["stage3_terminal"] for c in cells]
    rho = spearman([c["self_margin"] for c in cells], outcome)
    zs = within_world_z(cells, "route_margin")
    zs_self = within_world_z(cells, "self_margin")
    out = {
        "cells": len(cells), "passing": sum(c["passed"] for c in cells),
        "test_a_route_margin": concordance(cells, "route_margin"),
        "test_a_self_margin": concordance(cells, "self_margin"),
        "test_a_stage2_terminal_reference": concordance(cells, "stage2_terminal"),
        "test_b_self_margin": {
            "spearman_pooled": rho,
            "permutation_fraction": permutation_fraction([c["self_margin"] for c in cells], outcome, rho),
            "spearman_by_run": {run: spearman([c["self_margin"] for c in cells if c["run"] == run],
                                              [c["stage3_terminal"] for c in cells if c["run"] == run])
                                for run in SEEDS},
            "separation": separation(cells, "self_margin"),
            "range": [float(min(c["self_margin"] for c in cells)), float(max(c["self_margin"] for c in cells))],
        },
        "within_world_z_route_margin_spearman": spearman([z for z, _ in zs], [o for _, o in zs]) if zs else float("nan"),
        "within_world_z_self_margin_spearman": (spearman([z for z, _ in zs_self], [o for _, o in zs_self])
                                                if zs_self else float("nan")),
        "self_vs_route_margin_spearman": spearman([c["self_margin"] for c in cells],
                                                 [c["route_margin"] for c in cells]),
        "recovered_routes_range": [min(c["recovered_routes"] for c in cells),
                                   max(c["recovered_routes"] for c in cells)],
    }
    return out


def guards(cells: list[dict], analysis: dict) -> list[str]:
    problems = []
    if len(cells) != 24:
        problems.append(f"{len(cells)} cells, expected 24")
    for c in cells:
        if c["recovered_routes"] < RECOVERY_MIN:
            problems.append(f"{c['run']} w{c['world']} s{c['stream']}: generating route recovered only "
                            f"{c['recovered_routes']}/{ROUTES}; self_margin is not measuring identifiability")
        if not math.isfinite(c["self_margin"]) or c["self_margin"] <= 0:
            problems.append(f"{c['run']} w{c['world']} s{c['stream']}: self_margin {c['self_margin']} not positive")
    lo, hi = analysis["test_b_self_margin"]["range"]
    if hi - lo <= 0:
        problems.append("self_margin is constant across cells, cannot discriminate")
    for key in ("test_a_route_margin", "test_a_self_margin"):
        a = analysis[key]
        if not a.get("available"):
            problems.append(f"{key}: {a.get('reason')}")
        elif a["n"] != 7:
            problems.append(f"{key}: {a['n']} worlds, expected 7")
        elif a["ties"]:
            problems.append(f"{key}: tied margins in {a['ties']}")
    return problems


def triage(analysis: dict) -> str:
    """The plan's registered rule, on Test A (route_margin) and Test B."""
    a = analysis["test_a_route_margin"]
    b = analysis["test_b_self_margin"]
    k = a.get("k", 0)
    rho, frac = b["spearman_pooled"], b["permutation_fraction"]
    strong_b = math.isfinite(rho) and abs(rho) >= 0.5 and math.isfinite(frac) and frac <= 0.05
    if k >= 5 or strong_b:
        return "SURVIVES"
    if k <= 3 and (not math.isfinite(rho) or abs(rho) < 0.3):
        return "DISSOLVES"
    return "MIXED"


def main() -> int:
    torch.set_num_threads(1)
    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    geometry = json.loads(GEOMETRY_CENSUS.read_text())["cells"]
    reports = {"SO2": json.loads(SO2_REPORT.read_text()), "SO3": json.loads(SO3_REPORT.read_text()),
               "SO4": json.loads(SO4_REPORT.read_text())}
    cells, problems = collect(geometry, reports)
    analysis = analyse(cells)
    problems += guards(cells, analysis)
    if problems:
        print(json.dumps({"problems": problems}, indent=1))
        return 1
    analysis["triage"] = triage(analysis)
    atomic_json(OUTPUT, {"plan": PLAN.as_posix(), "status": "TIER 0 DESCRIPTIVE - not a verdict",
                         "probe": {"states": PROBE_STATES, "seed": PROBE_SEED, "routes": ROUTES,
                                   "route_seed": ROUTE_SEED},
                         "permutations": PERMUTATIONS, "permutation_seed": PERM_SEED,
                         "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, GEOMETRY_CENSUS)},
                         "created_utc": now(), "cells": cells, "analysis": analysis})
    a, b = analysis["test_a_route_margin"], analysis["test_b_self_margin"]
    print(f"TEST A route_margin concordance k={a['k']}/{a['n']} (expected {a['expected']:.2f}, "
          f"binomial upper tail {a['binomial_upper_tail']:.4f})")
    print(f"TEST A self_margin  concordance k={analysis['test_a_self_margin']['k']}/7 | "
          f"stage2-error reference k={analysis['test_a_stage2_terminal_reference']['k']}/7")
    print(f"TEST B self_margin rho {b['spearman_pooled']:+.3f} perm {b['permutation_fraction']:.4f} "
          f"sep {b['separation']['ratio']:.2f} range {b['range'][0]:.4f}-{b['range'][1]:.4f} "
          f"by run {({k: round(v, 3) for k, v in b['spearman_by_run'].items()})}")
    print(f"within-world z: route_margin rho {analysis['within_world_z_route_margin_spearman']:+.3f} | "
          f"self_margin rho {analysis['within_world_z_self_margin_spearman']:+.3f}")
    print(f"self vs route margin rho {analysis['self_vs_route_margin_spearman']:+.3f} | "
          f"recovered routes {analysis['recovered_routes_range']}")
    print("TRIAGE:", analysis["triage"])
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
