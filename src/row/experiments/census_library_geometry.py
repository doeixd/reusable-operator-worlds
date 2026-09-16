"""Library-geometry census (Tier 0, descriptive; `LIBRARY_GEOMETRY_CENSUS_PLAN.md`).

Is slot DISTINGUISHABILITY the predictor that stage-2 prefix error misses? The
world-quality census found prefix error predicts stage-3 outcome except in SO4
world 8, where the best-prefix stream failed worst. This census measures the
geometry of every unchanged-protocol cell's frozen stage-2 library and asks
whether any geometric measure predicts the outcome and orders world 8's streams.

Reads frozen stage-2 models plus `reports/world_quality_census.json` for the
already-published outcomes. Trains nothing, opens no world, produces no verdict.
"""
from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.audit_so3_consolidation import restore_prefix_stage2
from row.experiments.census_world_quality import spearman
from row.experiments.score_so2_online_gate import restore_stage
from row.experiments.so1_storage import atomic_json, digest, now

PLAN = Path("LIBRARY_GEOMETRY_CENSUS_PLAN.md")
OUTPUT = Path("reports/library_geometry_census.json")
WORLD_CENSUS = Path("reports/world_quality_census.json")
SO2_REPORT = Path("reports/so2_online_gate.json")
SO3_REPORT = Path("reports/so3_consolidation.json")
SO4_REPORT = Path("reports/so4_b2_retest.json")
PROBE_STATES = 512
PROBE_SEED = 2031
PERMUTATIONS = 2000
PERM_SEED = 2029
THRESHOLD = 0.05
MEASURES = ("min_pair", "mean_pair", "route_margin", "effective_rank", "slot_norm_spread")
MIN_PAIR_SPREAD = 2.0
W8_STREAM_ORDER = (1, 0)  # worst stage-3, best stage-3: s1 = 0.1731, s0 = 0.0161


def probe(world_seed: int, stream: int, d: int) -> torch.Tensor:
    values = np.random.default_rng(np.random.SeedSequence([PROBE_SEED, world_seed, stream])).normal(
        size=(PROBE_STATES, d))
    return torch.tensor(values, dtype=torch.float32)


def slot_outputs(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([operator(x) for operator in model.library])  # (slots, n, d)


def geometry(model, cfg, world, x: torch.Tensor) -> dict:
    """The five registered measures, all properties of the stage-2 library alone."""
    outputs = slot_outputs(model, x)
    norms = torch.linalg.vector_norm(outputs.reshape(len(outputs), -1), dim=1)
    pairs = []
    for i, j in itertools.combinations(range(len(outputs)), 2):
        pairs.append(float(torch.linalg.vector_norm(outputs[i] - outputs[j]) / max(float(norms[i]), 1e-12)))
    flat = outputs.reshape(len(outputs), -1).numpy().astype(np.float64)
    singular = np.linalg.svd(flat, compute_uv=False)
    participation = float((singular.sum() ** 2) / max(float((singular ** 2).sum()), 1e-300))
    frozen = FrozenLibrary(model)
    margins = []
    with torch.no_grad():
        for task in world.tasks:
            if task.task_id not in model.task_codes:
                continue
            mse = frozen.all_route_support_mse(torch.tensor(task.train_x, dtype=torch.float32),
                                               torch.tensor(task.train_y, dtype=torch.float32))
            best, second = torch.topk(mse, 2, largest=False).values.tolist()
            margins.append((second - best) / max(best, 1e-12))
    return {"min_pair": float(min(pairs)), "mean_pair": float(np.mean(pairs)),
            "route_margin": float(np.median(margins)), "effective_rank": participation,
            "slot_norm_spread": float(norms.max() / torch.clamp(norms.min(), min=1e-12)),
            "slots": len(outputs), "scored_tasks": len(margins)}


def load_stage2(run: str, world_seed: int, stream: int, record: dict, model_seed: int):
    """Stage-2 model with ALL of its state; SO2 records carry no task_ids, SO3/SO4 do."""
    cfg, world, _, _ = stage_setup(world_seed, 2, model_seed)
    if run == "SO2":
        path = Path(f"artifacts/so2_online_gate/cells/STAGED_w{world_seed}/stage2")
        model = restore_stage(path, cfg, world, record["novel_probe_available"])
    else:
        root = "so3_consolidation" if run == "SO3" else "so4_b2_retest"
        path = Path(f"artifacts/{root}/prefixes/w{world_seed}_s{stream}/stage2")
        model = restore_prefix_stage2(path, cfg, record)
    return cfg, world, model, path


def collect(rows: list[dict], reports: dict) -> tuple[list[dict], list[str]]:
    seeds = {"SO2": 5000, "SO3": 6000, "SO4": 7000}
    out, problems = [], []
    for row in rows:
        run, w, s = row["run"], row["world"], row["stream"]
        if run == "SO2":
            record = reports["SO2"]["cells"][f"STAGED_w{w}"]["stages"]["2"]
        else:
            record = reports[run]["prefixes"][f"w{w}_s{s}"]["stages"]["2"]
        cfg, world, model, path = load_stage2(run, w, s, record, seeds[run])
        if library_sha(model) != record["library_sha256"]:
            problems.append(f"{run} w{w} s{s}: stage-2 library hash differs from its run record")
        measures = geometry(model, cfg, world, probe(w, s, cfg.world.state_dim))
        out.append({"run": run, "world": w, "stream": s, "path": path.as_posix(),
                    "stage2_terminal": row["stage2_terminal"], "stage3_terminal": row["stage3_terminal"],
                    "passed": row["passed"], **measures})
    return out, problems


def separation(cells: list[dict], key: str) -> dict:
    failing = [c[key] for c in cells if not c["passed"]]
    passing = [c[key] for c in cells if c["passed"]]
    med_f = float(np.median(failing)) if failing else float("nan")
    med_p = float(np.median(passing)) if passing else float("nan")
    return {"failing_median": med_f, "passing_median": med_p,
            "ratio": med_f / med_p if passing and med_p > 0 else float("nan")}


def permutation_fraction(x, y, rho: float) -> float:
    if not math.isfinite(rho):
        return float("nan")
    # One-sided on the OBSERVED sign, matching `census_world_quality.permutation_fraction`
    # (the plan registers "the same permutation null"). A negative rho is tested for being
    # as negative, not as large in magnitude.
    rng = np.random.default_rng(np.random.SeedSequence([PERM_SEED]))
    y = np.asarray(y, dtype=float)
    sign = 1.0 if rho >= 0 else -1.0
    count = sum(1 for _ in range(PERMUTATIONS)
                if sign * spearman(x, rng.permutation(y)) >= sign * rho - 1e-12)
    return (count + 1) / (PERMUTATIONS + 1)


def w8_ordering(cells: list[dict], key: str) -> dict:
    """Does the measure order SO4 world 8's streams as their stage-3 outcomes did?"""
    w8 = {c["stream"]: c for c in cells if c["run"] == "SO4" and c["world"] == 8}
    if set(w8) != {0, 1, 2}:
        return {"available": False}
    worst, best = W8_STREAM_ORDER
    # "Correct" means the measure ranks the worst-outcome stream as the least favourable.
    # For min_pair / route_margin / effective_rank, lower is less favourable; for
    # mean_pair it is also lower; for slot_norm_spread, higher is less favourable.
    lower_is_worse = key != "slot_norm_spread"
    values = {s: c[key] for s, c in w8.items()}
    if lower_is_worse:
        ordered = values[worst] == min(values.values()) and values[best] == max(values.values())
    else:
        ordered = values[worst] == max(values.values()) and values[best] == min(values.values())
    return {"available": True, "values": values, "orders_w8": bool(ordered)}


def analyse(cells: list[dict]) -> dict:
    outcome = [c["stage3_terminal"] for c in cells]
    mixed_keys = {("SO3", 4), ("SO3", 5), ("SO4", 7), ("SO4", 8)}
    mixed = [c for c in cells if (c["run"], c["world"]) in mixed_keys]
    median_error = float(np.median([c["stage2_terminal"] for c in cells]))
    low_error = [c for c in cells if c["stage2_terminal"] <= median_error]
    out = {"cells": len(cells), "passing": sum(c["passed"] for c in cells),
           "median_stage2_terminal": median_error, "measures": {}}
    for key in MEASURES:
        values = [c[key] for c in cells]
        rho = spearman(values, outcome)
        out["measures"][key] = {
            "spearman_pooled": rho,
            "permutation_fraction": permutation_fraction(values, outcome, rho),
            "spearman_by_run": {run: spearman([c[key] for c in cells if c["run"] == run],
                                              [c["stage3_terminal"] for c in cells if c["run"] == run])
                                for run in ("SO2", "SO3", "SO4")},
            "separation": separation(cells, key),
            "mixed_worlds_spearman": spearman([c[key] for c in mixed], [c["stage3_terminal"] for c in mixed]),
            "mixed_worlds_separation": separation(mixed, key),
            "low_error_spearman": spearman([c[key] for c in low_error],
                                           [c["stage3_terminal"] for c in low_error]),
            "low_error_cells": len(low_error),
            "w8": w8_ordering(cells, key),
            "range": [float(min(values)), float(max(values))],
        }
    return out


def anchor(cells: list[dict], reports: dict) -> list[str]:
    """The plan's registered bitwise anchor: one reconstructed stage-2 model must
    reproduce the terminal median its own run recorded, to 1e-6."""
    from row.experiments.audit_rotated_g5r_interference import score
    from types import SimpleNamespace
    target = next(c for c in cells if (c["run"], c["world"], c["stream"]) == ("SO4", 6, 0))
    record = reports["SO4"]["prefixes"]["w6_s0"]["stages"]["2"]
    cfg, world, model, _ = load_stage2("SO4", 6, 0, record, 7000)
    trained = SimpleNamespace(tasks=[t for t in world.tasks if t.task_id in model.task_codes])
    recomputed = score(model, trained)["median"]
    if abs(recomputed - record["terminal_median"]) > 1e-6:
        return [f"anchor: re-scored stage-2 median {recomputed} vs recorded {record['terminal_median']}"]
    return []


def guards(cells: list[dict], analysis: dict) -> list[str]:
    problems = []
    if len(cells) != 24:
        problems.append(f"{len(cells)} cells, expected 24")
    for c in cells:
        if not all(math.isfinite(c[k]) for k in MEASURES):
            problems.append(f"{c['run']} w{c['world']} s{c['stream']}: non-finite measure")
        if c["min_pair"] > c["mean_pair"] + 1e-12:
            problems.append(f"{c['run']} w{c['world']} s{c['stream']}: min_pair exceeds mean_pair")
    for key in MEASURES:
        lo, hi = analysis["measures"][key]["range"]
        if hi - lo <= 0:
            problems.append(f"{key}: constant across cells, cannot discriminate")
    lo, hi = analysis["measures"]["min_pair"]["range"]
    if lo > 0 and hi / lo < MIN_PAIR_SPREAD:
        problems.append(f"min_pair varies only {hi / lo:.2f}x across cells (< {MIN_PAIR_SPREAD}x): "
                        "the measure cannot discriminate at this scale")
    return problems


def triage(analysis: dict) -> str:
    """The plan's registered rule."""
    explains = any(
        (m["spearman_pooled"] == m["spearman_pooled"] and abs(m["spearman_pooled"]) >= 0.5
         and m["permutation_fraction"] <= 0.05 and m["w8"].get("orders_w8"))
        for m in analysis["measures"].values())
    if explains:
        return "GEOMETRY-EXPLAINS"
    nothing = (all(not (abs(m["spearman_pooled"]) > 0.3) or m["spearman_pooled"] != m["spearman_pooled"]
                   for m in analysis["measures"].values())
               or not any(m["w8"].get("orders_w8") for m in analysis["measures"].values()))
    return "GEOMETRY-ADDS-NOTHING" if nothing else "MIXED"


def main() -> int:
    torch.set_num_threads(1)
    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    rows = json.loads(WORLD_CENSUS.read_text())["rows"]
    reports = {"SO2": json.loads(SO2_REPORT.read_text()), "SO3": json.loads(SO3_REPORT.read_text()),
               "SO4": json.loads(SO4_REPORT.read_text())}
    cells, problems = collect(rows, reports)
    analysis = analyse(cells)
    problems += guards(cells, analysis)
    problems += anchor(cells, reports)
    if problems:
        print(json.dumps({"problems": problems}, indent=1))
        return 1
    analysis["triage"] = triage(analysis)
    atomic_json(OUTPUT, {"plan": PLAN.as_posix(), "status": "TIER 0 DESCRIPTIVE - not a verdict",
                         "probe": {"states": PROBE_STATES, "seed": PROBE_SEED},
                         "permutations": PERMUTATIONS, "permutation_seed": PERM_SEED,
                         "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, WORLD_CENSUS)},
                         "created_utc": now(), "cells": cells, "analysis": analysis})
    for key in MEASURES:
        m = analysis["measures"][key]
        s, ms = m["separation"], m["mixed_worlds_separation"]
        print(f"{key:18s} rho {m['spearman_pooled']:+.3f} perm {m['permutation_fraction']:.4f} "
              f"sep {s['ratio']:.2f} | mixed rho {m['mixed_worlds_spearman']:+.3f} sep {ms['ratio']:.2f} "
              f"| low-error rho {m['low_error_spearman']:+.3f} | w8 ordered {m['w8'].get('orders_w8')}")
    print(f"cells {analysis['cells']} passing {analysis['passing']}")
    print("TRIAGE:", analysis["triage"])
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
