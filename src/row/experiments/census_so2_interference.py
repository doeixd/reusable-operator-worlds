"""SO2 interference census (Tier 0, descriptive; `SO2_INTERFERENCE_CENSUS_PLAN.md`).

Reads only the frozen SO2 artifacts. No training. Refuses to write a report
unless every consistency guard in the plan passes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.score_so2_online_gate import restore_stage
from row.experiments.so1_storage import atomic_json, now
from row.metrics import nmse

CELLS = Path("artifacts/so2_online_gate/cells")
OUTPUT = Path("reports/so2_interference_census.json")
WORLDS = (0, 1, 2)
MODEL_SEED = 5000
TOL = 1e-6
THRESHOLD = 0.05


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def task_summaries(path: Path) -> list[dict]:
    with (path / "metrics.jsonl").open(encoding="utf-8") as handle:
        rows = [row for row in map(json.loads, handle) if row.get("record_type") == "task_summary"]
    return sorted(rows, key=lambda r: r["task_index"])


def score_tasks(model, tasks) -> dict[str, float]:
    """Hard-route query NMSE per task in eval mode, as G5R Stage D's `score`."""
    model.eval()
    with torch.no_grad():
        return {t.task_id: float(nmse(model(torch.tensor(t.eval_x, dtype=torch.float32), t.task_id).numpy(),
                                      t.eval_y))
                for t in tasks}


def transplant(library_model, code_model, code_cfg):
    """A fresh model at `code_cfg`'s program length: `library_model`'s library,
    `code_model`'s task codes and temperature. Nothing else carries over."""
    host = build_fast(code_cfg)
    host.library.load_state_dict(library_model.library.state_dict())
    for task_id, code in code_model.task_codes.items():
        host.task_codes[task_id] = torch.nn.Parameter(code.detach().clone())
    host.temperature = code_model.temperature
    return host


def slot_drift(before, after, world_seed: int, d: int) -> dict:
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence([2026, world_seed])).normal(size=(512, d)),
                         dtype=torch.float32)
    with torch.no_grad():
        changes = [float(torch.norm(b(probe) - a(probe)) / torch.norm(a(probe)))
                   for a, b in zip(before.library, after.library)]
    return {"median": float(np.median(changes)), "max": float(np.max(changes)), "per_slot": changes}


def cell_census(arm: str, world_seed: int, problems: list[str]) -> dict:
    name = f"{arm}_w{world_seed}"
    result = json.loads((CELLS / name / "result.json").read_text())["result"]
    loaded, out = {}, {"stages": {}}
    for stage in sorted(int(s) for s in result["stages"]):
        record = result["stages"][str(stage)]
        cfg, world, _, _ = stage_setup(world_seed, stage, MODEL_SEED)
        path = CELLS / name / f"stage{stage}"
        model = restore_stage(path, cfg, world, record["novel_probe_available"])
        trained = [t for t in world.tasks if t.task_id in model.task_codes]
        loaded[stage] = (cfg, trained, model)

        summaries = task_summaries(path)
        ids = [r["task_id"] for r in summaries]
        end = [r["final_nmse"] for r in summaries]
        zero = [r["zero_shot_nmse"] for r in summaries]
        terminal = score_tasks(model, trained)
        term = [terminal[i] for i in ids]

        # Guards: terminal recompute, end-of-task source, exact self-transplant.
        worst = max(abs(terminal[i] - record["terminal_per_task"][i]) for i in ids)
        if worst > TOL:
            problems.append(f"{name} s{stage}: terminal recompute differs by {worst:.2e}")
        if abs(float(np.median(end)) - record["end_of_task_median"]) > TOL:
            problems.append(f"{name} s{stage}: end-of-task median not reproduced")
        own = score_tasks(transplant(model, model, cfg), trained)
        if max(abs(own[i] - terminal[i]) for i in ids) > TOL:
            problems.append(f"{name} s{stage}: self-transplant does not reproduce terminal")

        ratio = [math.log10(max(t, 1e-12) / max(e, 1e-12)) for t, e in zip(term, end)]
        out["stages"][str(stage)] = {
            "tasks": len(ids),
            "terminal_median": float(np.median(term)), "end_of_task_median": float(np.median(end)),
            "log10_terminal_over_end_quartile_means": [float(np.mean(c)) for c in np.array_split(ratio, 4)],
            "spearman_position_vs_log_ratio": spearman(list(range(len(ids))), ratio),
            "fraction_degraded_over_2x": float(np.mean([r > math.log10(2) for r in ratio])),
            "fraction_improved_over_2x": float(np.mean([r < -math.log10(2) for r in ratio])),
            "lost_threshold": int(sum(e <= THRESHOLD < t for t, e in zip(term, end))),
            "gained_threshold": int(sum(t <= THRESHOLD < e for t, e in zip(term, end))),
            "zero_shot_quartile_medians": [float(np.median(c)) for c in np.array_split(zero, 4)],
            "end_of_task_quartile_medians": [float(np.median(c)) for c in np.array_split(end, 4)],
            "terminal_quartile_medians": [float(np.median(c)) for c in np.array_split(term, 4)],
        }

    if arm == "STAGED":
        out["backward_transplant"] = {}
        for early, late in ((1, 2), (1, 3), (2, 3)):
            cfg_e, trained_e, model_e = loaded[early]
            moved = list(score_tasks(transplant(loaded[late][2], model_e, cfg_e), trained_e).values())
            out["backward_transplant"][f"s{early}_under_s{late}"] = {
                "median": float(np.median(moved)), "below_threshold": int(sum(v <= THRESHOLD for v in moved)),
                "own_terminal_median": out["stages"][str(early)]["terminal_median"],
                "own_below_threshold": result["stages"][str(early)]["terminal_below_threshold"],
            }
        d = loaded[1][0].world.state_dim
        out["library_drift"] = {f"s{a}_to_s{b}": slot_drift(loaded[a][2], loaded[b][2], world_seed, d)
                                for a, b in ((1, 2), (2, 3))}
    return out


def main() -> int:
    torch.set_num_threads(1)
    problems, cells = [], {}
    for arm in ("STAGED", "PLAIN"):
        for w in WORLDS:
            cells[f"{arm}_w{w}"] = cell_census(arm, w, problems)
            print(f"[{arm}_w{w}] done", flush=True)
    if problems:
        print(json.dumps(problems, indent=1))
        return 1
    atomic_json(OUTPUT, {"plan": "SO2_INTERFERENCE_CENSUS_PLAN.md", "status": "TIER 0 DESCRIPTIVE - not a verdict",
                         "git_commit": git_commit(), "source_run": "071fdaf", "created_utc": now(), "cells": cells})
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
