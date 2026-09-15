"""Independent SO2-P recomputation from durable arm records and saved models.

Rebuilds every arm's terminal model from disk, re-scores every task with its own
NMSE, recomputes library hashes, drift from SO2's saved stage-2 library, the
anchor, the three gates and the frozen triage rule without the runner's
decision functions. Exits nonzero on any disagreement.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.score_so2_online_gate import restore_stage

REPORT = Path("reports/so2p_plasticity.json")
CELLS = Path("artifacts/so2p_plasticity/cells")
SO2_CELL = Path("artifacts/so2_online_gate/cells/STAGED_w1")
WORLD, MODEL_SEED, THRESHOLD, TOL = 1, 5000, 0.05, 1e-6
CHANGES = {"BASE": None, "LR_1/2": ("global_learning_rate", 0.0005), "LR_1/4": ("global_learning_rate", 0.00025),
           "LR_1/10": ("global_learning_rate", 0.0001), "REPLAY_2x": ("replay_examples_per_task", 8),
           "REPLAY_4x": ("replay_examples_per_task", 16)}


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def library_hash(model) -> str:
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()


def query_nmse(model, task) -> float:
    with torch.no_grad():
        p = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id).numpy().astype(np.float64)
    y = np.asarray(task.eval_y, dtype=np.float64)
    return float(np.mean((y - p) ** 2) / np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))


def drift_median(before, after) -> float:
    probe = torch.tensor(np.random.default_rng(np.random.SeedSequence([2026, WORLD])).normal(size=(512, 16)),
                         dtype=torch.float32)
    with torch.no_grad():
        return float(np.median([float(torch.norm(b(probe) - a(probe)) / torch.norm(a(probe)))
                                for a, b in zip(before.library, after.library)]))


def end_of_task(path: Path) -> list[tuple[str, float]]:
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = sorted((r for r in rows if r.get("record_type") == "task_summary"), key=lambda r: r["task_index"])
    return [(r["task_id"], float(r["final_nmse"])) for r in rows]


def main() -> int:
    torch.set_num_threads(1)
    report = json.loads(REPORT.read_text())
    so2 = json.loads((SO2_CELL / "result.json").read_text())["result"]
    problems, arms = [], {}
    cfg2, world2, _, _ = stage_setup(WORLD, 2, MODEL_SEED)
    stage2 = restore_stage(SO2_CELL / "stage2", cfg2, world2, so2["stages"]["2"]["novel_probe_available"])
    base_cfg, world, _, _ = stage_setup(WORLD, 3, MODEL_SEED)
    for arm, change in CHANGES.items():
        stored = json.loads((CELLS / arm.replace("/", "_") / "result.json").read_text())
        rec = stored["result"]
        if sha_json(rec) != stored["result_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
            problems.append(f"{arm}: record hash/commit mismatch")
        cfg = base_cfg if change is None else replace(
            base_cfg, discrete_model=replace(base_cfg.discrete_model, **{change[0]: change[1]}))
        model = restore_stage(CELLS / arm.replace("/", "_") / "stage3", cfg, world, True)
        model.eval()
        terminal = {t.task_id: query_nmse(model, t) for t in world.tasks if t.task_id in model.task_codes}
        eot = end_of_task(CELLS / arm.replace("/", "_") / "stage3")
        term = [terminal[i] for i, _ in eot]
        end = [e for _, e in eot]
        a = {"sha": library_hash(model), "terminal_median": float(np.median(term)),
             "terminal_below": int(sum(v <= THRESHOLD for v in term)), "eot_median": float(np.median(end)),
             "lost": int(sum(e <= THRESHOLD < t for t, e in zip(term, end))),
             "anchor": abs(terminal[eot[-1][0]] - eot[-1][1]), "drift": drift_median(stage2, model),
             "per_task": terminal}
        arms[arm] = a
        for key, recorded in (("terminal_median", rec["terminal_median"]), ("eot_median", rec["end_of_task_median"]),
                              ("drift", rec["drift_from_stage2"]["median"])):
            if abs(a[key] - recorded) > TOL:
                problems.append(f"{arm}: {key} {a[key]} vs recorded {recorded}")
        for key, recorded in (("terminal_below", rec["terminal_below"]), ("lost", rec["lost_threshold"]),
                              ("sha", rec["library_sha256"])):
            if a[key] != recorded:
                problems.append(f"{arm}: {key} {a[key]} vs recorded {recorded}")
    base = arms["BASE"]
    rec3 = so2["stages"]["3"]
    g0 = base["sha"] == rec3["library_sha256"] and all(
        abs(base["per_task"][k] - v) <= TOL for k, v in rec3["terminal_per_task"].items())
    g1 = all(arms[n]["sha"] != base["sha"] for n in CHANGES if n != "BASE")
    g2 = all(a["anchor"] <= TOL for a in arms.values())
    others = [arms[n] for n in CHANGES if n != "BASE"]
    if not (g0 and g1 and g2):
        label = "UNINFORMATIVE"
    elif any(o["terminal_median"] <= THRESHOLD and o["terminal_below"] >= 32 and o["drift"] < base["drift"]
             and o["eot_median"] <= 2 * base["eot_median"] for o in others):
        label = "LIVE"
    elif any((o["terminal_median"] <= base["terminal_median"] / 2 or o["lost"] <= base["lost"] / 2)
             and o["drift"] < base["drift"] for o in others):
        label = "PARTIAL"
    elif all(arms[x]["drift"] > arms[y]["drift"] for x, y in
             (("BASE", "LR_1/2"), ("LR_1/2", "LR_1/4"), ("LR_1/4", "LR_1/10"))):
        label = "DISFAVOURED"
    else:
        label = "UNINFORMATIVE"
    if report.get("triage") != label or report.get("complete") is not True:
        problems.append(f"triage {report.get('triage')} vs recomputed {label}")
    if report.get("gates") != {"G0_reproduction": g0, "G1_non_vacuity": g1, "G2_anchor": g2}:
        problems.append(f"gates {report.get('gates')} vs recomputed {(g0, g1, g2)}")
    print(json.dumps({"triage": label, "gates": [g0, g1, g2],
                      "arms": {n: {k: v for k, v in a.items() if k not in ("per_task", "sha")} for n, a in arms.items()},
                      "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
