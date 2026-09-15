"""Independent SO4 recomputation from durable records and saved models.

Reloads every prefix stage model, STAGED stage-3 model and PLAIN model;
re-scores every task with its own NMSE; recomputes library hashes, the transfer
chain, anchors, stream distinctness, world medians, the stream sub-clause,
each world's margin from its twelve recorded program pairs (adaptations are
not repeated), and the frozen ladder, without the runner's decision functions.
G0/G1 are rechecked from their durable records. Exits nonzero on disagreement.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.score_so2_online_gate import restore_stage

REPORT = Path("reports/so4_b2_retest.json")
ROOT = Path("artifacts/so4_b2_retest")
SO2_RESULT = Path("artifacts/so2_online_gate/cells/STAGED_w1/result.json")
WORLDS, STREAMS, MODEL_SEED, THRESHOLD, MARGIN, TOL, PAIRS = (6, 7, 8, 9), (0, 1, 2), 7000, 0.05, 0.75, 1e-6, 12


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def library_hash(model) -> str:
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()


def query_nmse(model, task) -> float:
    with torch.no_grad():
        p = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id).numpy().astype(np.float64)
    y = np.asarray(task.eval_y, dtype=np.float64)
    return float(np.mean((y - p) ** 2) / np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))


def last_end_of_task(path: Path) -> tuple[str, float]:
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = sorted((r for r in rows if r.get("record_type") == "task_summary"), key=lambda r: r["task_index"])
    return rows[-1]["task_id"], float(rows[-1]["final_nmse"])


def rescore(path: Path, cfg, world, probe: bool) -> dict:
    model = restore_stage(path, cfg, world, probe)
    model.eval()
    per_task = {t.task_id: query_nmse(model, t) for t in world.tasks if t.task_id in model.task_codes}
    last_id, last_end = last_end_of_task(path)
    return {"sha": library_hash(model), "median": float(np.median(list(per_task.values()))),
            "below": int(sum(v <= THRESHOLD for v in per_task.values())), "per_task": per_task,
            "anchor": abs(per_task[last_id] - last_end), "ids": set(per_task)}


def record(kind: str, name: str, report: dict, problems: list) -> dict:
    stored = json.loads((ROOT / "records" / kind / f"{name}.json").read_text())
    if sha_json(stored["result"]) != stored["result_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"] \
            or stored["stamp"]["protocol_sha256"] != report["protocol_sha256"]:
        problems.append(f"{kind}/{name}: record hash/commit/protocol mismatch")
    return stored["result"]


def main() -> int:
    torch.set_num_threads(1)
    report = json.loads(REPORT.read_text())
    problems, harness = [], True
    so2 = json.loads(SO2_RESULT.read_text())["result"]["stages"]["3"]
    for variant in ("omitted", "explicit"):
        g = record("gate", f"G0_{variant}", report, problems)
        harness &= g["library_sha256"] == so2["library_sha256"] and g["worst_per_task_abs_diff"] <= TOL
    medians, cells = {w: {} for w in WORLDS}, {}
    for w in WORLDS:
        for s in STREAMS:
            prefix = record("prefix", f"w{w}_s{s}", report, problems)
            shas, ids = {}, {}
            for stage in (1, 2):
                cfg, world, _, _ = stage_setup(w, stage, MODEL_SEED)
                r = rescore(ROOT / "prefixes" / f"w{w}_s{s}" / f"stage{stage}", cfg, world,
                            prefix["stages"][str(stage)]["novel_probe_available"])
                shas[stage], ids[stage] = r["sha"], r["ids"]
                harness &= r["anchor"] <= TOL and r["sha"] == prefix["stages"][str(stage)]["library_sha256"]
            harness &= prefix["stages"]["2"]["library_sha256_at_start"] == shas[1] and not ids[1] & ids[2]
            c = record("cell", f"w{w}_s{s}", report, problems)
            cfg3, world3, _, _ = stage_setup(w, 3, MODEL_SEED)
            r = rescore(ROOT / "cells" / f"w{w}_s{s}" / "stage3", cfg3, world3, c["novel_probe_available"])
            cells[(w, s)] = r
            medians[w][s] = r["median"]
            harness &= r["anchor"] <= TOL and c["library_sha256_at_start"] == shas[2] and not (r["ids"] & (ids[1] | ids[2]))
            if r["sha"] != c["library_sha256"] or abs(r["median"] - c["terminal_median"]) > TOL:
                problems.append(f"w{w}_s{s}: rescored {r['median']} / sha vs record {c['terminal_median']}")
        p = record("plain", f"w{w}", report, problems)
        cfg3, world3, _, _ = stage_setup(w, 3, MODEL_SEED)
        rp = rescore(ROOT / "plain" / f"w{w}" / "stage3", cfg3, world3, p["novel_probe_available"])
        harness &= rp["anchor"] <= TOL and p["library_sha256_at_start"] is None
        if abs(rp["median"] - p["terminal_median"]) > TOL:
            problems.append(f"plain w{w}: rescored {rp['median']} vs {p['terminal_median']}")
    g1 = record("gate", "G1", report, problems)
    shared = cells[(6, 0)]
    harness &= g1["library_sha256"] == shared["sha"] and all(
        abs(g1["terminal_per_task"][k] - v) <= TOL for k, v in shared["per_task"].items())
    harness &= all(len({cells[(w, s)]["sha"] for s in STREAMS}) == len(STREAMS) for w in WORLDS)

    margins = {}
    for w in WORLDS:
        pairs = [record("margin", f"w{w}_p{i}", report, problems) for i in range(PAIRS)]
        if sorted(p["index"] for p in pairs) != list(range(PAIRS)) or any(p["steps"] != 2000 for p in pairs):
            problems.append(f"w{w}: margin pairs incomplete or not at ADAPT_STEPS")
        scratch = math.exp(sum(math.log(max(p["scratch"], 1e-12)) for p in pairs) / PAIRS)
        trained = math.exp(sum(math.log(max(p["trained"], 1e-12)) for p in pairs) / PAIRS)
        margins[w] = math.log(scratch) - math.log(trained)
        harness &= math.isfinite(margins[w])

    W = {w: float(np.median(list(medians[w].values()))) for w in WORLDS}
    passing = [w for w in WORLDS if W[w] <= THRESHOLD]
    if not harness:
        label = "HARNESS_FAILED"
    else:
        terminal = len(passing) >= 3
        streams_ok = all(max(medians[w].values()) <= 2 * THRESHOLD for w in passing)  # SO4 Amendment 1
        margins_ok = sum(1 for w in WORLDS if margins[w] >= MARGIN) >= 3
        label = ("SO4_PASSES" if terminal and streams_ok and margins_ok else
                 "SO4_ACQUIRES_ONLY" if terminal and streams_ok else
                 "SO4_STREAM_FRAGILE" if terminal else "SO4_FAILS")
    recorded = report.get("classification", {})
    if recorded.get("program") != label or report.get("complete") is not True:
        problems.append(f"program {recorded.get('program')} vs recomputed {label}")
    for w in WORLDS:
        if abs(float(report["margins"][str(w)]) - margins[w]) > 1e-9:
            problems.append(f"w{w}: margin {report['margins'][str(w)]} vs recomputed {margins[w]}")
    print(json.dumps({"program": label, "harness": harness, "world_medians": W, "stream_medians": medians,
                      "margins": margins, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
