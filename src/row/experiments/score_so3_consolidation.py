"""Independent SO3 recomputation from durable records and saved models.

Reloads every saved stage-3 model and both stage models of every prefix,
re-scores every task with its own NMSE, recomputes library hashes, the transfer
chain, anchors, G2 non-vacuity and the frozen per-arm ladder and program label
without the runner's decision functions. G0 and G1 are re-checked from their
durable records against SO2's record and the shared cell. Exits nonzero on any
disagreement.
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

REPORT = Path("reports/so3_consolidation.json")
ROOT = Path("artifacts/so3_consolidation")
SO2_RESULT = Path("artifacts/so2_online_gate/cells/STAGED_w1/result.json")
WORLDS, STREAMS, MODEL_SEED, THRESHOLD, TOL = (3, 4, 5), (0, 1, 2), 6000, 0.05, 1e-6
CHANGES = {"BASE": None, "LR_HALF": ("global_learning_rate", 0.0005), "STORE_8": ("replay_examples_per_task", 8)}


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
    # G0 from durable gate records against SO2's record.
    so2 = json.loads(SO2_RESULT.read_text())["result"]["stages"]["3"]
    for variant in ("omitted", "explicit"):
        g = record("gate", f"G0_{variant}", report, problems)
        harness &= g["library_sha256"] == so2["library_sha256"] and g["worst_per_task_abs_diff"] <= TOL
    medians, cells = {a: {w: {} for w in WORLDS} for a in CHANGES}, {}
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
                if abs(r["median"] - prefix["stages"][str(stage)]["terminal_median"]) > TOL:
                    problems.append(f"prefix w{w}_s{s} stage {stage}: terminal {r['median']} vs recorded")
            harness &= prefix["stages"]["2"]["library_sha256_at_start"] == shas[1] and not ids[1] & ids[2]
            base_cfg, world3, _, _ = stage_setup(w, 3, MODEL_SEED)
            for arm, change in CHANGES.items():
                name = f"w{w}_s{s}_{arm}"
                c = record("cell", name, report, problems)
                cfg = base_cfg if change is None else replace(
                    base_cfg, discrete_model=replace(base_cfg.discrete_model, **{change[0]: change[1]}))
                r = rescore(ROOT / "cells" / name / "stage3", cfg, world3, c["novel_probe_available"])
                cells[name] = r
                medians[arm][w][s] = r["median"]
                harness &= r["anchor"] <= TOL and c["library_sha256_at_start"] == shas[2]
                harness &= not (r["ids"] & (ids[1] | ids[2]))
                if r["sha"] != c["library_sha256"]:
                    problems.append(f"{name}: saved model differs from record")
                if abs(r["median"] - c["terminal_median"]) > TOL or r["below"] != c["terminal_below"]:
                    problems.append(f"{name}: terminal {r['median']}/{r['below']} vs recorded "
                                    f"{c['terminal_median']}/{c['terminal_below']}")
                if getattr(cfg.discrete_model, "replay_examples_per_task") != c["training"]["replay_examples_per_task"] \
                        or cfg.discrete_model.global_learning_rate != c["training"]["global_learning_rate"]:
                    problems.append(f"{name}: resolved training fields differ from the registered arm")
    # G1 from its durable gate record against the shared cell.
    g1 = record("gate", "G1", report, problems)
    shared = cells["w3_s0_BASE"]
    harness &= g1["library_sha256"] == shared["sha"] and all(
        abs(g1["terminal_per_task"][k] - v) <= TOL for k, v in shared["per_task"].items())
    # G2 non-vacuity.
    harness &= all(len({cells[f"w{w}_s{s}_{a}"]["sha"] for a in CHANGES}) == 3 for w in WORLDS for s in STREAMS)
    harness &= all(len({cells[f"w{w}_s{s}_{a}"]["sha"] for s in STREAMS}) == 3 for w in WORLDS for a in CHANGES)

    wm = {a: {w: float(np.median(list(medians[a][w].values()))) for w in WORLDS} for a in CHANGES}
    labels = {}
    if not harness:
        program = "HARNESS_FAILED"
    else:
        for arm in ("LR_HALF", "STORE_8"):
            n_pass = sum(1 for w in WORLDS if wm[arm][w] <= THRESHOLD and wm[arm][w] < wm["BASE"][w])
            n_half = sum(1 for w in WORLDS if wm[arm][w] <= 0.5 * wm["BASE"][w])
            if arm == "STORE_8":
                n_robust = sum(1 for w in WORLDS if wm[arm][w] < min(medians["BASE"][w].values()))
                labels["STORE_8_STREAM_ROBUST"] = n_robust >= 2
                if n_pass >= 2:
                    labels[arm] = "STORE_8_PASSES" if n_robust >= 2 else "STORE_8_STREAM_CONFOUNDED"
                    continue
            labels[arm] = (f"{arm}_PASSES" if n_pass >= 2 else f"{arm}_PARTIAL" if n_half >= 2 else f"{arm}_FAILS")
        if "PASSES" in labels["LR_HALF"] or labels["STORE_8"] == "STORE_8_PASSES":
            program = "SO3_PASSES"
        elif labels["LR_HALF"].endswith("PARTIAL") or labels["STORE_8"] in ("STORE_8_PARTIAL",
                                                                              "STORE_8_STREAM_CONFOUNDED"):
            program = "SO3_PARTIAL"
        else:
            program = "SO3_FAILS"
    recorded = report.get("classification", {})
    if recorded.get("program") != program or report.get("complete") is not True:
        problems.append(f"program {recorded.get('program')} vs recomputed {program}")
    for key in ("LR_HALF", "STORE_8", "STORE_8_STREAM_ROBUST"):
        if key in labels and recorded.get(key) != labels[key]:
            problems.append(f"{key} {recorded.get(key)} vs recomputed {labels[key]}")
    print(json.dumps({"program": program, "labels": labels, "harness": harness, "world_medians": wm,
                      "stream_medians": medians, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
