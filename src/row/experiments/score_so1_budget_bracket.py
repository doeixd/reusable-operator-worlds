"""Independent SO1 scorer: no imports from the runner's decision functions."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from row.config import load_config
from row.models import FastRotatedDiscreteLibraryLearner
from row.rotated_world import generate_rotated_world
from row.experiments.so1_storage import (
    atomic_json, digest, fingerprint, resolved, restore_model, world_digest,
)

LEVELS = (16384, 32768, 65536, 131072, 262144)
WORLDS = ("0", "1", "2")
THRESHOLD = 0.05
V2, V3 = "SO1-budget-bracket-v2-restart", "SO1-budget-bracket-v3-relaunch"
V2_INPUTS = {"SO1_BUDGET_BRACKET_PLAN.md", "SO1_RESTART_AMENDMENT.md", "reports/rotated_g5r_interference.json"}
V3_INPUTS = V2_INPUTS | {"SO1_ANCHOR_DIAGNOSTIC_AMENDMENT.md", "reports/so1_anchor_diagnostic.json",
                         "reports/so1_budget_bracket.json"}
CORNERS = ("O_b2_g16384", "O_b64_g262144")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "non-finite metric")


def persistence_from_scores(checkpoints):
    points = sorted((int(k), v["median"]) for k, v in checkpoints.items())
    crossings = [i for i, (_, value) in enumerate(points) if value <= THRESHOLD]
    if not crossings:
        return None, "not crossed"
    first = crossings[0]
    after = points[first + 1:first + 3]
    label = ("crossed, persistence unobservable" if len(after) < 2 else
             "persistent" if max(v for _, v in after) <= THRESHOLD else "crossed, not persistent")
    return points[first][0], label


def summarize_cells(cells):
    """Independent estimands; called only after a valid complete oracle grid."""
    passes = {k: sum(c["terminal_median"] <= THRESHOLD for c in worlds.values()) >= 2
              for k, worlds in cells.items()}
    envelopes = {}
    for b in (2, 64):
        passing, persistent = [], []
        for g in LEVELS:
            key = f"O_b{b}_g{g}"
            if passes[key]:
                passing.append(g)
            if sum(cells[key][w]["terminal_median"] <= THRESHOLD and
                   persistence_from_scores(cells[key][w]["checkpoints"])[1] == "persistent"
                   for w in WORLDS) >= 2:
                persistent.append(g)
        envelopes[str(b)] = {"lowest_passing": min(passing, default=None),
                             "lowest_passing_excluding_unobservable": min(persistent, default=None)}
    pairs = {str(g): {w: cells[f"O_b2_g{g}"][w]["terminal_median"] -
                        cells[f"O_b64_g{g}"][w]["terminal_median"] for w in WORLDS} for g in LEVELS}
    monotonic = {str(b): {w: all(cells[f"O_b{b}_g{a}"][w]["terminal_median"] >=
                                cells[f"O_b{b}_g{z}"][w]["terminal_median"]
                                for a, z in zip(LEVELS, LEVELS[1:])) for w in WORLDS} for b in (2, 64)}
    learned_keys = [f"L_b{b}_g{envelopes[str(b)]['lowest_passing']}" for b in (2, 64)
                    if envelopes[str(b)]["lowest_passing"] is not None]
    classification = ("NO_ORACLE_CELL_PASSES" if not learned_keys else
                      "ORACLE_AND_LEARNED_PASS" if any(passes[k] for k in learned_keys) else
                      "ORACLE_PASSES_LEARNED_FAILS")
    predictions = {
        "P1": envelopes["64"]["lowest_passing"] == 262144,
        "P2": envelopes["2"]["lowest_passing"] is not None and envelopes["2"]["lowest_passing"] <= 131072,
        "P3": all(sum(v < 0 for v in pairs[str(g)].values()) >= 2 for g in LEVELS[:3]),
        "P4": all(cells[f"O_b64_g{g}"]["0"]["terminal_median"] > THRESHOLD for g in LEVELS),
        "P5": all(persistence_from_scores(cells[f"O_b{b}_g{g}"][w]["checkpoints"])[1] != "persistent"
                  for b in (2, 64) for g in LEVELS[:3] for w in WORLDS),
        "stage2_fails_all_worlds": None if not learned_keys else
            all(cells[k][w]["terminal_median"] > THRESHOLD for k in learned_keys for w in WORLDS),
    }
    return {"envelope": envelopes, "paired_differences_b2_minus_b64": pairs,
            "dose_monotonicity": monotonic, "cell_passes": passes,
            "classification": classification, "predictions": predictions}


def builder(cfg):
    m = cfg.discrete_model
    return FastRotatedDiscreteLibraryLearner(
        d=cfg.world.state_dim, operator_slots=m.operator_slots, operator_rank=m.operator_rank,
        task_steps=m.task_steps, alpha=m.operator_alpha_init,
        initial_temperature=m.initial_temperature, final_temperature=m.final_temperature,
        seed=m.seed, learnable_alpha=m.learnable_alpha, activation=m.operator_activation)


def validate_cell(cell, path, stamp, cfg, world):
    finite_tree(cell)
    stored = json.loads((path / "result.json").read_text())
    require(stored["complete"] is True and stored["stamp"] == stamp, "cell stamp mismatch")
    require(stored["result"] == cell and stored["result_sha256"] == fingerprint(cell), "result digest mismatch")
    require(set(stored["artifact_sha256"]) == {"model.pt", "model_state.json", "config.yaml", "fingerprint.json"}, "missing artifact")
    for name, sha in stored["artifact_sha256"].items():
        require(digest(path / name) == sha, f"artifact hash mismatch: {path / name}")
    require(yaml.safe_load((path / "config.yaml").read_text()) == resolved(cfg), "resolved config mismatch")
    manifest = json.loads((path / "fingerprint.json").read_text())
    require(manifest == {"stamp": stamp, "resolved": resolved(cfg)}, "manifest mismatch")
    require(cell["world_sha256"] == world_digest(world), "fixed world arrays/order changed")
    require(cell["started_from_seed"] == cfg.discrete_model.seed, "wrong model seed")
    u = stamp["updates"]
    wanted = {str(c) for c in (0, u // 8, u // 4, u // 2, 3 * u // 4, u)}
    require(set(cell["checkpoints"]) == wanted, "checkpoint count/locations differ")
    ids = {t.task_id for t in world.tasks}
    for checkpoint in cell["checkpoints"].values():
        require(set(checkpoint["per_task"]) == ids, "checkpoint task set differs")
        values = list(checkpoint["per_task"].values())
        for label, value in (("mean", np.mean(values)), ("median", np.median(values)),
                             ("minimum", min(values)), ("maximum", max(values))):
            require(checkpoint[label] == value, f"checkpoint {label} mismatch")
        for threshold in (0.02, 0.05, 0.1):
            require(checkpoint[f"below_{threshold}"] == sum(v <= threshold for v in values), "threshold count mismatch")
    require(cell["final_per_task"] == cell["checkpoints"][str(u)]["per_task"], "terminal task scores differ")
    require(cell["terminal_median"] == float(np.median(list(cell["final_per_task"].values()))), "terminal median mismatch")
    require(cell["finite"] is True and cell["shared_relative_change"] > 0 and cell["reload_exact"] is True,
            "learning/reload non-vacuity failure")
    require(cell["passes"] == (cell["terminal_median"] <= THRESHOLD), "pass flag mismatch")
    require((cell["first_crossing"], cell["persistence"]) == persistence_from_scores(cell["checkpoints"]), "persistence mismatch")
    trajectory = [c["median"] for _, c in sorted(cell["checkpoints"].items(), key=lambda x: int(x[0]))]
    require(cell["monotone"] == all(b <= a for a, b in zip(trajectory, trajectory[1:])), "trajectory monotonicity mismatch")
    require(cell["oracle_routes"] == stamp["oracle"] and cell["updates"] == u and cell["batch"] == stamp["batch"], "cell construction mismatch")
    require(cell["cell_index"] == stamp["cell_index"] and cell["sampling_index"] == stamp["sampling_index"], "sampling stream mismatch")
    require(cell["gradients"] == u * stamp["batch"] and cell["implementation"] == "batched_rotation_v1", "budget/implementation mismatch")
    model = restore_model(path, cfg, world, builder)
    require(model.temperature == cfg.discrete_model.final_temperature and not model.training, "terminal runtime state mismatch")
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "non-finite model state")
    if stamp["oracle"]:
        require(cell["pinned_routes_preserved"] is True and cell["pinned_one_hot_at_1.0"] is True, "oracle pinning failed")
        # Reconstruct canonical same-determinant slot assignment independently.
        available = {sign: [s for s in range(cfg.discrete_model.operator_slots)
                           if (-1 if (cfg.world.state_dim - ((cfg.discrete_model.seed + 997 * s) & 1)) % 2 else 1) == sign]
                     for sign in (-1, 1)}
        mapping = {i: available[1 if np.linalg.det(p.Q) > 0 else -1].pop(0) for i, p in enumerate(world.library)}
        for task in world.tasks:
            code = model.task_codes[task.task_id]
            require(not code.requires_grad, "oracle code trainable")
            route = tuple(mapping[int(p)] for p in task.program.primitive_ids)
            require(tuple(code.argmax(-1).tolist()) == route, "saved oracle route mismatch")
            target = torch.full_like(code, -100)
            for position, slot in enumerate(route):
                target[position, slot] = 100
            require(torch.equal(code, target), "saved oracle logits not pinned")
    else:
        require(cell["code_relative_change"] > 0 and all(c.requires_grad for c in model.task_codes.values()), "learned code non-vacuity failure")
    with torch.no_grad():
        for task in world.tasks:
            prediction = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id).numpy()
            y = np.asarray(task.eval_y, dtype=np.float64)
            error = float(np.mean((y - prediction) ** 2))
            variance = float(np.mean((y - np.mean(y, axis=0, keepdims=True)) ** 2))
            require(error / variance == cell["final_per_task"][task.task_id], "independent model NMSE mismatch")
    return stored["finished_utc"]


def validate_report(report_path, config_path, exit_path):
    torch.set_num_threads(1)
    report = json.loads(report_path.read_text())
    finite_tree(report)
    p = report["protocol"]
    require(p["id"] in (V2, V3), "wrong protocol")
    v3 = p["id"] == V3
    if v3:
        require(p["updates_divisor"] == 1, "smoke protocol is never scientific")
    require(p["worlds"] == [0, 1, 2] and p["batches"] == [2, 64] and p["gradient_levels"] == list(LEVELS), "registered grid changed")
    require(p["threshold"] == THRESHOLD and p["anchor_tolerance"] == 0.02 and p["worlds_required"] == 2
            and p["persistence_later_checkpoints"] == 2, "registered thresholds changed")
    require(report["protocol_sha256"] == fingerprint(p), "protocol digest mismatch")
    for path, sha in p["input_sha256"].items():
        require(digest(path) == sha, f"input changed: {path}")
    require(set(p["input_sha256"]) == (V3_INPUTS if v3 else V2_INPUTS), "missing protocol inputs")
    base = load_config(config_path)
    configs = {w: replace(base, world=replace(base.world, seed=int(w))) for w in WORLDS}
    require(p["resolved_configs"] == {w: resolved(c) for w, c in configs.items()}, "full resolved configuration changed")
    launch = report["launch"]
    require(digest(launch["manifest_path"]) == launch["manifest_sha256"], "run manifest hash mismatch")
    manifest = json.loads(Path(launch["manifest_path"]).read_text())
    require(manifest["git_commit"] == report["git_commit"] and manifest["protocol_sha256"] == report["protocol_sha256"]
            and manifest["started_utc"] == report["started_utc"], "run manifest mismatch")
    require(digest(launch["gate_path"]) == launch["gate_sha256"], "gate hash mismatch")
    gate = json.loads(Path(launch["gate_path"]).read_text())
    require(gate["git_commit"] == report["git_commit"] and gate["protocol_sha256"] == report["protocol_sha256"], "stale gate")
    require(gate["gate"] == "PASS" and gate["memory_passes"] is True and gate["bitwise_identical_cells"] == gate["expected_cells"] == 9,
            "pool gate failed/incomplete")
    require(gate["serial"] == gate["pooled"] and len(gate["serial"]) == 9, "pool equivalence mismatch")
    exit_record = json.loads(exit_path.read_text())
    require(exit_record["git_commit"] == report["git_commit"], "wrong exit record")
    started, finished = map(datetime.fromisoformat, (report["started_utc"], report["finished_utc"]))
    require(finished >= started and datetime.fromisoformat(exit_record["finished_utc"]) >= finished, "stale exit/report")
    anchor_only = not v3 and report.get("classification") == "ANCHOR_FAILED_NOTHING_READ"
    cells = report["cells"]
    oracle_keys = {f"O_b{b}_g{g}" for b in (2, 64) for g in LEVELS}
    anchors = {"O_b2_g16384": "C_lo", "O_b64_g262144": "C_hi"}
    require(set(cells) == set(anchors) if anchor_only else oracle_keys <= set(cells), "expected oracle cell set missing")
    artifact_root = Path(launch["artifact_root"])
    worlds = {w: generate_rotated_world(configs[w].world) for w in WORLDS}
    count = 0
    for key, rows in cells.items():
        match = re.fullmatch(r"([OL])_b(2|64)_g(\d+)", key)
        require(match is not None, "unexpected cell key")
        oracle = match[1] == "O"
        b, g = int(match[2]), int(match[3])
        require(g in LEVELS and set(rows) == set(WORLDS), "incomplete cell worlds/budget")
        sampling_index = 100 + (0 if b == 2 else 5) + LEVELS.index(g)
        index = sampling_index if oracle else (110 if b == 2 else 111)
        for w, cell in rows.items():
            stamp = {"key": key, "world": int(w), "oracle": oracle, "updates": g // b, "batch": b,
                     "cell_index": index, "sampling_index": sampling_index, "git_commit": report["git_commit"],
                     "protocol_sha256": report["protocol_sha256"]}
            timestamp = validate_cell(cell, artifact_root / key / f"world_{w}", stamp, configs[w], worlds[w])
            require(started <= datetime.fromisoformat(timestamp) <= finished, "artifact freshness failed")
            count += 1
    stage_d = json.loads(Path("reports/rotated_g5r_interference.json").read_text())
    computed_anchors = {}
    for key, name in anchors.items():
        rows = {}
        for w in WORLDS:
            a = cells[key][w]["terminal_median"]
            b = stage_d["cells"][name][w]["terminal_median"]
            same = (a <= THRESHOLD) == (b <= THRESHOLD)
            rows[w] = {"so1": a, "stage_d": b, "abs_error": abs(a-b), "same_verdict": same,
                       "passes": abs(a-b) <= 0.02 and same}
        computed_anchors[key] = {"stage_d_cell": name, "worlds": rows, "passes": all(r["passes"] for r in rows.values())}
    extra = {}
    if v3:
        # Relaunch anchor: the matched-stream diagnostic, re-read independently.
        diag = json.loads(Path("reports/so1_anchor_diagnostic.json").read_text())
        require(diag["complete"] is True and diag["classification"] == "IMPLEMENTATION_EQUIVALENT"
                and diag["frozen_plan"] == "SO1_ANCHOR_DIAGNOSTIC_AMENDMENT.md", "diagnostic anchor not satisfied")
        errors = {w: max(r["d_impl0"], r["d_impl100"]) for w, r in diag["per_world"].items()}
        require(all(e <= 0.02 for e in errors.values()) and max(r["d_repro"] for r in diag["per_world"].values()) <= 1e-6,
                "diagnostic numbers do not support its classification")
        require(report["anchor"] == {"source": "reports/so1_anchor_diagnostic.json",
                                     "sha256": digest("reports/so1_anchor_diagnostic.json"),
                                     "diagnostic_commit": diag["git_commit"], "classification": diag["classification"],
                                     "matched_stream_abs_error": errors, "passes": True}, "anchor record mismatch")
        require(report["cross_stream_corner_comparison"] == computed_anchors, "corner comparison arithmetic mismatch")
        require(report["resampling_spread_disclosure"]["per_world"] ==
                {w: {"range": r["spread"], "sd": r["fast_sd"]} for w, r in diag["per_world"].items()}, "spread disclosure mismatch")
        first = json.loads(Path("reports/so1_budget_bracket.json").read_text())["cells"]
        extra["corner_reproduction_of_first_attempt"] = {
            key: {w: cells[key][w]["final_per_task"] == first[key][w]["final_per_task"] for w in WORLDS}
            for key in CORNERS}
    else:
        require(report["anchor"] == computed_anchors, "anchor arithmetic mismatch")
    if anchor_only:
        require(not all(a["passes"] for a in computed_anchors.values()), "false anchor failure")
        require(report["complete"] is False and exit_record["exit_code"] != 0, "failed anchor must not look complete")
        require(not any(k in report for k in ("envelope", "paired_differences_b2_minus_b64", "dose_monotonicity")), "analysis crossed failed anchor gate")
        return {"status": "VERIFIED_INSTRUMENT_GATE_FAILURE", "scientific_result_accepted": False,
                "verified_cells": count, "anchor": computed_anchors}
    require(report["complete"] is True and exit_record["exit_code"] == 0, "run not successfully complete")
    if not v3:
        require(all(a["passes"] for a in computed_anchors.values()), "anchor gate failed")
    estimates = summarize_cells(cells)
    expected_learned = {f"L_b{b}_g{estimates['envelope'][str(b)]['lowest_passing']}" for b in (2, 64)
                        if estimates["envelope"][str(b)]["lowest_passing"] is not None}
    require(set(cells) == oracle_keys | expected_learned, "unexpected learned cells")
    for key in estimates.keys() - {"predictions"}:
        require(report[key] == estimates[key], f"independent {key} mismatch")
    return {"status": "VERIFIED_COMPLETE", "scientific_result_accepted": True,
            "verified_cells": count, **estimates, **extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/so1_budget_bracket.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--exit-record", type=Path, default=Path("artifacts/so1_restart/exit.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/so1_restart/independent_score.json"))
    args = parser.parse_args()
    result = validate_report(args.report, args.config, args.exit_record)
    result["report_sha256"] = digest(args.report)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
