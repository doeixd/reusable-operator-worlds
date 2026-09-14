"""Independent SO2 recomputation from durable cells and saved stage models.

Re-scores each arm's terminal model on the canonical world with an independent
NMSE, rebuilds the library-transfer chain from the saved stage models,
recomputes the export margin's geometric means from the recorded per-program
values (the adaptations themselves are not repeated), and recomputes the
ordered classification without the runner's decision function.
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
from row.experiments.audit_so1_budget_bracket import build_fast
REPORT = Path("reports/so2_online_gate.json")
CELLS = Path("artifacts/so2_online_gate/cells")
WORLDS = (0, 1, 2)
THRESHOLD, MARGIN, MODEL_SEED = 0.05, 0.75, 5000


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def library_sha(model) -> str:
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()


def geo(values) -> float:
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))


def restore_stage(path: Path, cfg, world, probe_available: bool):
    """Rebuild a saved stage model with ALL of its state.

    The lifetime's terminal novel-composition probe (a diagnostic run after the
    terminal metrics, training only its own code) leaves
    `task_novel_composition_0` in the model wherever an unseen program exists.
    A strict load must register that code too, or the saved stage-3 model
    cannot be reconstructed.
    """
    model = build_fast(cfg)
    for task in world.tasks:
        model.begin_task(task.task_id)
    if probe_available:
        model.begin_task("task_novel_composition_0")
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True), strict=True)
    state = json.loads((path / "model_state.json").read_text())
    if set(state["requires_grad"]) != set(dict(model.named_parameters())):
        raise ValueError(f"{path}: incomplete requires_grad state")
    model.temperature = state["temperature"]
    return model


def main() -> int:
    torch.set_num_threads(1)
    report = json.loads(REPORT.read_text())
    problems, medians, margins, harness = [], {}, {}, True
    for arm in ("STAGED", "PLAIN"):
        for w in WORLDS:
            directory = CELLS / f"{arm}_w{w}"
            stored = json.loads((directory / "result.json").read_text())
            result = stored["result"]
            if sha_json(result) != stored["result_sha256"] or stored["stamp"]["git_commit"] != report["git_commit"]:
                problems.append(f"{arm}_w{w}: record hash/commit mismatch")
            harness &= result["model_seed"] == MODEL_SEED
            harness &= all(s["anchor_abs_error"] <= 1e-6 for s in result["stages"].values())
            shas = {}
            for stage in sorted(int(s) for s in result["stages"]):
                cfg, world, _, _ = stage_setup(w, stage, MODEL_SEED)
                model = restore_stage(directory / f"stage{stage}", cfg, world,
                                      result["stages"][str(stage)]["novel_probe_available"])
                shas[stage] = library_sha(model)
                if shas[stage] != result["stages"][str(stage)]["library_sha256"]:
                    problems.append(f"{arm}_w{w}: stage {stage} saved model differs from record")
                if stage == 3:
                    model.eval()
                    errors = []
                    with torch.no_grad():
                        for task in world.tasks:
                            y = np.asarray(task.eval_y, dtype=np.float64)
                            p = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id).numpy().astype(np.float64)
                            errors.append(np.mean((y - p) ** 2) / np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))
                    medians[(arm, w)] = float(np.median(errors))
            if arm == "STAGED":
                for stage in (2, 3):
                    harness &= result["stages"][str(stage)]["library_sha256_at_start"] == shas[stage - 1]
            else:
                harness &= result["stages"]["3"]["library_sha256_at_start"] is None
            rows = result["margin"]["rows"]
            recomputed = math.log(geo([r["scratch"] for r in rows])) - math.log(geo([r["trained"] for r in rows]))
            if abs(recomputed - result["margin"]["margin"]) > 1e-9:
                problems.append(f"{arm}_w{w}: margin arithmetic {result['margin']['margin']} vs {recomputed}")
            if len(rows) != report["protocol"]["held_out"]:
                problems.append(f"{arm}_w{w}: expected {report['protocol']['held_out']} held-out programs")
            margins[(arm, w)] = recomputed
            harness &= all(math.isfinite(v) for v in (result["terminal_median"], recomputed))
    staged_median = {w: medians[("STAGED", w)] for w in WORLDS}
    staged_margin = {w: margins[("STAGED", w)] for w in WORLDS}
    acquires = sum(staged_median[w] <= THRESHOLD for w in WORLDS) >= 2
    exports = sum(staged_margin[w] >= MARGIN for w in WORLDS) >= 2
    label = ("HARNESS_FAILED" if not harness else
             "SO2_PASSES" if acquires and exports else
             "SO2_ACQUIRES_ONLY" if acquires else "SO2_FAILS")
    for (arm, w), value in medians.items():
        recorded = json.loads((CELLS / f"{arm}_w{w}" / "result.json").read_text())["result"]["terminal_median"]
        if abs(value - recorded) > 1e-6:
            problems.append(f"{arm}_w{w}: re-scored terminal {value} vs recorded {recorded}")
    if report.get("classification") != label or report.get("complete") is not True:
        problems.append(f"classification {report.get('classification')} vs recomputed {label}")
    print(json.dumps({"classification": label, "harness": harness,
                      "staged_terminal": staged_median, "staged_margin": staged_margin,
                      "plain_terminal": {w: medians[("PLAIN", w)] for w in WORLDS},
                      "plain_margin": {w: margins[("PLAIN", w)] for w in WORLDS},
                      "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
