"""Development-only H28-C oracle-core adapter pilot.

The core is supplied, the adapter family is restricted, and fitting uses only
support examples. This script is not a confirmatory runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from row.world import Primitive

ROOT = Path(__file__).resolve().parents[1]
VERSION = "h28-c-adapter-pilot-v0"
PLAN = ROOT / "H28_C_ADAPTER_PILOT_PLAN.md"
PROGRAMS = ((0, 1, 2), (5, 3, 1))
SUPPORT_OPS = (0, 1, 2, 3)
QUERY_OPS = (0, 1, 2, 3, 4, 5)
PLAN_STEPS = tuple(0.4 / (2 ** i) for i in range(10))
TOL = 1e-10


def givens(angles):
    matrix = np.eye(16)
    for angle, (i, j) in zip(angles, ((0, 1), (2, 3), (4, 5), (6, 7))):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.eye(16)
        rotation[i, i] = rotation[j, j] = c
        rotation[i, j], rotation[j, i] = -s, s
        matrix = rotation @ matrix
    return matrix


def fixture():
    library = tuple(Primitive.random(0, k, 16, 8, .35) for k in range(6))
    angles = np.random.default_rng(np.random.SeedSequence([0, 2821, 1])).uniform(-.7, .7, size=(2, 4))
    matrices = {"identity": np.eye(16), "context_1": givens(angles[0]), "context_2": givens(angles[1])}
    support_z = np.random.default_rng(np.random.SeedSequence([0, 2821, 2])).normal(size=(16, 16))
    query_z = np.random.default_rng(np.random.SeedSequence([0, 2821, 3])).normal(size=(32, 16))
    return library, matrices, {"support": support_z, "query": query_z}, angles


def observed_target(primitive, z, matrix):
    return primitive(z) @ matrix.T


def predict(primitive, observed, matrix):
    return primitive(observed @ matrix) @ matrix.T


def loss(prediction, target):
    return float(np.mean((prediction-target) ** 2))


def errors(library, matrices, data, angles):
    rows = []
    for context, matrix in matrices.items():
        for op in QUERY_OPS:
            target = observed_target(library[op], data["query"], matrix)
            fitted = predict(library[op], data["query"], matrix)
            rows.append({"context": context, "operation": op,
                         "canonical_query_mse": loss(fitted @ np.linalg.inv(matrix).T,
                                                       target @ np.linalg.inv(matrix).T),
                         "observed_query_mse": loss(fitted, target)})
    return rows


def fit_context(library, matrix, support_z, starts):
    support_observed = support_z @ matrix.T
    support = [(op, support_observed, observed_target(library[op], support_z, matrix)) for op in SUPPORT_OPS]
    evaluations = 0

    def objective(angles):
        nonlocal evaluations
        evaluations += 1
        candidate = givens(angles)
        return float(np.mean([loss(predict(library[op], x, candidate), y) for op, x, y in support]))

    best = None
    traces = []
    for start_index, initial in enumerate(starts):
        angles = np.asarray(initial, dtype=np.float64).copy()
        current = objective(angles)
        accepted = 0
        for step in PLAN_STEPS:
            for _ in range(4):
                for coordinate in range(4):
                    candidates = []
                    for direction in (0, -1, 1):
                        proposal = angles.copy() if direction == 0 else angles.copy()
                        if direction:
                            proposal[coordinate] = np.clip(proposal[coordinate] + direction * step, -np.pi/2, np.pi/2)
                        candidates.append((objective(proposal), direction, proposal))
                    candidates.sort(key=lambda item: (item[0], 0 if item[1] == 0 else 1 if item[1] == -1 else 2))
                    chosen_loss, direction, chosen = candidates[0]
                    if chosen_loss < current:
                        angles, current, accepted = chosen, chosen_loss, accepted + 1
        traces.append({"start": start_index, "angles": angles.tolist(), "support_mse": current,
                       "accepted_moves": accepted, "objective_evaluations": evaluations})
        if best is None or current < best["support_mse"]:
            best = {"angles": angles.copy(), "support_mse": current, "start": start_index}
    return best, traces, evaluations


def run(smoke=False):
    library, matrices, data, true_angles = fixture()
    names = ("identity",) if smoke else tuple(matrices)
    starts = (np.zeros(4), np.full(4, .4), np.full(4, -.4))
    records, fitting = [], {}
    for context in names:
        matrix = matrices[context]
        if context == "identity":
            fitted = {"angles": [0., 0., 0., 0.], "support_mse": 0., "start": 0}
            traces, evaluations = [], 0
        else:
            fitted, traces, evaluations = fit_context(library, matrix, data["support"], starts)
        fitting[context] = {"chosen": {"angles": fitted["angles"], "support_mse": fitted["support_mse"], "start": fitted["start"]},
                            "traces": traces, "objective_evaluations": evaluations}
        fitting[context]["chosen"]["angles"] = list(np.asarray(fitted["angles"], dtype=float))
        chosen_matrix = givens(fitted["angles"])
        for op in QUERY_OPS:
            target = observed_target(library[op], data["query"], matrix)
            observed_query = data["query"] @ matrix.T
            pred = predict(library[op], observed_query, chosen_matrix)
            oracle = predict(library[op], observed_query, matrix)
            baseline = predict(library[op], observed_query, np.eye(16))
            canonical_target = target @ np.linalg.inv(matrix).T
            energy = float(np.mean(canonical_target ** 2))
            records.append({"context": context, "operation": op, "support_operation": op in SUPPORT_OPS,
                            "query_canonical_nmse": loss(pred @ np.linalg.inv(matrix).T, canonical_target) / energy,
                            "oracle_canonical_nmse": loss(oracle @ np.linalg.inv(matrix).T, canonical_target) / energy,
                            "no_adapter_canonical_nmse": loss(baseline @ np.linalg.inv(matrix).T, canonical_target) / energy})
    return {"version": VERSION, "smoke": smoke, "contexts": names,
            "true_angles_not_exposed_to_fit": True,
            "true_angles_sha256": hashlib.sha256(np.asarray(true_angles).tobytes()).hexdigest(),
            "records": records, "fitting": fitting,
            "data_shapes": {key: list(value.shape) for key, value in data.items()},
            "core_changed": False, "query_used_for_fit": False,
            "economic_value_measured": False, "learned_core": False}


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-check", action="store_true", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve existing output: {args.output}")
    started = time.perf_counter()
    result = run(args.smoke)
    result["seconds"] = time.perf_counter() - started
    inputs = ("H28_C_ADAPTER_PILOT_PLAN.md", "tools/h28_adapter_pilot.py", "src/row/world.py")
    report = {"status": "PROVISIONAL_DEVELOPMENT_CHECK", "accepted_scientific_result": False,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
              "input_sha256": {path: digest(ROOT/path) for path in inputs}, "result": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "contexts": result["contexts"], "seconds": result["seconds"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
