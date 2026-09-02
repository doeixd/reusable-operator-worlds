"""RF0b: oracle ceiling for semantic canonicalization of E6 motifs.

Refit RF0a's shallow raw-symbol decoder, apply it at depth 8/10, and ask
whether all three positions of the planted teacher motif are jointly recovered.
No model, route, or world is trained.  See ``RF0B_SEMANTIC_MOTIF_PLAN.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e5_synthesizer import fatal
from row.experiments.audit_e6_corpus import ngrams, plant_corpus
from row.experiments.audit_e6a_macro_economics import (
    CORPUS,
    DEPTHS,
    PLANT_FRACTION,
    PRIMARY_L,
    WORLDS,
    motif_survival,
)
from row.experiments.audit_rf0a_semantic_recoverability import (
    ARMS,
    CLASSES,
    DISPLAYED_E6_MEANS,
    E6A_REPORT,
    LAMBDAS,
    NULL_DRAWS,
    OccurrenceData,
    feature_matrix,
    fit_ridge,
    load_e6a_route,
    permute_within_position,
    reproduce_e6,
    role_vector,
    select_probe,
    sha256_file,
    validate_cache_completeness,
)

PLAN = Path("RF0B_SEMANTIC_MOTIF_PLAN.md")
RF0A_REPORT = Path("reports/rf0a_semantic_recoverability.json")
DEFAULT_OUTPUT = Path("reports/rf0b_semantic_motif.json")
ABSOLUTE_GATE = 0.30
MATERIALITY = 0.20
FAIL_ABSOLUTE = 0.15
FAIL_DELTA = 0.10
ROLE_GAP = 0.20


def empty_occurrences(programs, routes, carries, sites, depth: int) -> OccurrenceData:
    labels, symbols, roles, positions, tasks, planted = [], [], [], [], [], []
    for task_index, (program, route, carry, site) in enumerate(zip(
        programs, routes, carries, sites, strict=True
    )):
        for position, (label, symbol) in enumerate(zip(program, route, strict=True)):
            labels.append(int(label))
            symbols.append(int(symbol))
            roles.append(role_vector(position, depth))
            positions.append(position)
            tasks.append(task_index)
            planted.append(bool(carry and site <= position < site + PRIMARY_L))
    rows = len(labels)
    return OccurrenceData(
        labels=np.asarray(labels, dtype=np.int64),
        raw_symbols=np.asarray(symbols, dtype=np.int64),
        roles=np.asarray(roles, dtype=np.float64),
        fingerprints=np.zeros((rows, 1), dtype=np.float64),
        neighbours=np.zeros((rows, 0), dtype=np.float64),
        native=np.zeros((rows, 0), dtype=np.float64),
        trace=np.zeros((rows, 0), dtype=np.float64),
        positions=np.asarray(positions, dtype=np.int64),
        task_indices=np.asarray(tasks, dtype=np.int64),
        planted=np.asarray(planted, dtype=bool),
    )


def regenerate_world_depth(world: int, depth: int, world_config: dict) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([970, world, depth]))
    motif, programs, carries, sites = plant_corpus(
        rng,
        int(world_config["teacher_primitives"]),
        CORPUS,
        depth,
        PRIMARY_L,
        int(CORPUS * PLANT_FRACTION),
    )
    routes = [load_e6a_route(world, depth, index) for index in range(CORPUS)]
    data = empty_occurrences(programs, routes, carries, sites, depth)
    return {
        "motif": tuple(int(value) for value in motif),
        "programs": programs,
        "carries": carries,
        "sites": sites,
        "routes": routes,
        "data": data,
    }


def fit_arm(data: dict[int, dict], arm: str):
    train, valid = data[4]["data"], data[6]["data"]
    x_train, x_valid = feature_matrix(train, arm), feature_matrix(valid, arm)
    selected, path = select_probe(x_train, train.labels, x_valid, valid.labels)
    model = fit_ridge(
        np.concatenate([x_train, x_valid]),
        np.concatenate([train.labels, valid.labels]),
        selected,
    )
    predictions = {
        depth: model.predict(feature_matrix(data[depth]["data"], arm))
        for depth in (8, 10)
    }
    return selected, path, model, predictions


def phi(left: np.ndarray, right: np.ndarray) -> float | None:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denominator = float(left.std() * right.std())
    if denominator == 0.0:
        return None
    return float(np.mean((left - left.mean()) * (right - right.mean())) / denominator)


def motif_score(regenerated: dict, flat_predictions: np.ndarray) -> dict:
    depth = len(regenerated["routes"][0])
    predictions = np.asarray(flat_predictions, dtype=np.int64).reshape(CORPUS, depth)
    labels = np.asarray(regenerated["programs"], dtype=np.int64)
    planted_indices = [index for index, carry in enumerate(regenerated["carries"]) if carry]
    unplanted_indices = [index for index, carry in enumerate(regenerated["carries"]) if not carry]
    if len(planted_indices) != 64 or len(unplanted_indices) != 64:
        raise ValueError("RF0b requires 64 planted and 64 unplanted tasks")
    correctness, decoded = [], []
    for index in planted_indices:
        site = int(regenerated["sites"][index])
        if not 0 <= site <= depth - PRIMARY_L:
            raise ValueError("invalid planted site")
        predicted = predictions[index, site:site + PRIMARY_L]
        target = labels[index, site:site + PRIMARY_L]
        correctness.append(predicted == target)
        decoded.append(tuple(int(value) for value in predicted))
    correct = np.asarray(correctness, dtype=bool)
    exact = np.all(correct, axis=1)
    marginal = correct.mean(axis=0)
    product = float(np.prod(marginal))
    tally = Counter(decoded)
    modal, modal_count = tally.most_common(1)[0]
    motif = tuple(int(value) for value in regenerated["motif"])
    false_occurrences = []
    for index in unplanted_indices:
        grams = set(ngrams(predictions[index].tolist(), PRIMARY_L))
        false_occurrences.append(motif in grams)
    return {
        "planted_tasks": len(planted_indices),
        "unplanted_tasks": len(unplanted_indices),
        "exact_recovery": float(exact.mean()),
        "exact_count": int(exact.sum()),
        "relative_position_accuracy": marginal.tolist(),
        "independence_product": product,
        "exact_minus_independence": float(exact.mean() - product),
        "pairwise_phi": {
            "0_1": phi(correct[:, 0], correct[:, 1]),
            "0_2": phi(correct[:, 0], correct[:, 2]),
            "1_2": phi(correct[:, 1], correct[:, 2]),
        },
        "modal_decoded_motif": list(modal),
        "modal_count": int(modal_count),
        "modal_recurrence": float(modal_count / len(decoded)),
        "modal_equals_teacher_motif": bool(modal == motif),
        "unplanted_teacher_motif_occurrence": float(np.mean(false_occurrences)),
    }


def reproduce_rf0a_arm(data: dict[int, dict], arm: str, rf0a_world: dict) -> dict:
    from row.experiments.audit_rf0a_semantic_recoverability import score_probe

    record = score_probe(
        data[4]["data"], data[6]["data"],
        {8: data[8]["data"], 10: data[10]["data"]}, arm,
    )
    expected = rf0a_world["arms"][arm]
    fatal(record == expected, f"RF0a {arm} arm did not reproduce exactly")
    return record


def permutation_null(data: dict[int, dict], world: int) -> dict:
    train, valid = data[4]["data"], data[6]["data"]
    x_train, x_valid = feature_matrix(train, "Z"), feature_matrix(valid, "Z")
    x_refit = np.concatenate([x_train, x_valid])
    x_deep = {depth: feature_matrix(data[depth]["data"], "Z") for depth in (8, 10)}
    draws = {8: [], 10: []}
    for draw in range(NULL_DRAWS):
        rng = np.random.default_rng(np.random.SeedSequence([986, world, draw]))
        train_labels = permute_within_position(train.labels, train.positions, rng)
        valid_labels = permute_within_position(valid.labels, valid.positions, rng)
        selected, _ = select_probe(x_train, train_labels, x_valid, valid_labels)
        model = fit_ridge(
            x_refit, np.concatenate([train_labels, valid_labels]), selected
        )
        for depth in (8, 10):
            predictions = model.predict(x_deep[depth])
            draws[depth].append(motif_score(data[depth], predictions)["exact_recovery"])
    return {
        str(depth): {
            "draws": values,
            "mean": float(np.mean(values)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
        for depth, values in draws.items()
    }


def synthetic_controls() -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([987, 0]))
    motifs = rng.integers(0, CLASSES, size=(600, PRIMARY_L))
    correct = motifs.copy()
    shuffled = rng.permutation(motifs.reshape(-1)).reshape(motifs.shape)
    positive = float(np.mean(np.all(correct == motifs, axis=1)))
    negative = float(np.mean(np.all(shuffled == motifs, axis=1)))
    return {
        "positive_exact_recovery": positive,
        "negative_exact_recovery": negative,
        "passed": bool(positive == 1.0 and negative <= 0.10),
    }


def classify(worlds: dict[str, dict], controls: dict) -> dict:
    scoreable = [int(world) for world, result in worlds.items()
                 if result["structurally_scoreable"]]
    if len(scoreable) < 2 or not controls["passed"]:
        return {"classification": "RF0B UNSCOREABLE", "scoreable_worlds": scoreable}

    def full_gate(world: int, depth: int) -> bool:
        cell = worlds[str(world)]["depths"][str(depth)]
        return bool(
            cell["Z"]["exact_recovery"] >= ABSOLUTE_GATE
            and cell["delta_sem"] >= MATERIALITY
            and cell["Z_minus_R_exact"] >= ROLE_GAP
            and cell["permutation_pass"]
        )

    passing = {
        depth: [world for world in scoreable if full_gate(world, depth)]
        for depth in (8, 10)
    }
    both = [world for world in scoreable if full_gate(world, 8) and full_gate(world, 10)]
    if len(both) >= 2:
        return {"classification": "SEMANTIC CANONICALIZATION CEILING EXISTS",
                "passing_worlds": both, "passing_by_depth": passing,
                "scoreable_worlds": scoreable}
    if len(passing[8]) >= 2 and len(passing[10]) < 2:
        return {"classification": "SEMANTIC MOTIF HORIZON BETWEEN 8 AND 10",
                "passing_by_depth": passing, "scoreable_worlds": scoreable}
    failed = [world for world in scoreable if all(
        worlds[str(world)]["depths"][str(depth)]["Z"]["exact_recovery"] < FAIL_ABSOLUTE
        and worlds[str(world)]["depths"][str(depth)]["delta_sem"] < FAIL_DELTA
        for depth in (8, 10)
    )]
    if len(failed) >= 2:
        return {"classification": "LOCAL DECODABILITY DOES NOT RESTORE MOTIFS",
                "failing_worlds": failed, "passing_by_depth": passing,
                "scoreable_worlds": scoreable}
    return {"classification": "RF0B UNRESOLVED", "passing_by_depth": passing,
            "scoreable_worlds": scoreable}


def audit_world(world: int, rf0a: dict, e6_report: dict) -> dict:
    artifact = Path("artifacts/e1_disc") / f"world_{world}"
    raw_config = yaml.safe_load((artifact / "config.yaml").read_text(encoding="utf-8"))
    data = {
        depth: regenerate_world_depth(world, depth, raw_config["world"])
        for depth in DEPTHS
    }
    e6 = {
        str(depth): reproduce_e6(world, depth, data[depth], e6_report)
        for depth in DEPTHS
    }
    rf0a_z = reproduce_rf0a_arm(data, "Z", rf0a["worlds"][str(world)])
    rf0a_r = reproduce_rf0a_arm(data, "R", rf0a["worlds"][str(world)])
    z_lambda, _, _, z_predictions = fit_arm(data, "Z")
    r_lambda, _, _, r_predictions = fit_arm(data, "R")
    fatal(z_lambda == rf0a_z["selected_lambda"], "Z lambda mismatch after reproduction")
    fatal(r_lambda == rf0a_r["selected_lambda"], "R lambda mismatch after reproduction")
    null = permutation_null(data, world)
    depths = {}
    for depth in (8, 10):
        z_score = motif_score(data[depth], z_predictions[depth])
        r_score = motif_score(data[depth], r_predictions[depth])
        literal = e6[str(depth)]["survival_rate"]
        depths[str(depth)] = {
            "teacher_motif": list(data[depth]["motif"]),
            "literal_E6_survival": literal,
            "Z": z_score,
            "R": r_score,
            "delta_sem": z_score["exact_recovery"] - literal,
            "Z_minus_R_exact": z_score["exact_recovery"] - r_score["exact_recovery"],
            "permutation_null": null[str(depth)],
            "permutation_pass": bool(
                z_score["exact_recovery"] > null[str(depth)]["p99"]
            ),
        }
    coverage = all(
        sorted(int(value) for value in np.unique(data[depth]["data"].labels))
        == list(range(CLASSES)) for depth in (4, 6)
    )
    structural = bool(
        coverage and all(check["passed"] for check in e6.values())
        and all(cell["Z"]["planted_tasks"] == 64 and cell["Z"]["unplanted_tasks"] == 64
                for cell in depths.values())
    )
    return {
        "artifact": str(artifact),
        "model_checkpoint_sha256": sha256_file(artifact / "model.pt"),
        "rf0a_Z_reproduced": True,
        "rf0a_R_reproduced": True,
        "selected_lambdas": {"Z": z_lambda, "R": r_lambda},
        "e6_reproduction": e6,
        "shallow_label_coverage": coverage,
        "structurally_scoreable": structural,
        "depths": depths,
    }


def verify_launch_state() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command).returncode != 0:
            raise SystemExit("FATAL: RF0b launch requires no tracked or staged changes")
    source = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(source)],
                      capture_output=True).returncode != 0:
        raise SystemExit("FATAL: RF0b implementation is not committed")


def write_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    if subprocess.run(["python", "-m", "row.experiments.score_rf0a"]).returncode != 0:
        raise SystemExit("RF0a scorer failed")
    verify_launch_state()
    fatal(not args.output.exists(), f"refuse to overwrite {args.output}")
    cache = validate_cache_completeness()
    rf0a = json.loads(RF0A_REPORT.read_text(encoding="utf-8"))
    fatal(rf0a["decision"]["classification"] == "RAW LOCAL SEMANTICS SURVIVE",
          "RF0a did not authorize RF0b")
    e6_report = json.loads(E6A_REPORT.read_text(encoding="utf-8"))
    controls = synthetic_controls()
    fatal(controls["passed"], "synthetic joint-recovery controls failed")
    started = datetime.now(timezone.utc)
    payload = {
        "completed": False,
        "frozen_plan": str(PLAN),
        "plan_sha256": sha256_file(PLAN),
        "git_commit": git_commit(),
        "started_at_utc": started.isoformat(),
        "protocol": {
            "worlds": list(WORLDS), "depths": list(DEPTHS), "corpus": CORPUS,
            "planted_per_cell": 64, "motif_length": PRIMARY_L,
            "fit_depth": 4, "validation_depth": 6, "test_depths": [8, 10],
            "lambdas": list(LAMBDAS), "null_draws": NULL_DRAWS,
            "absolute_gate": ABSOLUTE_GATE, "materiality": MATERIALITY,
            "role_gap": ROLE_GAP,
            "inputs": "cached raw routes and regenerated teacher programs only",
            "claim": "oracle semantic-canonicalization ceiling",
        },
        "provenance": {
            "rf0a_report": str(RF0A_REPORT),
            "rf0a_report_sha256": sha256_file(RF0A_REPORT),
            "e6a_report": str(E6A_REPORT),
            "e6a_report_sha256": sha256_file(E6A_REPORT),
            "cache_completeness": cache,
        },
        "synthetic_controls": controls,
        "worlds": {},
    }
    for world in WORLDS:
        print(f"[RF0b] world {world}", flush=True)
        result = audit_world(world, rf0a, e6_report)
        payload["worlds"][str(world)] = result
        print("  " + "  ".join(
            f"d{depth} C={result['depths'][str(depth)]['Z']['exact_recovery']:.3f} "
            f"literal={result['depths'][str(depth)]['literal_E6_survival']:.3f} "
            f"delta={result['depths'][str(depth)]['delta_sem']:.3f}"
            for depth in (8, 10)
        ), flush=True)
    displayed = {
        str(depth): round(100.0 * float(np.mean([
            payload["worlds"][str(world)]["e6_reproduction"][str(depth)]["survival_rate"]
            for world in WORLDS
        ])), 2)
        for depth in DEPTHS
    }
    fatal(displayed == {str(depth): value for depth, value in DISPLAYED_E6_MEANS.items()},
          f"displayed E6 survival means differ: {displayed}")
    payload["e6_world_mean_survival_percent"] = displayed
    payload["decision"] = classify(payload["worlds"], controls)
    completed = datetime.now(timezone.utc)
    payload["completed_at_utc"] = completed.isoformat()
    payload["elapsed_seconds"] = (completed - started).total_seconds()
    payload["completed"] = True
    write_atomic(payload, args.output)
    print(f"[RF0b] {payload['decision']['classification']}", flush=True)
    print(f"[RF0b] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
