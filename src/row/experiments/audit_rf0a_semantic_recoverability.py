"""RF0a: oracle census of local semantic recoverability across E6 depths.

The frozen learner library and E6A route cache are observations.  Teacher
primitive identities are probe labels only.  No route, learner parameter, or
world is optimized here; depth 8/10 labels are opened only for final scoring.
See ``RF0A_SEMANTIC_RECOVERABILITY_PLAN.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams, plant_corpus
from row.experiments.audit_e6a_macro_economics import (
    CACHE as E6A_CACHE,
    CORPUS,
    DEPTHS,
    PLANT_FRACTION,
    PRIMARY_L,
    WORLDS,
    fingerprint as e6a_fingerprint,
    motif_survival,
)
from row.support_split_world import _build_tasks

PLAN = Path("RF0A_SEMANTIC_RECOVERABILITY_PLAN.md")
RF0_PLAN = Path("RF0_ROLE_FILLER_PROTOCOL.md")
E6A_REPORT = Path("reports/e6a_macro_economics.json")
DEFAULT_OUTPUT = Path("reports/rf0a_semantic_recoverability.json")
ARMS = ("R", "Z", "ZR", "FR", "ZRN", "LOCAL", "TRACE")
LAMBDAS = (1e-4, 1e-2, 1.0, 1e2)
CLASSES = 6
SLOTS = 12
NULL_DRAWS = 200
ABSOLUTE_GATE = 0.60
MATERIALITY = 0.20
ROLE_LEAK_GATE = 0.30
DISPLAYED_E6_MEANS = {4: 91.15, 6: 52.60, 8: 13.02, 10: 7.29}


@dataclass(frozen=True)
class OccurrenceData:
    """Learner-side features plus oracle labels retained only in memory."""

    labels: np.ndarray
    raw_symbols: np.ndarray
    roles: np.ndarray
    fingerprints: np.ndarray
    neighbours: np.ndarray
    native: np.ndarray
    trace: np.ndarray
    positions: np.ndarray
    task_indices: np.ndarray
    planted: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    ridge_lambda: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64)[:, self.keep] - self.mean) / self.scale
        scores = z @ self.coefficients + self.intercept
        return np.argmax(scores, axis=1).astype(np.int64)


@dataclass(frozen=True)
class RidgeDecomposition:
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    x_bar: np.ndarray
    y_bar: np.ndarray
    u: np.ndarray
    singular: np.ndarray
    vt: np.ndarray
    y_weighted: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_vector(position: int, depth: int) -> np.ndarray:
    if depth <= 0:
        raise ValueError("depth must be positive")
    boundary = np.zeros(3, dtype=np.float64)
    if position == 0:
        boundary[0] = 1.0
    elif position == depth - 1:
        boundary[2] = 1.0
    else:
        boundary[1] = 1.0
    denominator = max(depth - 1, 1)
    continuous = np.array(
        [position / denominator, (depth - 1 - position) / denominator, depth / 10.0],
        dtype=np.float64,
    )
    return np.concatenate([boundary, continuous])


def one_hot(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= width):
        raise ValueError("one-hot index outside declared width")
    result = np.zeros((len(values), width), dtype=np.float64)
    result[np.arange(len(values)), values] = 1.0
    return result


def interaction(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", left, right).reshape(len(left), -1)


def feature_matrix(data: OccurrenceData, arm: str) -> np.ndarray:
    z = one_hot(data.raw_symbols, SLOTS)
    role = data.roles.astype(np.float64, copy=False)
    fp = data.fingerprints.astype(np.float64, copy=False)
    zr = np.concatenate([z, role, interaction(z, role)], axis=1)
    fr = np.concatenate([fp, role, interaction(fp, role)], axis=1)
    if arm == "R":
        result = role
    elif arm == "Z":
        result = z
    elif arm == "ZR":
        result = zr
    elif arm == "FR":
        result = fr
    elif arm == "ZRN":
        result = np.concatenate([zr, data.neighbours], axis=1)
    elif arm == "LOCAL":
        result = np.concatenate([fr, data.neighbours, data.native], axis=1)
    elif arm == "TRACE":
        result = np.concatenate(
            [fr, data.neighbours, data.native, data.trace], axis=1
        )
    else:
        raise ValueError(f"unknown arm {arm}")
    if result.ndim != 2 or len(result) != data.rows or not np.all(np.isfinite(result)):
        raise ValueError(f"nonfinite or malformed features for {arm}")
    return result


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray,
                      classes: int = CLASSES) -> tuple[float, list[float], list[list[int]]]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        confusion[label, prediction] += 1
    totals = confusion.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("balanced accuracy requires every class")
    recalls = np.diag(confusion) / totals
    return float(np.mean(recalls)), recalls.tolist(), confusion.tolist()


def _preprocess_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    full_mean = x.mean(axis=0)
    full_scale = x.std(axis=0)
    keep = np.flatnonzero(full_scale > 0.0)
    if not len(keep):
        raise ValueError("all probe features are constant")
    mean = full_mean[keep]
    scale = full_scale[keep]
    return (x[:, keep] - mean) / scale, mean, scale, keep


def decompose_ridge(x: np.ndarray, labels: np.ndarray,
                    classes: int = CLASSES) -> RidgeDecomposition:
    labels = np.asarray(labels, dtype=np.int64)
    standardized, mean, scale, keep = _preprocess_fit(x)
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("ridge fit requires every class")
    weights = 1.0 / counts[labels]
    weights /= weights.mean()
    targets = one_hot(labels, classes)
    total_weight = float(weights.sum())
    x_bar = np.sum(weights[:, None] * standardized, axis=0) / total_weight
    y_bar = np.sum(weights[:, None] * targets, axis=0) / total_weight
    root = np.sqrt(weights)[:, None]
    x_weighted = root * (standardized - x_bar)
    y_weighted = root * (targets - y_bar)
    u, singular, vt = np.linalg.svd(x_weighted, full_matrices=False)
    if not len(singular) or not np.all(np.isfinite(singular)):
        raise ValueError("ridge SVD is empty or nonfinite")
    return RidgeDecomposition(
        mean, scale, keep, x_bar, y_bar, u, singular, vt, y_weighted
    )


def ridge_from_decomposition(decomposition: RidgeDecomposition,
                             ridge_lambda: float) -> RidgeModel:
    singular = decomposition.singular
    cutoff = 1e-10 * singular[0]
    gain = np.zeros_like(singular)
    eligible = singular > cutoff
    gain[eligible] = singular[eligible] / (
        singular[eligible] ** 2 + float(ridge_lambda)
    )
    coefficients = (decomposition.vt.T * gain) @ (
        decomposition.u.T @ decomposition.y_weighted
    )
    intercept = decomposition.y_bar - decomposition.x_bar @ coefficients
    arrays = (decomposition.mean, decomposition.scale, coefficients, intercept)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("nonfinite ridge model")
    return RidgeModel(
        decomposition.mean, decomposition.scale, decomposition.keep,
        coefficients, intercept, float(ridge_lambda)
    )


def fit_ridge(x: np.ndarray, labels: np.ndarray, ridge_lambda: float,
              classes: int = CLASSES) -> RidgeModel:
    """Weighted multiclass ridge with an unpenalized intercept and SVD solve."""
    return ridge_from_decomposition(decompose_ridge(x, labels, classes), ridge_lambda)


def select_probe(x_fit: np.ndarray, y_fit: np.ndarray, x_valid: np.ndarray,
                 y_valid: np.ndarray, lambdas: Iterable[float] = LAMBDAS,
                 classes: int = CLASSES) -> tuple[float, list[dict]]:
    records = []
    decomposition = decompose_ridge(x_fit, y_fit, classes)
    for ridge_lambda in lambdas:
        model = ridge_from_decomposition(decomposition, ridge_lambda)
        score, _, _ = balanced_accuracy(y_valid, model.predict(x_valid), classes)
        records.append({"lambda": float(ridge_lambda), "balanced_accuracy": score})
    best_score = max(record["balanced_accuracy"] for record in records)
    # Frozen tie rule: greatest validation accuracy, then largest lambda.
    selected = max(
        record["lambda"] for record in records
        if abs(record["balanced_accuracy"] - best_score) <= 1e-15
    )
    return float(selected), records


def score_probe(train: OccurrenceData, valid: OccurrenceData,
                tests: dict[int, OccurrenceData], arm: str) -> dict:
    x_fit, x_valid = feature_matrix(train, arm), feature_matrix(valid, arm)
    selected, path = select_probe(x_fit, train.labels, x_valid, valid.labels)
    x_refit = np.concatenate([x_fit, x_valid], axis=0)
    y_refit = np.concatenate([train.labels, valid.labels], axis=0)
    model = fit_ridge(x_refit, y_refit, selected)
    fit_score, _, _ = balanced_accuracy(train.labels, model.predict(x_fit))
    valid_score, _, _ = balanced_accuracy(valid.labels, model.predict(x_valid))
    deep = {}
    for depth, data in tests.items():
        score, recalls, confusion = balanced_accuracy(
            data.labels, model.predict(feature_matrix(data, arm))
        )
        deep[str(depth)] = {
            "balanced_accuracy": score,
            "per_class_recall": recalls,
            "confusion": confusion,
        }
    return {
        "selected_lambda": selected,
        "candidate_validation": path,
        "feature_count_before_drop": int(x_refit.shape[1]),
        "feature_count_after_drop": int(len(model.keep)),
        "fit_balanced_accuracy": fit_score,
        "validation_balanced_accuracy_after_refit": valid_score,
        "deep": deep,
    }


def _coordinate_summary(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array.mean(axis=0), array.std(axis=0)])


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


@torch.no_grad()
def trace_route(model, support_x: np.ndarray, route: list[int]) -> list[np.ndarray]:
    z = torch.tensor(support_x, dtype=torch.float32)
    states = [z.detach().cpu().numpy().copy()]
    for slot in route:
        z = model.library[int(slot)](z)
        states.append(z.detach().cpu().numpy().copy())
    return states


@torch.no_grad()
def common_fingerprints(model, depth4_tasks, depth4_routes) -> tuple[np.ndarray, dict]:
    bank_parts = []
    for task, route in zip(depth4_tasks[:16], depth4_routes[:16], strict=True):
        bank_parts.extend(trace_route(model, task.train_x[:32], route))
    bank = np.concatenate(bank_parts, axis=0)
    if bank.shape != (2560, 16):
        raise ValueError(f"common bank has {bank.shape}, expected (2560, 16)")
    tensor = torch.tensor(bank, dtype=torch.float32)
    flattened = []
    for slot in range(SLOTS):
        output = model.library[slot](tensor)
        flattened.append((output - tensor).detach().cpu().numpy().reshape(-1))
    matrix = np.stack(flattened, axis=0).astype(np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] <= 0 or not np.all(np.isfinite(singular)):
        raise ValueError("invalid common-state fingerprint SVD")
    rank = min(11, int(np.sum(singular > 1e-8 * singular[0])))
    if not 1 <= rank <= 11:
        raise ValueError(f"fingerprint rank {rank} outside [1,11]")
    coordinates = u[:, :rank] * singular[:rank]
    return coordinates, {
        "rows": int(bank.shape[0]),
        "state_dim": int(bank.shape[1]),
        "bank_sha256": hashlib.sha256(bank.tobytes()).hexdigest(),
        "identical_bank_for_all_slots": True,
        "rank": rank,
        "singular_values": singular[:12].tolist(),
    }


def load_e6a_route(world: int, depth: int, index: int) -> list[int]:
    path = E6A_CACHE / f"w{world}_d{depth}_{index}.json"
    if not path.exists():
        raise SystemExit(f"FATAL: missing E6A cache cell {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != e6a_fingerprint():
        raise SystemExit(f"FATAL: E6A protocol mismatch at {path}")
    route = payload.get("value", {}).get("route")
    reduction = payload.get("value", {}).get("support_reduction")
    fatal(isinstance(route, list) and len(route) == depth, f"bad route at {path}")
    fatal(all(isinstance(value, int) and 0 <= value < SLOTS for value in route),
          f"bad route symbols at {path}")
    fatal(isinstance(reduction, (int, float)) and np.isfinite(reduction) and reduction > 0,
          f"bad support reduction at {path}")
    return [int(value) for value in route]


def validate_cache_completeness() -> dict:
    expected = {
        f"w{world}_d{depth}_{index}.json"
        for world in WORLDS for depth in DEPTHS for index in range(CORPUS)
    }
    actual = {path.name for path in E6A_CACHE.glob("w*_d*_*.json")}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    fatal(not missing and not extra and len(actual) == 1536,
          f"E6A cache mismatch: {len(missing)} missing, {len(extra)} extra")
    return {"expected_cells": 1536, "actual_cells": len(actual), "passed": True}


def regenerate_depth(cell: dict, world: int, depth: int) -> dict:
    config = cell["config"]
    rng = np.random.default_rng(np.random.SeedSequence([970, world, depth]))
    planted_count = int(CORPUS * PLANT_FRACTION)
    motif, programs, carries, sites = plant_corpus(
        rng, config.world.teacher_primitives, CORPUS, depth, PRIMARY_L, planted_count
    )
    tasks = _build_tasks(
        config.world,
        cell["world"].library,
        programs,
        [f"task_e6a_d{depth}_{index}" for index in range(CORPUS)],
        index_offset=70000 + depth * 200,
    )
    routes = [load_e6a_route(world, depth, index) for index in range(CORPUS)]
    return {
        "motif": motif,
        "programs": programs,
        "carries": carries,
        "sites": sites,
        "tasks": tasks,
        "routes": routes,
    }


def reproduce_e6(world: int, depth: int, regenerated: dict, e6_report: dict) -> dict:
    tally = Counter(
        gram for route in regenerated["routes"] for gram in ngrams(route, PRIMARY_L)
    )
    macro, count = tally.most_common(1)[0]
    survival = motif_survival(
        regenerated["routes"], regenerated["carries"], regenerated["sites"],
        macro, PRIMARY_L,
    )
    expected = e6_report["depths"][str(depth)]["worlds"][str(world)][
        "by_macro_len"
    ][str(PRIMARY_L)]
    expected_survival = expected["motif_survival"]
    fatal(list(macro) == expected["macro"], f"E6 macro mismatch at w{world} d{depth}")
    fatal(survival["hits_at_planted_site"] == expected_survival["hits_at_planted_site"],
          f"E6 planted hits mismatch at w{world} d{depth}")
    fatal(survival["survival_rate"] == expected_survival["survival_rate"],
          f"E6 survival mismatch at w{world} d{depth}")
    return {
        "top_gram": list(macro),
        "overlapping_corpus_count": int(count),
        "registered_nonoverlapping_uses": int(expected["total_uses_in_corpus"]),
        "hits_at_planted_site": int(survival["hits_at_planted_site"]),
        "planted": int(survival["planted"]),
        "survival_rate": float(survival["survival_rate"]),
        "passed": True,
    }


def occurrence_data(model, depth: int, regenerated: dict,
                    fingerprints: np.ndarray) -> OccurrenceData:
    labels, symbols, roles, neighbours = [], [], [], []
    natives, traces, positions, task_indices, planted_rows = [], [], [], [], []
    for task_index, (program, route, task, carries, site) in enumerate(zip(
        regenerated["programs"], regenerated["routes"], regenerated["tasks"],
        regenerated["carries"], regenerated["sites"], strict=True
    )):
        states = trace_route(model, task.train_x, route)
        post_summary = [_coordinate_summary(state) for state in states[1:]]
        trace_summary = np.zeros((10, 32), dtype=np.float64)
        trace_summary[:depth] = np.stack(post_summary)
        mask = np.zeros(10, dtype=np.float64)
        mask[:depth] = 1.0
        histogram = np.bincount(route, minlength=SLOTS).astype(np.float64) / depth
        bigrams = np.zeros((SLOTS, SLOTS), dtype=np.float64)
        for left, right in zip(route[:-1], route[1:]):
            bigrams[left, right] += 1.0
        if depth > 1:
            bigrams /= depth - 1
        task_trace = np.concatenate([trace_summary.reshape(-1), mask, histogram,
                                     bigrams.reshape(-1)])
        for position, (teacher_primitive, slot) in enumerate(zip(program, route, strict=True)):
            incoming, outgoing = states[position], states[position + 1]
            update = outgoing - incoming
            native = np.concatenate([
                _coordinate_summary(incoming),
                _coordinate_summary(outgoing),
                _coordinate_summary(update),
                np.array([_rms(incoming), _rms(outgoing), _rms(update)]),
            ])
            left = route[position - 1] if position > 0 else SLOTS
            right = route[position + 1] if position + 1 < depth else SLOTS
            neighbour = np.concatenate([
                one_hot(np.array([left]), SLOTS + 1)[0],
                one_hot(np.array([right]), SLOTS + 1)[0],
            ])
            labels.append(int(teacher_primitive))
            symbols.append(int(slot))
            roles.append(role_vector(position, depth))
            neighbours.append(neighbour)
            natives.append(native)
            traces.append(task_trace)
            positions.append(position)
            task_indices.append(task_index)
            planted_rows.append(bool(carries and site <= position < site + PRIMARY_L))
    result = OccurrenceData(
        labels=np.asarray(labels, dtype=np.int64),
        raw_symbols=np.asarray(symbols, dtype=np.int64),
        roles=np.asarray(roles, dtype=np.float64),
        fingerprints=fingerprints[np.asarray(symbols, dtype=np.int64)],
        neighbours=np.asarray(neighbours, dtype=np.float64),
        native=np.asarray(natives, dtype=np.float64),
        trace=np.asarray(traces, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.int64),
        task_indices=np.asarray(task_indices, dtype=np.int64),
        planted=np.asarray(planted_rows, dtype=bool),
    )
    expected_rows = CORPUS * depth
    if result.rows != expected_rows:
        raise ValueError(f"depth {depth} has {result.rows}, expected {expected_rows}")
    return result


def permute_within_position(labels: np.ndarray, positions: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int64).copy()
    for position in sorted(set(int(value) for value in positions)):
        indices = np.flatnonzero(positions == position)
        result[indices] = rng.permutation(result[indices])
    return result


def permutation_null(train: OccurrenceData, valid: OccurrenceData,
                     tests: dict[int, OccurrenceData], world: int) -> dict:
    x_fit = feature_matrix(train, "ZR")
    x_valid = feature_matrix(valid, "ZR")
    x_refit = np.concatenate([x_fit, x_valid], axis=0)
    deep_x = {depth: feature_matrix(data, "ZR") for depth, data in tests.items()}
    draws = {depth: [] for depth in tests}
    for draw in range(NULL_DRAWS):
        rng = np.random.default_rng(np.random.SeedSequence([983, world, draw]))
        fit_labels = permute_within_position(train.labels, train.positions, rng)
        valid_labels = permute_within_position(valid.labels, valid.positions, rng)
        selected, _ = select_probe(x_fit, fit_labels, x_valid, valid_labels)
        model = fit_ridge(
            x_refit, np.concatenate([fit_labels, valid_labels]), selected
        )
        for depth, data in tests.items():
            score, _, _ = balanced_accuracy(data.labels, model.predict(deep_x[depth]))
            draws[depth].append(score)
    return {
        str(depth): {
            "draws": values,
            "p99": float(np.percentile(values, 99)),
            "mean": float(np.mean(values)),
        }
        for depth, values in draws.items()
    }


def synthetic_controls() -> dict:
    """Frozen positive and negative non-vacuity checks for the ZR fitter."""
    def make(repeats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.repeat(np.arange(CLASSES, dtype=np.int64), repeats)
        symbols = labels // 2
        boundary = labels % 2
        roles = np.zeros((len(labels), 6), dtype=np.float64)
        roles[np.arange(len(labels)), np.where(boundary == 0, 0, 2)] = 1.0
        roles[:, 3] = boundary
        roles[:, 4] = 1 - boundary
        roles[:, 5] = 0.4
        z = one_hot(symbols, SLOTS)
        zr = np.concatenate([z, roles, interaction(z, roles)], axis=1)
        return labels, z, zr

    train_y, train_z, train_zr = make(20)
    valid_y, valid_z, valid_zr = make(10)
    test_y, test_z, test_zr = make(30)
    selected_z, _ = select_probe(train_z, train_y, valid_z, valid_y)
    selected_zr, _ = select_probe(train_zr, train_y, valid_zr, valid_y)
    z_model = fit_ridge(np.concatenate([train_z, valid_z]),
                        np.concatenate([train_y, valid_y]), selected_z)
    zr_model = fit_ridge(np.concatenate([train_zr, valid_zr]),
                         np.concatenate([train_y, valid_y]), selected_zr)
    z_score, _, _ = balanced_accuracy(test_y, z_model.predict(test_z))
    zr_predictions = zr_model.predict(test_zr)
    zr_score, _, _ = balanced_accuracy(test_y, zr_predictions)
    negative_labels = np.random.default_rng(
        np.random.SeedSequence([984, 0])
    ).permutation(test_y)
    negative_score, _, _ = balanced_accuracy(negative_labels, zr_predictions)
    return {
        "positive": {
            "Z": z_score,
            "ZR": zr_score,
            "delta": zr_score - z_score,
            "passed": bool(zr_score >= 0.95 and zr_score - z_score >= 0.40),
        },
        "negative": {
            "balanced_accuracy": negative_score,
            "passed": bool(negative_score <= 0.30),
        },
    }


def _both_depths(world_result: dict, arm: str, predicate) -> bool:
    return all(predicate(
        world_result["arms"][arm]["deep"][str(depth)]["balanced_accuracy"], depth
    ) for depth in (8, 10))


def classify(worlds: dict[str, dict], controls: dict) -> dict:
    scoreable = [
        int(world) for world, result in worlds.items()
        if result["non_vacuity"]["base_scoreable"]
    ]
    if len(scoreable) < 2 or not controls["positive"]["passed"] or not controls["negative"]["passed"]:
        return {"classification": "UNSCOREABLE", "scoreable_worlds": scoreable}

    def worlds_where(test) -> list[int]:
        return [world for world in scoreable if test(worlds[str(world)])]

    raw = worlds_where(lambda result: _both_depths(
        result, "Z", lambda score, _depth: score >= ABSOLUTE_GATE
    ))
    if len(raw) >= 2:
        return {"classification": "RAW LOCAL SEMANTICS SURVIVE", "passing_worlds": raw,
                "scoreable_worlds": scoreable}

    role = worlds_where(lambda result: all(
        result["arms"]["ZR"]["deep"][str(depth)]["balanced_accuracy"] >= ABSOLUTE_GATE
        and result["contrasts"][str(depth)]["ZR_minus_Z"] >= MATERIALITY
        and result["non_vacuity"]["permutation_pass"][str(depth)]
        for depth in (8, 10)
    ))
    if len(role) >= 2:
        return {"classification": "ROLE-CONDITIONED LOCAL SEMANTICS",
                "passing_worlds": role, "scoreable_worlds": scoreable}

    geometric = worlds_where(lambda result: all(
        result["arms"]["FR"]["deep"][str(depth)]["balanced_accuracy"] >= ABSOLUTE_GATE
        and result["arms"]["FR"]["deep"][str(depth)]["balanced_accuracy"]
        - max(result["arms"][arm]["deep"][str(depth)]["balanced_accuracy"]
              for arm in ("Z", "ZR")) >= MATERIALITY
        and result["non_vacuity"]["permutation_pass"][str(depth)]
        for depth in (8, 10)
    ))
    if len(geometric) >= 2:
        return {"classification": "FUNCTION-GEOMETRIC LOCAL SEMANTICS",
                "passing_worlds": geometric, "scoreable_worlds": scoreable}

    dynamic = worlds_where(lambda result: all(
        result["arms"]["LOCAL"]["deep"][str(depth)]["balanced_accuracy"] >= ABSOLUTE_GATE
        and result["arms"]["LOCAL"]["deep"][str(depth)]["balanced_accuracy"]
        - max(result["arms"][arm]["deep"][str(depth)]["balanced_accuracy"]
              for arm in ("ZR", "FR")) >= MATERIALITY
        and result["non_vacuity"]["permutation_pass"][str(depth)]
        for depth in (8, 10)
    ))
    if len(dynamic) >= 2:
        return {"classification": "DYNAMIC LOCAL SEMANTICS",
                "passing_worlds": dynamic, "scoreable_worlds": scoreable}

    trajectory = worlds_where(lambda result: all(
        result["arms"]["TRACE"]["deep"][str(depth)]["balanced_accuracy"] >= ABSOLUTE_GATE
        and result["arms"]["TRACE"]["deep"][str(depth)]["balanced_accuracy"]
        - result["arms"]["LOCAL"]["deep"][str(depth)]["balanced_accuracy"] >= MATERIALITY
        and result["non_vacuity"]["permutation_pass"][str(depth)]
        for depth in (8, 10)
    ))
    if len(trajectory) >= 2:
        return {"classification": "TRAJECTORY-DISTRIBUTED SEMANTICS",
                "passing_worlds": trajectory, "scoreable_worlds": scoreable}

    any_recoverable = worlds_where(lambda result: any(
        _both_depths(result, arm, lambda score, _depth: score >= ABSOLUTE_GATE)
        for arm in ARMS
    ))
    if len(any_recoverable) < 2:
        return {"classification": "TEACHER-LOCAL IDENTITY NOT RECOVERABLE",
                "scoreable_worlds": scoreable, "recoverable_worlds": any_recoverable}
    return {"classification": "UNRESOLVED", "scoreable_worlds": scoreable,
            "recoverable_worlds": any_recoverable}


def audit_world(cell: dict, world: int, e6_report: dict) -> tuple[dict, dict[int, float]]:
    regenerated = {depth: regenerate_depth(cell, world, depth) for depth in DEPTHS}
    e6_checks = {
        str(depth): reproduce_e6(world, depth, regenerated[depth], e6_report)
        for depth in DEPTHS
    }
    fingerprints, bank = common_fingerprints(
        cell["model"], regenerated[4]["tasks"], regenerated[4]["routes"]
    )
    data = {
        depth: occurrence_data(cell["model"], depth, regenerated[depth], fingerprints)
        for depth in DEPTHS
    }
    labels_present = {
        str(depth): sorted(int(value) for value in np.unique(rows.labels))
        for depth, rows in data.items()
    }
    label_coverage = all(values == list(range(CLASSES)) for values in labels_present.values())
    arms = {
        arm: score_probe(data[4], data[6], {8: data[8], 10: data[10]}, arm)
        for arm in ARMS
    }
    null = permutation_null(data[4], data[6], {8: data[8], 10: data[10]}, world)
    contrasts = {}
    role_leakage = {}
    permutation_pass = {}
    for depth in (8, 10):
        scores = {
            arm: arms[arm]["deep"][str(depth)]["balanced_accuracy"] for arm in ARMS
        }
        contrasts[str(depth)] = {
            "ZR_minus_Z": scores["ZR"] - scores["Z"],
            "FR_minus_Z": scores["FR"] - scores["Z"],
            "FR_minus_ZR": scores["FR"] - scores["ZR"],
            "LOCAL_minus_best_ZR_FR": scores["LOCAL"] - max(scores["ZR"], scores["FR"]),
            "TRACE_minus_LOCAL": scores["TRACE"] - scores["LOCAL"],
        }
        role_leakage[str(depth)] = bool(scores["R"] <= ROLE_LEAK_GATE)
        permutation_pass[str(depth)] = bool(scores["ZR"] > null[str(depth)]["p99"])
    base_scoreable = bool(
        label_coverage and all(role_leakage.values()) and bank["identical_bank_for_all_slots"]
        and 1 <= bank["rank"] <= 11 and all(check["passed"] for check in e6_checks.values())
    )
    result = {
        "artifact": str(cell["path"]),
        "model_checkpoint_sha256": sha256_file(cell["path"] / "model.pt"),
        "rows_by_depth": {str(depth): data[depth].rows for depth in DEPTHS},
        "feature_dimensions": {
            arm: int(feature_matrix(data[4], arm).shape[1]) for arm in ARMS
        },
        "common_function_bank": bank,
        "e6_reproduction": e6_checks,
        "label_coverage": {"classes_by_depth": labels_present, "passed": label_coverage},
        "arms": arms,
        "contrasts": contrasts,
        "permutation_null_ZR": null,
        "non_vacuity": {
            "role_leakage_pass": role_leakage,
            "permutation_pass": permutation_pass,
            "base_scoreable": base_scoreable,
        },
    }
    return result, {depth: e6_checks[str(depth)]["survival_rate"] for depth in DEPTHS}


def verify_launch_state() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command).returncode != 0:
            raise SystemExit("FATAL: RF0a launch requires no tracked or staged changes")
    source_path = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(source_path)],
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise SystemExit("FATAL: RF0a implementation is not committed")


def write_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(985)
    np.random.seed(985)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    verify_launch_state()
    fatal(not args.output.exists(), f"refuse to overwrite existing RF0a report {args.output}")
    validate_cache = validate_cache_completeness()
    e6_report = json.loads(E6A_REPORT.read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc)
    controls = synthetic_controls()
    fatal(controls["positive"]["passed"], "synthetic positive control failed")
    fatal(controls["negative"]["passed"], "synthetic negative control failed")
    payload = {
        "completed": False,
        "frozen_plan": str(PLAN),
        "frozen_rf0_protocol": str(RF0_PLAN),
        "plan_sha256": sha256_file(PLAN),
        "rf0_protocol_sha256": sha256_file(RF0_PLAN),
        "git_commit": git_commit(),
        "started_at_utc": started.isoformat(),
        "protocol": {
            "worlds": list(WORLDS), "depths": list(DEPTHS), "corpus": CORPUS,
            "fit_depth": 4, "validation_depth": 6, "test_depths": [8, 10],
            "arms": list(ARMS), "lambdas": list(LAMBDAS),
            "null_draws": NULL_DRAWS, "classes": CLASSES, "slots": SLOTS,
            "absolute_gate": ABSOLUTE_GATE, "materiality": MATERIALITY,
            "teacher_labels": "probe fitting/scoring only; absent from features",
            "inputs": "support x only; no support y, query x, or query y",
        },
        "provenance": {
            "e6a_report": str(E6A_REPORT),
            "e6a_report_sha256": sha256_file(E6A_REPORT),
            "e6a_cache_protocol": e6a_fingerprint(),
            "cache_completeness": validate_cache,
        },
        "synthetic_controls": controls,
        "worlds": {},
    }
    survival_by_depth = {depth: [] for depth in DEPTHS}
    cells = {world: load_cell(world, args.config) for world in WORLDS}
    for world in WORLDS:
        print(f"[RF0a] world {world}: reconstructing features and probes", flush=True)
        result, survival = audit_world(cells[world], world, e6_report)
        payload["worlds"][str(world)] = result
        for depth, value in survival.items():
            survival_by_depth[depth].append(value)
        print("  " + "  ".join(
            f"d{depth} Z={result['arms']['Z']['deep'][str(depth)]['balanced_accuracy']:.3f} "
            f"ZR={result['arms']['ZR']['deep'][str(depth)]['balanced_accuracy']:.3f} "
            f"FR={result['arms']['FR']['deep'][str(depth)]['balanced_accuracy']:.3f} "
            f"LOCAL={result['arms']['LOCAL']['deep'][str(depth)]['balanced_accuracy']:.3f} "
            f"TRACE={result['arms']['TRACE']['deep'][str(depth)]['balanced_accuracy']:.3f}"
            for depth in (8, 10)
        ), flush=True)
    displayed = {
        str(depth): round(100.0 * float(np.mean(values)), 2)
        for depth, values in survival_by_depth.items()
    }
    fatal(displayed == {str(key): value for key, value in DISPLAYED_E6_MEANS.items()},
          f"displayed E6 survival means differ: {displayed}")
    payload["e6_world_mean_survival_percent"] = displayed
    payload["decision"] = classify(payload["worlds"], controls)
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    payload["completed"] = True
    write_atomic(payload, args.output)
    print(f"[RF0a] {payload['decision']['classification']}", flush=True)
    print(f"[RF0a] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
