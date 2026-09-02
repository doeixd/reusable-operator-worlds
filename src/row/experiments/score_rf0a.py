"""Independent integrity scorer for the completed RF0a aggregate report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from row.experiments.audit_rf0a_semantic_recoverability import (
    ARMS,
    CLASSES,
    CORPUS,
    DEPTHS,
    DISPLAYED_E6_MEANS,
    E6A_CACHE,
    E6A_REPORT,
    LAMBDAS,
    NULL_DRAWS,
    PLAN,
    RF0_PLAN,
    SLOTS,
    WORLDS,
    classify,
    e6a_fingerprint,
    sha256_file,
)

DEFAULT_REPORT = Path("reports/rf0a_semantic_recoverability.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def score_confusion(record: dict, expected_rows: int) -> None:
    confusion = np.asarray(record["confusion"], dtype=np.int64)
    require(confusion.shape == (CLASSES, CLASSES), "wrong confusion shape")
    require(np.all(confusion >= 0), "negative confusion entry")
    require(int(confusion.sum()) == expected_rows, "confusion row count mismatch")
    totals = confusion.sum(axis=1)
    require(np.all(totals > 0), "empty teacher class")
    recalls = np.diag(confusion) / totals
    require(np.allclose(recalls, record["per_class_recall"], rtol=0, atol=1e-15),
            "per-class recall mismatch")
    require(math.isclose(float(recalls.mean()), record["balanced_accuracy"],
                         rel_tol=0, abs_tol=1e-15), "balanced accuracy mismatch")


def validate_report(path: Path = DEFAULT_REPORT) -> dict:
    require(path.exists(), f"missing report {path}")
    require(not path.with_suffix(path.suffix + ".tmp").exists(), "temporary report remains")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("completed") is True, "report is not atomically complete")
    require(payload["frozen_plan"] == str(PLAN), "wrong RF0a plan")
    require(payload["frozen_rf0_protocol"] == str(RF0_PLAN), "wrong RF0 protocol")
    require(payload["plan_sha256"] == sha256_file(PLAN), "RF0a plan hash mismatch")
    require(payload["rf0_protocol_sha256"] == sha256_file(RF0_PLAN),
            "RF0 protocol hash mismatch")
    require(payload["provenance"]["e6a_report_sha256"] == sha256_file(E6A_REPORT),
            "E6A report hash mismatch")
    require(payload["provenance"]["e6a_cache_protocol"] == e6a_fingerprint(),
            "E6A protocol mismatch")
    require(payload["provenance"]["cache_completeness"] == {
        "expected_cells": 1536, "actual_cells": 1536, "passed": True
    }, "cache completeness record mismatch")
    require(len(list(E6A_CACHE.glob("w*_d*_*.json"))) == 1536,
            "current E6A cache cell count mismatch")
    require(payload["protocol"]["worlds"] == list(WORLDS), "wrong worlds")
    require(payload["protocol"]["depths"] == list(DEPTHS), "wrong depths")
    require(payload["protocol"]["arms"] == list(ARMS), "wrong arms")
    require(payload["protocol"]["lambdas"] == list(LAMBDAS), "wrong lambda grid")
    require(payload["protocol"]["null_draws"] == NULL_DRAWS, "wrong null count")
    require(payload["protocol"]["slots"] == SLOTS, "wrong slot count")
    require(payload["protocol"]["classes"] == CLASSES, "wrong class count")
    require(payload["e6_world_mean_survival_percent"] == {
        str(depth): value for depth, value in DISPLAYED_E6_MEANS.items()
    }, "E6 displayed means mismatch")
    controls = payload["synthetic_controls"]
    require(controls["positive"]["passed"] and controls["negative"]["passed"],
            "synthetic control failure")
    require(controls["positive"]["ZR"] >= 0.95, "positive absolute gate failure")
    require(controls["positive"]["delta"] >= 0.40, "positive delta gate failure")
    require(controls["negative"]["balanced_accuracy"] <= 0.30,
            "negative gate failure")
    require(set(payload["worlds"]) == {str(world) for world in WORLDS},
            "wrong world key set")

    for world in WORLDS:
        result = payload["worlds"][str(world)]
        checkpoint = Path(result["artifact"]) / "model.pt"
        require(result["model_checkpoint_sha256"] == sha256_file(checkpoint),
                f"checkpoint hash mismatch in world {world}")
        require(result["rows_by_depth"] == {
            str(depth): CORPUS * depth for depth in DEPTHS
        }, f"row counts mismatch in world {world}")
        bank = result["common_function_bank"]
        require(bank["rows"] == 2560 and bank["state_dim"] == 16,
                f"common bank shape mismatch in world {world}")
        require(bank["identical_bank_for_all_slots"] is True,
                f"common bank identity failed in world {world}")
        require(1 <= bank["rank"] <= 11, f"fingerprint rank invalid in world {world}")
        rank = bank["rank"]
        expected_features = {
            "R": 6, "Z": 12, "ZR": 90, "FR": 7 * rank + 6,
            "ZRN": 116, "LOCAL": 7 * rank + 131,
            "TRACE": 7 * rank + 617,
        }
        require(result["feature_dimensions"] == expected_features,
                f"feature dimensions mismatch in world {world}")
        require(set(result["arms"]) == set(ARMS), f"arm set mismatch in world {world}")
        require(result["label_coverage"]["passed"], f"label coverage failed in world {world}")
        require(result["non_vacuity"]["base_scoreable"],
                f"world {world} is not base-scoreable")
        for depth in DEPTHS:
            require(result["e6_reproduction"][str(depth)]["passed"],
                    f"E6 reproduction failed at world {world} depth {depth}")
        for arm in ARMS:
            record = result["arms"][arm]
            require(record["selected_lambda"] in LAMBDAS,
                    f"unregistered lambda in world {world} arm {arm}")
            require(len(record["candidate_validation"]) == len(LAMBDAS),
                    f"incomplete lambda path in world {world} arm {arm}")
            for depth in (8, 10):
                score_confusion(record["deep"][str(depth)], CORPUS * depth)
        for depth in (8, 10):
            key = str(depth)
            z = result["arms"]["Z"]["deep"][key]["balanced_accuracy"]
            zr = result["arms"]["ZR"]["deep"][key]["balanced_accuracy"]
            require(math.isclose(result["contrasts"][key]["ZR_minus_Z"], zr - z,
                                 rel_tol=0, abs_tol=1e-15),
                    f"role contrast mismatch in world {world} depth {depth}")
            null = result["permutation_null_ZR"][key]
            require(len(null["draws"]) == NULL_DRAWS,
                    f"null count mismatch in world {world} depth {depth}")
            require(math.isclose(null["mean"], float(np.mean(null["draws"])),
                                 rel_tol=0, abs_tol=1e-15), "null mean mismatch")
            require(math.isclose(null["p99"], float(np.percentile(null["draws"], 99)),
                                 rel_tol=0, abs_tol=1e-15), "null p99 mismatch")
            require(result["non_vacuity"]["permutation_pass"][key] == (zr > null["p99"]),
                    f"null verdict mismatch in world {world} depth {depth}")

    recomputed = classify(payload["worlds"], controls)
    require(payload["decision"] == recomputed, "decision ladder mismatch")
    require(all_finite(payload), "nonfinite numeric value in report")
    started = datetime.fromisoformat(payload["started_at_utc"])
    completed = datetime.fromisoformat(payload["completed_at_utc"])
    require(completed >= started, "completion precedes start")
    require(payload["elapsed_seconds"] >= 0, "negative elapsed time")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=started.tzinfo)
    require(started <= mtime <= completed + timedelta(seconds=2),
            "report mtime outside recorded run")
    commit = payload["git_commit"]
    require(subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True).returncode == 0,
            "launch commit is missing")
    source = "src/row/experiments/audit_rf0a_semantic_recoverability.py"
    committed_source = subprocess.run(
        ["git", "show", f"{commit}:{source}"], capture_output=True, check=True
    ).stdout
    require(digest_bytes(committed_source) == digest_bytes(Path(source).read_bytes()),
            "current RF0a source differs from launch commit")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        payload = validate_report(args.report)
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"RF0a SCORER FAILED: {error}")
        raise SystemExit(1) from error
    print(
        f"RF0a scorer OK: {len(payload['worlds'])} worlds, "
        f"{len(ARMS)} arms, {NULL_DRAWS} null draws/cell; "
        f"decision={payload['decision']['classification']}"
    )


if __name__ == "__main__":
    main()
