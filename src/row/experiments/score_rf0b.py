"""Independent integrity scorer for the RF0b semantic-motif report."""
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
    E6A_CACHE,
    E6A_REPORT,
    DISPLAYED_E6_MEANS,
    LAMBDAS,
    NULL_DRAWS,
    sha256_file,
)
from row.experiments.audit_rf0b_semantic_motif import (
    ABSOLUTE_GATE,
    CORPUS,
    DEFAULT_OUTPUT,
    DEPTHS,
    MATERIALITY,
    PLAN,
    RF0A_REPORT,
    ROLE_GAP,
    WORLDS,
    classify,
)
from row.experiments.score_rf0a import all_finite


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_motif_arm(record: dict, label: str) -> None:
    require(record["planted_tasks"] == 64, f"{label}: wrong planted count")
    require(record["unplanted_tasks"] == 64, f"{label}: wrong unplanted count")
    require(record["exact_count"] in range(65), f"{label}: invalid exact count")
    require(math.isclose(record["exact_recovery"], record["exact_count"] / 64,
                         rel_tol=0, abs_tol=1e-15), f"{label}: exact rate mismatch")
    marginal = np.asarray(record["relative_position_accuracy"], dtype=np.float64)
    require(marginal.shape == (3,) and np.all((0 <= marginal) & (marginal <= 1)),
            f"{label}: invalid marginal accuracy")
    product = float(np.prod(marginal))
    require(math.isclose(record["independence_product"], product,
                         rel_tol=0, abs_tol=1e-15), f"{label}: product mismatch")
    require(math.isclose(record["exact_minus_independence"],
                         record["exact_recovery"] - product,
                         rel_tol=0, abs_tol=1e-15), f"{label}: dependence mismatch")
    require(len(record["modal_decoded_motif"]) == 3, f"{label}: bad modal motif")
    require(record["modal_count"] in range(1, 65), f"{label}: bad modal count")
    require(math.isclose(record["modal_recurrence"], record["modal_count"] / 64,
                         rel_tol=0, abs_tol=1e-15), f"{label}: modal rate mismatch")
    require(0 <= record["unplanted_teacher_motif_occurrence"] <= 1,
            f"{label}: bad unplanted occurrence rate")
    for value in record["pairwise_phi"].values():
        require(value is None or -1 <= value <= 1, f"{label}: invalid phi")


def validate_report(path: Path = DEFAULT_OUTPUT) -> dict:
    require(path.exists(), f"missing report {path}")
    require(not path.with_suffix(path.suffix + ".tmp").exists(), "temporary report remains")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("completed") is True, "report is incomplete")
    require(payload["frozen_plan"] == str(PLAN), "wrong frozen plan")
    require(payload["plan_sha256"] == sha256_file(PLAN), "plan hash mismatch")
    require(payload["provenance"]["rf0a_report_sha256"] == sha256_file(RF0A_REPORT),
            "RF0a input hash mismatch")
    require(payload["provenance"]["e6a_report_sha256"] == sha256_file(E6A_REPORT),
            "E6A input hash mismatch")
    require(payload["provenance"]["cache_completeness"] == {
        "expected_cells": 1536, "actual_cells": 1536, "passed": True
    }, "cache completeness mismatch")
    require(len(list(E6A_CACHE.glob("w*_d*_*.json"))) == 1536,
            "current E6A cache count mismatch")
    require(payload["protocol"]["worlds"] == list(WORLDS), "world protocol mismatch")
    require(payload["protocol"]["depths"] == list(DEPTHS), "depth protocol mismatch")
    require(payload["protocol"]["corpus"] == CORPUS, "corpus protocol mismatch")
    require(payload["protocol"]["lambdas"] == list(LAMBDAS), "lambda grid mismatch")
    require(payload["protocol"]["null_draws"] == NULL_DRAWS, "null count mismatch")
    require(payload["protocol"]["absolute_gate"] == ABSOLUTE_GATE,
            "absolute gate mismatch")
    require(payload["protocol"]["materiality"] == MATERIALITY,
            "materiality mismatch")
    require(payload["protocol"]["role_gap"] == ROLE_GAP, "role gap mismatch")
    require(payload["e6_world_mean_survival_percent"] == {
        str(depth): value for depth, value in DISPLAYED_E6_MEANS.items()
    }, "displayed E6 means mismatch")
    controls = payload["synthetic_controls"]
    require(controls["passed"] and controls["positive_exact_recovery"] == 1.0
            and controls["negative_exact_recovery"] <= 0.10,
            "synthetic controls failed")
    require(set(payload["worlds"]) == {str(world) for world in WORLDS},
            "wrong world keys")
    rf0a = json.loads(RF0A_REPORT.read_text(encoding="utf-8"))
    e6 = json.loads(E6A_REPORT.read_text(encoding="utf-8"))
    require(rf0a["decision"]["classification"] == "RAW LOCAL SEMANTICS SURVIVE",
            "RF0a gate no longer holds")
    for world in WORLDS:
        result = payload["worlds"][str(world)]
        require(result["rf0a_Z_reproduced"] and result["rf0a_R_reproduced"],
                f"world {world}: RF0a reproduction failed")
        require(result["selected_lambdas"]["Z"]
                == rf0a["worlds"][str(world)]["arms"]["Z"]["selected_lambda"],
                f"world {world}: Z lambda differs from RF0a")
        require(result["selected_lambdas"]["R"]
                == rf0a["worlds"][str(world)]["arms"]["R"]["selected_lambda"],
                f"world {world}: R lambda differs from RF0a")
        checkpoint = Path(result["artifact"]) / "model.pt"
        require(result["model_checkpoint_sha256"] == sha256_file(checkpoint),
                f"world {world}: checkpoint hash mismatch")
        require(result["structurally_scoreable"] and result["shallow_label_coverage"],
                f"world {world}: structural checks failed")
        for depth in DEPTHS:
            require(result["e6_reproduction"][str(depth)]["passed"],
                    f"world {world} depth {depth}: E6 reproduction failed")
        for depth in (8, 10):
            key = str(depth)
            cell = result["depths"][key]
            validate_motif_arm(cell["Z"], f"world {world} depth {depth} Z")
            validate_motif_arm(cell["R"], f"world {world} depth {depth} R")
            registered = e6["depths"][key]["worlds"][str(world)][
                "by_macro_len"
            ]["3"]["motif_survival"]["survival_rate"]
            require(cell["literal_E6_survival"] == registered,
                    f"world {world} depth {depth}: literal baseline mismatch")
            require(math.isclose(cell["delta_sem"],
                                 cell["Z"]["exact_recovery"] - registered,
                                 rel_tol=0, abs_tol=1e-15), "semantic delta mismatch")
            require(math.isclose(cell["Z_minus_R_exact"],
                                 cell["Z"]["exact_recovery"]
                                 - cell["R"]["exact_recovery"],
                                 rel_tol=0, abs_tol=1e-15), "role gap mismatch")
            null = cell["permutation_null"]
            require(len(null["draws"]) == NULL_DRAWS, "wrong null draw count")
            require(math.isclose(null["mean"], float(np.mean(null["draws"])),
                                 rel_tol=0, abs_tol=1e-15), "null mean mismatch")
            require(math.isclose(null["p99"], float(np.percentile(null["draws"], 99)),
                                 rel_tol=0, abs_tol=1e-15), "null p99 mismatch")
            require(null["max"] == max(null["draws"]), "null max mismatch")
            require(cell["permutation_pass"]
                    == (cell["Z"]["exact_recovery"] > null["p99"]),
                    "permutation verdict mismatch")
    require(payload["decision"] == classify(payload["worlds"], controls),
            "decision ladder mismatch")
    require(all_finite(payload), "nonfinite report value")
    started = datetime.fromisoformat(payload["started_at_utc"])
    completed = datetime.fromisoformat(payload["completed_at_utc"])
    require(completed >= started and payload["elapsed_seconds"] >= 0,
            "invalid timestamps")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=started.tzinfo)
    require(started <= mtime <= completed + timedelta(seconds=2), "stale report mtime")
    commit = payload["git_commit"]
    require(subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True).returncode == 0,
            "missing launch commit")
    source = "src/row/experiments/audit_rf0b_semantic_motif.py"
    committed = subprocess.run(["git", "show", f"{commit}:{source}"],
                               capture_output=True, check=True).stdout
    require(digest_bytes(committed) == digest_bytes(Path(source).read_bytes()),
            "RF0b source differs from launch commit")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        payload = validate_report(args.report)
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"RF0b SCORER FAILED: {error}")
        raise SystemExit(1) from error
    print(
        f"RF0b scorer OK: 3 worlds, 2 depths, {NULL_DRAWS} null draws/cell; "
        f"decision={payload['decision']['classification']}"
    )


if __name__ == "__main__":
    main()
