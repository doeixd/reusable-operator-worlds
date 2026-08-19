"""Resolved experiment fingerprints and artifact validation."""

from __future__ import annotations

import hashlib
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


FINGERPRINT_SCHEMA_VERSION = 1


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def resolved_config_sha256(resolved: dict[str, Any]) -> str:
    canonical = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_fingerprint(
    resolved: dict[str, Any],
    model_family: str,
    git_commit: str,
    *,
    backfilled: bool = False,
) -> dict[str, Any]:
    model = resolved[f"{model_family}_model"]
    world = resolved["world"]
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "resolved_config_sha256": resolved_config_sha256(resolved),
        "git_commit": git_commit,
        "model_family": model_family,
        "world_seed": int(world["seed"]),
        "model_seed": int(model["seed"]),
        "configured_rho": float(world["reuse_rho"]),
        "program_length": int(world["program_length"]),
        "hidden_width": model.get("hidden_width"),
        "task_embedding_dim": model.get("task_embedding_dim"),
        "operator_rank": model.get("operator_rank"),
        "operator_slots": model.get("operator_slots"),
        "step_code_dim": model.get("step_code_dim"),
        "hypernetwork_hidden_dim": model.get("hypernetwork_hidden_dim"),
        "global_learning_rate": model.get("global_learning_rate"),
        "task_learning_rate": model.get("task_learning_rate"),
        "replay_ratio": model.get("replay_ratio"),
        "learnable_alpha": model.get("learnable_alpha"),
        "operator_activation": model.get("operator_activation"),
        "backfilled_from_resolved_config": backfilled,
    }


def write_fingerprint(
    output: Path,
    resolved: dict[str, Any],
    model_family: str,
    git_commit: str,
    *,
    backfilled: bool = False,
) -> dict[str, Any]:
    fingerprint = build_fingerprint(
        resolved, model_family, git_commit, backfilled=backfilled
    )
    (output / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
    )
    return fingerprint


def _without_output(resolved: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in resolved.items() if key != "output"}


def _difference_paths(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    actual = json.loads(json.dumps(actual))
    expected = json.loads(json.dumps(expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(set(actual) | set(expected)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in actual or key not in expected:
                differences.append(path)
            else:
                differences.extend(_difference_paths(actual[key], expected[key], path))
        return differences
    return [] if actual == expected else [prefix]


def validate_artifact(
    output: Path,
    expected_resolved: dict[str, Any],
    model_family: str,
    *,
    ignore_output_directory: bool = False,
    backfill_missing_fingerprint: bool = True,
) -> dict[str, Any]:
    resolved_path = output / "config.yaml"
    actual_resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    # Normalize output.directory before comparison: absolute and relative
    # spellings of the same directory are the same provenance (a detached
    # driver and an in-repo shell may launch identical runs with different
    # path representations). Normalization applies to comparison COPIES
    # only — the stored fingerprint hashes the original resolved config,
    # which must remain untouched.
    def _normalized_copy(resolved: dict[str, Any]) -> dict[str, Any]:
        copied = copy.deepcopy(resolved)
        output_section = copied.get("output")
        if isinstance(output_section, dict) and "directory" in output_section:
            output_section["directory"] = str(
                Path(str(output_section["directory"])).resolve()
            )
        return copied

    actual_resolved_cmp = _normalized_copy(actual_resolved)
    expected_resolved_cmp = _normalized_copy(expected_resolved)
    actual_comparable = (
        _without_output(actual_resolved_cmp)
        if ignore_output_directory
        else actual_resolved_cmp
    )
    expected_comparable = (
        _without_output(expected_resolved_cmp)
        if ignore_output_directory
        else expected_resolved_cmp
    )
    differences = _difference_paths(actual_comparable, expected_comparable)
    if differences:
        raise ValueError(
            f"{output} resolved configuration does not match expectation "
            f"(fields={differences})"
        )

    commit_path = output / "git_commit.txt"
    git_commit = (
        commit_path.read_text(encoding="utf-8").strip()
        if commit_path.exists()
        else "unknown"
    )
    fingerprint_path = output / "fingerprint.json"
    if not fingerprint_path.exists():
        if not backfill_missing_fingerprint:
            raise ValueError(f"{output} has no fingerprint.json")
        return write_fingerprint(
            output,
            actual_resolved,
            model_family,
            git_commit,
            backfilled=True,
        )

    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    expected_hash = resolved_config_sha256(actual_resolved)
    if fingerprint.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
        raise ValueError(f"{output} has an unsupported fingerprint schema")
    if fingerprint.get("model_family") != model_family:
        raise ValueError(f"{output} fingerprint has the wrong model family")
    if fingerprint.get("resolved_config_sha256") != expected_hash:
        raise ValueError(f"{output} fingerprint does not match config.yaml")
    if fingerprint.get("git_commit") != git_commit:
        raise ValueError(f"{output} fingerprint does not match git_commit.txt")
    return fingerprint
