"""Aggregate paired continuous-versus-dense ROW pilot artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 12345) -> list[float]:
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(10_000, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def _effect_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "paired_values": values,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "bootstrap_95_percent_ci_of_mean": _bootstrap_mean_ci(array),
    }


def run(continuous_paths: list[Path], dense_paths: list[Path]) -> dict[str, object]:
    if len(continuous_paths) != len(dense_paths) or not continuous_paths:
        raise ValueError("continuous and dense artifact lists must have equal nonzero length")
    continuous = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in continuous_paths
    ]
    dense = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in dense_paths
    ]
    prequential = [
        float(d["cumulative_prequential_nll"]) - float(c["cumulative_prequential_nll"])
        for c, d in zip(continuous, dense, strict=True)
    ]
    final_nmse = [
        float(d["final_nmse"]["median"]) - float(c["final_nmse"]["median"])
        for c, d in zip(continuous, dense, strict=True)
    ]
    novel_32 = [
        float(d["novel_composition"]["nmse_by_support"]["32"])
        - float(c["novel_composition"]["nmse_by_support"]["32"])
        for c, d in zip(continuous, dense, strict=True)
    ]
    retained_bits = [
        float(d["retained_description"]["total_retained_bits"])
        - float(c["retained_description"]["total_retained_bits"])
        for c, d in zip(continuous, dense, strict=True)
    ]
    return {
        "worlds": len(continuous),
        "sign_convention": "positive values favor the continuous basis",
        "dense_minus_continuous_prequential_nll": _effect_summary(prequential),
        "dense_minus_continuous_final_median_nmse": _effect_summary(final_nmse),
        "dense_minus_continuous_novel_32_shot_nmse": _effect_summary(novel_32),
        "dense_minus_continuous_retained_bits": _effect_summary(retained_bits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, nargs="+", required=True)
    parser.add_argument("--dense", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.continuous, args.dense)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
