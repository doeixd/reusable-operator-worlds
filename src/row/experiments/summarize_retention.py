"""Run and aggregate current int8 retained-description checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from row.config import load_config
from row.experiments.learned_lifetime import resolved_learned_config
from row.experiments.quantize_artifact import run as quantize
from row.provenance import validate_artifact


DEVELOPMENT_WORLDS = tuple(range(10))


def _aggregate(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    bits = {int(row["total_retained_bits"]) for row in rows}
    madds = {int(row["inference_multiply_adds"]) for row in rows}
    if len(bits) != 1 or len(madds) != 1:
        raise ValueError(f"{model} retention accounting changed across worlds")
    return {
        "model": model,
        "worlds": len(rows),
        "total_retained_bits": bits.pop(),
        "shared_weight_bits": int(rows[0]["shared_weight_bits"]),
        "task_state_bits": int(rows[0]["task_state_bits"]),
        "inference_multiply_adds": madds.pop(),
        "mean_quantized_minus_float_nmse": fmean(
            float(row["quantized_minus_float_nmse_mean"]) for row in rows
        ),
        "maximum_task_nmse_increase": max(
            float(row["maximum_task_nmse_increase"]) for row in rows
        ),
        "per_world": [
            {
                key: row[key]
                for key in (
                    "world_seed",
                    "float_final_nmse_mean",
                    "quantized_final_nmse_mean",
                    "quantized_minus_float_nmse_mean",
                    "maximum_task_nmse_increase",
                )
            }
            for row in rows
        ],
    }


def _artifact_paths(
    baseline_root: Path,
    hyper_stage_one_root: Path,
    hyper_stage_two_root: Path,
    dense24_root: Path,
) -> dict[str, list[tuple[int, Path]]]:
    return {
        "continuous": [
            (world, baseline_root / f"world_{world}" / "continuous")
            for world in DEVELOPMENT_WORLDS
        ],
        "dense32": [
            (world, baseline_root / f"world_{world}" / "dense")
            for world in DEVELOPMENT_WORLDS
        ],
        "hypernetwork": [
            (
                world,
                (hyper_stage_one_root if world < 3 else hyper_stage_two_root)
                / "hypernetwork"
                / "global_3em03_task_5em02"
                / f"world_{world}",
            )
            for world in DEVELOPMENT_WORLDS
        ],
        "dense24": [
            (
                world,
                dense24_root
                / "dense"
                / "global_1em03_task_5em02"
                / f"world_{world}",
            )
            for world in DEVELOPMENT_WORLDS
        ],
    }


def summarize(
    baseline_root: Path,
    hyper_stage_one_root: Path,
    hyper_stage_two_root: Path,
    dense24_root: Path,
    discrete_artifact: Path | None,
) -> dict[str, Any]:
    model_reports = []
    paths = _artifact_paths(
        baseline_root, hyper_stage_one_root, hyper_stage_two_root, dense24_root
    )
    for label, artifacts in paths.items():
        rows = []
        for world, artifact in artifacts:
            result = quantize(artifact)
            rows.append({"world_seed": world, **result})
        model_reports.append(_aggregate(label, rows))

    if discrete_artifact is not None:
        raw = yaml.safe_load(
            (discrete_artifact / "config.yaml").read_text(encoding="utf-8")
        )
        config = load_config(Path("configs/v1.yaml"))
        config = replace(
            config,
            world=replace(config.world, seed=0),
            discrete_model=replace(
                config.discrete_model,
                temperature_schedule="per_task",
            ),
            evaluation=replace(
                config.evaluation,
                lifetime_checkpoints=(),
                checkpoint_novel_tasks=1,
                extended_diagnostics=False,
            ),
            output_directory=discrete_artifact,
        )
        validate_artifact(
            discrete_artifact,
            resolved_learned_config(config, "discrete", "forward"),
            "discrete",
            backfill_missing_fingerprint=True,
        )
        if raw != yaml.safe_load(
            (discrete_artifact / "config.yaml").read_text(encoding="utf-8")
        ):
            raise ValueError("validation unexpectedly changed discrete config")
        model_reports.append(
            _aggregate("discrete_per_task", [{"world_seed": 0, **quantize(discrete_artifact)}])
        )

    return {
        "scope": "current learnable-alpha artifacts; development worlds; 8-bit proxy",
        "quantization": "symmetric signed int8 per tensor with dequantized behavioral evaluation",
        "scale_overhead": "excluded consistently from V1 proxy",
        "models": model_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("artifacts/checkpoints_development"),
    )
    parser.add_argument(
        "--hyper-stage-one",
        type=Path,
        default=Path("artifacts/tuning/hypernetwork_stage1"),
    )
    parser.add_argument(
        "--hyper-stage-two",
        type=Path,
        default=Path("artifacts/tuning/hypernetwork_stage2"),
    )
    parser.add_argument(
        "--dense24",
        type=Path,
        default=Path("artifacts/controls/dense24_worlds0_9"),
    )
    parser.add_argument(
        "--discrete",
        type=Path,
        default=Path("artifacts/high_priority_controls/discrete_per_task"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/retention/current-retention.json"),
    )
    args = parser.parse_args()
    result = summarize(
        args.baselines,
        args.hyper_stage_one,
        args.hyper_stage_two,
        args.dense24,
        args.discrete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = [
        {
            key: row[key]
            for key in (
                "model",
                "worlds",
                "total_retained_bits",
                "inference_multiply_adds",
                "mean_quantized_minus_float_nmse",
            )
        }
        for row in result["models"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
