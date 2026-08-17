"""Quantize a learned-model artifact and verify retained behavior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from row.metrics import nmse
from row.models import ContinuousBasisLearner, DenseLearner, DiscreteLibraryLearner
from row.world import World, WorldConfig


def symmetric_int8_dequantize(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return values representable by a signed per-tensor 8-bit quantizer."""
    if not tensor.is_floating_point():
        return tensor.clone(), 1.0
    maximum = float(torch.max(torch.abs(tensor)))
    if maximum == 0.0:
        return tensor.clone(), 1.0
    scale = maximum / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127)
    return quantized * scale, scale


def _build_from_artifact(
    raw: dict[str, object], kind: str
) -> DenseLearner | ContinuousBasisLearner | DiscreteLibraryLearner:
    world_raw = raw["world"]
    assert isinstance(world_raw, dict)
    d = int(world_raw["state_dim"])
    alpha = float(world_raw["alpha"])
    if kind == "dense":
        model_raw = raw["dense_model"]
        assert isinstance(model_raw, dict)
        return DenseLearner(
            d=d,
            task_embedding_dim=int(model_raw["task_embedding_dim"]),
            hidden_width=int(model_raw["hidden_width"]),
            residual_blocks=int(model_raw["residual_blocks"]),
            seed=int(model_raw["seed"]),
        )
    if kind == "continuous":
        model_raw = raw["continuous_model"]
        assert isinstance(model_raw, dict)
        return ContinuousBasisLearner(
            d=d,
            operator_slots=int(model_raw["operator_slots"]),
            operator_rank=int(model_raw["operator_rank"]),
            task_steps=int(model_raw["task_steps"]),
            alpha=alpha,
            seed=int(model_raw["seed"]),
        )
    model_raw = raw["discrete_model"]
    assert isinstance(model_raw, dict)
    return DiscreteLibraryLearner(
        d=d,
        operator_slots=int(model_raw["operator_slots"]),
        operator_rank=int(model_raw["operator_rank"]),
        task_steps=int(model_raw["task_steps"]),
        alpha=alpha,
        initial_temperature=float(model_raw["initial_temperature"]),
        final_temperature=float(model_raw["final_temperature"]),
        seed=int(model_raw["seed"]),
    )


@torch.no_grad()
def _task_scores(
    model: DenseLearner | ContinuousBasisLearner | DiscreteLibraryLearner,
    world: World,
) -> np.ndarray:
    model.eval()
    scores = []
    for task in world.tasks:
        prediction = model(
            torch.as_tensor(task.eval_x, dtype=torch.float32), task.task_id
        ).cpu().numpy()
        scores.append(nmse(prediction, task.eval_y))
    return np.asarray(scores)


def _inference_multiply_adds(raw: dict[str, object], kind: str) -> int:
    world_raw = raw["world"]
    assert isinstance(world_raw, dict)
    d = int(world_raw["state_dim"])
    if kind == "dense":
        model_raw = raw["dense_model"]
        assert isinstance(model_raw, dict)
        width = int(model_raw["hidden_width"])
        embedding = int(model_raw["task_embedding_dim"])
        blocks = int(model_raw["residual_blocks"])
        return blocks * ((d + embedding) * width + width * d)
    model_raw = raw[f"{kind}_model"]
    assert isinstance(model_raw, dict)
    slots = int(model_raw["operator_slots"])
    rank = int(model_raw["operator_rank"])
    steps = int(model_raw["task_steps"])
    if kind == "discrete":
        return steps * (d * rank + rank * d)
    operator_madds = steps * slots * (d * rank + rank * d)
    mixture_madds = steps * slots * d
    return operator_madds + mixture_madds


def run(artifact: Path) -> dict[str, object]:
    raw = yaml.safe_load((artifact / "config.yaml").read_text(encoding="utf-8"))
    summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    kind = str(summary["model"])
    world_raw = raw["world"]
    assert isinstance(world_raw, dict)
    world = World.generate(WorldConfig(**world_raw))
    model = _build_from_artifact(raw, kind)
    for task in world.tasks:
        model.begin_task(task.task_id)
    checkpoint = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    checkpoint_keys = checkpoint["model_state_dict"].keys()
    novel_keys = [
        key.removeprefix("task_codes.")
        for key in checkpoint_keys
        if key.startswith("task_codes.task_novel_composition")
    ]
    for novel_key in novel_keys:
        model.begin_task(novel_key)
    model.load_state_dict(checkpoint["model_state_dict"])
    float_scores = _task_scores(model, world)

    quantized_state: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for name, value in model.state_dict().items():
        if kind == "discrete" and name.startswith("task_codes."):
            # Hardened routes are retained as exact categorical indices, not as
            # quantized training logits. One-hot logits reconstruct those routes
            # losslessly for behavioral evaluation.
            indices = torch.argmax(value, dim=-1, keepdim=True)
            quantized_state[name] = torch.zeros_like(value).scatter_(-1, indices, 1.0)
            scales[name] = 1.0
        else:
            quantized_state[name], scales[name] = symmetric_int8_dequantize(value)
    model.load_state_dict(quantized_state)
    quantized_scores = _task_scores(model, world)

    shared_count = int(summary["shared_parameter_count"])
    task_count = int(summary["task_state_scalar_count"])
    if kind == "discrete":
        active = int(summary["routing"]["active_operators"])
        route_bits_per_step = math.ceil(math.log2(active)) if active > 1 else 0
        task_state_bits = len(world.tasks) * int(world.config.program_length) * route_bits_per_step
    else:
        task_state_bits = 8 * task_count
    result: dict[str, object] = {
        "quantization": "symmetric signed int8, per tensor, dequantized evaluation",
        "bits_per_scalar": 8,
        "shared_weight_bits": 8 * shared_count,
        "task_state_bits": task_state_bits,
        "total_retained_bits": 8 * shared_count + task_state_bits,
        "scale_overhead_bits_excluded": True,
        "inference_multiply_adds": _inference_multiply_adds(raw, kind),
        "float_final_nmse_mean": float(np.mean(float_scores)),
        "quantized_final_nmse_mean": float(np.mean(quantized_scores)),
        "quantized_minus_float_nmse_mean": float(np.mean(quantized_scores - float_scores)),
        "maximum_task_nmse_increase": float(np.max(quantized_scores - float_scores)),
        "tensor_scales": scales,
    }
    (artifact / "quantization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary["retained_description"] = result
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {"model_state_dict": quantized_state, "quantization": result},
        artifact / "model_int8_dequantized.pt",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = run(args.artifact)
    print(
        f"{args.artifact}: {result['total_retained_bits']} bits; "
        f"quantized-minus-float NMSE={result['quantized_minus_float_nmse_mean']:.6g}; "
        f"inference multiply-adds={result['inference_multiply_adds']}"
    )


if __name__ == "__main__":
    main()
