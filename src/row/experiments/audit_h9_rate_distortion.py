"""How many bits does a trained H9 shared-residual learner actually need?

V2 recorded the shared-residual envelope losing the two-part cell because
it retains ~130k bits of task state. That figure is the cost of ONE code:
8 bits for every task scalar, whatever the scalar contains. This audit asks
the separable question — is 130k bits the information content of those
residuals, or an artifact of fixed-width storage? — by compressing a
trained H9 artifact post hoc, with no change to how it learned, and
reporting the rate-distortion frontier R(epsilon).

Three codes are swept, cheapest first:
  * magnitude pruning plus a presence bitmap,
  * per-task-step RANK reduction (2 -> 1 -> 0) chosen by reconstruction
    error, which is the representation PROMOTE will need anyway,
  * uniform quantization at fewer bits per retained scalar.

If R_H9(epsilon) is far below 130k bits at negligible distortion, then V2's
compression failure was substantially a storage-format failure, and the
wake-side variational machinery is less load-bearing than assumed. If it is
not, task information really is that large and must be attacked by changing
what the learner stores, not how it is serialized.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import nmse
from row.mixed_world import CANONICAL_PROFILE, MixedWorldFactory

BITS_PER_SCALAR = 8


def _load_world(config, world_seed: int):
    factory = MixedWorldFactory(list(CANONICAL_PROFILE))
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        return factory.generate(replace(config.world, seed=world_seed))
    finally:
        learned_lifetime.World = original


def _rebuild(config, artifact: Path, world):
    model = _build_model(config, "shared_residual")
    state = torch.load(artifact / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _scores(model, world) -> np.ndarray:
    return np.array(
        [
            nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)
            for task in world.tasks
        ]
    )


@torch.no_grad()
def _prune(model, fraction: float):
    """Zero the smallest-magnitude task scalars; pay one bitmap bit each."""

    values = torch.cat(
        [
            torch.cat((model.task_codes[t].abs().reshape(-1), model.task_residuals[t].abs().reshape(-1)))
            for t in model.task_codes
        ]
    )
    if fraction <= 0.0:
        threshold = -1.0
    else:
        threshold = float(torch.quantile(values, fraction))
    retained = 0
    total = 0
    for task_id in model.task_codes:
        for tensor in (model.task_codes[task_id], model.task_residuals[task_id]):
            keep = tensor.abs() > threshold
            tensor.mul_(keep.to(tensor.dtype))
            retained += int(keep.sum())
            total += int(keep.numel())
    return retained, total


@torch.no_grad()
def _reduce_rank(model, target_rank: int):
    """Drop task-step residual components below `target_rank` by energy."""

    retained = 0
    total = 0
    for task_id in model.task_codes:
        _, residual_u, residual_v, residual_b = model._unpack(task_id)
        u = residual_u.clone()
        v = residual_v.clone()
        b = residual_b.clone()
        for step in range(model.task_steps):
            energy = torch.tensor(
                [
                    float(torch.norm(u[step, :, k]) * torch.norm(v[step, k, :]))
                    for k in range(model.residual_rank)
                ]
            )
            order = torch.argsort(energy, descending=True)
            for position, component in enumerate(order):
                if position >= target_rank:
                    u[step, :, component] = 0.0
                    v[step, component, :] = 0.0
                    b[step, component] = 0.0
        kept = min(target_rank, model.residual_rank)
        # Retained scalars: routes always, plus kept rank components.
        retained += model.route_size + model.task_steps * kept * (2 * model.d + 1)
        total += model.route_size + model.task_steps * model.residual_rank * (
            2 * model.d + 1
        )
        model.task_residuals[task_id].copy_(
            torch.cat((u.reshape(-1), v.reshape(-1), b.reshape(-1)))
        )
    return retained, total


@torch.no_grad()
def _quantize(model, bits: int):
    levels = 2 ** (bits - 1) - 1
    for task_id in model.task_codes:
        for tensor in (model.task_codes[task_id], model.task_residuals[task_id]):
            scale = float(tensor.abs().max())
            if scale <= 0.0:
                continue
            step = scale / levels
            tensor.copy_(torch.round(tensor / step).clamp(-levels, levels) * step)


def audit(config, artifact: Path, world_seed: int, margin: float) -> dict[str, object]:
    world = _load_world(config, world_seed)
    reference = _rebuild(config, artifact, world)
    base = _scores(reference, world)
    shared_bits = BITS_PER_SCALAR * int(reference.shared_parameter_count)
    total_task_scalars = int(reference.task_state_scalar_count)
    dense_task_bits = BITS_PER_SCALAR * total_task_scalars
    points = []

    def _record(label: str, model, task_bits: int) -> None:
        scores = _scores(model, world)
        points.append(
            {
                "code": label,
                "task_bits": int(task_bits),
                "total_bits": int(task_bits + shared_bits),
                "mean_nmse_increase": float(np.mean(scores - base)),
                "within_margin": bool(float(np.mean(scores - base)) <= margin),
            }
        )

    for fraction in (0.25, 0.5, 0.75, 0.9, 0.95):
        model = copy.deepcopy(reference)
        retained, total = _prune(model, fraction)
        _record(f"prune@{fraction:g}", model, BITS_PER_SCALAR * retained + total)
    for target_rank in (1, 0):
        model = copy.deepcopy(reference)
        retained, _ = _reduce_rank(model, target_rank)
        _record(f"rank{target_rank}", model, BITS_PER_SCALAR * retained)
    for bits in (4, 2):
        model = copy.deepcopy(reference)
        _quantize(model, bits)
        _record(f"int{bits}", model, bits * total_task_scalars)
    # Best combination: rank reduction then coarse quantization.
    for target_rank in (1,):
        for bits in (4, 2):
            model = copy.deepcopy(reference)
            retained, _ = _reduce_rank(model, target_rank)
            _quantize(model, bits)
            _record(f"rank{target_rank}+int{bits}", model, bits * retained)

    within = [point for point in points if point["within_margin"]]
    best = min(within, key=lambda point: point["total_bits"]) if within else None
    return {
        "world_seed": world_seed,
        "mean_float_nmse": float(np.mean(base)),
        "dense_two_part_task_bits": dense_task_bits,
        "dense_two_part_total_bits": dense_task_bits + shared_bits,
        "non_inferiority_margin": margin,
        "points": points,
        "best_within_margin": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--root", type=Path, default=Path("artifacts/v2_mixed/canonical"))
    parser.add_argument("--margin", type=float, default=1e-4)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_h9_rate_distortion.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for world_seed in args.worlds:
        artifact = args.root / f"world_{world_seed}" / "shared_residual"
        row = audit(config, artifact, world_seed, args.margin)
        rows.append(row)
        best = row["best_within_margin"]
        print(f"world {world_seed}: dense {row['dense_two_part_total_bits']:,} bits")
        for point in row["points"]:
            print(
                f"    {point['code']:>14s} {point['total_bits']:>8,} bits  "
                f"dNMSE {point['mean_nmse_increase']:+.6f}  within={point['within_margin']}"
            )
        if best:
            print(
                f"    best within margin: {best['code']} at {best['total_bits']:,} bits "
                f"({100 * best['total_bits'] / row['dense_two_part_total_bits']:.0f}% of dense)"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
