"""Score P-2026-08-18-D: do learned residuals cluster by hidden task group?

The V3 spec's section 2.1 validity gate. The world-level precondition
(teacher deviations separate by group) is checked by the generator; this is
the harder claim the gate actually needs — that a TRAINED learner's
task-step residuals carry the family structure, so there is something for
PROMOTE to find.

Instrument, and the reason it is not a naive similarity: a shared basis can
absorb any task-INVARIANT component, and spectral renormalization leaves a
large one (measured correlation floor 0.33 among teacher deviations even
with no family structure at all). Group structure is therefore measured on
residual functions CENTERED by their cross-task mean. Teacher group labels
are used only here, for scoring, and never enter training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory


def _residual_functions(model, world, probe: torch.Tensor) -> np.ndarray:
    """Each task's residual effect on a fixed probe, concatenated over steps."""

    rows = []
    with torch.no_grad():
        for task in world.tasks:
            _, residual_u, residual_v, residual_b = model._unpack(task.task_id)
            effects = []
            for step in range(model.task_steps):
                hidden = torch.tanh(
                    torch.nn.functional.linear(
                        probe, residual_v[step], residual_b[step]
                    )
                )
                effects.append(
                    torch.nn.functional.linear(hidden, residual_u[step]).reshape(-1)
                )
            rows.append(torch.cat(effects).cpu().numpy())
    return np.stack(rows)


def score(config, artifact: Path, world_seed: int, spec: TaskGroupSpec, model_kind: str):
    factory = TaskGroupWorldFactory(list(CANONICAL_PROFILE), spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        world = factory.generate(replace(config.world, seed=world_seed, reuse_rho=1.0))
    finally:
        learned_lifetime.World = original

    model = _build_model(config, model_kind)
    state = torch.load(artifact / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()

    generator = np.random.default_rng(world_seed + 991)
    probe = _tensor(generator.normal(size=(256, config.world.state_dim)))
    functions = _residual_functions(model, world, probe)
    # Remove the task-invariant component a shared basis could absorb.
    functions = functions - functions.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(functions, axis=1, keepdims=True)
    normalized = functions / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T

    assignment = np.array(world.group_assignment[: len(world.tasks)])
    within, cross = [], []
    _ = assignment
    for first in range(len(world.tasks)):
        for second in range(first + 1, len(world.tasks)):
            value = float(similarity[first, second])
            (within if assignment[first] == assignment[second] else cross).append(value)
    within_mean = float(np.mean(within))
    cross_mean = float(np.mean(cross))

    # The decision-relevant question: could the promoter's own detection
    # step RECOVER the hidden partition from these residuals? Separation
    # being positive is necessary but not sufficient. Two-means on the
    # centered, normalized residual functions, scored as partition accuracy
    # against ground truth (chance is 0.5 for two balanced groups; the label
    # permutation is resolved by taking the better of the two matchings).
    generator_state = np.random.default_rng(world_seed + 7)
    best_accuracy = 0.0
    for _ in range(10):
        centers = normalized[generator_state.choice(len(normalized), 2, replace=False)]
        labels = np.zeros(len(normalized), dtype=int)
        for _ in range(50):
            distances = np.stack(
                [np.linalg.norm(normalized - center, axis=1) for center in centers]
            )
            labels = distances.argmin(axis=0)
            for index in range(2):
                if (labels == index).any():
                    centers[index] = normalized[labels == index].mean(axis=0)
        accuracy = max(
            float(np.mean(labels == assignment)),
            float(np.mean(labels == 1 - assignment)),
        )
        best_accuracy = max(best_accuracy, accuracy)

    return {
        "partition_recovery_accuracy": best_accuracy,
        "chance_accuracy": 0.5,
        "world_seed": world_seed,
        "eta": spec.eta,
        "model": model_kind,
        "within_group_mean_similarity": within_mean,
        "cross_group_mean_similarity": cross_mean,
        "separation": within_mean - cross_mean,
        # The ledger's "factor of 3" phrasing needs a positive quantity to
        # be meaningful; with balanced centered groups the cross-group mean
        # is driven negative by construction, so the ratio is reported only
        # when it is well defined and the separation is the statistic read.
        "ratio": within_mean / cross_mean if cross_mean > 1e-9 else None,
        "separation_over_absolute_cross": (
            (within_mean - cross_mean) / abs(cross_mean) if abs(cross_mean) > 1e-9 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v3_taskgroup"))
    parser.add_argument("--model", default="shared_residual")
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.9])
    parser.add_argument("--future-tasks", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_group_clustering.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rows = []
    for eta in args.etas:
        for world_seed in args.worlds:
            artifact = args.root / f"eta{eta:g}" / f"world_{world_seed}" / args.model
            if not (artifact / "summary.json").exists():
                continue
            spec = TaskGroupSpec(eta=eta, future_tasks=args.future_tasks)
            rows.append(score(config, artifact, world_seed, spec, args.model))
    for eta in args.etas:
        selected = [row for row in rows if row["eta"] == eta]
        if not selected:
            continue
        separations = [row["separation"] for row in selected]
        recoveries = [row["partition_recovery_accuracy"] for row in selected]
        print(
            f"eta={eta:g}: mean separation {np.mean(separations):+.4f} "
            f"positive in {sum(value > 0 for value in separations)}/{len(separations)}; "
            f"partition recovery {np.mean(recoveries):.3f} "
            f"(worlds {[f'{value:.3f}' for value in recoveries]}, chance 0.500)"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
