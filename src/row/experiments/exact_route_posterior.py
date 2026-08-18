"""Exact Bayesian route posterior over a frozen discrete library (V2 Model 7a).

Enumerates all K^L hard routes over a trained library, maintains an online
posterior per task from its demonstrations, and scores predict-before-update
prequential cost with Bayesian model averaging. This is a deliberately
ADVANTAGED bound: the posterior receives the final trained library, whereas
online learners trained theirs concurrently. If even this bound trails the
continuous learner, premature commitment (H7) is refuted; if it wins, the
discrete deficit is an inference cost, not a representation cost.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml

from row.experiments.quantize_artifact import _build_from_artifact
from row.metrics import gaussian_nll
from row.world import World, WorldConfig


@torch.no_grad()
def enumerate_route_outputs(operators, x: torch.Tensor, steps: int) -> torch.Tensor:
    """Return [K^steps, d] outputs f_route(x) for one input, route applied in order."""
    levels = [x.unsqueeze(0)]  # [1, d]
    for _ in range(steps):
        prev = levels[-1]  # [K^(l-1)... flattened, d]
        outs = torch.stack([op(prev) for op in operators], dim=1)  # [prev, K, d]
        levels.append(outs.reshape(-1, outs.shape[-1]))
    return levels[-1]  # [K^steps, d]; index = base-K digits, FIRST applied is most significant


def route_index_to_tuple(index: int, slots: int, steps: int) -> tuple[int, ...]:
    digits = []
    for _ in range(steps):
        digits.append(index % slots)
        index //= slots
    return tuple(reversed(digits))  # first applied first


@torch.no_grad()
def run(artifact: Path, output: Path) -> dict[str, object]:
    raw = yaml.safe_load((artifact / "config.yaml").read_text(encoding="utf-8"))
    model = _build_from_artifact(raw, "discrete")
    world_raw = raw["world"]
    world = World.generate(WorldConfig(**world_raw))
    for task in world.tasks:
        model.begin_task(task.task_id)
    try:
        ckpt = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        ckpt = torch.load(artifact / "model.pt", map_location="cpu", weights_only=False)
    for key in ckpt["model_state_dict"]:
        if key.startswith("task_codes.task_novel_composition"):
            model.begin_task(key.removeprefix("task_codes."))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    operators = list(model.library)

    slots = len(operators)
    steps = int(raw["discrete_model"]["task_steps"])
    n_routes = slots**steps
    sigma = float(raw["evaluation"]["gaussian_sigma"])
    d = int(world_raw["state_dim"])
    log_norm = -0.5 * d * math.log(2 * math.pi * sigma * sigma)
    inv_two_sigma2 = 1.0 / (2 * sigma * sigma)

    # route index (base-slots, first applied most significant) -> tuple
    route_tuples = [route_index_to_tuple(i, slots, steps) for i in range(n_routes)]
    # note: enumerate_route_outputs applies ops level by level; flat index at
    # level l varies fastest in the LAST applied operator, so index digits in
    # base `slots` read (first applied ... last applied) from most to least
    # significant, matching route_index_to_tuple.

    cumulative_bma_point_nll = 0.0
    cumulative_mixture_log_loss = 0.0
    cumulative_map_point_nll = 0.0
    task_rows = []
    entropy_curves = []
    for task_index, task in enumerate(world.tasks):
        log_post = torch.full((n_routes,), -math.log(n_routes), dtype=torch.float64)
        true_route = task.program.primitive_ids
        entropies = []
        map_matches_final = False
        concentration_step = None
        for t in range(world.config.examples_per_task):
            x = torch.as_tensor(task.train_x[t], dtype=torch.float32)
            y = task.train_y[t]
            outputs = enumerate_route_outputs(operators, x, steps).double()  # [R, d]
            # predict BEFORE update
            weights = torch.softmax(log_post, dim=0)
            bma_mean = (weights.unsqueeze(1) * outputs).sum(dim=0).numpy()
            cumulative_bma_point_nll += gaussian_nll(
                bma_mean[None, :], y[None, :], sigma
            )
            map_index = int(torch.argmax(log_post))
            cumulative_map_point_nll += gaussian_nll(
                outputs[map_index].numpy()[None, :], y[None, :], sigma
            )
            sq = ((outputs - torch.as_tensor(y, dtype=torch.float64)) ** 2).sum(dim=1)
            log_lik = log_norm - inv_two_sigma2 * sq
            cumulative_mixture_log_loss += -float(
                torch.logsumexp(log_post + log_lik, dim=0)
            )
            # update
            log_post = log_post + log_lik
            log_post = log_post - torch.logsumexp(log_post, dim=0)
            probs = torch.softmax(log_post, dim=0)
            entropy = float(-(probs * torch.clamp(log_post, min=-745)).sum())
            entropies.append(entropy)
            if concentration_step is None and entropy < 0.1:
                concentration_step = t + 1
        map_index = int(torch.argmax(log_post))
        map_matches_final = route_tuples[map_index] == tuple(true_route)
        task_rows.append(
            {
                "task_index": task_index,
                "true_route": list(true_route),
                "map_route": list(route_tuples[map_index]),
                "map_equals_true": map_matches_final,
                "examples_to_entropy_below_0.1_nat": concentration_step,
                "entropy_after_1": entropies[0],
                "entropy_after_4": entropies[3],
                "entropy_after_16": entropies[15],
            }
        )
        entropy_curves.append(entropies)

    mean_entropy_curve = np.mean(np.array(entropy_curves), axis=0)
    summary = {
        "scope": (
            "post-hoc exact route posterior over the frozen per-task-annealed "
            "discrete library; world seed "
            f"{world.config.seed}, exact reuse; deliberately advantaged bound "
            "(final library, not concurrent training)"
        ),
        "routes_enumerated": n_routes,
        "cumulative_prequential_gaussian_log_loss_bma_point": cumulative_bma_point_nll,
        "cumulative_prequential_gaussian_log_loss_map_point": cumulative_map_point_nll,
        "cumulative_prequential_mixture_log_loss": cumulative_mixture_log_loss,
        "map_equals_true_route_fraction": float(
            np.mean([row["map_equals_true"] for row in task_rows])
        ),
        "median_examples_to_entropy_below_0.1_nat": float(
            np.median(
                [
                    row["examples_to_entropy_below_0.1_nat"]
                    for row in task_rows
                    if row["examples_to_entropy_below_0.1_nat"] is not None
                ]
            )
        ),
        "mean_entropy_curve_first_16": [float(v) for v in mean_entropy_curve[:16]],
        "reference_losses_world_0": {
            "continuous_decoupled_tanh": -170967.3,
            "discrete_per_task_anneal_online": -146146.1,
            "discrete_global_anneal_online": -137321.0,
            "dense_c": -166521.2,
        },
        "tasks": task_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "exact-route-posterior.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/high_priority_controls/discrete_per_task"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2_route_posterior")
    )
    args = parser.parse_args()
    summary = run(args.artifact, args.output)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("tasks",)}, indent=2
        )[:2000]
    )


if __name__ == "__main__":
    main()
