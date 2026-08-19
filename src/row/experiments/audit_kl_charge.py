"""Is each task's code charged once, or once per replay appearance?

The MDL conversion beta * 2*sigma^2/(N*d) is correct if a task's KL is
spread over exactly the N observations whose likelihood it accompanies. A
ROW lifetime is not one pass over an isolated dataset: replay re-exposes
completed tasks, and the batch penalty is the mean over the unique tasks
present, so the integrated pressure on a given task's code can differ from
the intended single charge — and can differ SYSTEMATICALLY WITH TASK INDEX,
since early tasks sit in the replay buffer far longer than late ones.

This audit reproduces the replay bookkeeping exactly (same seeds, same
sampling policy, no model or torch required) and reports, per task, the
integrated KL coefficient actually applied over the lifetime as a multiple
of the intended one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from row.config import load_config
from row.experiments.learned_lifetime import TaskReplayBuffer
from row.world import World


def audit(config, world_seed: int) -> dict[str, object]:
    from dataclasses import replace

    world = World.generate(replace(config.world, seed=world_seed))
    selected = config.variational_model
    replay = TaskReplayBuffer(selected.seed + 1)
    examples = config.world.examples_per_task
    replay_count = int(round(selected.replay_ratio))

    # Weight applied to a task's KL in one update: description_penalty is
    # the MEAN over the unique task IDs present in the batch.
    index_of = {t.task_id: i for i, t in enumerate(world.tasks)}
    # Two policies, and for each the KL pressure AND the data pressure a
    # task receives. The decisive quantity is their RATIO: repeated visits
    # under SGD are more optimization steps toward the same objective, not
    # a higher price, so long as code pressure and likelihood pressure move
    # together. A ratio that varies with task position is a real
    # position-dependent coding penalty; a raw charge that varies while the
    # ratio is flat is not.
    batch_mean_kl: Counter[int] = Counter()
    current_only_kl: Counter[int] = Counter()
    data_pressure: Counter[int] = Counter()
    for task in world.tasks:
        for _ in range(examples):
            items = replay.sample(replay_count)
            present = [task.task_id, *(item[2] for item in items)]
            unique = list(dict.fromkeys(present))
            # Data term is an MSE mean over batch elements, so each task's
            # likelihood weight is its share of the batch.
            for task_id in present:
                data_pressure[index_of[task_id]] += 1.0 / len(present)
            # Superseded policy: mean over unique tasks in the batch.
            for task_id in unique:
                batch_mean_kl[index_of[task_id]] += 1.0 / len(unique)
            # Alternative policy: only the current task, full weight.
            current_only_kl[index_of[task.task_id]] += 1.0
        replay.add_task(task, selected.replay_examples_per_task)

    # Intended: the equivalent of one full charge per the task's own N
    # examples, i.e. an integrated coefficient of exactly `examples`.
    count = len(world.tasks)
    data = np.array([data_pressure[i] for i in range(count)])
    batch_mean = np.array([batch_mean_kl[i] for i in range(count)])
    current_only = np.array([current_only_kl[i] for i in range(count)])
    ratios = batch_mean / data
    charge_ratios = batch_mean / examples
    current_only_ratios = current_only / data
    def _tilt(values):
        quarter = max(1, len(values) // 4)
        return float(np.mean(values[:quarter]) / np.mean(values[-quarter:]))

    return {
        "world_seed": world_seed,
        "adopted_policy": "batch mean over unique tasks present",
        "pressure_ratio": {
            "note": "integrated KL weight divided by integrated data weight",
            "mean": float(np.mean(ratios)),
            "minimum": float(np.min(ratios)),
            "maximum": float(np.max(ratios)),
            "first_to_last_quarter_tilt": _tilt(ratios),
        },
        "raw_charge_multiple_of_intended": {
            "note": "integrated KL coefficient over the intended single charge",
            "mean": float(np.mean(charge_ratios)),
            "minimum": float(np.min(charge_ratios)),
            "maximum": float(np.max(charge_ratios)),
            "first_to_last_quarter_tilt": _tilt(charge_ratios),
        },
        "rejected_current_task_only_policy": {
            "note": "pressure ratio if only the current task were charged",
            "mean": float(np.mean(current_only_ratios)),
            "minimum": float(np.min(current_only_ratios)),
            "maximum": float(np.max(current_only_ratios)),
            "first_to_last_quarter_tilt": _tilt(current_only_ratios),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("reports/v3_kl_charge_audit.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = [audit(config, world_seed) for world_seed in args.worlds]
    for row in rows:
        pressure = row["pressure_ratio"]
        charge = row["raw_charge_multiple_of_intended"]
        rejected = row["rejected_current_task_only_policy"]
        print(f"world {row['world_seed']}:")
        print(
            f"    pressure ratio (decisive): mean {pressure['mean']:.3f} "
            f"range {pressure['minimum']:.3f}-{pressure['maximum']:.3f} "
            f"tilt {pressure['first_to_last_quarter_tilt']:.3f}x"
        )
        print(
            f"    raw charge multiple:       mean {charge['mean']:.2f} "
            f"range {charge['minimum']:.2f}-{charge['maximum']:.2f} "
            f"tilt {charge['first_to_last_quarter_tilt']:.2f}x"
        )
        print(
            f"    if current-task-only:      mean {rejected['mean']:.3f} "
            f"range {rejected['minimum']:.3f}-{rejected['maximum']:.3f} "
            f"tilt {rejected['first_to_last_quarter_tilt']:.2f}x"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
