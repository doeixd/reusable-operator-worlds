"""H11.3: do promoted abstractions make future related tasks cheaper?

Compression without forward benefit is storage optimization. This probes the
held-out future block — tasks drawn from the same hidden families that never
entered any lifetime — and measures what it costs each learner to acquire
them.

Protocol, matched exactly across learners: deep-copy the trained model so
the probe cannot alter it, freeze the shared library, introduce a future
task, let it adapt its own task state (and, for the promoting learner,
select a library entry or none from its own early examples), and record the
adaptation curve. The unpromoted learner gets the identical budget and the
identical data; the only difference is whether a library exists to reuse.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import gaussian_nll, nmse
from row.mixed_world import CANONICAL_PROFILE
from row.models import PromotingSharedResidualLearner
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory

SUPPORTS = (0, 1, 2, 4, 8, 16, 32)


def _load(config, path: Path, kind: str, world_seed: int, spec: TaskGroupSpec, slots: int):
    factory = TaskGroupWorldFactory(list(CANONICAL_PROFILE), spec)
    original = learned_lifetime.World
    learned_lifetime.World = factory
    try:
        world = factory.generate(replace(config.world, seed=world_seed, reuse_rho=1.0))
    finally:
        learned_lifetime.World = original
    local = replace(
        config,
        shared_residual_model=replace(config.shared_residual_model, operator_slots=slots),
    )
    model = _build_model(local, kind)
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    if isinstance(model, PromotingSharedResidualLearner):
        count = sum(1 for key in state if key.startswith("abstractions."))
        for index in range(count):
            model.abstractions.append(
                torch.nn.Parameter(state[f"abstractions.{index}"].clone(), requires_grad=False)
            )
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()
    return model, world


def adapt(model, task, config, task_lr: float, residual_lr: float, reuse_at: int = 8):
    """Acquire one future task with the shared library frozen."""

    probe = copy.deepcopy(model)
    for parameter in probe.shared_parameters():
        parameter.requires_grad_(False)
    if isinstance(probe, PromotingSharedResidualLearner):
        for parameter in probe.abstractions:
            parameter.requires_grad_(False)
    handles = probe.begin_task(task.task_id)
    route, residual = handles[0], handles[1]
    optimizer = torch.optim.Adam(
        [
            {"params": [route], "lr": task_lr},
            {"params": [residual], "lr": residual_lr},
        ]
    )
    curve, nats = {}, 0.0
    for step in range(max(SUPPORTS) + 1):
        if step in SUPPORTS:
            probe.eval()
            with torch.no_grad():
                prediction = probe(_tensor(task.eval_x), task.task_id).cpu().numpy()
            curve[step] = nmse(prediction, task.eval_y)
        if step == max(SUPPORTS):
            break
        if isinstance(probe, PromotingSharedResidualLearner) and step == reuse_at:
            probe.select_reference(
                task.task_id,
                _tensor(task.train_x[:step]),
                _tensor(task.train_y[:step]),
            )
        probe.eval()
        with torch.no_grad():
            online = probe(_tensor(task.train_x[step : step + 1]), task.task_id).cpu().numpy()
        nats += gaussian_nll(
            online, task.train_y[step : step + 1], config.evaluation.gaussian_sigma
        )
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            probe(_tensor(task.train_x[step : step + 1]), task.task_id),
            _tensor(task.train_y[step : step + 1]),
        )
        loss.backward()
        optimizer.step()
    reference = (
        probe.task_reference.get(task.task_id)
        if isinstance(probe, PromotingSharedResidualLearner)
        else None
    )
    return curve, nats, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/v3_taskgroup/eta0.9_k6_onset16_frozen16_newprim"),
    )
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--onset", type=int, default=16)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("reports/v3_future_block.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=2, eta=0.9, future_tasks=8, family_onset=args.onset,
        new_primitive_families=True,
    )
    task_lr = config.shared_residual_model.task_learning_rate
    residual_lr = config.shared_residual_model.residual_learning_rate
    rows = []
    for world_seed in args.worlds:
        entry = {"world_seed": world_seed}
        for kind, label in (("shared_residual", "unpromoted"), ("promoting", "promoted")):
            path = args.root / f"world_{world_seed}" / kind
            if not (path / "summary.json").exists():
                continue
            model, world = _load(config, path, kind, world_seed, spec, args.slots)
            curves, nats, reused = [], [], 0
            for task in world.future_tasks:
                curve, cost, reference = adapt(model, task, config, task_lr, residual_lr)
                curves.append(curve)
                nats.append(cost)
                reused += int(reference is not None)
            entry[label] = {
                "mean_adaptation_nats": float(np.mean(nats)),
                "mean_nmse_by_support": {
                    str(s): float(np.mean([c[s] for c in curves])) for s in SUPPORTS
                },
                "tasks_reusing_library": reused,
                "future_tasks": len(curves),
            }
        rows.append(entry)

    print("H11.3 FUTURE BLOCK: do promoted abstractions make new related tasks cheaper?")
    deltas, nats_gain = [], []
    for row in rows:
        if "promoted" not in row or "unpromoted" not in row:
            continue
        u, p = row["unpromoted"], row["promoted"]
        d32 = u["mean_nmse_by_support"]["32"] - p["mean_nmse_by_support"]["32"]
        dn = u["mean_adaptation_nats"] - p["mean_adaptation_nats"]
        deltas.append(d32)
        nats_gain.append(dn)
        print(
            f"  world {row['world_seed']}: 32-shot NMSE {u['mean_nmse_by_support']['32']:.5f} -> "
            f"{p['mean_nmse_by_support']['32']:.5f} ({d32:+.5f}) | "
            f"adaptation nats {u['mean_adaptation_nats']:.0f} -> {p['mean_adaptation_nats']:.0f} "
            f"({dn:+.0f}) | reuse {p['tasks_reusing_library']}/{p['future_tasks']}"
        )
    if deltas:
        print(
            f"  mean 32-shot improvement {np.mean(deltas):+.5f} "
            f"(positive {sum(d > 0 for d in deltas)}/{len(deltas)}); "
            f"mean adaptation-nats improvement {np.mean(nats_gain):+.0f} "
            f"(positive {sum(n > 0 for n in nats_gain)}/{len(nats_gain)})"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
