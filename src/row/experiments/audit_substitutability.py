"""The promotion validity gate, stated operationally.

Three properties have to hold before an explicit promoter can be tested,
and the earlier proxies (parameter similarity, residual clustering,
recoverable family identity) each certified worlds where one of them was
missing:

    load-bearing?   removing the private computation must cost something
    compressible?   one fitted shared function must recover much of it
    family-specific? it must beat a single global abstraction

The comparison is always four-way — private residual, family abstraction,
global abstraction, zero — because only the spread between them separates
genuine abstraction from generic compression from deletion.

Every fit is LEAVE-ONE-OUT: the abstraction substituted into task tau is
fitted without tau's own residual. Fitting jointly and then reporting that
the fit explains its own members measures compression of a fixed
collection; leave-one-out measures cross-task REUSE, which is what
promotion claims and what a future task will actually get.

Two probe distributions, because they ask different questions: a common
Gaussian domain asks whether the operators are equivalent, while
on-trajectory states ask whether the substitution preserves the
computation the task actually performs. Substitutability on trajectory is
the one that matters for promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from row.config import load_config
from row.experiments import learned_lifetime
from row.experiments.learned_lifetime import _build_model, _tensor
from row.metrics import nmse
from row.mixed_world import CANONICAL_PROFILE
from row.task_group_world import TaskGroupSpec, TaskGroupWorldFactory


def _split(model, flat):
    a = model.residual_u_size
    b = a + model.residual_v_size
    return (
        flat[:a].reshape(model.task_steps, model.d, model.residual_rank),
        flat[a:b].reshape(model.task_steps, model.residual_rank, model.d),
        flat[b:].reshape(model.task_steps, model.residual_rank),
    )


def _innovation(z, u, v, b, step):
    return torch.nn.functional.linear(
        torch.tanh(torch.nn.functional.linear(z, v[step], b[step])), u[step]
    )


def _fit(model, member_ids, probe, steps=300, lr=0.02):
    initial = torch.stack(
        [model.task_residuals[t].detach() for t in member_ids]
    ).mean(dim=0)
    candidate = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([candidate], lr=lr)
    with torch.no_grad():
        targets = torch.stack(
            [
                torch.stack(
                    [
                        _innovation(probe, *_split(model, model.task_residuals[t].detach()), s)
                        for s in range(model.task_steps)
                    ]
                )
                for t in member_ids
            ]
        )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        u, v, b = _split(model, candidate)
        predicted = torch.stack(
            [_innovation(probe, u, v, b, s) for s in range(model.task_steps)]
        )
        torch.mean(torch.square(predicted.unsqueeze(0) - targets)).backward()
        optimizer.step()
    return candidate.detach()


@torch.no_grad()
def _task_nmse(model, task) -> float:
    return nmse(model(_tensor(task.eval_x), task.task_id).cpu().numpy(), task.eval_y)


@torch.no_grad()
def _trajectory_probe(model, world, task_indices, samples=256):
    """States actually arriving at each step, pooled over tasks."""

    states = []
    for index in task_indices[:8]:
        task = world.tasks[index]
        z = _tensor(task.eval_x[: max(1, samples // 8)])
        states.append(z)
    return torch.cat(states)[:samples]


def audit(config, path: Path, world_seed: int, spec: TaskGroupSpec, slots: int, steps: int):
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
    model = _build_model(local, "shared_residual")
    state = torch.load(path / "model.pt", weights_only=True)["model_state_dict"]
    for key in state:
        if key.startswith("task_codes."):
            model.begin_task(key.split(".", 1)[1])
    model.load_state_dict(state)
    model.eval()

    assignment = np.array(world.group_assignment[: len(world.tasks)])
    post = list(range(spec.family_onset, len(world.tasks)))
    common = _tensor(
        np.random.default_rng(world_seed + 991).normal(size=(256, config.world.state_dim))
    )
    trajectory = _trajectory_probe(model, world, post)

    results = {}
    for probe_name, probe in (("common", common), ("trajectory", trajectory)):
        private, family, glob, zero = [], [], [], []
        for index in post:
            task = world.tasks[index]
            group = int(assignment[index])
            family_members = [
                world.tasks[i].task_id
                for i in post
                if assignment[i] == group and i != index
            ]
            all_members = [world.tasks[i].task_id for i in post if i != index]
            family_fit = _fit(model, family_members, probe, steps=steps)
            global_fit = _fit(model, all_members, probe, steps=steps)

            private.append(_task_nmse(model, task))
            for label, value, bucket in (
                ("family", family_fit, family),
                ("global", global_fit, glob),
                ("zero", None, zero),
            ):
                clone = copy.deepcopy(model)
                with torch.no_grad():
                    if value is None:
                        clone.task_residuals[task.task_id].zero_()
                    else:
                        clone.task_residuals[task.task_id].copy_(value)
                clone.eval()
                bucket.append(_task_nmse(clone, task))
        private_mean = float(np.mean(private))
        zero_mean = float(np.mean(zero))
        span = zero_mean - private_mean
        results[probe_name] = {
            "private_residual_nmse": private_mean,
            "family_abstraction_nmse": float(np.mean(family)),
            "global_abstraction_nmse": float(np.mean(glob)),
            "zero_residual_nmse": zero_mean,
            "load_bearing_span": span,
            "family_capture_fraction": (
                (zero_mean - float(np.mean(family))) / span if span > 0 else 0.0
            ),
            "global_capture_fraction": (
                (zero_mean - float(np.mean(glob))) / span if span > 0 else 0.0
            ),
        }
        results[probe_name]["family_advantage"] = (
            results[probe_name]["family_capture_fraction"]
            - results[probe_name]["global_capture_fraction"]
        )
    return {"world_seed": world_seed, "leave_one_out": True, **results}


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
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--new-primitive-families", action="store_true", default=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_substitutability.json")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    spec = TaskGroupSpec(
        groups=args.groups,
        eta=args.eta,
        future_tasks=8,
        family_onset=args.onset,
        new_primitive_families=args.new_primitive_families,
    )
    rows = []
    for world_seed in args.worlds:
        rows.append(
            audit(
                config,
                args.root / f"world_{world_seed}" / "shared_residual",
                world_seed,
                spec,
                args.slots,
                args.steps,
            )
        )
    print("LEAVE-ONE-OUT SUBSTITUTABILITY GATE")
    for probe in ("common", "trajectory"):
        print(f"  probe: {probe}")
        for row in rows:
            entry = row[probe]
            print(
                f"    world {row['world_seed']}: private {entry['private_residual_nmse']:.5f} | "
                f"family {entry['family_abstraction_nmse']:.5f} | "
                f"global {entry['global_abstraction_nmse']:.5f} | "
                f"zero {entry['zero_residual_nmse']:.5f} || "
                f"capture family {entry['family_capture_fraction']*100:.1f}% "
                f"global {entry['global_capture_fraction']*100:.1f}% "
                f"advantage {entry['family_advantage']*100:+.1f}pts"
            )
        advantages = [row[probe]["family_advantage"] for row in rows]
        print(
            f"    mean family advantage {np.mean(advantages)*100:+.1f} points, "
            f"positive {sum(a > 0 for a in advantages)}/{len(advantages)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
