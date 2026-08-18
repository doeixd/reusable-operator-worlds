"""Wake/sleep consolidating continuous learner (V2 Model 8, step 003).

Wake: the standard continuous-basis online protocol. Sleep, at the
checkpoint task counts: enumerate all hard routes over the learner's own
basis for each completed uncompiled task, form the exact route posterior
from that task's training data, and compile the task to its MAP route iff
the posterior has concentrated (entropy below H_THRESHOLD) and the hard
route's evaluation NMSE is within KAPPA of the soft mixture's. Compiled
tasks thereafter execute hard routes (9 bits retained state instead of
192); shared operators keep training throughout, including through
compiled tasks appearing in replay.

Gate thresholds are FROZEN (PROGRESS.md, 2026-08-18): H_THRESHOLD = 0.1
nat, KAPPA = 1.5. The pre-registered shape prediction: gate firing rate
non-decreasing in rho across the full grid.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.config import ExperimentConfig, load_config
from row.experiments.learned_lifetime import (
    TaskReplayBuffer,
    _novel_data,
    _tensor,
)
from row.metrics import gaussian_nll, nmse
from row.models import ContinuousBasisLearner
from row.world import Task, World

H_THRESHOLD = 0.1
KAPPA = 1.5
SLEEP_CHECKPOINTS = (8, 16, 32, 64)


class ConsolidatingContinuous(nn.Module):
    """Continuous basis with per-task compilation to hard routes."""

    def __init__(self, base: ContinuousBasisLearner) -> None:
        super().__init__()
        self.base = base
        self.compiled: dict[str, tuple[int, ...]] = {}

    def begin_task(self, task_id: str) -> nn.Parameter:
        return self.base.begin_task(task_id)

    def forward(self, x: Tensor, task_id: str) -> Tensor:
        route = self.compiled.get(task_id)
        if route is None:
            return self.base(x, task_id)
        z = x
        for slot in route:
            z = self.base.basis[slot](z)
        return z

    def forward_tasks(self, x: Tensor, task_ids) -> Tensor:
        return torch.cat(
            [
                self.forward(sample.unsqueeze(0), task_id)
                for sample, task_id in zip(x, task_ids, strict=True)
            ],
            dim=0,
        )

    def shared_parameters(self):
        return self.base.shared_parameters()


@torch.no_grad()
def _route_outputs_batched(basis, x: Tensor, steps: int) -> Tensor:
    """[K^steps, B, d] outputs for a batch, first-applied most significant."""
    level = x.unsqueeze(0)  # [1, B, d]
    for _ in range(steps):
        flat = level.reshape(-1, level.shape[-1])
        level = torch.stack([op(flat) for op in basis], dim=1)  # [prev, K, d]
        level = level.reshape(-1, x.shape[0], x.shape[1])
    return level


def _index_to_route(index: int, slots: int, steps: int) -> tuple[int, ...]:
    digits = []
    for _ in range(steps):
        digits.append(index % slots)
        index //= slots
    return tuple(reversed(digits))


@torch.no_grad()
def _sleep(
    model: ConsolidatingContinuous,
    world: World,
    completed: list[int],
    sigma: float,
    rows: list[dict[str, object]],
    tasks_completed: int,
) -> dict[str, object]:
    basis = list(model.base.basis)
    slots = len(basis)
    steps = model.base.task_steps
    d = world.config.state_dim
    log_norm = -0.5 * d * math.log(2 * math.pi * sigma * sigma)
    inv = 1.0 / (2 * sigma * sigma)
    examined = compiled_now = 0
    for task_index in completed:
        task = world.tasks[task_index]
        if task.task_id in model.compiled:
            continue
        examined += 1
        x = _tensor(task.train_x)
        y = torch.as_tensor(task.train_y, dtype=torch.float64)
        outputs = _route_outputs_batched(basis, x, steps).double()  # [R,B,d]
        sq = ((outputs - y.unsqueeze(0)) ** 2).sum(dim=2)  # [R,B]
        log_lik = (log_norm - inv * sq).sum(dim=1)  # [R]
        log_post = log_lik - torch.logsumexp(log_lik, dim=0)
        probs = torch.exp(log_post)
        entropy = float(-(probs * torch.clamp(log_post, min=-745)).sum())
        if entropy >= H_THRESHOLD:
            continue
        map_index = int(torch.argmax(log_post))
        route = _index_to_route(map_index, slots, steps)
        eval_x = _tensor(task.eval_x)
        z = eval_x
        for slot in route:
            z = basis[slot](z)
        hard_nmse = nmse(z.numpy(), task.eval_y)
        soft_nmse = nmse(model.base(eval_x, task.task_id).numpy(), task.eval_y)
        if hard_nmse <= KAPPA * soft_nmse:
            model.compiled[task.task_id] = route
            compiled_now += 1
    record = {
        "record_type": "sleep",
        "tasks_completed": tasks_completed,
        "tasks_examined": examined,
        "compiled_this_sleep": compiled_now,
        "compiled_total": len(model.compiled),
        "firing_rate_cumulative": len(model.compiled) / max(1, tasks_completed),
    }
    rows.append(record)
    return record


def run(
    config: ExperimentConfig, order: str = "forward", force_compile: bool = False
) -> dict[str, object]:
    torch.set_num_threads(1)
    world = World.generate(config.world)
    mc = config.continuous_model
    base = ContinuousBasisLearner(
        d=config.world.state_dim,
        operator_slots=mc.operator_slots,
        operator_rank=mc.operator_rank,
        task_steps=mc.task_steps,
        alpha=mc.operator_alpha_init,
        seed=mc.seed,
        learnable_alpha=mc.learnable_alpha,
        activation=mc.operator_activation,
    )
    model = ConsolidatingContinuous(base)
    optimizer = torch.optim.AdamW(
        model.shared_parameters(), lr=mc.global_learning_rate,
        weight_decay=mc.weight_decay,
    )
    replay = TaskReplayBuffer(mc.seed + 1)
    sigma = config.evaluation.gaussian_sigma
    rows: list[dict[str, object]] = []
    cumulative_nll = 0.0
    nll_by_task: list[float] = []
    completed: list[int] = []
    sleeps: list[dict[str, object]] = []
    for lifetime_index, task in enumerate(world.tasks):
        code = model.begin_task(task.task_id)
        optimizer.add_param_group(
            {"params": [code], "lr": mc.task_learning_rate, "weight_decay": 0.0}
        )
        task_nll = 0.0
        for n_seen in range(config.world.examples_per_task):
            x = _tensor(task.train_x[n_seen : n_seen + 1])
            model.eval()
            with torch.no_grad():
                prediction = model(x, task.task_id).cpu().numpy()
            online = gaussian_nll(
                prediction, task.train_y[n_seen : n_seen + 1], sigma
            )
            cumulative_nll += online
            task_nll += online
            replay_items = replay.sample(int(round(mc.replay_ratio)))
            batch_x = [task.train_x[n_seen], *(i[0] for i in replay_items)]
            batch_y = [task.train_y[n_seen], *(i[1] for i in replay_items)]
            ids = [task.task_id, *(i[2] for i in replay_items)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_tasks(_tensor(np.stack(batch_x)), ids)
            loss = torch.nn.functional.mse_loss(out, _tensor(np.stack(batch_y)))
            loss.backward()
            optimizer.step()
        nll_by_task.append(task_nll)
        replay.add_task(task, mc.replay_examples_per_task)
        completed.append(lifetime_index)
        tasks_completed = lifetime_index + 1
        if tasks_completed in SLEEP_CHECKPOINTS:
            if force_compile:
                _force_all(model, world, completed)
                sleeps.append(
                    {
                        "record_type": "sleep",
                        "tasks_completed": tasks_completed,
                        "forced": True,
                        "compiled_total": len(model.compiled),
                    }
                )
            else:
                sleeps.append(
                    _sleep(model, world, completed, sigma, rows, tasks_completed)
                )

    model.eval()
    final_scores = []
    for task in world.tasks:
        with torch.no_grad():
            pred = model(_tensor(task.eval_x), task.task_id).numpy()
        final_scores.append(nmse(pred, task.eval_y))
    novel = _adapt_novel(model, world, config, mc.task_learning_rate)
    compiled_n = len(model.compiled)
    task_state_bits = compiled_n * mc.task_steps * math.ceil(
        math.log2(mc.operator_slots)
    ) + (len(world.tasks) - compiled_n) * mc.task_steps * mc.operator_slots * 8
    summary: dict[str, object] = {
        "model": "consolidating_continuous",
        "gate": {"h_threshold": H_THRESHOLD, "kappa": KAPPA,
                 "forced": force_compile},
        "cumulative_prequential_gaussian_log_loss": cumulative_nll,
        "per_task_prequential_nll": nll_by_task,
        "final_nmse_mean": float(np.mean(final_scores)),
        "final_nmse_median": float(np.median(final_scores)),
        "compiled_tasks": compiled_n,
        "compiled_fraction": compiled_n / len(world.tasks),
        "sleep_records": sleeps,
        "task_state_bits_mixed": task_state_bits,
        "novel_composition": novel,
        "world": world.config_dict(),
    }
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out / "compiled_routes.json").write_text(
        json.dumps({k: list(v) for k, v in model.compiled.items()}, indent=2),
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def _force_all(model: ConsolidatingContinuous, world: World, completed) -> None:
    basis = list(model.base.basis)
    steps = model.base.task_steps
    for task_index in completed:
        task = world.tasks[task_index]
        if task.task_id in model.compiled:
            continue
        x = _tensor(task.train_x)
        y = torch.as_tensor(task.train_y, dtype=torch.float64)
        outputs = _route_outputs_batched(basis, x, steps).double()
        sq = ((outputs - y.unsqueeze(0)) ** 2).sum(dim=2).sum(dim=1)
        model.compiled[task.task_id] = _index_to_route(
            int(torch.argmin(sq)), len(basis), steps
        )


def _adapt_novel(model, world, config, task_lr) -> dict[str, object]:
    route, train_x, train_y, eval_x, eval_y = _novel_data(world, config, 0)
    for p in model.shared_parameters():
        p.requires_grad_(False)
    novel_id = "task_novel_composition_0"
    code = model.begin_task(novel_id)
    optimizer = torch.optim.Adam([code], lr=task_lr)
    curve: dict[str, float] = {}
    for n_seen in range(33):
        if n_seen in {0, 1, 2, 4, 8, 16, 32}:
            model.eval()
            with torch.no_grad():
                pred = model(_tensor(eval_x), novel_id).numpy()
            curve[str(n_seen)] = nmse(pred, eval_y)
        if n_seen == 32:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(_tensor(train_x[n_seen : n_seen + 1]), novel_id)
        loss = torch.nn.functional.mse_loss(
            pred, _tensor(train_y[n_seen : n_seen + 1])
        )
        loss.backward()
        optimizer.step()
    for p in model.shared_parameters():
        p.requires_grad_(True)
    return {"teacher_route": list(route), "nmse_by_support": curve}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument("--reuse-rho", type=float, default=1.0)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        world=replace(
            config.world, seed=args.world_seed, reuse_rho=args.reuse_rho
        ),
        output_directory=args.output,
    )
    if (args.output / "summary.json").exists():
        print("summary exists; skipping")
        return
    summary = run(config, force_compile=args.force_compile)
    print(
        f"consolidating rho={args.reuse_rho} world={args.world_seed}: "
        f"loss={summary['cumulative_prequential_gaussian_log_loss']:.1f} "
        f"compiled={summary['compiled_tasks']}/64 "
        f"novel32={summary['novel_composition']['nmse_by_support']['32']:.4f}"
    )


if __name__ == "__main__":
    main()
