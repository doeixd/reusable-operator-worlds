"""Within-lifetime amortized program inference (V2 Model 7b/7c, step 006).

Wake: the standard continuous protocol. Sleep (after 8, 16, and 32
completed tasks): train a small DeepSets compiler C_phi on THIS lifetime's
solved tasks only — input is a set of (x, y) demonstrations, output is a
per-step mixture code, and training is behavioral (query-set prediction
loss through the frozen basis), respecting code non-identifiability. After
the first sleep, every new task's code is warm-started from
C_phi(first demonstration) after example 1 arrives; ordinary online
gradient descent continues from there (the 7c hybrid).

7b-dream: half of each compiler training batch is FANTASY tasks — random
hard routes over the learner's own current basis, executed to generate
synthetic demonstrations — at matched total compiler training compute.
Pre-registered falsifier: the dream benefit must shrink as rho falls.

H10 metric: cumulative prequential cost of tasks 9-64 versus the plain
continuous baseline on identical worlds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.config import load_config
from row.experiments.learned_lifetime import TaskReplayBuffer, _tensor
from row.metrics import gaussian_nll, nmse
from row.models import ContinuousBasisLearner
from row.world import World

SLEEPS = (8, 16, 32)
COMPILER_STEPS_PER_SLEEP = 300
DEMONSTRATIONS = 16
QUERIES = 16


class SetCompiler(nn.Module):
    def __init__(self, d: int, task_steps: int, slots: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(
                nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
            )
            self.head = nn.Sequential(
                nn.Linear(64, 64), nn.GELU(), nn.Linear(64, task_steps * slots)
            )
        self.task_steps = task_steps
        self.slots = slots

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        pooled = self.encoder(torch.cat((x, y), dim=-1)).mean(dim=0)
        return self.head(pooled).reshape(self.task_steps, self.slots)


def _mixture_forward(base: ContinuousBasisLearner, logits: Tensor, x: Tensor) -> Tensor:
    coefficients = torch.softmax(logits, dim=-1)
    z = x
    for step in range(base.task_steps):
        candidates = torch.stack([op(z) for op in base.basis], dim=0)
        z = torch.sum(
            coefficients[step].view(base.operator_slots, 1, 1) * candidates, dim=0
        )
    return z


def _train_compiler(
    compiler: SetCompiler,
    base: ContinuousBasisLearner,
    real_tasks: list,
    dream: bool,
    rng: np.random.Generator,
    steps: int,
) -> None:
    optimizer = torch.optim.Adam(compiler.parameters(), lr=1e-3)
    for p in base.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        use_dream = dream and rng.random() < 0.5
        if use_dream:
            route = rng.integers(0, base.operator_slots, size=base.task_steps)
            x = torch.as_tensor(
                rng.normal(size=(DEMONSTRATIONS + QUERIES, base.basis[0].V.shape[1])),
                dtype=torch.float32,
            )
            with torch.no_grad():
                z = x
                for slot in route:
                    z = base.basis[int(slot)](z)
                y = z
            demo = DEMONSTRATIONS
        else:
            task = real_tasks[int(rng.integers(0, len(real_tasks)))]
            n = len(task.train_x)
            demo = min(DEMONSTRATIONS, max(1, n // 2))
            queries = min(QUERIES, n - demo)
            span = n - demo - queries
            start = 0 if span <= 0 else int(rng.integers(0, span + 1))
            x = _tensor(task.train_x[start : start + demo + queries])
            y = _tensor(task.train_y[start : start + demo + queries])
        logits = compiler(x[:demo], y[:demo])
        prediction = _mixture_forward(base, logits, x[demo:])
        loss = torch.nn.functional.mse_loss(prediction, y[demo:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for p in base.parameters():
        p.requires_grad_(True)


def run(config, dream: bool) -> dict[str, object]:
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
    compiler = SetCompiler(
        config.world.state_dim, mc.task_steps, mc.operator_slots, mc.seed + 31
    )
    compiler_trained = False
    rng = np.random.default_rng(np.random.SeedSequence([config.world.seed, 71]))
    optimizer = torch.optim.AdamW(
        base.shared_parameters(), lr=mc.global_learning_rate,
        weight_decay=mc.weight_decay,
    )
    replay = TaskReplayBuffer(mc.seed + 1)
    sigma = config.evaluation.gaussian_sigma
    cumulative = 0.0
    nll_by_task: list[float] = []
    for lifetime_index, task in enumerate(world.tasks):
        code = base.begin_task(task.task_id)
        optimizer.add_param_group(
            {"params": [code], "lr": mc.task_learning_rate, "weight_decay": 0.0}
        )
        task_nll = 0.0
        for n_seen in range(config.world.examples_per_task):
            x = _tensor(task.train_x[n_seen : n_seen + 1])
            base.eval()
            with torch.no_grad():
                prediction = base(x, task.task_id).numpy()
            online = gaussian_nll(prediction, task.train_y[n_seen : n_seen + 1], sigma)
            cumulative += online
            task_nll += online
            if n_seen == 0 and compiler_trained:
                with torch.no_grad():
                    demo_n = 1
                    logits = compiler(
                        _tensor(task.train_x[:demo_n]), _tensor(task.train_y[:demo_n])
                    )
                    code.copy_(logits)
            replay_items = replay.sample(int(round(mc.replay_ratio)))
            batch_x = [task.train_x[n_seen], *(i[0] for i in replay_items)]
            batch_y = [task.train_y[n_seen], *(i[1] for i in replay_items)]
            ids = [task.task_id, *(i[2] for i in replay_items)]
            base.train()
            optimizer.zero_grad(set_to_none=True)
            out = base.forward_tasks(_tensor(np.stack(batch_x)), ids)
            loss = torch.nn.functional.mse_loss(out, _tensor(np.stack(batch_y)))
            loss.backward()
            optimizer.step()
        nll_by_task.append(task_nll)
        replay.add_task(task, mc.replay_examples_per_task)
        if (lifetime_index + 1) in SLEEPS:
            _train_compiler(
                compiler, base, list(world.tasks[: lifetime_index + 1]),
                dream, rng, COMPILER_STEPS_PER_SLEEP,
            )
            compiler_trained = True

    summary = {
        "model": "compiler_continuous" + ("_dream" if dream else ""),
        "cumulative_prequential_gaussian_log_loss": cumulative,
        "per_task_prequential_nll": nll_by_task,
        "post_sleep_nll_tasks_9_to_64": float(sum(nll_by_task[8:])),
        "compiler": {
            "sleeps": list(SLEEPS),
            "steps_per_sleep": COMPILER_STEPS_PER_SLEEP,
            "dream": dream,
            "dream_fraction": 0.5 if dream else 0.0,
            "demonstrations_for_warm_start": 1,
        },
        "world": {"seed": config.world.seed, "reuse_rho": config.world.reuse_rho},
    }
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--reuse-rho", type=float, default=1.0)
    parser.add_argument("--dream", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        print("summary exists; skipping")
        return
    config = load_config(args.config)
    config = replace(
        config,
        world=replace(config.world, seed=args.world_seed, reuse_rho=args.reuse_rho),
        output_directory=args.output,
    )
    summary = run(config, dream=args.dream)
    print(
        f"compiler{'-dream' if args.dream else ''} rho={args.reuse_rho} "
        f"world={args.world_seed}: loss={summary['cumulative_prequential_gaussian_log_loss']:.1f} "
        f"post-sleep={summary['post_sleep_nll_tasks_9_to_64']:.1f}"
    )


if __name__ == "__main__":
    main()
