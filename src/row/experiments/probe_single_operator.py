"""EXPLORATORY Tier-1 probe: can library and routes co-form when routing is clustering?

NOT PREREGISTERED and never a verdict (AGENTS.md compute-economy tiers). J1
showed that a hard commitment made on a random library locks in, and J0 that
gradient routing needs an already-good library. Both concern length-3
programs, where a task's route is one of 1,728. This probe asks the same
question at the easiest possible setting: length-ONE tasks, where a route is
just "which of 12 slots", many tasks share an operation, and co-formation
reduces to clustering tasks onto slots.

One development world (seed 1), a reduced budget, three arms: ORACLE (pinned
true assignment), SEARCH (re-chosen every RESEARCH updates by exhaustive
support-only search), LEARNED (the ordinary soft task codes). Its purpose is
to decide whether a length curriculum (J1c) is worth freezing.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_interference import (
    _assignment, _flat_shared, _relative_change, oracle_routes, pin_code, score,
)
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.audit_j1_search_loop import ari
from row.experiments.learned_lifetime import _shared_optimizer, _training_values
from row.experiments.so1_storage import atomic_json, environment, log_line, now
from row.rotated_world import Program

OUTPUT = Path("reports/probe_single_operator.json")
ROOT = Path("artifacts/probe_single_operator")
WORLD_SEED = 1
TASKS = 60
LENGTH = 1
UPDATES = 16384
BATCH = 2
RESEARCH = 512
STREAM = 1707


def probe_config():
    base = load_config("configs/v1.yaml")
    return replace(base,
                   world=CurriculumWorldConfig.from_world(replace(base.world, seed=WORLD_SEED),
                                                         program_length=LENGTH, tasks=TASKS, stage="single-operator"),
                   discrete_model=replace(base.discrete_model, task_steps=LENGTH))


def run_arm(mode: str, updates: int = UPDATES, research: int = RESEARCH) -> dict:
    torch.set_num_threads(1)
    cfg = probe_config()
    world = generate_curriculum_world(cfg.world)
    global_lr, task_lr, weight_decay, _, _, _, _ = _training_values(cfg, "rotated_discrete")
    model = build_fast(cfg)
    rounds = []

    def search() -> dict:
        library = FrozenLibrary(model)
        chosen = {}
        for task in world.tasks:
            support = library.all_route_support_mse(torch.tensor(task.train_x, dtype=torch.float32),
                                                   torch.tensor(task.train_y, dtype=torch.float32))
            chosen[task.task_id] = tuple(unflatten(int(torch.argmin(support)), library.slots, library.steps))
        return chosen

    oracle = mode == "oracle"
    routes = oracle_routes(world, _assignment(cfg, world)) if oracle else (search() if mode == "search" else None)
    codes = {}
    for task in world.tasks:
        code = model.begin_task(task.task_id)
        if routes is not None:
            pin_code(code, routes[task.task_id])
        codes[task.task_id] = code
    optimizer = _shared_optimizer(model, global_lr, weight_decay)
    if mode == "learned":
        optimizer.add_param_group({"params": list(codes.values()), "lr": task_lr, "weight_decay": 0.0})
    initial = _flat_shared(model)
    all_x = torch.tensor(np.concatenate([t.train_x for t in world.tasks]), dtype=torch.float32)
    all_y = torch.tensor(np.concatenate([t.train_y for t in world.tasks]), dtype=torch.float32)
    all_ids = [t.task_id for t in world.tasks for _ in range(len(t.train_x))]
    rng = np.random.default_rng(np.random.SeedSequence([STREAM, WORLD_SEED, 0]))
    trajectory = {"0": score(model, world)["median"]}
    started = time.perf_counter()
    for update in range(1, updates + 1):
        if mode == "search" and update > 1 and (update - 1) % research == 0:
            new = search()
            rounds.append(float(np.mean([new[t] != routes[t] for t in routes])))
            routes = new
            for task_id, code in codes.items():
                pin_code(code, routes[task_id])
        model.set_training_progress((update - 1) / max(1, updates - 1))
        model.train()
        idx = rng.integers(0, len(all_x), size=BATCH)
        indices = torch.tensor(idx, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            model.forward_tasks(all_x.index_select(0, indices), [all_ids[int(i)] for i in idx]),
            all_y.index_select(0, indices))
        loss.backward()
        optimizer.step()
        if update in {updates // 8, updates // 4, updates // 2, updates}:
            trajectory[str(update)] = score(model, world)["median"]
    final = score(model, world)
    hard = model.hard_routes()
    teacher = {t.task_id: int(t.program.primitive_ids[0]) for t in world.tasks}
    ids = list(teacher)
    return {"mode": mode, "updates": updates, "terminal_median": final["median"], "trajectory": trajectory,
            "below_0.05": final["below_0.05"], "tasks": len(ids),
            "ari_slot_vs_primitive": ari([hard[t][0] for t in ids], [teacher[t] for t in ids]),
            "slots_used": len({hard[t][0] for t in ids}),
            "route_change_rounds": rounds,
            "shared_relative_change": _relative_change(initial, _flat_shared(model)),
            "seconds": round(time.perf_counter() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--research", type=int, default=RESEARCH)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    out = {"status": "EXPLORATORY, not preregistered, never a verdict", "git_commit": git_commit(),
           "world": {"seed": WORLD_SEED, "tasks": TASKS, "program_length": LENGTH, "generator": "curriculum_world"},
           "budget": {"updates": args.updates, "batch": BATCH, "research_every": args.research, "stream": STREAM},
           "environment": environment(), "started_utc": now(), "arms": {}}
    for mode in ("oracle", "search", "learned"):
        log_line(ROOT / "run.log", f"[{mode}] start")
        result = run_arm(mode, args.updates, args.research)
        out["arms"][mode] = result
        atomic_json(OUTPUT, out)
        log_line(ROOT / "run.log", f"[{mode}] median {result['terminal_median']:.4f} "
                                   f"below_0.05 {result['below_0.05']}/{result['tasks']} "
                                   f"ARI {result['ari_slot_vs_primitive']:.3f} slots {result['slots_used']} "
                                   f"({result['seconds']}s)")
    out["finished_utc"] = now()
    atomic_json(OUTPUT, out)
    print(json.dumps({m: {k: r[k] for k in ("terminal_median", "below_0.05", "ari_slot_vs_primitive", "slots_used")}
                      for m, r in out["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
