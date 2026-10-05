"""Depth-4 order-free online stream (decision 17, 2026-10-04): the O2 `SHUFFLED` construction one level deeper.

Additive; nothing existing changes. A depth-4 stream holds, in one stream-seeded random order:
- 64 canonical length-4 tasks: `generate_rotated_world` with `program_length=4` (distinct programs; the cap is
  6**4 = 1,296), task ids from the canonical stream 21;
- 60 length-1, 64 length-2 and 64 length-3 tasks: `generate_curriculum_world` (programs with replacement), task ids
  from stream 41 keyed by length, exactly as the depth-3 line's anchors.
252 tasks. The learner is `PlannedDepthRotatedLearner` with `task_steps=4`; each task runs at its own depth. Scored
on the 64 canonical length-4 tasks. Model seed 5000; replay seed None at stream 0, else SeedSequence([7500, w, s]);
order seed [1930, w] at stream 0, else [1930, w, s] (1930 is unused by every earlier order stream: 1920, 1921).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments import o2_online_reliability as o2
from row.models.online_variable_depth import PlannedDepthRotatedLearner
from row.rotated_world import generate_rotated_world

DEPTH = 4
MODEL_SEED = 5000
ORDER_BASE = 1930
ANCHORS = {1: 60, 2: 64, 3: 64}
CANONICAL_TASKS = 64


def config4(world_seed: int):
    base = load_config('configs/v1.yaml')
    cfg = dataclasses.replace(base, world=dataclasses.replace(base.world, seed=world_seed, program_length=DEPTH,
                                                              tasks=CANONICAL_TASKS))
    return dataclasses.replace(cfg, discrete_model=dataclasses.replace(cfg.discrete_model, task_steps=DEPTH,
                                                                       seed=MODEL_SEED))


def order_entropy(world: int, stream: int) -> list[int]:
    return [ORDER_BASE, world] if stream == 0 else [ORDER_BASE, world, stream]


def build_stream(world_seed: int, stream: int):
    cfg = config4(world_seed)
    canonical_world = generate_rotated_world(cfg.world)
    tasks = list(canonical_world.tasks)
    for length, n in ANCHORS.items():
        wcfg = CurriculumWorldConfig.from_world(cfg.world, program_length=length, tasks=n, stage=f'length-{length}')
        tasks += list(generate_curriculum_world(wcfg).tasks)
    order = np.random.default_rng(np.random.SeedSequence(order_entropy(world_seed, stream))).permutation(len(tasks))
    stream_tasks = [tasks[int(i)] for i in order]
    if len({t.task_id for t in stream_tasks}) != len(stream_tasks):
        raise ValueError('task ids collide across lengths')
    plan = {t.task_id: len(t.program.primitive_ids) for t in stream_tasks}
    return cfg, canonical_world, stream_tasks, plan, list(canonical_world.tasks)


def planned_model(cfg, plan):
    sel = cfg.discrete_model
    return PlannedDepthRotatedLearner(
        d=cfg.world.state_dim, operator_slots=sel.operator_slots, operator_rank=sel.operator_rank,
        task_steps=DEPTH, alpha=sel.operator_alpha_init, initial_temperature=sel.initial_temperature,
        final_temperature=sel.final_temperature, seed=sel.seed, learnable_alpha=sel.learnable_alpha,
        activation=sel.operator_activation, depth_plan=plan)


def replay_seed_for(world: int, stream: int):
    return o2.replay_seed_for(world, stream)
