"""Order-free online stream at any depth >= 4 (decision 17's depth rung). Generalizes `d4_stream` without touching it.

A depth-D stream holds, in one stream-seeded random order:
- 64 canonical length-D tasks: `generate_rotated_world` with `program_length=D` (distinct programs), scored;
- 60 length-1 and 64 tasks of every length 2..D-1: `generate_curriculum_world` (programs with replacement), task ids
  from stream 41 keyed by length, the depth-3 line's anchor construction.
Learner `PlannedDepthRotatedLearner` with `task_steps=D`. Model seed 5000; replay seed None at stream 0, else
SeedSequence([7500, w, s]); order seed [1926 + D, w] at stream 0, else [1926 + D, w, s] (D=4 gives d4_stream's 1930).
At D = 4 this reproduces `d4_stream.build_stream` exactly (test).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments import o2_online_reliability as o2
from row.models.online_variable_depth import PlannedDepthRotatedLearner
from row.rotated_world import generate_rotated_world

MODEL_SEED = 5000
CANONICAL_TASKS = 64


def anchors(depth):
    return {1: 60, **{length: 64 for length in range(2, depth)}}


def config(depth, world_seed):
    base = load_config('configs/v1.yaml')
    cfg = dataclasses.replace(base, world=dataclasses.replace(base.world, seed=world_seed, program_length=depth,
                                                              tasks=CANONICAL_TASKS))
    return dataclasses.replace(cfg, discrete_model=dataclasses.replace(cfg.discrete_model, task_steps=depth,
                                                                       seed=MODEL_SEED))


def order_entropy(depth, world, stream):
    base = 1926 + depth
    return [base, world] if stream == 0 else [base, world, stream]


def build_stream(depth, world_seed, stream):
    cfg = config(depth, world_seed)
    canonical_world = generate_rotated_world(cfg.world)
    tasks = list(canonical_world.tasks)
    for length, n in anchors(depth).items():
        wcfg = CurriculumWorldConfig.from_world(cfg.world, program_length=length, tasks=n, stage=f'length-{length}')
        tasks += list(generate_curriculum_world(wcfg).tasks)
    order = np.random.default_rng(np.random.SeedSequence(order_entropy(depth, world_seed, stream))).permutation(len(tasks))
    stream_tasks = [tasks[int(i)] for i in order]
    if len({t.task_id for t in stream_tasks}) != len(stream_tasks):
        raise ValueError('task ids collide across lengths')
    plan = {t.task_id: len(t.program.primitive_ids) for t in stream_tasks}
    return cfg, canonical_world, stream_tasks, plan, list(canonical_world.tasks)


def planned_model(depth, cfg, plan):
    sel = cfg.discrete_model
    return PlannedDepthRotatedLearner(
        d=cfg.world.state_dim, operator_slots=sel.operator_slots, operator_rank=sel.operator_rank,
        task_steps=depth, alpha=sel.operator_alpha_init, initial_temperature=sel.initial_temperature,
        final_temperature=sel.final_temperature, seed=sel.seed, learnable_alpha=sel.learnable_alpha,
        activation=sel.operator_activation, depth_plan=plan)


def replay_seed_for(world, stream):
    return o2.replay_seed_for(world, stream)
