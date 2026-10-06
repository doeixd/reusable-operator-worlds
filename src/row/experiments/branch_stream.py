"""Branch tasks for online streams (SUCCESSOR_LADDER.md, B1-online). Additive.

`branch_tasks(world_seed, n, examples, evaluation)`: n tasks `IF(w.x > 0, A, B)` on the rotated world's library:
w a random unit hyperplane, A != B random length-2 teacher programs, the decision on the INPUT (B1's INPUT
variant). Each task's `program` is A (its LENGTH sets the task's planned depth, 2; its identity is never shown to
the learner). Task ids from stream 43 (canonical ids use 21, anchors 41, keyed by length). Draws per task from
seed [5600, world, i]; inputs from N(0, I). The teacher executes the targets.

`stream(world, stream, n_branch)`: O2's SHUFFLED depth-3 stream (60 length-1, 64 length-2, 64 canonical length-3
tasks) with n_branch branch tasks added, all in one random order seeded [1945, world] at stream 0, else
[1945, world, stream] (`order_entropy`); the canonical 64 remain the scored straight-line set.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from row.experiments import o2_online_reliability as o2
from row.world import Program, Task, _draw_opaque_task_ids, _rng

ORDER_BASE = 1945


def branch_tasks(world, n, examples, evaluation, d):
    library = world.library
    ids = _draw_opaque_task_ids(_rng(world.config.seed, 43), n)
    tasks, meta = [], {}
    for i, tid in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([5600, world.config.seed, i]))
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        A = tuple(int(o) for o in rng.integers(0, len(library), size=2))
        B = A
        while B == A:
            B = tuple(int(o) for o in rng.integers(0, len(library), size=2))
        pa, pb = Program(A), Program(B)
        tx, ex = rng.normal(size=(examples, d)), rng.normal(size=(evaluation, d))

        def target(x):
            return np.where((x @ w > 0)[:, None], pa.execute(library, x), pb.execute(library, x))

        tasks.append(Task(task_id=tid, program=pa, teacher_library=library, train_x=tx, train_y=target(tx),
                          eval_x=ex, eval_y=target(ex)))
        meta[tid] = {'A': list(A), 'B': list(B), 'w': w.tolist()}
    return tasks, meta


def order_entropy(world, stream):
    return [ORDER_BASE, world] if stream == 0 else [ORDER_BASE, world, stream]


def stream(world_seed, s, n_branch):
    cfg3, world3, st, plan, canonical = o2.build_stream('SHUFFLED', world_seed, s)
    btasks, meta = branch_tasks(world3, n_branch, cfg3.world.examples_per_task, cfg3.world.evaluation_examples,
                                cfg3.world.state_dim)
    tasks = list(st) + btasks
    order = np.random.default_rng(np.random.SeedSequence(order_entropy(world_seed, s))).permutation(len(tasks))
    mixed = [tasks[int(i)] for i in order]
    if len({t.task_id for t in mixed}) != len(mixed):
        raise ValueError('task ids collide')
    plan = {t.task_id: len(t.program.primitive_ids) for t in mixed}
    return cfg3, dataclasses.replace(world3, tasks=tuple(mixed)), mixed, plan, canonical, btasks, meta
