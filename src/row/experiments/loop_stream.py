"""Loop tasks for online streams (SUCCESSOR_LADDER.md, B2-online). Additive.

`loop_tasks(world, n, examples, evaluation, d)`: n tasks `z = x; repeat at most K times: stop if w.z <= 0, else
z = P(z)` on the rotated world's library (the B2 Tier 0 census construction): P one teacher primitive, w a random unit
hyperplane, K = `K`. Each task's `program` is (P,) (its LENGTH, 1, sets the planned depth of the loop body; the
identity of P is never shown to the learner). Task ids from stream 47 (canonical 21, anchors 41, branches 43).
Draws per task from seed [5800, world, i]; inputs N(0, I). The teacher executes the targets.

`stream(world, stream, n_loop)`: O2's SHUFFLED depth-3 stream (60 length-1, 64 length-2, 64 canonical length-3) with
n_loop loop tasks added, all in one random order seeded [1947, world] at stream 0, else [1947, world, stream]; the
canonical 64 remain the scored straight-line set.

`body_depth=2` (B2-depth2): the body is P2(P1(z)) with P1 != P2 two teacher primitives (`program` (P1, P2), planned
depth 2); ids from stream 49, draws [5810, world, i], meta 'P' = [P1, P2]. Default 1 is the construction above, bitwise.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from row.experiments import o2_online_reliability as o2
from row.world import Task, Program, _draw_opaque_task_ids, _rng

ORDER_BASE = 1947
K = 6


def teacher_loop(P, w, x):
    z = np.array(x, dtype=np.float64)
    k = np.zeros(len(z), dtype=int)
    for _ in range(K):
        go = z @ w > 0
        if not go.any():
            break
        z[go] = P(z[go])
        k += go
    return z, k


def body_fn(library, ps):
    if len(ps) == 1:
        return library[ps[0]]
    return lambda z: library[ps[1]](library[ps[0]](z))


def loop_tasks(world, n, examples, evaluation, d, body_depth=1):
    library = world.library
    ids = _draw_opaque_task_ids(_rng(world.config.seed, 47 if body_depth == 1 else 49), n)
    tasks, meta = [], {}
    for i, tid in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([5800 if body_depth == 1 else 5810, world.config.seed, i]))
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        if body_depth == 1:
            ps = (int(rng.integers(0, len(library))),)
        else:
            ps = tuple(int(v) for v in rng.choice(len(library), size=2, replace=False))
        tx, ex = rng.normal(size=(examples, d)), rng.normal(size=(evaluation, d))
        ty, _ = teacher_loop(body_fn(library, ps), w, tx)
        ey, ek = teacher_loop(body_fn(library, ps), w, ex)
        tasks.append(Task(task_id=tid, program=Program(ps), teacher_library=library, train_x=tx, train_y=ty,
                          eval_x=ex, eval_y=ey))
        meta[tid] = {'P': ps[0] if body_depth == 1 else list(ps), 'w': w.tolist(), 'eval_count_hist': np.bincount(ek, minlength=K + 1).tolist()}
    return tasks, meta


def order_entropy(world, stream):
    return [ORDER_BASE, world] if stream == 0 else [ORDER_BASE, world, stream]


def stream(world_seed, s, n_loop, body_depth=1):
    cfg3, world3, st, plan, canonical = o2.build_stream('SHUFFLED', world_seed, s)
    ltasks, meta = loop_tasks(world3, n_loop, cfg3.world.examples_per_task, cfg3.world.evaluation_examples,
                              cfg3.world.state_dim, body_depth)
    tasks = list(st) + ltasks
    order = np.random.default_rng(np.random.SeedSequence(order_entropy(world_seed, s))).permutation(len(tasks))
    mixed = [tasks[int(i)] for i in order]
    if len({t.task_id for t in mixed}) != len(mixed):
        raise ValueError('task ids collide')
    plan = {t.task_id: len(t.program.primitive_ids) for t in mixed}
    return cfg3, dataclasses.replace(world3, tasks=tuple(mixed)), mixed, plan, canonical, ltasks, meta
