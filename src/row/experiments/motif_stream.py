"""Online stream on a MOTIF (hierarchical) task distribution over the rotated family (decision 23; M0 generator).

One stream holds, in one stream-seeded random order (316 tasks, D2's count):
- 60 length-1 and 64 length-2 anchor tasks: `generate_curriculum_world`, exactly as `dn_stream` builds them;
- 192 SHALLOW motif tasks: one of the world's 6 motifs (distinct non-constant length-3 primitive sequences) with
  0-2 random filler primitives split around it (depth 3-5), scored at the end (`scored`).
Held out (never in the stream): 24 DEEP tasks, three motifs drawn with replacement (depth 9).
Motifs from SeedSequence([6400, w]); shallow/deep programs [6401, w]; examples [6402, w, kind, i]; task ids
[6403, w] checked unique against the anchors; order [6404, w] at stream 0, else [6404, w, s].
Learner `PlannedDepthRotatedLearner` with `task_steps=5` (`dn_stream.planned_model`).
"""
from __future__ import annotations

import numpy as np

from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments import dn_stream as dn
from row.rotated_world import rotated_library
from row.world import Program, Task, World, _draw_opaque_task_ids

DEPTH = 5
DEEP_DEPTH = 9
K_MOTIFS, MOTIF_LEN = 6, 3
N_SHALLOW, N_DEEP, MAX_FILLER = 192, 24, 2
ANCHORS = {1: 60, 2: 64}


def motifs_of(w, n_prims):
    rng = np.random.default_rng(np.random.SeedSequence([6400, w]))
    out = []
    while len(out) < K_MOTIFS:
        m = tuple(int(v) for v in rng.integers(0, n_prims, MOTIF_LEN))
        if len(set(m)) > 1 and m not in out:
            out.append(m)
    return out


def programs(w, n_prims, n_deep=N_DEEP):
    motifs = motifs_of(w, n_prims)
    rng = np.random.default_rng(np.random.SeedSequence([6401, w]))
    shallow = []
    for _ in range(N_SHALLOW):
        m = motifs[int(rng.integers(0, K_MOTIFS))]
        f = int(rng.integers(0, MAX_FILLER + 1))
        left = int(rng.integers(0, f + 1))
        fill = tuple(int(v) for v in rng.integers(0, n_prims, f))
        shallow.append(fill[:left] + m + fill[left:])
    deep = [sum((motifs[int(k)] for k in rng.integers(0, K_MOTIFS, DEEP_DEPTH // MOTIF_LEN)), ())
            for _ in range(n_deep)]
    return motifs, shallow, deep


def make_tasks(cfg, library, progs, kind, ids):
    d = cfg.world.state_dim
    out = []
    for i, (prog, tid) in enumerate(zip(progs, ids, strict=True)):
        rng = np.random.default_rng(np.random.SeedSequence([6402, cfg.world.seed, kind, i]))
        tx = rng.normal(size=(cfg.world.examples_per_task, d))
        ex = rng.normal(size=(cfg.world.evaluation_examples, d))
        p = Program(tuple(prog))
        out.append(Task(task_id=tid, program=p, teacher_library=library, train_x=tx,
                        train_y=p.execute(library, tx), eval_x=ex, eval_y=p.execute(library, ex)))
    return out


def build_stream(w, s):
    cfg = dn.config(DEPTH, w)
    library = rotated_library(cfg.world)
    anchors = []
    for length, n in ANCHORS.items():
        wcfg = CurriculumWorldConfig.from_world(cfg.world, program_length=length, tasks=n, stage=f'length-{length}')
        anchors += list(generate_curriculum_world(wcfg).tasks)
    motifs, shallow, deep = programs(w, len(library))
    ids = _draw_opaque_task_ids(np.random.default_rng(np.random.SeedSequence([6403, w])), N_SHALLOW + N_DEEP,
                                forbidden={t.task_id for t in anchors})
    shallow_tasks = make_tasks(cfg, library, shallow, 0, ids[:N_SHALLOW])
    deep_tasks = make_tasks(cfg, library, deep, 1, ids[N_SHALLOW:])
    tasks = anchors + shallow_tasks
    entropy = [6404, w] if s == 0 else [6404, w, s]
    order = np.random.default_rng(np.random.SeedSequence(entropy)).permutation(len(tasks))
    st = [tasks[int(i)] for i in order]
    if len({t.task_id for t in st + deep_tasks}) != len(st) + len(deep_tasks):
        raise ValueError('task ids collide')
    plan = {t.task_id: len(t.program.primitive_ids) for t in st}
    world = World(config=cfg.world, library=library, tasks=tuple(st))
    return cfg, world, st, plan, shallow_tasks, deep_tasks, motifs


def build_stream_with_deep(w, s, n_deep):
    """M2: the same stream with the first `n_deep` deep (depth-9) tasks IN it. Shallow tasks, anchors, motifs and
    the first deep programs are identical to `build_stream`; order [6406, w] at stream 0, else [6406, w, s].
    Learner `task_steps` = DEEP_DEPTH (`dn.config(DEEP_DEPTH, w)`)."""
    cfg = dn.config(DEEP_DEPTH, w)
    library = rotated_library(cfg.world)
    anchors = []
    for length, n in ANCHORS.items():
        wcfg = CurriculumWorldConfig.from_world(cfg.world, program_length=length, tasks=n, stage=f'length-{length}')
        anchors += list(generate_curriculum_world(wcfg).tasks)
    motifs, shallow, deep = programs(w, len(library), n_deep)
    ids = _draw_opaque_task_ids(np.random.default_rng(np.random.SeedSequence([6403, w])), N_SHALLOW + n_deep,
                                forbidden={t.task_id for t in anchors})
    shallow_tasks = make_tasks(cfg, library, shallow, 0, ids[:N_SHALLOW])
    deep_tasks = make_tasks(cfg, library, deep, 1, ids[N_SHALLOW:])
    tasks = anchors + shallow_tasks + deep_tasks
    entropy = [6406, w] if s == 0 else [6406, w, s]
    order = np.random.default_rng(np.random.SeedSequence(entropy)).permutation(len(tasks))
    st = [tasks[int(i)] for i in order]
    if len({t.task_id for t in st}) != len(st):
        raise ValueError('task ids collide')
    plan = {t.task_id: len(t.program.primitive_ids) for t in st}
    world = World(config=cfg.world, library=library, tasks=tuple(st))
    return cfg, world, st, plan, shallow_tasks, deep_tasks, motifs
