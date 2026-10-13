"""Generality-program generator, version g1 (`GENERALITY_PROGRAM.md`, Track G, G0).

The rotated primitive with two new knobs, the hidden GAIN `g` and the SHARED rotation component `m`:

    P_k(z) = Q_k (z + a U_k tanh(g V_k z + b_k)),   Q_k = polar(G_k + m G_shared)

`G_k` is the current generator's own rotation draw (stream 12) and `G_shared` one draw common to the world
(stream 50, unused elsewhere): at large `m` every `Q_k` is near one shared rotation, so primitives become similar
and a wrong program can nearly fit (route inference gets harder). `a` (`alpha`), the number of primitives `K`
(`teacher_primitives`) and the state width `d` (`state_dim`) are already `WorldConfig` fields and are passed through
unchanged. Every knob stays inside the learner's operator family (`RotatedLearnedOperator` learns `U`, `V`, `b`,
`alpha` and `Q` freely, so `g` is absorbed by `V`), so a learner failure on a g1 world cannot be a representational
mismatch the way the GELU control was.

At `g = 1, m = 0` every function here reproduces `rotated_world` / `curriculum_world` BITWISE (the multiplication is skipped,
not performed with 1.0), which `tests/test_general_world.py` asserts on real worlds. Nothing here changes
`WorldConfig`, `World`, `Program` or any existing generator: `g` arrives as an argument and callers record it in
their own artifacts, as `mixed_world` records its rho profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from row.rotated_world import RotatedPrimitive
from row.world import Array, Program, Task, World, _draw_opaque_task_ids, _opaque_task_ids, _rng, _sample_programs

VERSION = 'g1'


@dataclass(frozen=True)
class GainPrimitive:
    """`RotatedPrimitive` with the pre-activation scaled by `gain`; identical draws, identical `Q`."""

    base: RotatedPrimitive
    gain: float = 1.0

    def __call__(self, z: Array) -> Array:
        if self.gain == 1.0:
            return self.base(z)
        p = self.base
        z = np.asarray(z, dtype=np.float64)
        hidden = np.tanh(self.gain * (z @ p.V.T) + p.b)
        return (z + p.alpha * (hidden @ p.U.T)) @ p.Q.T


def _shared_rotation(p: RotatedPrimitive, seed: int, k: int, d: int, shared: float) -> RotatedPrimitive:
    # Polar factor (nearest orthogonal matrix), not QR: QR of a sum converges to the shared factor far more slowly
    # (mean pairwise ||Q_i - Q_j|| 3.2 against 2.4 at m = 4, world 70). Polar of a Gaussian is Haar like QR's, so
    # m > 0 is a fresh, equally random rotation family; m = 0 skips this function and keeps the QR draw bitwise.
    u, _, vt = np.linalg.svd(_rng(seed, 12, k).normal(size=(d, d)) + shared * _rng(seed, 50).normal(size=(d, d)))
    Q = u @ vt
    return RotatedPrimitive(U=p.U, V=p.V, b=p.b, Q=Q, alpha=p.alpha)


def general_library(config, gain: float = 1.0, shared: float = 0.0) -> tuple:
    """The g1 library. At `gain == 1, shared == 0` it is exactly `rotated_library(config)`."""
    if config.reuse_rho != 1.0:
        raise ValueError('general worlds require reuse_rho == 1.0 (see generate_rotated_world)')
    base = tuple(RotatedPrimitive.random(seed=config.seed, primitive_index=k, d=config.state_dim,
                                         rank=config.teacher_rank, alpha=config.alpha)
                 for k in range(config.teacher_primitives))
    if shared != 0.0:
        base = tuple(_shared_rotation(p, config.seed, k, config.state_dim, float(shared)) for k, p in enumerate(base))
    if gain == 1.0:
        return base
    return tuple(GainPrimitive(p, float(gain)) for p in base)


def _tasks(config, library, programs, task_ids):
    tasks = []
    for index, (program, task_id) in enumerate(zip(programs, task_ids, strict=True)):
        train_x = _rng(config.seed, 30, index).normal(size=(config.examples_per_task, config.state_dim))
        eval_x = _rng(config.seed, 31, index).normal(size=(config.evaluation_examples, config.state_dim))
        tasks.append(Task(task_id=task_id, program=program, teacher_library=library,
                          train_x=train_x, train_y=program.execute(library, train_x),
                          eval_x=eval_x, eval_y=program.execute(library, eval_x)))
    return tuple(tasks)


def generate_general_world(config, gain: float = 1.0, shared: float = 0.0) -> World:
    """`generate_rotated_world` on the g1 library (distinct programs, streams 20/21/30/31)."""
    library = general_library(config, gain, shared)
    return World(config=config, library=library,
                 tasks=_tasks(config, library, _sample_programs(config), _opaque_task_ids(config)))


def generate_general_curriculum_world(config, gain: float = 1.0, shared: float = 0.0) -> World:
    """`generate_curriculum_world` on the g1 library (programs with replacement, streams 40/41)."""
    library = general_library(config, gain, shared)
    draws = _rng(config.seed, 40).integers(0, config.teacher_primitives, size=(config.tasks, config.program_length))
    task_ids = _draw_opaque_task_ids(_rng(config.seed, 41, config.program_length), config.tasks)
    programs = [Program(tuple(int(p) for p in row)) for row in draws]
    return World(config=config, library=library, tasks=_tasks(config, library, programs, task_ids))
