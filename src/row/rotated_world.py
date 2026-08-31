"""The rotated substrate's generator: `P(z) = Q (z + a U tanh(V z + b))`.

`ROTATED_SUBSTRATE_SPEC.md`. A generator VARIANT in the manner of
`mixed_world.py` and `task_group_world.py` -- `world.py`, `WorldConfig` and every
existing resolved-config fingerprint are untouched. Adding a field to the shared
config would invalidate every fingerprint in the project.

Why the rotation. The standard primitive `tanh(z + a U tanh(Vz + b))` at
`a = 0.35` is a weak perturbation of the identity, with two measured consequences:
iterates CONVERGE (`P^5` and `P^6` differ by 1.1%, so one fixed repeat count
approximates any distribution of counts to NMSE <= 0.12) and operators barely
DIFFER (one fixed operator approximates a choice between two others to 0.7%
error). Loops are unidentifiable and branches unnecessary -- one property seen
twice. An orthogonal `Q` removes the fixed point and separates the operators:
measured iteration necessity rises from 0.03 to 1.08-1.58 and branch necessity
from 0.007 to 0.93-1.04, with state norms bounded at x1.03.

`RotatedPrimitive` reduces EXACTLY to `world.Primitive` when `Q = I`, which
`tests/test_rotated_world.py` asserts rather than assumes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from row.config import WorldConfig
from row.world import (Array, Primitive, Program, Task, World, _opaque_task_ids,
                       _rng, _sample_programs, _spectral_normalize)


@dataclass(frozen=True)
class RotatedPrimitive:
    """`world.Primitive` with an orthogonal map applied after the residual step.

    The outer `tanh` of the standard primitive is REPLACED by `Q`, not composed
    with it: squashing is what creates the fixed point, and keeping it would
    reintroduce the convergence the rotation exists to remove. `Q` orthogonal
    keeps states bounded without it.
    """

    U: Array
    V: Array
    b: Array
    Q: Array
    alpha: float = 0.35

    @classmethod
    def random(cls, seed: int, primitive_index: int, d: int, rank: int,
               alpha: float) -> "RotatedPrimitive":
        # The U/V/b draws use the SAME stream and order as `Primitive.random`, so
        # a rotated world's residual parameters match a standard world's exactly
        # and `Q` is the only difference between the two substrates.
        generator = _rng(seed, 10, primitive_index)
        V = _spectral_normalize(generator.normal(size=(rank, d)))
        U = _spectral_normalize(generator.normal(size=(d, rank)))
        b = generator.normal(scale=0.2, size=rank)
        rotation = _rng(seed, 12, primitive_index).normal(size=(d, d))
        Q, r = np.linalg.qr(rotation)
        # Fix the QR sign convention so `Q` is deterministic across LAPACK builds.
        Q = Q * np.sign(np.diag(r))
        return cls(U=U, V=V, b=b, Q=Q, alpha=alpha)

    def __call__(self, z: Array) -> Array:
        z = np.asarray(z, dtype=np.float64)
        hidden = np.tanh(z @ self.V.T + self.b)
        return (z + self.alpha * (hidden @ self.U.T)) @ self.Q.T


def rotated_library(config: WorldConfig) -> tuple[RotatedPrimitive, ...]:
    return tuple(
        RotatedPrimitive.random(seed=config.seed, primitive_index=k,
                                d=config.state_dim, rank=config.teacher_rank,
                                alpha=config.alpha)
        for k in range(config.teacher_primitives))


def generate_rotated_world(config: WorldConfig) -> World:
    """A `World` whose library is rotated. Everything else follows `world.py`.

    Programs, task ids, example streams and the per-task library derivation are
    the standard ones, so a rotated world differs from a standard world in the
    primitive family and in nothing else.
    """
    from row.world import _task_library

    # `_task_library` constructs STANDARD `Primitive` objects when `reuse_rho < 1`,
    # which would silently discard `Q` and hand back an unrotated world. Fail
    # loudly rather than produce a substrate that looks rotated and is not.
    if config.reuse_rho != 1.0:
        raise ValueError(
            "generate_rotated_world requires reuse_rho == 1.0: the per-task "
            "library derivation rebuilds standard Primitives and would drop the "
            "rotation. Extend _task_library before using rho < 1 here.")

    library = rotated_library(config)
    programs = _sample_programs(config)
    task_ids = _opaque_task_ids(config)
    tasks = []
    for index, (program, task_id) in enumerate(zip(programs, task_ids, strict=True)):
        task_library = _task_library(config, library, index)
        train_x = _rng(config.seed, 30, index).normal(
            size=(config.examples_per_task, config.state_dim))
        eval_x = _rng(config.seed, 31, index).normal(
            size=(config.evaluation_examples, config.state_dim))
        # `_sample_programs` already yields `Program` objects.
        executor = program if isinstance(program, Program) else Program(tuple(program))
        tasks.append(Task(task_id=task_id, program=executor,
                          teacher_library=task_library,
                          train_x=train_x, train_y=executor.execute(task_library, train_x),
                          eval_x=eval_x, eval_y=executor.execute(task_library, eval_x)))
    return World(config=config, library=library, tasks=tuple(tasks))
