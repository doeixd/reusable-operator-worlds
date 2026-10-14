"""Operator-FORM generators for decision 20 (a) (`G0F_FORM_CENSUS_PLAN.md`, Track G).

G0 showed that no knob of the rotated form `Q (z + a U tanh(V z + b))` makes a computed state hard to predict
linearly from the input (S1 >= 0.958): an identity skip plus a bounded residual plus an orthogonal map stays close to
an orthogonal map. These generators drop that structure, one way each, at d = 16 and full width:

- `tanh` (no skip): `P(z) = c_g Q tanh(g R z + b)`; `R`, `Q` Haar rotations, `b ~ N(0, 0.2^2)`, and
  `c_g = 1 / sqrt(E[tanh(g x)^2])` (x standard normal, 64-point Gauss-Hermite) so a standard-normal input keeps unit
  per-coordinate variance. Small `g` is near the linear map `c_g g Q R`; large `g` is near a sign map.
- `gate` (multiplicative): `P(z) = Q (z * 2 sigmoid(g (W z + c)))`, `W ~ N(0, 1/d)`, `c ~ N(0, 0.2^2)`; each
  coordinate is scaled by a state-dependent factor in (0, 2).
- `fold` (piecewise linear): `P(z) = Q f_j(...f_1(z))`, `f_i(z) = z - 2 min(0, w_i . z) w_i` with unit `w_i`: a
  reflection of the half-space `w_i . z < 0`. Norm-preserving exactly, continuous, linear on each of at most `2^j`
  cones.

Every draw comes from `SeedSequence([seed, 120, part, k])` (stream 120, verified unused elsewhere in `src/`); the
same function with `seed = MODEL_SEED` draws an UNTRAINED same-form library. Nothing here changes `WorldConfig`,
`World` or an existing generator; the census records the form and knob in its own artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from row.general_world import _tasks
from row.world import Array, Program, World, _opaque_task_ids, _rng, _sample_programs

VERSION = 'f1'
STREAM = 120
FORMS = ('tanh', 'gate', 'fold')


def _haar(seed: int, part: int, k: int, d: int) -> Array:
    q, r = np.linalg.qr(_rng(seed, STREAM, part, k).normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def tanh_gain_scale(g: float) -> float:
    x, w = np.polynomial.hermite_e.hermegauss(64)
    return float(1.0 / np.sqrt(np.sum(w * np.tanh(g * x) ** 2) / np.sum(w)))


def _sigmoid(x: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class FormPrimitive:
    form: str
    knob: float
    Q: Array
    A: Array      # tanh: R; gate: W; fold: the j unit normals as rows
    b: Array      # tanh: b; gate: c; fold: unused (zeros)
    scale: float = 1.0

    @classmethod
    def random(cls, form: str, knob: float, seed: int, k: int, d: int) -> 'FormPrimitive':
        Q = _haar(seed, 2, k, d)
        if form == 'tanh':
            g = _rng(seed, STREAM, 3, k)
            return cls(form, float(knob), Q, _haar(seed, 1, k, d), g.normal(scale=0.2, size=d), tanh_gain_scale(knob))
        if form == 'gate':
            g = _rng(seed, STREAM, 4, k)
            return cls(form, float(knob), Q, g.normal(scale=1.0 / np.sqrt(d), size=(d, d)), g.normal(scale=0.2, size=d))
        if form == 'fold':
            j = int(knob)
            if j != knob or j < 1:
                raise ValueError('fold knob is a positive integer number of folds')
            w = _rng(seed, STREAM, 5, k).normal(size=(j, d))
            return cls(form, float(knob), Q, w / np.linalg.norm(w, axis=1, keepdims=True), np.zeros(d))
        raise ValueError(f'unknown form {form}')

    def __call__(self, z: Array) -> Array:
        z = np.asarray(z, dtype=np.float64)
        if self.form == 'tanh':
            return self.scale * np.tanh(self.knob * (z @ self.A.T) + self.b) @ self.Q.T
        if self.form == 'gate':
            return (z * 2.0 * _sigmoid(self.knob * (z @ self.A.T + self.b))) @ self.Q.T
        for w in self.A:
            z = z - 2.0 * np.minimum(0.0, z @ w)[:, None] * w[None, :]
        return z @ self.Q.T


def form_library(form: str, knob: float, seed: int, count: int, d: int) -> tuple:
    return tuple(FormPrimitive.random(form, knob, seed, k, d) for k in range(count))


def generate_form_world(config, form: str, knob: float) -> World:
    """Distinct programs and inputs exactly as `generate_rotated_world` (streams 20/21/30/31); only the library differs."""
    library = form_library(form, knob, config.seed, config.teacher_primitives, config.state_dim)
    return World(config=config, library=library,
                 tasks=_tasks(config, library, _sample_programs(config), _opaque_task_ids(config)))


def generate_form_curriculum_world(config, form: str, knob: float) -> World:
    """Programs with replacement (streams 40/41), as `generate_curriculum_world`, on the form library."""
    from row.world import _draw_opaque_task_ids
    library = form_library(form, knob, config.seed, config.teacher_primitives, config.state_dim)
    draws = _rng(config.seed, 40).integers(0, config.teacher_primitives, size=(config.tasks, config.program_length))
    task_ids = _draw_opaque_task_ids(_rng(config.seed, 41, config.program_length), config.tasks)
    programs = [Program(tuple(int(p) for p in row)) for row in draws]
    return World(config=config, library=library, tasks=_tasks(config, library, programs, task_ids))
