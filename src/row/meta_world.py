"""H20 testbed: families whose ABSTRACTIONS are related to each other.

V3's task-group world gives each family a fresh primitive drawn
independently of every other, so a schema over abstractions has nothing
to describe. This generator adds the one knob H20 needs — how much of
each family operator lies in a subspace shared with the other families —
while holding what each operator individually costs and is worth fixed.

    theta_f(r) = sqrt(r) * B alpha_f  +  sqrt(1 - r) * C_f

WHY THIS IS A FUNCTIONAL MIXTURE AND NOT A PARAMETER ONE. A primitive
computes `tanh(z + alpha * h(z) U^T)` with `h(z) = tanh(z V^T + b)`. If
`V` and `b` are SHARED across the family primitives — they are, here —
then `h` is one fixed function of the input and the residual
contribution is LINEAR in `U`. A mixture of `U` matrices is therefore
exactly a mixture of the functions they compute, evaluated at every
input simultaneously. That is what lets this generator claim a
functional construction rather than a parameter-space one that would
have to argue away gauge freedom (V5 spec, H20).

Two consequences the spec depends on:

  * at r = 0 the family operators are independent draws, which must
    reproduce the structureless control: a schema buys nothing;
  * at r = 1 every family operator lies in one K-dimensional functional
    family, so a schema of rank K describes them all.

Norm is held across r by construction: `B_k` and `C_f` are each
spectrally normalized before mixing and `alpha_f` is a unit vector, so
`E||theta_f||` does not depend on r. The balance gates verify that
empirically rather than trusting the algebra.

Provenance stays OUTSIDE `WorldConfig`, as in `mixed_world` and
`task_group_world`: adding a field there would invalidate every existing
resolved-config fingerprint. Runners record `MetaFamilySpec.as_dict()`
in their own artifact files.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from row.world import (
    Primitive,
    Task,
    World,
    WorldConfig,
    _opaque_task_ids,
    _rng,
    _sample_programs,
    _spectral_normalize,
)


class MetaFamilySpec:
    """F recurring families whose operators share a functional subspace."""

    def __init__(
        self,
        families: int = 4,
        tasks_per_family: int = 16,
        r_meta: float = 0.0,
        subspace_rank: int = 2,
        family_onset: int = 8,
    ) -> None:
        if families < 1:
            raise ValueError("families must be positive")
        if tasks_per_family < 1:
            raise ValueError("tasks_per_family must be positive")
        if not 0.0 <= r_meta <= 1.0:
            raise ValueError("r_meta must lie in [0, 1]")
        if subspace_rank < 1:
            raise ValueError("subspace_rank must be positive")
        if family_onset < 0:
            raise ValueError("family_onset must be nonnegative")
        # The scale knob is F at fixed m, never F at fixed N: raising F
        # inside a fixed task budget starves every candidate and shrinks
        # the library, which is the confound that invalidated the first
        # F-sweep (review 32).
        self.families = int(families)
        self.tasks_per_family = int(tasks_per_family)
        self.r_meta = float(r_meta)
        # K is the dimension of the shared functional family. A schema
        # of rank K is sufficient at r_meta = 1 BY CONSTRUCTION, so a
        # fitted schema that cannot reach it is a fitting failure and
        # not a property of the world.
        self.subspace_rank = int(subspace_rank)
        self.family_onset = int(family_onset)

    @property
    def total_tasks(self) -> int:
        return self.family_onset + self.families * self.tasks_per_family

    def family_of(self, task_index: int) -> int | None:
        """Which family this task belongs to, or None before the onset."""

        if task_index < self.family_onset:
            return None
        offset = task_index - self.family_onset
        family = offset // self.tasks_per_family
        return family if family < self.families else None

    def as_dict(self) -> dict[str, object]:
        return {
            "families": self.families,
            "tasks_per_family": self.tasks_per_family,
            "r_meta": self.r_meta,
            "subspace_rank": self.subspace_rank,
            "family_onset": self.family_onset,
            "total_tasks": self.total_tasks,
        }


def family_operators(config: WorldConfig, spec: MetaFamilySpec) -> tuple[Primitive, ...]:
    """The F family primitives, sharing hidden features by construction."""

    d, rank = config.state_dim, config.teacher_rank
    shared = _rng(config.seed, 60)
    # ONE hidden map for every family. This is what makes the mixture
    # below a mixture of functions rather than of coordinates.
    V = _spectral_normalize(shared.normal(size=(rank, d)))
    b = shared.normal(scale=0.2, size=rank)
    basis = [
        _spectral_normalize(_rng(config.seed, 61, k).normal(size=(d, rank)))
        for k in range(spec.subspace_rank)
    ]
    # Orthonormal basis for the shared subspace, in MATRIX space. The
    # span is what carries the relatedness; orthonormalizing it changes
    # no operator's family membership and makes the algebra below exact.
    flat = np.stack([B.ravel() for B in basis])
    shared_basis, _ = np.linalg.qr(flat.T)
    shared_basis = shared_basis.T[: spec.subspace_rank]
    # One scale for every family at every r_meta, so magnitude cannot
    # co-vary with the knob. Taken from a spectrally normalized draw so
    # the family operators sit at the same scale as the base library.
    target = float(np.linalg.norm(
        _spectral_normalize(_rng(config.seed, 63).normal(size=(d, rank)))
    ))

    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    operators = []
    for family in range(spec.families):
        generator = _rng(config.seed, 62, family)
        coefficients = _unit(generator.normal(size=spec.subspace_rank))
        shared_part = _unit(coefficients @ shared_basis)
        private = generator.normal(size=d * rank)
        # Project the private part OUT of the shared subspace. Without
        # this the two components have a random nonzero inner product,
        # so ||theta_f|| wobbles with the draw and the balance gate
        # fails for a reason that has nothing to do with relatedness —
        # measured at 31.6% spread on contribution before the fix.
        private = private - (private @ shared_basis.T) @ shared_basis
        private = _unit(private)
        mixed = (
            np.sqrt(spec.r_meta) * shared_part
            + np.sqrt(1.0 - spec.r_meta) * private
        )
        # Orthogonality makes this exact rather than expected:
        # ||mixed||^2 = r + (1 - r) = 1 for every family and every r.
        U = (target * mixed).reshape(d, rank)
        operators.append(Primitive(U=U, V=V, b=b, alpha=config.alpha))
    return tuple(operators)


def generate_meta_world(config: WorldConfig, spec: MetaFamilySpec) -> World:
    """A world with `spec.families` recurring innovation families."""

    if config.tasks != spec.total_tasks:
        raise ValueError(
            f"config.tasks ({config.tasks}) must equal family_onset + "
            f"families * tasks_per_family ({spec.total_tasks})"
        )
    library = tuple(
        Primitive.random(
            seed=config.seed,
            primitive_index=k,
            d=config.state_dim,
            rank=config.teacher_rank,
            alpha=config.alpha,
        )
        for k in range(config.teacher_primitives)
    )
    operators = family_operators(config, spec)
    programs = _sample_programs(config)
    task_ids = _opaque_task_ids(config)

    tasks = []
    for index, (program, task_id) in enumerate(zip(programs, task_ids, strict=True)):
        task_library = library
        family = spec.family_of(index)
        if family is not None:
            task_library = (*library, operators[family])
            # Fixed step position, as in the V3 testbed: varying it per
            # task hides family structure inside step placement and a
            # family mean then captures almost none of what its members
            # share.
            steps = list(program.primitive_ids)
            steps[config.program_length - 1] = len(task_library) - 1
            program = replace(program, primitive_ids=tuple(steps))
        train_rng = _rng(config.seed, 30, index)
        eval_rng = _rng(config.seed, 31, index)
        train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
        eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
        tasks.append(
            Task(
                task_id=task_id,
                program=program,
                teacher_library=task_library,
                train_x=train_x,
                train_y=program.execute(task_library, train_x),
                eval_x=eval_x,
                eval_y=program.execute(task_library, eval_x),
            )
        )

    world = World(config=config, library=library, tasks=tuple(tasks))
    object.__setattr__(world, "meta_family_spec", spec)
    object.__setattr__(world, "family_operators", operators)
    return world


class MetaWorldFactory:
    """Drop-in for `World` inside runners: `factory.generate(config)`."""

    def __init__(self, spec: MetaFamilySpec) -> None:
        self.spec = spec

    def generate(self, config: WorldConfig) -> World:
        return generate_meta_world(replace(config), self.spec)
