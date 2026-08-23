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
        held_out_per_family: int = 1,
        held_out_families: int = 2,
        schema_groups: int = 1,
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
        # Extra members of each family, GENERATED BUT NEVER PLACED IN
        # `world.tasks`. V6 measures whether a representation makes a
        # related future task cheap to learn, and a "future" task the
        # lifetime already trained on is not future: measured, such a
        # task starts at 0.006 support loss and adaptation changes
        # nothing, so every arm ties at the same saturated number.
        self.held_out_per_family = int(held_out_per_family)
        # Whole FAMILIES the lifetime never sees, drawn from the same
        # shared subspace. A held-out member of a SEEN family turned out
        # to be nearly free for every arm (0.005 support loss zero-shot,
        # flat in support size), so it cannot discriminate. An unseen
        # family is the honest future: its operator lies in the same
        # functional subspace, so a representation that captured the
        # subspace should acquire it cheaply and one that did not should
        # not.
        self.held_out_families = int(held_out_families)
        # H47 B2: G disjoint shared subspaces. Families are assigned to
        # groups contiguously (families 0..F/G-1 -> group 0, ...); held-out
        # families round-robin, one per group when held_out_families == G.
        # G = 1 is the original generator, bit for bit.
        if schema_groups < 1:
            raise ValueError("schema_groups must be positive")
        if self.families % int(schema_groups):
            raise ValueError("families must be divisible by schema_groups")
        if self.held_out_families and self.held_out_families % int(schema_groups):
            raise ValueError("held_out_families must be divisible by schema_groups")
        self.schema_groups = int(schema_groups)

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

    def group_of_family(self, family: int) -> int:
        """Schema group of a family index (trained or held-out)."""

        if family < self.families:
            return family // (self.families // self.schema_groups)
        return (family - self.families) % self.schema_groups

    def as_dict(self) -> dict[str, object]:
        if self.schema_groups != 1:
            return {**self._base_dict(), "schema_groups": self.schema_groups}
        return self._base_dict()

    def _base_dict(self) -> dict[str, object]:
        return {
            "families": self.families,
            "tasks_per_family": self.tasks_per_family,
            "r_meta": self.r_meta,
            "subspace_rank": self.subspace_rank,
            "family_onset": self.family_onset,
            "held_out_per_family": self.held_out_per_family,
            "held_out_families": self.held_out_families,
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
    # One orthonormal basis per schema group, in MATRIX space. Group 0
    # draws exactly the original seeds (61, k); later groups draw
    # (61, g*rank + k) and are projected out of every earlier group's
    # span before orthonormalization, so the G subspaces are disjoint and
    # G = 1 is the original construction bit for bit.
    group_bases = []
    for g in range(spec.schema_groups):
        basis = [
            _spectral_normalize(
                _rng(config.seed, 61, g * spec.subspace_rank + k).normal(size=(d, rank))
            )
            for k in range(spec.subspace_rank)
        ]
        flat = np.stack([B.ravel() for B in basis])
        for earlier in group_bases:
            flat = flat - (flat @ earlier.T) @ earlier
        q, _ = np.linalg.qr(flat.T)
        group_bases.append(q.T[: spec.subspace_rank])
    shared_basis = group_bases[0]
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
    for family in range(spec.families + spec.held_out_families):
        generator = _rng(config.seed, 62, family)
        family_basis = group_bases[spec.group_of_family(family)]
        coefficients = _unit(generator.normal(size=spec.subspace_rank))
        shared_part = _unit(coefficients @ family_basis)
        private = generator.normal(size=d * rank)
        # Project the private part OUT of the shared subspace. Without
        # this the two components have a random nonzero inner product,
        # so ||theta_f|| wobbles with the draw and the balance gate
        # fails for a reason that has nothing to do with relatedness —
        # measured at 31.6% spread on contribution before the fix.
        private = private - (private @ family_basis.T) @ family_basis
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

    # Held-out family members: extra programs from an EXTENDED sampler,
    # so the lifetime's own task list is bit-identical to a run with
    # held_out_per_family = 0 and existing artifacts stay valid.
    held_out: list[Task] = []
    if spec.held_out_per_family:
        extra = spec.families * spec.held_out_per_family
        extended = replace(config, tasks=config.tasks + extra)
        extra_programs = _sample_programs(extended)[config.tasks:]
        extra_ids = _opaque_task_ids(extended)[config.tasks:]
        for offset, (program, task_id) in enumerate(
            zip(extra_programs, extra_ids, strict=True)
        ):
            family = offset % spec.families
            task_library = (*library, operators[family])
            steps = list(program.primitive_ids)
            steps[config.program_length - 1] = len(task_library) - 1
            program = replace(program, primitive_ids=tuple(steps))
            index = config.tasks + offset
            train_rng = _rng(config.seed, 30, index)
            eval_rng = _rng(config.seed, 31, index)
            train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
            eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
            held_out.append(
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

    novel: list[Task] = []
    if spec.held_out_families:
        extra = spec.held_out_families * spec.held_out_per_family
        base = config.tasks + spec.families * spec.held_out_per_family
        extended = replace(config, tasks=base + extra)
        novel_programs = _sample_programs(extended)[base:]
        novel_ids = _opaque_task_ids(extended)[base:]
        for offset, (program, task_id) in enumerate(
            zip(novel_programs, novel_ids, strict=True)
        ):
            operator = operators[spec.families + offset % spec.held_out_families]
            task_library = (*library, operator)
            steps = list(program.primitive_ids)
            steps[config.program_length - 1] = len(task_library) - 1
            program = replace(program, primitive_ids=tuple(steps))
            index = base + offset
            train_rng = _rng(config.seed, 30, index)
            eval_rng = _rng(config.seed, 31, index)
            train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
            eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
            novel.append(
                Task(
                    task_id=task_id, program=program, teacher_library=task_library,
                    train_x=train_x, train_y=program.execute(task_library, train_x),
                    eval_x=eval_x, eval_y=program.execute(task_library, eval_x),
                )
            )

    # UNSEEN UNRELATED tasks: base-primitive programs, no family
    # operator, never in the lifetime. H31 previously used PRE-ONSET
    # tasks, which the lifetime trains on, so "related" was novel and
    # "unrelated" was familiar — an asymmetry that cannot separate
    # structural specificity from generic plasticity (review 55).
    unrelated: list[Task] = []
    if spec.held_out_per_family:
        count = spec.held_out_per_family * spec.families
        base = (config.tasks + spec.families * spec.held_out_per_family
                + spec.held_out_families * spec.held_out_per_family)
        extended = replace(config, tasks=base + count)
        programs_u = _sample_programs(extended)[base:]
        ids_u = _opaque_task_ids(extended)[base:]
        for offset, (program, task_id) in enumerate(
            zip(programs_u, ids_u, strict=True)
        ):
            index = base + offset
            train_rng = _rng(config.seed, 30, index)
            eval_rng = _rng(config.seed, 31, index)
            train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
            eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
            unrelated.append(
                Task(
                    task_id=task_id, program=program, teacher_library=library,
                    train_x=train_x, train_y=program.execute(library, train_x),
                    eval_x=eval_x, eval_y=program.execute(library, eval_x),
                )
            )

    world = World(config=config, library=library, tasks=tuple(tasks))
    object.__setattr__(world, "unseen_unrelated_tasks", tuple(unrelated))
    object.__setattr__(world, "novel_family_tasks", tuple(novel))
    object.__setattr__(world, "held_out_family_tasks", tuple(held_out))
    object.__setattr__(world, "held_out_family_index",
                       tuple(i % spec.families for i in range(len(held_out))))
    object.__setattr__(world, "meta_family_spec", spec)
    object.__setattr__(world, "family_operators", operators)
    object.__setattr__(world, "family_group",
                       tuple(spec.group_of_family(f) for f in range(spec.families + spec.held_out_families)))
    return world


class MetaWorldFactory:
    """Drop-in for `World` inside runners: `factory.generate(config)`."""

    def __init__(self, spec: MetaFamilySpec) -> None:
        self.spec = spec

    def generate(self, config: WorldConfig) -> World:
        return generate_meta_world(replace(config), self.spec)
