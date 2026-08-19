"""V3 promotion testbed: worlds with hidden task-group families.

Family components are assigned per TASK GROUP rather than per primitive, so
the structure cross-cuts tasks and a task-invariant shared basis cannot
absorb it. That absorbability is exactly the design flaw that made
Benchmark E's hierarchy measure negative in V2 (V2 spec section 3), and it
is what prediction P-2026-08-18-D is about.

The construction reuses the Benchmark D epsilon machinery: a task's
deviation from the base primitive is split into a family-shared direction
and a task-private one, mixed by `eta`. The private draw comes from the
untouched Benchmark D stream, so a task-group world at eta = 0 reproduces
the canonical mixed world BIT-EXACTLY. The structureless control is
therefore the same generator at eta = 0 rather than a separate one whose
equivalence would have to be argued.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from row.mixed_world import MixedWorld
from row.world import (
    Primitive,
    Task,
    WorldConfig,
    _opaque_task_ids,
    _rng,
    _sample_programs,
    _spectral_normalize,
)


class TaskGroupSpec:
    """Hidden task-group family structure for the promotion testbed."""

    def __init__(
        self,
        groups: int = 2,
        eta: float = 0.0,
        block_size: int | None = None,
        future_tasks: int = 8,
        resample_future: bool = False,
    ) -> None:
        if groups < 1:
            raise ValueError("groups must be positive")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [0, 1]")
        if block_size is not None and block_size < 1:
            raise ValueError("block size must be positive when given")
        if future_tasks < 0:
            raise ValueError("future task count must be nonnegative")
        self.groups = int(groups)
        self.eta = float(eta)
        # None: one stable family direction for the whole lifetime (the
        # promotion testbed). An integer: the direction is redrawn every
        # `block_size` tasks, which makes the instability OBSERVABLE from
        # sequential evidence and is the drifting-family refusal control
        # (V3 spec 2.3).
        self.block_size = block_size
        self.future_tasks = int(future_tasks)
        # Regime-change world: identical lifetime history, future family
        # directions redrawn. Observationally indistinguishable from the
        # testbed, so it carries a same-decision prediction, never a
        # refusal requirement.
        self.resample_future = bool(resample_future)

    def as_dict(self) -> dict[str, object]:
        return {
            "groups": self.groups,
            "eta": self.eta,
            "block_size": self.block_size,
            "future_tasks": self.future_tasks,
            "resample_future": self.resample_future,
        }


def _group_assignment(config: WorldConfig, groups: int, count: int) -> tuple[int, ...]:
    """Balanced, deterministic, and never exposed to any learner."""

    order = _rng(config.seed, 13).permutation(count)
    assignment = [0] * count
    for position, task_index in enumerate(order):
        assignment[int(task_index)] = position % groups
    return tuple(assignment)


def _task_group_library(
    config: WorldConfig,
    base_library,
    task_index: int,
    profile,
    spec: TaskGroupSpec,
    group: int,
    block: int,
):
    result = []
    for primitive_index, base in enumerate(base_library):
        rho = float(profile[primitive_index])
        if rho == 1.0:
            result.append(base)
            continue
        # Task-private draw: the untouched Benchmark D stream, so eta = 0
        # reproduces the canonical mixed world exactly.
        private = _rng(config.seed, 11, task_index, primitive_index)
        private_v = _spectral_normalize(
            private.normal(size=(config.teacher_rank, config.state_dim))
        )
        private_u = _spectral_normalize(
            private.normal(size=(config.state_dim, config.teacher_rank))
        )
        private_b = private.normal(scale=0.2, size=config.teacher_rank)
        if spec.eta > 0.0:
            family = _rng(config.seed, 12, group, block, primitive_index)
            family_v = _spectral_normalize(
                family.normal(size=(config.teacher_rank, config.state_dim))
            )
            family_u = _spectral_normalize(
                family.normal(size=(config.state_dim, config.teacher_rank))
            )
            family_b = family.normal(scale=0.2, size=config.teacher_rank)
            share = float(np.sqrt(1.0 - spec.eta * spec.eta))
            epsilon_v = spec.eta * family_v + share * private_v
            epsilon_u = spec.eta * family_u + share * private_u
            epsilon_b = spec.eta * family_b + share * private_b
        else:
            epsilon_v, epsilon_u, epsilon_b = private_v, private_u, private_b
        scale = float(np.sqrt(1.0 - rho * rho))
        result.append(
            Primitive(
                U=_spectral_normalize(rho * base.U + scale * epsilon_u),
                V=_spectral_normalize(rho * base.V + scale * epsilon_v),
                b=rho * base.b + scale * epsilon_b,
                alpha=config.alpha,
            )
        )
    return tuple(result)


class TaskGroupWorld(MixedWorld):
    """Mixed world with hidden task-group families and a held-out future block."""

    def library_for_task(self, task_index: int):  # type: ignore[override]
        if 0 <= task_index < len(self.tasks):
            return self.tasks[task_index].teacher_library
        return _task_group_library(
            self.config,
            self.library,
            task_index,
            self.rho_profile,
            self.task_group_spec,
            self.group_assignment[task_index % len(self.group_assignment)],
            0,
        )


def generate_task_group_world(
    config: WorldConfig, profile, spec: TaskGroupSpec
) -> TaskGroupWorld:
    if len(profile) != config.teacher_primitives:
        raise ValueError("profile length must match teacher_primitives")
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
    programs = _sample_programs(config)
    task_ids = _opaque_task_ids(config)
    total = len(programs)
    assignment = _group_assignment(config, spec.groups, total + spec.future_tasks)

    def _block_of(task_index: int) -> int:
        if spec.block_size is None:
            return 0
        return task_index // spec.block_size

    def _build(task_index: int, program, task_id: str, future: bool) -> Task:
        block = _block_of(task_index)
        if future and spec.resample_future:
            # Fresh directions for the held-out block only; the lifetime
            # history is identical to the testbed's.
            block = -1
        task_library = _task_group_library(
            config, library, task_index, profile, spec, assignment[task_index], block
        )
        train_rng = _rng(config.seed, 30, task_index)
        eval_rng = _rng(config.seed, 31, task_index)
        train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
        eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
        return Task(
            task_id=task_id,
            program=program,
            teacher_library=task_library,
            train_x=train_x,
            train_y=program.execute(task_library, train_x),
            eval_x=eval_x,
            eval_y=program.execute(task_library, eval_x),
        )

    tasks = [
        _build(index, program, task_id, future=False)
        for index, (program, task_id) in enumerate(zip(programs, task_ids, strict=True))
    ]
    # The future block extends the program and ID samplers and never enters
    # `world.tasks`, so no lifetime can train on it.
    future: list[Task] = []
    if spec.future_tasks:
        future_config = replace(config, tasks=total + spec.future_tasks)
        future_programs = _sample_programs(future_config)[total:]
        future_ids = _opaque_task_ids(future_config)[total:]
        future = [
            _build(total + offset, program, task_id, future=True)
            for offset, (program, task_id) in enumerate(
                zip(future_programs, future_ids, strict=True)
            )
        ]

    world = TaskGroupWorld(config=config, library=library, tasks=tuple(tasks))
    object.__setattr__(world, "rho_profile", tuple(float(v) for v in profile))
    object.__setattr__(world, "task_group_spec", spec)
    object.__setattr__(world, "group_assignment", assignment)
    object.__setattr__(world, "future_tasks", tuple(future))
    return world


class TaskGroupWorldFactory:
    """Drop-in for `World` inside runners: `factory.generate(config)`."""

    def __init__(self, profile, spec: TaskGroupSpec) -> None:
        self.profile = tuple(float(v) for v in profile)
        self.spec = spec

    def generate(self, config: WorldConfig) -> TaskGroupWorld:
        return generate_task_group_world(replace(config), self.profile, self.spec)


def teacher_group_clustering(
    world: TaskGroupWorld, probe_examples: int = 512, max_tasks: int = 32
) -> dict[str, object]:
    """World-level precondition for the section 2.1 validity gate.

    Compares within-group against cross-group functional similarity of the
    teacher deviations. This is a property of the WORLD, checked before any
    learner runs; the gate proper (P-2026-08-18-D) asks the same question of
    a trained learner's residuals, which is the harder claim.
    """

    generator = _rng(world.config.seed, 42)
    probe = generator.normal(size=(probe_examples, world.config.state_dim))
    count = min(max_tasks, len(world.tasks))
    # Average per-primitive correlations over primitives that actually have
    # a deviation. Concatenating all primitives instead would splice the
    # exact-zero blocks of any rho = 1 primitive into every task vector,
    # which manufactures a large spurious correlation floor (measured 0.43)
    # purely because those coordinates are identically zero everywhere.
    base_effects = [
        world.library[k](probe) for k in range(world.config.teacher_primitives)
    ]
    active = [
        k
        for k in range(world.config.teacher_primitives)
        if float(world.rho_profile[k]) < 1.0
    ]
    deviations = [
        [
            (world.tasks[index].teacher_library[k](probe) - base_effects[k]).ravel()
            for k in active
        ]
        for index in range(count)
    ]
    # Remove the task-INVARIANT component of the deviation before asking
    # about groups. Spectral renormalization leaves every task sharing a
    # common offset from the base primitive (measured correlation floor
    # 0.33 even at eta = 0), and a task-invariant shared basis absorbs
    # exactly that component — which is why Benchmark E's absorbable
    # hierarchy measured negative. What promotion can exploit is the
    # structure that survives centering.
    for position in range(len(active)):
        common = np.mean(
            [deviations[index][position] for index in range(count)], axis=0
        )
        for index in range(count):
            deviations[index][position] = deviations[index][position] - common
    within: list[float] = []
    cross: list[float] = []
    for first in range(count):
        for second in range(first + 1, count):
            correlation = float(
                np.mean(
                    [
                        np.corrcoef(
                            deviations[first][position], deviations[second][position]
                        )[0, 1]
                        for position in range(len(active))
                    ]
                )
            )
            same = world.group_assignment[first] == world.group_assignment[second]
            (within if same else cross).append(correlation)
    within_mean = float(np.mean(within)) if within else 0.0
    cross_mean = float(np.mean(cross)) if cross else 0.0
    return {
        "tasks_compared": count,
        "within_group_mean_correlation": within_mean,
        "cross_group_mean_correlation": cross_mean,
        # SEPARATION is the statistic to read. A ratio is degenerate here:
        # centering balanced groups forces the cross-group mean negative, so
        # within/cross carries a sign flip rather than a magnitude. The ratio
        # is reported only for positive cross-group similarity.
        "separation": within_mean - cross_mean,
        "ratio": (
            within_mean / cross_mean if cross_mean > 1e-9 else None
        ),
    }
