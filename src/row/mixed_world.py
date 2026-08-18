"""Benchmark D: mixed-recurrence worlds with per-primitive reuse levels.

A per-primitive `rho` profile replaces the world's single `reuse_rho`.
Provenance stays OUTSIDE `WorldConfig` (adding a field there would
invalidate all existing resolved-config fingerprints); runners record the
profile in their own artifact files instead, mirroring the scrambled-ID
pattern.

Seed-scheme note (documented deviation from the V2 spec's draft plan): the
per-task epsilon draws in the homogeneous `_task_library` are independent
of `rho`, so a per-primitive profile needs no new SeedSequence component.
Consequence, verified by unit test: a UNIFORM profile (r, r, ..., r)
reproduces the homogeneous world at `reuse_rho = r` bit-exactly, so the
uniform-high and uniform-low anchor conditions are the existing
homogeneous artifacts rather than new runs.
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

CANONICAL_PROFILE = (1.0, 0.95, 0.8, 0.5, 0.2, 0.0)


class MixedWorld(World):
    """World whose per-task libraries use a per-primitive rho profile."""

    def library_for_task(self, task_index: int):  # type: ignore[override]
        if 0 <= task_index < len(self.tasks):
            return self.tasks[task_index].teacher_library
        return _mixed_task_library(
            self.config, self.library, task_index, self.rho_profile
        )


def _mixed_task_library(
    config: WorldConfig,
    base_library,
    task_index: int,
    profile,
):
    result = []
    for primitive_index, base in enumerate(base_library):
        rho = float(profile[primitive_index])
        if rho == 1.0:
            result.append(base)
            continue
        generator = _rng(config.seed, 11, task_index, primitive_index)
        epsilon_v = _spectral_normalize(
            generator.normal(size=(config.teacher_rank, config.state_dim))
        )
        epsilon_u = _spectral_normalize(
            generator.normal(size=(config.state_dim, config.teacher_rank))
        )
        epsilon_b = generator.normal(scale=0.2, size=config.teacher_rank)
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


def generate_mixed_world(config: WorldConfig, profile) -> MixedWorld:
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
    tasks = []
    for task_index, (program, task_id) in enumerate(
        zip(programs, task_ids, strict=True)
    ):
        task_library = _mixed_task_library(config, library, task_index, profile)
        train_rng = _rng(config.seed, 30, task_index)
        eval_rng = _rng(config.seed, 31, task_index)
        train_x = train_rng.normal(
            size=(config.examples_per_task, config.state_dim)
        )
        eval_x = eval_rng.normal(
            size=(config.evaluation_examples, config.state_dim)
        )
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
    world = MixedWorld(config=config, library=library, tasks=tuple(tasks))
    object.__setattr__(world, "rho_profile", tuple(float(v) for v in profile))
    return world


class MixedWorldFactory:
    """Drop-in for `World` inside runners: `factory.generate(config)`."""

    def __init__(self, profile) -> None:
        self.profile = tuple(float(v) for v in profile)

    def generate(self, config: WorldConfig) -> MixedWorld:
        return generate_mixed_world(replace(config), self.profile)


def per_primitive_recurrence(
    world: MixedWorld, probe_examples: int = 512, max_tasks: int = 16
) -> list[dict[str, float]]:
    """Measured residual-function correlation, per primitive."""
    generator = _rng(world.config.seed, 41)
    probe = generator.normal(size=(probe_examples, world.config.state_dim))
    identity_path = np.tanh(probe)
    task_count = min(max_tasks, len(world.tasks))
    rows = []
    for primitive_index in range(world.config.teacher_primitives):
        effects = [
            world.tasks[t].teacher_library[primitive_index](probe) - identity_path
            for t in range(task_count)
        ]
        correlations = []
        for first in range(task_count):
            for second in range(first + 1, task_count):
                correlations.append(
                    float(
                        np.corrcoef(
                            effects[first].ravel(), effects[second].ravel()
                        )[0, 1]
                    )
                )
        row = {
            "primitive_index": primitive_index,
            "measured_recurrence": float(np.mean(correlations)),
        }
        if hasattr(world, "rho_profile"):
            row["configured_rho"] = world.rho_profile[primitive_index]
        rows.append(row)
    return rows


def usage_by_task_index(world: MixedWorld) -> dict[str, object]:
    """Gate (d): primitive usage must not correlate with task index."""
    correlations = []
    for primitive_index in range(world.config.teacher_primitives):
        usage = np.array(
            [
                task.program.primitive_ids.count(primitive_index)
                for task in world.tasks
            ],
            dtype=np.float64,
        )
        index = np.arange(len(world.tasks), dtype=np.float64)
        if usage.std() == 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(usage, index)[0, 1]))
    return {
        "per_primitive_usage_index_correlation": correlations,
        "max_abs_correlation": float(np.max(np.abs(correlations))),
    }


HIERARCHICAL_WEIGHTS = (0.5, 0.7)  # global, family; task weight is the remainder
FAMILY_OF_PRIMITIVE = (0, 0, 0, 1, 1, 1)


def _hierarchical_task_library(config, base_library, family_library, task_index, weights):
    global_w, family_w = weights
    task_w = float(np.sqrt(max(0.0, 1.0 - global_w**2 - family_w**2)))
    result = []
    for primitive_index, base in enumerate(base_library):
        family = family_library[FAMILY_OF_PRIMITIVE[primitive_index]]
        generator = _rng(config.seed, 12, task_index, primitive_index)
        epsilon_v = _spectral_normalize(
            generator.normal(size=(config.teacher_rank, config.state_dim))
        )
        epsilon_u = _spectral_normalize(
            generator.normal(size=(config.state_dim, config.teacher_rank))
        )
        epsilon_b = generator.normal(scale=0.2, size=config.teacher_rank)
        result.append(
            Primitive(
                U=_spectral_normalize(
                    global_w * base.U + family_w * family.U + task_w * epsilon_u
                ),
                V=_spectral_normalize(
                    global_w * base.V + family_w * family.V + task_w * epsilon_v
                ),
                b=global_w * base.b + family_w * family.b + task_w * epsilon_b,
                alpha=config.alpha,
            )
        )
    return tuple(result)


class HierarchicalWorld(World):
    def library_for_task(self, task_index: int):  # type: ignore[override]
        if 0 <= task_index < len(self.tasks):
            return self.tasks[task_index].teacher_library
        return _hierarchical_task_library(
            self.config, self.library, self.family_library, task_index,
            self.hierarchy_weights,
        )


def generate_hierarchical_world(config: WorldConfig, weights=HIERARCHICAL_WEIGHTS):
    """Benchmark E: P_task,i mixes a global component, one of two family
    components, and a task-specific component, spectrally renormalized."""
    base_library = tuple(
        Primitive.random(seed=config.seed, primitive_index=k,
                         d=config.state_dim, rank=config.teacher_rank,
                         alpha=config.alpha)
        for k in range(config.teacher_primitives)
    )
    family_library = tuple(
        Primitive.random(seed=config.seed, primitive_index=100 + f,
                         d=config.state_dim, rank=config.teacher_rank,
                         alpha=config.alpha)
        for f in range(2)
    )
    programs = _sample_programs(config)
    task_ids = _opaque_task_ids(config)
    tasks = []
    for task_index, (program, task_id) in enumerate(
        zip(programs, task_ids, strict=True)
    ):
        library = _hierarchical_task_library(
            config, base_library, family_library, task_index, weights
        )
        train_rng = _rng(config.seed, 30, task_index)
        eval_rng = _rng(config.seed, 31, task_index)
        train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
        eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
        tasks.append(
            Task(task_id=task_id, program=program, teacher_library=library,
                 train_x=train_x, train_y=program.execute(library, train_x),
                 eval_x=eval_x, eval_y=program.execute(library, eval_x))
        )
    world = HierarchicalWorld(config=config, library=base_library, tasks=tuple(tasks))
    object.__setattr__(world, "family_library", family_library)
    object.__setattr__(world, "hierarchy_weights", tuple(float(w) for w in weights))
    return world


class HierarchicalWorldFactory:
    def __init__(self, weights=HIERARCHICAL_WEIGHTS) -> None:
        self.weights = tuple(float(w) for w in weights)

    def generate(self, config: WorldConfig):
        return generate_hierarchical_world(replace(config), self.weights)
