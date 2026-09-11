"""Curriculum worlds: rotated worlds whose programs may REPEAT.

Additive. `generate_rotated_world` samples a permutation of the distinct
programs, so a world of program length L admits at most `primitives ** L`
tasks: a length-1 rotated world could hold only six. A curriculum stage needs
many tasks per operation, so this module samples programs WITH REPLACEMENT
from its own stream and builds the tasks exactly as `generate_rotated_world`
does.

Nothing here changes `WorldConfig`, `World`, `Program`, or any existing
generator: the stage's length and task count arrive through the resolved
config the caller passes (`program_length`, `tasks`), and the replacement
sampling uses stream 40, unused elsewhere. Callers record the stage in their
own artifacts, as `mixed_world` records its rho profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from row.rotated_world import rotated_library
from row.world import Program, Task, World, _draw_opaque_task_ids, _rng, _task_library


@dataclass(frozen=True)
class CurriculumWorldConfig:
    """`WorldConfig`'s fields, minus the distinct-program cap on `tasks`.

    `WorldConfig` requires `tasks <= teacher_primitives ** program_length`
    because its sampler draws distinct programs; that guard is correct there
    and is left untouched. Sampling with replacement has no such bound, so a
    curriculum stage carries its own config type rather than relaxing the
    shared one. Field names match `WorldConfig` so the existing helpers,
    `resolved()` and the artifact writers work unchanged; `stage` is recorded
    for provenance and is not used by the generator.
    """
    seed: int = 0
    state_dim: int = 16
    teacher_rank: int = 8
    teacher_primitives: int = 6
    program_length: int = 3
    tasks: int = 64
    examples_per_task: int = 128
    evaluation_examples: int = 256
    reuse_rho: float = 1.0
    alpha: float = 0.35
    stage: str = "curriculum"

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.teacher_rank <= 0:
            raise ValueError("state_dim and teacher_rank must be positive")
        if self.teacher_primitives <= 0 or self.program_length <= 0:
            raise ValueError("teacher_primitives and program_length must be positive")
        if self.tasks <= 0:
            raise ValueError("tasks must be positive")
        if self.examples_per_task <= 0 or self.evaluation_examples <= 0:
            raise ValueError("dataset sizes must be positive")
        if self.reuse_rho != 1.0:
            raise ValueError("curriculum worlds require reuse_rho == 1.0")

    @classmethod
    def from_world(cls, config, *, program_length: int, tasks: int, stage: str = "curriculum"):
        return cls(seed=config.seed, state_dim=config.state_dim, teacher_rank=config.teacher_rank,
                   teacher_primitives=config.teacher_primitives, program_length=program_length,
                   tasks=tasks, examples_per_task=config.examples_per_task,
                   evaluation_examples=config.evaluation_examples, reuse_rho=config.reuse_rho,
                   alpha=config.alpha, stage=stage)


def generate_curriculum_world(config: CurriculumWorldConfig) -> World:
    """A rotated world whose `tasks` programs are drawn with replacement."""
    if config.reuse_rho != 1.0:
        raise ValueError("curriculum worlds require reuse_rho == 1.0 (see generate_rotated_world)")
    library = rotated_library(config)
    draws = _rng(config.seed, 40).integers(0, config.teacher_primitives,
                                           size=(config.tasks, config.program_length))
    task_ids = _draw_opaque_task_ids(_rng(config.seed, 21), config.tasks)
    tasks = []
    for index, task_id in enumerate(task_ids):
        program = Program(tuple(int(p) for p in draws[index]))
        task_library = _task_library(config, library, index)
        train_x = _rng(config.seed, 30, index).normal(size=(config.examples_per_task, config.state_dim))
        eval_x = _rng(config.seed, 31, index).normal(size=(config.evaluation_examples, config.state_dim))
        tasks.append(Task(task_id=task_id, program=program, teacher_library=task_library,
                          train_x=train_x, train_y=program.execute(task_library, train_x),
                          eval_x=eval_x, eval_y=program.execute(task_library, eval_x)))
    return World(config=config, library=library, tasks=tuple(tasks))
