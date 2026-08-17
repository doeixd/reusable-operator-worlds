"""Deterministic hidden operator worlds and fixed task datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _rng(seed: int, *components: int) -> np.random.Generator:
    """Create a process-independent deterministic random stream."""
    return np.random.default_rng(np.random.SeedSequence([seed, *components]))


def _spectral_normalize(matrix: Array) -> Array:
    norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    if norm == 0.0:
        raise ValueError("cannot normalize a zero matrix")
    return matrix / norm


@dataclass(frozen=True)
class WorldConfig:
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

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.teacher_rank <= 0:
            raise ValueError("state_dim and teacher_rank must be positive")
        if self.teacher_primitives <= 0 or self.program_length <= 0:
            raise ValueError("teacher_primitives and program_length must be positive")
        possible = self.teacher_primitives ** self.program_length
        if not 0 < self.tasks <= possible:
            raise ValueError(f"tasks must be in [1, {possible}]")
        if self.examples_per_task <= 0 or self.evaluation_examples <= 0:
            raise ValueError("dataset sizes must be positive")
        if self.reuse_rho != 1.0:
            raise NotImplementedError("the foundation milestone supports exact reuse (rho=1) only")


@dataclass(frozen=True)
class Primitive:
    U: Array
    V: Array
    b: Array
    alpha: float = 0.35

    @classmethod
    def random(cls, seed: int, primitive_index: int, d: int, rank: int, alpha: float) -> "Primitive":
        generator = _rng(seed, 10, primitive_index)
        V = _spectral_normalize(generator.normal(size=(rank, d)))
        U = _spectral_normalize(generator.normal(size=(d, rank)))
        b = generator.normal(scale=0.2, size=rank)
        return cls(U=U, V=V, b=b, alpha=alpha)

    def __call__(self, z: Array) -> Array:
        z = np.asarray(z, dtype=np.float64)
        hidden = np.tanh(z @ self.V.T + self.b)
        return np.tanh(z + self.alpha * (hidden @ self.U.T))


@dataclass(frozen=True)
class Program:
    primitive_ids: tuple[int, ...]

    def execute(self, library: tuple[Primitive, ...], x: Array) -> Array:
        z = np.asarray(x, dtype=np.float64)
        for primitive_id in self.primitive_ids:
            z = library[primitive_id](z)
        return z


@dataclass(frozen=True)
class Task:
    task_id: str
    program: Program
    train_x: Array
    train_y: Array
    eval_x: Array
    eval_y: Array


@dataclass(frozen=True)
class World:
    config: WorldConfig
    library: tuple[Primitive, ...]
    tasks: tuple[Task, ...]

    @classmethod
    def generate(cls, config: WorldConfig) -> "World":
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
        tasks: list[Task] = []
        for task_index, (program, task_id) in enumerate(zip(programs, task_ids, strict=True)):
            train_rng = _rng(config.seed, 30, task_index)
            eval_rng = _rng(config.seed, 31, task_index)
            train_x = train_rng.normal(size=(config.examples_per_task, config.state_dim))
            eval_x = eval_rng.normal(size=(config.evaluation_examples, config.state_dim))
            tasks.append(
                Task(
                    task_id=task_id,
                    program=program,
                    train_x=train_x,
                    train_y=program.execute(library, train_x),
                    eval_x=eval_x,
                    eval_y=program.execute(library, eval_x),
                )
            )
        return cls(config=config, library=library, tasks=tuple(tasks))

    def public_task(self, index: int) -> tuple[str, Array, Array]:
        """Return only information available to a non-oracle learner."""
        task = self.tasks[index]
        return task.task_id, task.train_x, task.train_y

    def programs_json(self) -> list[dict[str, object]]:
        return [
            {"task_index": i, "task_id": task.task_id, "primitive_ids": list(task.program.primitive_ids)}
            for i, task in enumerate(self.tasks)
        ]

    def diagnostics(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for i, task in enumerate(self.tasks):
            y = task.eval_y
            rows.append(
                {
                    "task_index": i,
                    "task_id": task.task_id,
                    "output_variance": float(np.var(y)),
                    "output_abs_mean": float(np.mean(np.abs(y))),
                    "saturation_fraction": float(np.mean(np.abs(y) > 0.98)),
                }
            )
        return rows

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def _sample_programs(config: WorldConfig) -> tuple[Program, ...]:
    all_programs = list(product(range(config.teacher_primitives), repeat=config.program_length))
    order = _rng(config.seed, 20).permutation(len(all_programs))[: config.tasks]
    return tuple(Program(tuple(all_programs[int(i)])) for i in order)


def _opaque_task_ids(config: WorldConfig) -> tuple[str, ...]:
    generator = _rng(config.seed, 21)
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < config.tasks:
        value = generator.bytes(8).hex()
        if value not in seen:
            seen.add(value)
            values.append(f"task_{value}")
    return tuple(values)
