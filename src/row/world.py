"""Deterministic hidden operator worlds and fixed task datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
        if not 0.0 <= self.reuse_rho <= 1.0:
            raise ValueError("reuse_rho must be in [0, 1]")


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
    teacher_library: tuple[Primitive, ...]
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
            task_library = _task_library(config, library, task_index)
            train_rng = _rng(config.seed, 30, task_index)
            eval_rng = _rng(config.seed, 31, task_index)
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
        return cls(config=config, library=library, tasks=tuple(tasks))

    def public_task(self, index: int) -> tuple[str, Array, Array]:
        """Return only information available to a non-oracle learner."""
        task = self.tasks[index]
        return task.task_id, task.train_x, task.train_y

    def with_scrambled_task_ids(self, scramble_seed: int) -> "World":
        """Reassign independent opaque IDs without changing task contents or order."""
        original_ids = {task.task_id for task in self.tasks}
        task_ids = _draw_opaque_task_ids(
            _rng(self.config.seed, 22, scramble_seed),
            len(self.tasks),
            forbidden=original_ids,
        )
        return replace(
            self,
            tasks=tuple(
                replace(task, task_id=task_id)
                for task, task_id in zip(self.tasks, task_ids, strict=True)
            ),
        )

    def library_for_task(self, task_index: int) -> tuple[Primitive, ...]:
        if 0 <= task_index < len(self.tasks):
            return self.tasks[task_index].teacher_library
        return _task_library(self.config, self.library, task_index)

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

    def functional_reuse_diagnostics(
        self, probe_examples: int = 512, max_tasks: int = 16
    ) -> dict[str, float | int]:
        """Measure recurrence of task-specific residual effects on fixed probes."""
        generator = _rng(self.config.seed, 41)
        probe = generator.normal(size=(probe_examples, self.config.state_dim))
        identity_path = np.tanh(probe)
        task_count = min(max_tasks, len(self.tasks))
        correlations: list[float] = []
        normalized_distances: list[float] = []
        for primitive_index in range(self.config.teacher_primitives):
            effects = [
                self.tasks[task_index].teacher_library[primitive_index](probe) - identity_path
                for task_index in range(task_count)
            ]
            for first in range(task_count):
                for second in range(first + 1, task_count):
                    left = effects[first].ravel()
                    right = effects[second].ravel()
                    correlations.append(float(np.corrcoef(left, right)[0, 1]))
                    denominator = 0.5 * (float(np.var(left)) + float(np.var(right)))
                    normalized_distances.append(
                        float(np.mean(np.square(left - right)) / denominator)
                    )
        return {
            "configured_rho": self.config.reuse_rho,
            "probe_examples": probe_examples,
            "tasks_compared": task_count,
            "mean_pairwise_residual_correlation": float(np.mean(correlations)),
            "mean_pairwise_residual_normalized_distance": float(
                np.mean(normalized_distances)
            ),
        }

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def _sample_programs(config: WorldConfig) -> tuple[Program, ...]:
    all_programs = list(product(range(config.teacher_primitives), repeat=config.program_length))
    order = _rng(config.seed, 20).permutation(len(all_programs))[: config.tasks]
    return tuple(Program(tuple(all_programs[int(i)])) for i in order)


def _opaque_task_ids(config: WorldConfig) -> tuple[str, ...]:
    return _draw_opaque_task_ids(_rng(config.seed, 21), config.tasks)


def _draw_opaque_task_ids(
    generator: np.random.Generator,
    count: int,
    forbidden: set[str] | None = None,
) -> tuple[str, ...]:
    values: list[str] = []
    seen = set() if forbidden is None else set(forbidden)
    while len(values) < count:
        value = generator.bytes(8).hex()
        task_id = f"task_{value}"
        if task_id not in seen:
            seen.add(task_id)
            values.append(task_id)
    return tuple(values)


def _task_library(
    config: WorldConfig,
    base_library: tuple[Primitive, ...],
    task_index: int,
) -> tuple[Primitive, ...]:
    if config.reuse_rho == 1.0:
        return base_library
    independent_scale = np.sqrt(1.0 - config.reuse_rho**2)
    result: list[Primitive] = []
    for primitive_index, base in enumerate(base_library):
        generator = _rng(config.seed, 11, task_index, primitive_index)
        epsilon_V = _spectral_normalize(
            generator.normal(size=(config.teacher_rank, config.state_dim))
        )
        epsilon_U = _spectral_normalize(
            generator.normal(size=(config.state_dim, config.teacher_rank))
        )
        epsilon_b = generator.normal(scale=0.2, size=config.teacher_rank)
        result.append(
            Primitive(
                U=_spectral_normalize(
                    config.reuse_rho * base.U + independent_scale * epsilon_U
                ),
                V=_spectral_normalize(
                    config.reuse_rho * base.V + independent_scale * epsilon_V
                ),
                b=config.reuse_rho * base.b + independent_scale * epsilon_b,
                alpha=config.alpha,
            )
        )
    return tuple(result)
