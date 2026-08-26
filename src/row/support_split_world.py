"""E2: a world whose lifetime sees a CHOSEN set of programs, with three held-out strata.

`E2_COMPOSITION_PLAN.md`. The ordinary generator samples `tasks` programs at
random from `K ** D`. E2 needs the training set specified in advance so that
whole program structures — and, for the new stratum, whole PLACEMENTS of a
primitive in a position — can be withheld.

Three strata over the programs the lifetime does not train on:

    H1  triple-novel:   no withheld placement, every adjacent pair seen
    H2  pair-novel:     no withheld placement, some adjacent pair never seen
    H3  position-novel: places a withheld primitive in its withheld position

`H3` is why this generator exists: it asks whether a learned operator keeps its
semantics in a program position it never occupied, which the ordinary sampler
cannot ask because it cannot guarantee the placement was absent.

Provenance stays OUTSIDE `WorldConfig` (adding a field there would invalidate
every existing resolved-config fingerprint, AGENTS.md); the spec is recorded by
the runner in its own artifact file. A world built with no withheld placements
and the ordinary program sample reproduces `World.generate` exactly.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from row.world import (Primitive, Program, Task, World, WorldConfig, _opaque_task_ids, _rng,
                       _task_library)


@dataclass(frozen=True)
class SupportSplitSpec:
    """Which placements are withheld, and how the training set is chosen."""

    holdouts: int = 3
    spec_seed: int = 764
    context_min: int = 3
    balance_max: float = 2.0
    min_stratum: int = 16
    max_attempts: int = 256

    def as_dict(self) -> dict:
        return {"holdouts": self.holdouts, "spec_seed": self.spec_seed,
                "context_min": self.context_min, "balance_max": self.balance_max,
                "min_stratum": self.min_stratum, "max_attempts": self.max_attempts}


def _pairs(program) -> set:
    return {(i, program[i], program[i + 1]) for i in range(len(program) - 1)}


def withheld_placements(config: WorldConfig, spec: SupportSplitSpec) -> list[tuple[int, int]]:
    """(primitive, position) pairs the lifetime never sees. Deterministic in the seeds."""
    rng = np.random.default_rng(np.random.SeedSequence([spec.spec_seed, config.seed]))
    k, d = config.teacher_primitives, config.program_length
    if spec.holdouts > min(k, d):
        raise ValueError("at most one withheld position per primitive, and one per position")
    primitives = [int(v) for v in rng.permutation(k)[: spec.holdouts]]
    positions = [int(v) for v in rng.permutation(d)[: spec.holdouts]]
    return list(zip(primitives, positions))


def split_programs(config: WorldConfig, spec: SupportSplitSpec) -> dict:
    """The frozen training set and the three held-out strata.

    Amendment 1: the fill ORDER is searched over seeded attempts until every
    stratum reaches `min_stratum` and the balance and context constraints hold.
    A purely neutral fill can cover every adjacent pair and leave `H2` EMPTY —
    it did so for world 0 on the first attempt — and E2 registers minimum
    stratum sizes as a design constraint. The search is over the SPLIT's
    structural properties only; no model, loss, or performance quantity enters
    it, and the accepted attempt index is recorded in the artifact.
    """

    for attempt in range(spec.max_attempts):
        result = _split_attempt(config, spec, attempt)
        d = result["diagnostics"]
        if (min(d["stratum_sizes"].values()) >= spec.min_stratum
                and d["balance_ratio"] <= spec.balance_max
                and d["context_min"] >= spec.context_min):
            d["accepted_attempt"] = attempt
            return result
    raise ValueError(f"no split satisfying the registered constraints in "
                     f"{spec.max_attempts} attempts for world {config.seed}")


def _split_attempt(config: WorldConfig, spec: SupportSplitSpec, attempt: int) -> dict:
    k, d, budget = config.teacher_primitives, config.program_length, config.tasks
    held = withheld_placements(config, spec)
    all_programs = [tuple(p) for p in itertools.product(range(k), repeat=d)]

    def touches(program) -> bool:
        return any(program[i] == p for p, i in held)

    pool = [p for p in all_programs if not touches(p)]
    if len(pool) < budget:
        raise ValueError(f"training pool {len(pool)} smaller than the {budget}-task lifetime")
    rng = np.random.default_rng(np.random.SeedSequence([spec.spec_seed, config.seed, 1, attempt]))
    order = [int(i) for i in rng.permutation(len(pool))]
    train: list[tuple[int, ...]] = []
    required = {(p, i) for p in range(k) for i in range(d) if (p, i) not in held}
    need = set(required)
    for index in order:                                   # cover every required placement first
        if not need:
            break
        program = pool[index]
        gain = {(p, i) for i, p in enumerate(program)} & need
        if gain:
            train.append(program)
            need -= gain
    for index in order:                                   # then fill neutrally to the budget
        if len(train) >= budget:
            break
        program = pool[index]
        if program not in train:
            train.append(program)
    if need:
        raise ValueError(f"could not cover required placements: {sorted(need)}")
    seen_pairs: set = set()
    for program in train:
        seen_pairs |= _pairs(program)
    train_set = set(train)
    strata: dict[str, list] = {"H1": [], "H2": [], "H3": []}
    for program in all_programs:
        if program in train_set:
            continue
        if touches(program):
            strata["H3"].append(program)
        elif _pairs(program) <= seen_pairs:
            strata["H1"].append(program)
        else:
            strata["H2"].append(program)
    counts = np.bincount(np.array([p for program in train for p in program]), minlength=k)
    contexts: dict[int, set] = {p: set() for p in range(k)}
    for program in train:
        for i, p in enumerate(program):
            contexts[p].add((i, program[:i], program[i + 1:]))
    return {
        "train": train, "strata": strata, "withheld_placements": held,
        "diagnostics": {
            "train_size": len(train),
            "balance_ratio": float(counts.max() / max(counts.min(), 1)),
            "context_min": int(min(len(v) for v in contexts.values())),
            "stratum_sizes": {name: len(v) for name, v in strata.items()},
            "required_placements_covered": True,
        },
    }


def _build_tasks(config: WorldConfig, library, programs, task_ids, index_offset: int = 0):
    tasks = []
    for local_index, (program, task_id) in enumerate(zip(programs, task_ids, strict=True)):
        task_index = index_offset + local_index
        task_library = _task_library(config, library, task_index)
        train_x = _rng(config.seed, 30, task_index).normal(
            size=(config.examples_per_task, config.state_dim))
        eval_x = _rng(config.seed, 31, task_index).normal(
            size=(config.evaluation_examples, config.state_dim))
        executor = Program(tuple(program))
        tasks.append(Task(task_id=task_id, program=executor, teacher_library=task_library,
                          train_x=train_x, train_y=executor.execute(task_library, train_x),
                          eval_x=eval_x, eval_y=executor.execute(task_library, eval_x)))
    return tuple(tasks)


class SupportSplitWorld(World):
    """A `World` whose tasks are the chosen training programs, plus held-out strata."""

    held_out: dict = field(default_factory=dict)


def generate_support_split_world(config: WorldConfig, spec: SupportSplitSpec) -> World:
    library = tuple(
        Primitive.random(seed=config.seed, primitive_index=k, d=config.state_dim,
                         rank=config.teacher_rank, alpha=config.alpha)
        for k in range(config.teacher_primitives)
    )
    split = split_programs(config, spec)
    task_ids = _opaque_task_ids(config)
    world = World(config=config, library=library,
                  tasks=_build_tasks(config, library, split["train"], task_ids))
    # Held-out tasks use task indices beyond the lifetime, so their example
    # streams and task libraries are disjoint from every trained task's.
    held_tasks: dict[str, tuple] = {}
    offset = len(split["train"])
    for name in ("H1", "H2", "H3"):
        programs = split["strata"][name]
        ids = [f"task_heldout_{name}_{i}" for i in range(len(programs))]
        held_tasks[name] = _build_tasks(config, library, programs, ids, index_offset=offset)
        offset += len(programs)
    object.__setattr__(world, "held_out", held_tasks) if hasattr(world, "__dataclass_fields__") \
        else setattr(world, "held_out", held_tasks)
    object.__setattr__(world, "split_diagnostics", split["diagnostics"]) \
        if hasattr(world, "__dataclass_fields__") else None
    return world


class SupportSplitFactory:
    """Drop-in for `learned_lifetime.World` (the established injection pattern)."""

    def __init__(self, spec: SupportSplitSpec) -> None:
        self.spec = spec

    def generate(self, config: WorldConfig) -> World:
        return generate_support_split_world(config, self.spec)
