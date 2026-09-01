"""Localize G5R's acquisition failure under G5R_DIAGNOSIS_PLAN.md."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from row.config import load_config
from row.models import RotatedDiscreteLibraryLearner, RotatedLearnedOperator
from row.rotated_world import generate_rotated_world

WORLDS = (0, 1, 2)
PRIMITIVES = 6
TRAIN_EXAMPLES = 512
QUERY_EXAMPLES = 2048
ADAM_STEPS = 2000
ADAM_LR = 0.001
WEIGHT_DECAY = 0.0001
LBFGS_STEPS = 500
LBFGS_LR = 1.0
CELL_THRESHOLD = 0.02
CELLS_PER_WORLD = 5
WORLDS_REQUIRED = 2
STAGE_C_STEPS = 4096
STAGE_C_BATCH = 64
STAGE_C_CHECKPOINTS = (0, 256, 1024, 4096)
STAGE_C_THRESHOLD = 0.05
ARMS = ("H-Adam", "H-LBFGS", "Q-Adam")


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def nmse(prediction: Tensor, target: Tensor) -> float:
    return float(
        torch.mean((prediction - target) ** 2)
        / (torch.var(target, unbiased=False) + 1e-12)
    )


def determinant_sign(matrix: np.ndarray | Tensor) -> int:
    value = (
        float(torch.linalg.det(matrix).detach())
        if isinstance(matrix, Tensor)
        else float(np.linalg.det(matrix))
    )
    if abs(value) < 0.5:
        raise ValueError(f"matrix is not recognizably orthogonal: det={value}")
    return 1 if value > 0 else -1


def slot_determinant(d: int, model_seed: int) -> int:
    reflections = d - (model_seed & 1)
    return 1 if reflections % 2 == 0 else -1


def assign_slots(library, d: int, model_seed: int, slots: int) -> dict[int, int]:
    """Lowest unused canonical slot in the teacher primitive's O(d) component."""
    available = {1: [], -1: []}
    for slot in range(slots):
        seed = model_seed + 997 * slot
        available[slot_determinant(d, seed)].append(slot)
    assignment = {}
    for primitive_index, primitive in enumerate(library):
        sign = determinant_sign(primitive.Q)
        if not available[sign]:
            raise ValueError(f"no unused learner slot for determinant {sign}")
        assignment[primitive_index] = available[sign].pop(0)
    return assignment


def householder_decompose(target: Tensor, reflections: int) -> Tensor:
    """Construct vectors whose row-wise Householder product equals ``target``.

    Left reduction gives ``I = H_k ... H_1 target`` and therefore
    ``target = H_1 ... H_k``. Duplicate reflection pairs pad to the learner's
    fixed count without changing the map.
    """
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target must be square")
    d = target.shape[0]
    if reflections not in {d - 1, d}:
        raise ValueError("reflections must be d - 1 or d")
    work = target.detach().to(dtype=torch.float64).clone()
    eye = torch.eye(d, dtype=torch.float64, device=work.device)
    vectors: list[Tensor] = []
    for column in range(d):
        x = work[column:, column]
        desired = torch.zeros_like(x)
        desired[0] = 1.0
        delta = x - desired
        if float(torch.linalg.vector_norm(delta)) <= 1e-10:
            continue
        vector = torch.zeros(d, dtype=torch.float64, device=work.device)
        vector[column:] = delta
        reflector = eye - 2.0 * torch.outer(vector, vector) / torch.sum(vector.square())
        work = reflector @ work
        vectors.append(vector / torch.linalg.vector_norm(vector))
    if float(torch.max(torch.abs(work - eye))) > 1e-7:
        raise ValueError("Householder reduction did not reach identity")
    if len(vectors) > reflections or (reflections - len(vectors)) % 2:
        raise ValueError(
            f"target needs {len(vectors)} reflections, incompatible with {reflections}"
        )
    pad = torch.zeros(d, dtype=torch.float64, device=work.device)
    pad[0] = 1.0
    while len(vectors) < reflections:
        vectors.extend((pad.clone(), pad.clone()))
    return torch.stack(vectors)


def copy_teacher(operator: RotatedLearnedOperator, teacher) -> None:
    target = torch.tensor(teacher.Q.T, dtype=torch.float64)
    vectors = householder_decompose(target, operator.rotation.vectors.shape[0])
    with torch.no_grad():
        operator.U.copy_(torch.tensor(teacher.U, dtype=operator.U.dtype))
        operator.V.copy_(torch.tensor(teacher.V, dtype=operator.V.dtype))
        operator.b.copy_(torch.tensor(teacher.b, dtype=operator.b.dtype))
        if isinstance(operator.alpha, Tensor):
            operator.alpha.copy_(torch.tensor(teacher.alpha, dtype=operator.alpha.dtype))
        else:
            operator.alpha = float(teacher.alpha)
        operator.rotation.vectors.copy_(
            vectors.to(dtype=operator.rotation.vectors.dtype)
        )


class ProjectedRotatedOperator(nn.Module):
    """Dense orthogonal row-map, retracted externally after every Adam step."""

    def __init__(self, reference: RotatedLearnedOperator) -> None:
        super().__init__()
        self.U = nn.Parameter(reference.U.detach().clone())
        self.V = nn.Parameter(reference.V.detach().clone())
        self.b = nn.Parameter(reference.b.detach().clone())
        self.alpha = nn.Parameter(reference.alpha.detach().clone())
        self.row_map = nn.Parameter(reference.rotation.matrix().detach().clone())
        self.activation = reference.activation

    def forward(self, z: Tensor) -> Tensor:
        hidden = torch.nn.functional.linear(z, self.V, self.b)
        hidden = (
            torch.tanh(hidden)
            if self.activation == "tanh"
            else torch.nn.functional.gelu(hidden)
        )
        residual = z + self.alpha * torch.nn.functional.linear(hidden, self.U)
        return residual @ self.row_map

    @torch.no_grad()
    def retract(self) -> None:
        left, _, right = torch.linalg.svd(self.row_map, full_matrices=False)
        self.row_map.copy_(left @ right)


def _adamw(model: nn.Module) -> torch.optim.AdamW:
    alpha = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and (name == "alpha" or name.endswith(".alpha"))
    ]
    alpha_ids = {id(p) for p in alpha}
    regular = [
        p for p in model.parameters() if p.requires_grad and id(p) not in alpha_ids
    ]
    groups = [{"params": regular, "weight_decay": WEIGHT_DECAY}]
    if alpha:
        groups.append({"params": alpha, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=ADAM_LR)


def examples(world_seed: int, primitive_index: int, teacher) -> tuple[Tensor, ...]:
    d = teacher.Q.shape[0]
    train_rng = np.random.default_rng(
        np.random.SeedSequence([1700, world_seed, primitive_index, 0])
    )
    query_rng = np.random.default_rng(
        np.random.SeedSequence([1700, world_seed, primitive_index, 1])
    )
    train_x_np = train_rng.normal(size=(TRAIN_EXAMPLES, d))
    query_x_np = query_rng.normal(size=(QUERY_EXAMPLES, d))
    return (
        torch.tensor(train_x_np, dtype=torch.float32),
        torch.tensor(teacher(train_x_np), dtype=torch.float32),
        torch.tensor(query_x_np, dtype=torch.float32),
        torch.tensor(teacher(query_x_np), dtype=torch.float32),
    )


def stage_a_cell(teacher, seed: int, data: tuple[Tensor, ...]) -> dict:
    d, rank = teacher.Q.shape[0], teacher.V.shape[0]
    operator = RotatedLearnedOperator(
        d, rank, 0.2, seed, learnable_alpha=True, activation="tanh"
    )
    copy_teacher(operator, teacher)
    _, _, query_x, query_y = data
    with torch.no_grad():
        query_nmse = nmse(operator(query_x), query_y)
        represented = operator.rotation.matrix().to(dtype=torch.float64)
        target = torch.tensor(teacher.Q.T, dtype=torch.float64)
        rotation_error = float(torch.max(torch.abs(represented - target)))
        orthogonality = float(
            torch.max(torch.abs(represented.T @ represented - torch.eye(d, dtype=torch.float64)))
        )
    return {
        "query_nmse": query_nmse,
        "rotation_max_abs_error": rotation_error,
        "orthogonality_max_abs_error": orthogonality,
        "passes": bool(
            math.isfinite(query_nmse)
            and query_nmse <= 1e-8
            and rotation_error <= 1e-6
            and orthogonality <= 1e-6
        ),
    }


def _endpoint(model: nn.Module, data: tuple[Tensor, ...]) -> tuple[float, float]:
    train_x, train_y, query_x, query_y = data
    model.eval()
    with torch.no_grad():
        return nmse(model(train_x), train_y), nmse(model(query_x), query_y)


def stage_b_cell(
    teacher,
    seed: int,
    data: tuple[Tensor, ...],
    arm: str,
    *,
    adam_steps: int = ADAM_STEPS,
    lbfgs_steps: int = LBFGS_STEPS,
) -> dict:
    d, rank = teacher.Q.shape[0], teacher.V.shape[0]
    reference = RotatedLearnedOperator(
        d, rank, 0.2, seed, learnable_alpha=True, activation="tanh"
    )
    model: nn.Module = (
        ProjectedRotatedOperator(reference) if arm == "Q-Adam" else reference
    )
    train_x, train_y, _, _ = data
    initial_train, initial_query = _endpoint(model, data)
    model.train()
    if arm in {"H-Adam", "Q-Adam"}:
        optimizer = _adamw(model)
        for _ in range(adam_steps):
            optimizer.zero_grad()
            loss = torch.mean((model(train_x) - train_y) ** 2)
            if not bool(torch.isfinite(loss)):
                break
            loss.backward()
            optimizer.step()
            if isinstance(model, ProjectedRotatedOperator):
                model.retract()
    elif arm == "H-LBFGS":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=LBFGS_LR,
            max_iter=lbfgs_steps,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((model(train_x) - train_y) ** 2)
            penalty = sum(
                torch.sum(parameter.square())
                for name, parameter in model.named_parameters()
                if name != "alpha" and not name.endswith(".alpha")
            )
            objective = loss + 0.5 * WEIGHT_DECAY * penalty
            objective.backward()
            return objective

        optimizer.step(closure)
    else:
        raise ValueError(f"unknown arm {arm}")
    final_train, final_query = _endpoint(model, data)
    rotation = (
        model.row_map.detach()
        if isinstance(model, ProjectedRotatedOperator)
        else model.rotation.matrix().detach()
    )
    orthogonality = float(
        torch.max(torch.abs(rotation.T @ rotation - torch.eye(d)))
    )
    finite = all(
        math.isfinite(value)
        for value in (initial_train, initial_query, final_train, final_query, orthogonality)
    ) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return {
        "initial_train_nmse": initial_train,
        "initial_query_nmse": initial_query,
        "final_train_nmse": final_train,
        "final_query_nmse": final_query,
        "orthogonality_max_abs_error": orthogonality,
        "finite": finite,
        "passes": bool(finite and final_query <= CELL_THRESHOLD),
    }


def oracle_forward(
    model: RotatedDiscreteLibraryLearner,
    x: Tensor,
    task_ids: Sequence[str],
    routes: dict[str, tuple[int, ...]],
) -> Tensor:
    output = torch.zeros_like(x)
    for task_id in dict.fromkeys(task_ids):
        indices = torch.tensor(
            [i for i, value in enumerate(task_ids) if value == task_id],
            dtype=torch.long,
            device=x.device,
        )
        z = x.index_select(0, indices)
        for slot in routes[task_id]:
            z = model.library[slot](z)
        output = output.index_copy(0, indices, z)
    return output


@torch.no_grad()
def stage_c_score(model, world, routes) -> dict:
    values = []
    model.eval()
    for task in world.tasks:
        x = torch.tensor(task.eval_x, dtype=torch.float32)
        y = torch.tensor(task.eval_y, dtype=torch.float32)
        prediction = oracle_forward(model, x, [task.task_id] * len(x), routes)
        values.append(nmse(prediction, y))
    array = np.asarray(values)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "below_0.02": int(np.sum(array <= 0.02)),
        "below_0.05": int(np.sum(array <= 0.05)),
        "below_0.1": int(np.sum(array <= 0.1)),
    }


def stage_c_world(
    config,
    world,
    assignment: dict[int, int],
    *,
    steps: int = STAGE_C_STEPS,
    batch_size: int = STAGE_C_BATCH,
    checkpoints_requested: tuple[int, ...] = STAGE_C_CHECKPOINTS,
) -> dict:
    selected = config.discrete_model
    model = RotatedDiscreteLibraryLearner(
        d=config.world.state_dim,
        operator_slots=selected.operator_slots,
        operator_rank=selected.operator_rank,
        task_steps=selected.task_steps,
        alpha=selected.operator_alpha_init,
        initial_temperature=selected.initial_temperature,
        final_temperature=selected.final_temperature,
        seed=selected.seed,
        learnable_alpha=selected.learnable_alpha,
        activation=selected.operator_activation,
    )
    routes = {
        task.task_id: tuple(assignment[int(p)] for p in task.program.primitive_ids)
        for task in world.tasks
    }
    for task in world.tasks:
        code = model.begin_task(task.task_id)
        with torch.no_grad():
            code.fill_(-50.0)
            for step, slot in enumerate(routes[task.task_id]):
                code[step, slot] = 50.0
        code.requires_grad_(False)
        represented_route = tuple(int(v) for v in torch.argmax(code, dim=-1))
        if represented_route != routes[task.task_id]:
            raise RuntimeError("hard oracle route did not survive code construction")
    optimizer = _adamw(model)
    all_x = torch.tensor(
        np.concatenate([task.train_x for task in world.tasks]), dtype=torch.float32
    )
    all_y = torch.tensor(
        np.concatenate([task.train_y for task in world.tasks]), dtype=torch.float32
    )
    all_ids = [
        task.task_id for task in world.tasks for _ in range(len(task.train_x))
    ]
    rng = np.random.default_rng(np.random.SeedSequence([1701, config.world.seed]))
    checkpoints = {"0": stage_c_score(model, world, routes)}
    model.train()
    for update in range(1, steps + 1):
        indices_np = rng.integers(0, len(all_x), size=batch_size)
        indices = torch.tensor(indices_np, dtype=torch.long)
        task_ids = [all_ids[int(i)] for i in indices_np]
        optimizer.zero_grad()
        prediction = oracle_forward(
            model, all_x.index_select(0, indices), task_ids, routes
        )
        loss = torch.mean((prediction - all_y.index_select(0, indices)) ** 2)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite Stage C loss at update {update}")
        loss.backward()
        optimizer.step()
        if update in checkpoints_requested:
            checkpoints[str(update)] = stage_c_score(model, world, routes)
            model.train()
    final = checkpoints[str(steps)]
    finite = all(
        math.isfinite(value)
        for record in checkpoints.values()
        for value in record.values()
        if isinstance(value, float)
    ) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return {
        "assignment": {str(k): v for k, v in assignment.items()},
        "checkpoints": checkpoints,
        "finite": finite,
        "passes": bool(finite and final["median"] <= STAGE_C_THRESHOLD),
    }


def protocol() -> dict:
    return {
        "worlds": list(WORLDS),
        "primitives": PRIMITIVES,
        "train_examples": TRAIN_EXAMPLES,
        "query_examples": QUERY_EXAMPLES,
        "adam_steps": ADAM_STEPS,
        "adam_lr": ADAM_LR,
        "weight_decay": WEIGHT_DECAY,
        "lbfgs_steps": LBFGS_STEPS,
        "lbfgs_lr": LBFGS_LR,
        "cell_threshold": CELL_THRESHOLD,
        "cells_per_world": CELLS_PER_WORLD,
        "worlds_required": WORLDS_REQUIRED,
        "stage_c_steps": STAGE_C_STEPS,
        "stage_c_batch": STAGE_C_BATCH,
        "stage_c_checkpoints": list(STAGE_C_CHECKPOINTS),
        "stage_c_threshold": STAGE_C_THRESHOLD,
        "arms": list(ARMS),
    }


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_clean_code(output: Path) -> None:
    """Allow only the resumable report itself to be untracked or modified."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    allowed = output.resolve()
    unexpected = []
    for line in result.stdout.splitlines():
        candidate = Path(line[3:].strip().strip('"')).resolve()
        if candidate != allowed:
            unexpected.append(line)
    if unexpected:
        raise SystemExit(
            "diagnosis must run from clean committed code; unexpected status: "
            + repr(unexpected)
        )


def arm_summary(cells: dict, arm: str) -> dict:
    counts = {}
    for world_seed in WORLDS:
        counts[str(world_seed)] = sum(
            bool(cells[f"w{world_seed}_p{primitive}_{arm}"]["passes"])
            for primitive in range(PRIMITIVES)
        )
    worlds_passing = sum(count >= CELLS_PER_WORLD for count in counts.values())
    return {
        "passing_cells_by_world": counts,
        "passing_worlds": worlds_passing,
        "passes": bool(worlds_passing >= WORLDS_REQUIRED),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/rotated_g5r_diagnosis.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("preregistration check failed")
    require_clean_code(args.output)

    expected_protocol = protocol()
    if args.output.exists():
        out = json.loads(args.output.read_text(encoding="utf-8"))
        if out.get("protocol") != expected_protocol:
            raise SystemExit("existing report has a different protocol")
        if out.get("git_commit") != git_commit():
            raise SystemExit("existing report came from a different git commit")
    else:
        out = {
            "frozen_plan": "G5R_DIAGNOSIS_PLAN.md",
            "git_commit": git_commit(),
            "protocol": expected_protocol,
            "stage_a": {"cells": {}},
            "stage_b": {"cells": {}},
            "stage_c": {"worlds": {}},
        }

    config = load_config(args.config)
    worlds = {
        seed: generate_rotated_world(replace(config.world, seed=seed))
        for seed in WORLDS
    }
    assignments = {
        seed: assign_slots(
            worlds[seed].library,
            config.world.state_dim,
            config.discrete_model.seed,
            config.discrete_model.operator_slots,
        )
        for seed in WORLDS
    }

    for world_seed in WORLDS:
        for primitive_index, teacher in enumerate(worlds[world_seed].library):
            key = f"w{world_seed}_p{primitive_index}"
            if key in out["stage_a"]["cells"]:
                continue
            slot = assignments[world_seed][primitive_index]
            seed = config.discrete_model.seed + 997 * slot
            data = examples(world_seed, primitive_index, teacher)
            result = stage_a_cell(teacher, seed, data)
            result.update({"world": world_seed, "primitive": primitive_index, "slot": slot})
            out["stage_a"]["cells"][key] = result
            write(out, args.output)
            print(f"[A w{world_seed} p{primitive_index}] NMSE {result['query_nmse']:.3g} "
                  f"rotation {result['rotation_max_abs_error']:.3g} -> "
                  f"{'PASS' if result['passes'] else 'FAIL'}", flush=True)

    stage_a_passes = len(out["stage_a"]["cells"]) == 18 and all(
        cell["passes"] for cell in out["stage_a"]["cells"].values()
    )
    out["stage_a"]["passes"] = stage_a_passes
    if not stage_a_passes:
        out["classification"] = "IMPLEMENTATION_OR_REPRESENTABILITY_DEFECT"
        out["complete"] = True
        write(out, args.output)
        print(out["classification"])
        return

    for world_seed in WORLDS:
        for primitive_index, teacher in enumerate(worlds[world_seed].library):
            slot = assignments[world_seed][primitive_index]
            seed = config.discrete_model.seed + 997 * slot
            data = examples(world_seed, primitive_index, teacher)
            for arm in ARMS:
                key = f"w{world_seed}_p{primitive_index}_{arm}"
                if key in out["stage_b"]["cells"]:
                    continue
                result = stage_b_cell(teacher, seed, data, arm)
                result.update({
                    "world": world_seed,
                    "primitive": primitive_index,
                    "slot": slot,
                    "arm": arm,
                })
                out["stage_b"]["cells"][key] = result
                write(out, args.output)
                print(f"[B {arm} w{world_seed} p{primitive_index}] query "
                      f"{result['final_query_nmse']:.5f} -> "
                      f"{'PASS' if result['passes'] else 'FAIL'}", flush=True)

    summaries = {arm: arm_summary(out["stage_b"]["cells"], arm) for arm in ARMS}
    out["stage_b"]["arms"] = summaries
    h_adam = summaries["H-Adam"]["passes"]
    h_lbfgs = summaries["H-LBFGS"]["passes"]
    q_adam = summaries["Q-Adam"]["passes"]
    if not h_adam:
        if h_lbfgs:
            classification = "ADAM_OPTIMIZER_FAILURE"
        elif q_adam:
            classification = "HOUSEHOLDER_PARAMETERIZATION_FAILURE"
        else:
            classification = "ROTATED_OPERATOR_FINDABILITY_FAILURE"
        out["classification"] = classification
        out["complete"] = True
        write(out, args.output)
        print(classification)
        return

    for world_seed in WORLDS:
        key = str(world_seed)
        if key in out["stage_c"]["worlds"]:
            continue
        local_config = replace(config, world=replace(config.world, seed=world_seed))
        result = stage_c_world(local_config, worlds[world_seed], assignments[world_seed])
        out["stage_c"]["worlds"][key] = result
        write(out, args.output)
        final = result["checkpoints"][str(STAGE_C_STEPS)]
        print(f"[C w{world_seed}] median {final['median']:.5f} -> "
              f"{'PASS' if result['passes'] else 'FAIL'}", flush=True)

    passing_worlds = sum(
        result["passes"] for result in out["stage_c"]["worlds"].values()
    )
    out["stage_c"]["passing_worlds"] = passing_worlds
    out["stage_c"]["passes"] = bool(passing_worlds >= WORLDS_REQUIRED)
    out["classification"] = (
        "ROUTE_INFERENCE_OR_ONLINE_INTERFERENCE_FAILURE"
        if out["stage_c"]["passes"]
        else "JOINT_LIBRARY_OR_COMPOSITION_FAILURE"
    )
    out["complete"] = True
    write(out, args.output)
    print(out["classification"])


if __name__ == "__main__":
    main()
