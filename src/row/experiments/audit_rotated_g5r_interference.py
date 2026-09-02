"""G5R Stage D: separate route inference from online interference.

Frozen in `G5R_INTERFERENCE_PLAN.md` (Amendment 1). A 2 x 2 over ROUTES
(oracle-pinned versus learned) and SCHEDULE (offline IID versus the unchanged
online lifetime loop) at the lifetime's own gradient budget, plus the Stage C
budget on the offline learned-route axis. Two cells already exist and are
reused, not rerun: C_hi (Stage C of the previous diagnosis, read from its
frozen report) and O_lr (the G5R lifetime artifacts, rescored here).

Development worlds 0-2 only. Teacher primitive identities enter only the
explicitly labelled oracle arms. Nothing here can revise G5R's 0/3 verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from row.config import load_config
from row.experiments.audit_e0_export import git_commit, load_model
from row.experiments.audit_rotated_g5r_diagnosis import (
    STAGE_C_BATCH,
    STAGE_C_STEPS,
    STAGE_C_THRESHOLD,
    WORLDS_REQUIRED,
    assign_slots,
    require_clean_code,
    write,
)
from row.experiments.learned_lifetime import (
    _shared_optimizer,
    _training_values,
    resolved_learned_config,
    run,
)
from row.metrics import nmse
from row.models import RotatedDiscreteLibraryLearner
from row.provenance import validate_artifact
from row.rotated_world import generate_rotated_world

WORLDS = (0, 1, 2)
LIFETIME_UPDATES = 8192
LIFETIME_BATCH = 2
THRESHOLD = STAGE_C_THRESHOLD
PIN_LOGIT = 100.0
ANCHOR_TOLERANCE = 1e-6
OFFLINE_CELLS = {
    # name: (oracle routes, updates, batch, cell index for the sampling stream)
    "C_lo": (True, LIFETIME_UPDATES, LIFETIME_BATCH, 0),
    "L_lo": (False, LIFETIME_UPDATES, LIFETIME_BATCH, 1),
    "L_hi": (False, STAGE_C_STEPS, STAGE_C_BATCH, 2),
}
CHECKPOINTS = (0, 256, 1024, 4096, 8192)
ARTIFACTS = Path("artifacts/g5r_interference/oracle_online")
G5R_ARTIFACTS = Path("artifacts/g5r_rotated")
STAGE_C_REPORT = Path("reports/rotated_g5r_diagnosis.json")
PROTOCOL_ID = "G5R-StageD-interference-v1"


def build_model(config) -> RotatedDiscreteLibraryLearner:
    selected = config.discrete_model
    return RotatedDiscreteLibraryLearner(
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


def oracle_routes(world, assignment: dict[int, int]) -> dict[str, tuple[int, ...]]:
    return {
        task.task_id: tuple(assignment[int(p)] for p in task.program.primitive_ids)
        for task in world.tasks
    }


def pin_code(code: Tensor, route: tuple[int, ...]) -> None:
    """Pin a task code to a hard route and freeze it (oracle arms only)."""
    with torch.no_grad():
        code.fill_(-PIN_LOGIT)
        for step, slot in enumerate(route):
            code[step, slot] = PIN_LOGIT
    code.requires_grad_(False)
    represented = tuple(int(v) for v in torch.argmax(code, dim=-1))
    if represented != tuple(route):
        raise RuntimeError("hard oracle route did not survive code construction")


def pinned_is_one_hot(code: Tensor, temperature: float) -> bool:
    probabilities = torch.softmax(code.detach() / temperature, dim=-1)
    target = torch.nn.functional.one_hot(
        torch.argmax(code.detach(), dim=-1), code.shape[-1]
    ).to(probabilities.dtype)
    return bool(torch.equal(probabilities, target))


@torch.no_grad()
def score(model, world) -> dict:
    """The shared Stage D scorer: hard-route query NMSE per training task.

    Uses `model(x, task_id)` in eval mode, exactly as the lifetime's own
    `_evaluate`, so the anchor against the G5R artifacts is meaningful.
    """
    model.eval()
    per_task = {}
    for task in world.tasks:
        prediction = model(torch.tensor(task.eval_x, dtype=torch.float32), task.task_id)
        per_task[task.task_id] = float(nmse(prediction.cpu().numpy(), task.eval_y))
    values = np.asarray(list(per_task.values()))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "below_0.02": int(np.sum(values <= 0.02)),
        "below_0.05": int(np.sum(values <= 0.05)),
        "below_0.1": int(np.sum(values <= 0.1)),
        "per_task": per_task,
    }


def _flat_shared(model) -> Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.shared_parameters()])


def _relative_change(before: Tensor, after: Tensor) -> float:
    return float(torch.norm(after - before) / torch.norm(before).clamp_min(1e-12))


def offline_cell(
    config,
    world,
    assignment: dict[int, int],
    *,
    oracle: bool,
    updates: int,
    batch: int,
    cell_index: int,
    checkpoints_requested: tuple[int, ...] = CHECKPOINTS,
) -> dict:
    global_lr, task_lr, weight_decay, _, _, _, _ = _training_values(
        config, "rotated_discrete"
    )
    model = build_model(config)
    routes = oracle_routes(world, assignment)
    codes = []
    for task in world.tasks:
        code = model.begin_task(task.task_id)
        if oracle:
            pin_code(code, routes[task.task_id])
        codes.append(code)
    optimizer = _shared_optimizer(model, global_lr, weight_decay)
    if not oracle:
        optimizer.add_param_group({"params": codes, "lr": task_lr, "weight_decay": 0.0})
    initial_shared = _flat_shared(model)
    initial_codes = torch.cat([c.detach().flatten().clone() for c in codes])

    all_x = torch.tensor(
        np.concatenate([task.train_x for task in world.tasks]), dtype=torch.float32
    )
    all_y = torch.tensor(
        np.concatenate([task.train_y for task in world.tasks]), dtype=torch.float32
    )
    all_ids = [task.task_id for task in world.tasks for _ in range(len(task.train_x))]
    rng = np.random.default_rng(
        np.random.SeedSequence([1702, config.world.seed, cell_index])
    )
    wanted = {c for c in checkpoints_requested if c <= updates} | {updates}
    checkpoints = {"0": score(model, world)}
    losses_finite = True
    for update in range(1, updates + 1):
        # The lifetime's global temperature schedule: progress runs from 0 at
        # the first update to 1 at the last.
        model.set_training_progress((update - 1) / max(1, updates - 1))
        model.train()
        indices_np = rng.integers(0, len(all_x), size=batch)
        indices = torch.tensor(indices_np, dtype=torch.long)
        task_ids = [all_ids[int(i)] for i in indices_np]
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_tasks(all_x.index_select(0, indices), task_ids)
        loss = torch.nn.functional.mse_loss(prediction, all_y.index_select(0, indices))
        if not bool(torch.isfinite(loss)):
            losses_finite = False
            raise RuntimeError(f"non-finite offline loss at update {update}")
        loss.backward()
        optimizer.step()
        if update in wanted:
            checkpoints[str(update)] = score(model, world)
    final = checkpoints[str(updates)]
    hard = {k: tuple(v) for k, v in model.hard_routes().items()}
    route_match = all(hard[t] == routes[t] for t in routes) if oracle else None
    final_codes = torch.cat([c.detach().flatten().clone() for c in codes])
    shared_change = _relative_change(initial_shared, _flat_shared(model))
    finite = (
        losses_finite
        and all(
            math.isfinite(v)
            for record in checkpoints.values()
            for v in record.values()
            if isinstance(v, float)
        )
        and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    )
    return {
        "oracle_routes": oracle,
        "updates": updates,
        "batch": batch,
        "checkpoints": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_task"}
            for k, v in checkpoints.items()
        },
        "final_per_task": final["per_task"],
        "terminal_median": final["median"],
        "shared_relative_change": shared_change,
        "code_relative_change": (
            None if oracle else _relative_change(initial_codes, final_codes)
        ),
        "pinned_routes_preserved": route_match,
        "pinned_one_hot_at_1.0": (
            all(pinned_is_one_hot(c, 1.0) for c in codes) if oracle else None
        ),
        "routing": model.routing_diagnostics(),
        "finite": finite,
        "passes": bool(
            finite
            and shared_change > 0.0
            and (route_match is None or route_match)
            and final["median"] <= THRESHOLD
        ),
    }


def online_stamp(config) -> dict:
    return {
        "protocol": PROTOCOL_ID,
        "cell": "O_or",
        "tasks": config.world.tasks,
        "world_seed": config.world.seed,
        "model_seed": config.discrete_model.seed,
        "slots": config.discrete_model.operator_slots,
        "teacher_family": "rotated",
        "learner_family": "rotated_discrete",
        "routes": "oracle-pinned via assign_slots; logits +/-100; requires_grad False",
        "loop": "learned_lifetime.run unchanged except task_code_hook",
    }


def run_online_oracle(config, world, assignment: dict[int, int]) -> Path:
    """Run (or skip if complete) the O_or lifetime for one world. One writer."""
    path = Path(config.output_directory)
    stamp = online_stamp(config)
    stamp_path = path / "g5r_interference_stamp.json"
    summary_path = path / "summary.json"
    if summary_path.exists():
        if not stamp_path.exists():
            raise SystemExit(f"FATAL: {path} has a summary but no Stage D stamp")
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stored != stamp:
            raise SystemExit(f"FATAL: {path} was produced under {stored}, not {stamp}")
        return path
    if stamp_path.exists():
        stored = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stored != stamp:
            raise SystemExit(f"FATAL: partial artifact {path} has wrong stamp {stored}")
    if path.exists() and any(path.iterdir()) and not stamp_path.exists():
        raise SystemExit(f"FATAL: refusing to write into non-empty unstamped {path}")
    path.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    routes = oracle_routes(world, assignment)

    def hook(model, task, world_task_index):
        pin_code(model.task_codes[task.task_id], routes[task.task_id])

    run(config, kind="rotated_discrete", world=world, task_code_hook=hook)
    return path


def _last_task_end_of_task(path: Path) -> tuple[str, float, float]:
    """(last task id, its end-of-task final_nmse, median end-of-task over tasks)."""
    finals = []
    last = None
    with (path / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") == "task_summary":
                finals.append(float(row["final_nmse"]))
                last = (row["task_id"], float(row["final_nmse"]))
    if last is None:
        raise RuntimeError(f"{path} has no task_summary rows")
    return last[0], last[1], float(np.median(finals))


def score_online_artifact(
    config, world, path: Path, *, oracle_assignment: dict[int, int] | None
) -> dict:
    """Reload a lifetime artifact and score it with the shared scorer."""
    expected = resolved_learned_config(
        replace(config, output_directory=path), "rotated_discrete", "forward"
    )
    validate_artifact(path, expected, "rotated_discrete", backfill_missing_fingerprint=False)
    model = load_model(replace(config, output_directory=path), path, "rotated_discrete")[0]
    terminal = score(model, world)
    last_id, last_value, end_of_task_median = _last_task_end_of_task(path)
    anchor_error = abs(terminal["per_task"][last_id] - last_value)
    result = {
        "artifact": str(path),
        "terminal_median": terminal["median"],
        "terminal": {k: v for k, v in terminal.items() if k != "per_task"},
        "final_per_task": terminal["per_task"],
        "end_of_task_median": end_of_task_median,
        "terminal_minus_end_of_task_median": terminal["median"] - end_of_task_median,
        "anchor_last_task_id": last_id,
        "anchor_abs_error": anchor_error,
        "anchor_passes": bool(anchor_error <= ANCHOR_TOLERANCE),
        "finite": all(bool(torch.isfinite(p).all()) for p in model.parameters())
        and all(math.isfinite(v) for v in terminal["per_task"].values()),
    }
    if oracle_assignment is not None:
        routes = oracle_routes(world, oracle_assignment)
        hard = {k: tuple(v) for k, v in model.hard_routes().items()}
        stored = json.loads((path / "hard_routes.json").read_text(encoding="utf-8"))
        result["pinned_routes_preserved"] = bool(
            all(hard.get(t) == routes[t] for t in routes)
            and all(tuple(stored[t]) == routes[t] for t in routes)
        )
        result["pinned_one_hot_at_1.0"] = all(
            pinned_is_one_hot(model.task_codes[t], 1.0) for t in routes
        )
        initial = build_model(config)
        result["shared_relative_change"] = _relative_change(
            _flat_shared(initial), _flat_shared(model)
        )
        result["passes"] = bool(
            result["finite"]
            and result["anchor_passes"]
            and result["pinned_routes_preserved"]
            and result["shared_relative_change"] > 0.0
            and terminal["median"] <= THRESHOLD
        )
    else:
        result["passes"] = bool(
            result["finite"] and result["anchor_passes"] and terminal["median"] <= THRESHOLD
        )
    return result


def stage_c_reused() -> dict:
    report = json.loads(STAGE_C_REPORT.read_text(encoding="utf-8"))
    worlds = report["stage_c"]["worlds"]
    out = {}
    for seed in WORLDS:
        cell = worlds[str(seed)]
        out[str(seed)] = {
            "terminal_median": cell["checkpoints"][str(STAGE_C_STEPS)]["median"],
            "checkpoints": cell["checkpoints"],
            "passes": bool(cell["passes"]),
            "source": str(STAGE_C_REPORT),
        }
    return out


def cell_passes(worlds: dict) -> bool:
    return sum(bool(worlds[str(s)]["passes"]) for s in WORLDS) >= WORLDS_REQUIRED


def classify(c_lo: bool, o_or: bool, l_lo: bool) -> str:
    """The registered decision ladder."""
    if not c_lo:
        return "BUDGET_LIMITED"
    if not o_or and l_lo:
        return "ONLINE_INTERFERENCE"
    if o_or and not l_lo:
        return "ROUTE_INFERENCE"
    if not o_or and not l_lo:
        return "BOTH_INDEPENDENTLY_SUFFICIENT"
    return "INTERACTION_ONLY"


def protocol(config) -> dict:
    return {
        "id": PROTOCOL_ID,
        "frozen_plan": "G5R_INTERFERENCE_PLAN.md (Amendment 1)",
        "worlds": list(WORLDS),
        "lifetime_updates": LIFETIME_UPDATES,
        "lifetime_batch": LIFETIME_BATCH,
        "stage_c_updates": STAGE_C_STEPS,
        "stage_c_batch": STAGE_C_BATCH,
        "threshold": THRESHOLD,
        "worlds_required": WORLDS_REQUIRED,
        "pin_logit": PIN_LOGIT,
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "checkpoints": list(CHECKPOINTS),
        "sampling_stream": "SeedSequence([1702, world, cell_index])",
        "global_lr": config.discrete_model.global_learning_rate,
        "task_lr": config.discrete_model.task_learning_rate,
        "weight_decay": config.discrete_model.weight_decay,
        "model_seed": config.discrete_model.seed,
        "slots": config.discrete_model.operator_slots,
        "temperature": [
            config.discrete_model.initial_temperature,
            config.discrete_model.final_temperature,
        ],
        "offline_cells": {
            k: {"oracle": v[0], "updates": v[1], "batch": v[2], "cell_index": v[3]}
            for k, v in OFFLINE_CELLS.items()
        },
    }


def world_config(base, seed: int):
    return replace(base, world=replace(base.world, seed=seed))


def online_config(base, seed: int):
    cfg = world_config(base, seed)
    return replace(cfg, output_directory=ARTIFACTS / f"world_{seed}")


def _assignment(config, world):
    return assign_slots(
        world.library,
        config.world.state_dim,
        config.discrete_model.seed,
        config.discrete_model.operator_slots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/rotated_g5r_interference.json")
    )
    parser.add_argument(
        "--lifetime-world",
        type=int,
        default=None,
        help="worker mode: run only the O_or lifetime for this world and exit",
    )
    parser.add_argument("--pool", type=int, default=3, help="concurrent O_or lifetimes")
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = load_config(args.config)

    if args.lifetime_world is not None:
        cfg = online_config(base, args.lifetime_world)
        world = generate_rotated_world(cfg.world)
        run_online_oracle(cfg, world, _assignment(cfg, world))
        return

    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("preregistration check failed")
    if subprocess.run(["python", "tools/check_invalid.py"]).returncode != 0:
        raise SystemExit("invalid-artifact check failed")
    require_clean_code(args.output)

    expected_protocol = protocol(base)
    if args.output.exists():
        out = json.loads(args.output.read_text(encoding="utf-8"))
        if out.get("protocol") != expected_protocol:
            raise SystemExit("existing report has a different protocol fingerprint")
        if out.get("git_commit") != git_commit():
            raise SystemExit("existing report came from a different git commit")
    else:
        out = {
            "frozen_plan": "G5R_INTERFERENCE_PLAN.md",
            "git_commit": git_commit(),
            "protocol": expected_protocol,
            "cells": {"C_hi": stage_c_reused(), "O_lr": {}, "O_or": {}},
            "complete": False,
        }
        for name in OFFLINE_CELLS:
            out["cells"][name] = {}
        write(out, args.output)

    worlds = {s: generate_rotated_world(world_config(base, s).world) for s in WORLDS}
    assignments = {s: _assignment(world_config(base, s), worlds[s]) for s in WORLDS}

    # Anchor first: the scorer must reproduce the G5R artifacts before anything
    # new is read.
    for seed in WORLDS:
        key = str(seed)
        if key in out["cells"]["O_lr"]:
            continue
        cfg = world_config(base, seed)
        result = score_online_artifact(
            cfg, worlds[seed], G5R_ARTIFACTS / f"world_{seed}", oracle_assignment=None
        )
        if not result["anchor_passes"]:
            raise SystemExit(
                f"ANCHOR FAILED on world {seed}: |error| = {result['anchor_abs_error']:.3g}"
            )
        out["cells"]["O_lr"][key] = result
        write(out, args.output)
        print(
            f"[O_lr w{seed}] terminal median {result['terminal_median']:.4f} "
            f"end-of-task median {result['end_of_task_median']:.4f} anchor OK"
        )

    for name, (oracle, updates, batch, cell_index) in OFFLINE_CELLS.items():
        for seed in WORLDS:
            key = str(seed)
            if key in out["cells"][name]:
                continue
            result = offline_cell(
                world_config(base, seed),
                worlds[seed],
                assignments[seed],
                oracle=oracle,
                updates=updates,
                batch=batch,
                cell_index=cell_index,
            )
            out["cells"][name][key] = result
            write(out, args.output)
            print(
                f"[{name} w{seed}] terminal median {result['terminal_median']:.4f} "
                f"{'PASS' if result['passes'] else 'FAIL'}"
            )

    pending = [s for s in WORLDS if str(s) not in out["cells"]["O_or"]]
    if pending:
        processes = {}
        for seed in pending:
            cfg = online_config(base, seed)
            path = Path(cfg.output_directory)
            if (path / "summary.json").exists():
                continue
            while len(processes) >= args.pool:
                finished = [s for s, p in processes.items() if p.poll() is not None]
                if not finished:
                    time.sleep(5)
                    continue
                if processes.pop(finished[0]).returncode != 0:
                    raise SystemExit(f"O_or lifetime for world {finished[0]} failed")
            log = ARTIFACTS / f"world_{seed}.log"
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            processes[seed] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "row.experiments.audit_rotated_g5r_interference",
                    "--config",
                    str(args.config),
                    "--lifetime-world",
                    str(seed),
                ],
                stdout=log.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
            )
            print(f"[O_or w{seed}] launched lifetime, log {log}")
        for seed, process in processes.items():
            if process.wait() != 0:
                raise SystemExit(f"O_or lifetime for world {seed} failed")
        for seed in pending:
            cfg = online_config(base, seed)
            path = run_online_oracle(cfg, worlds[seed], assignments[seed])
            result = score_online_artifact(
                cfg, worlds[seed], path, oracle_assignment=assignments[seed]
            )
            if not result["anchor_passes"]:
                raise SystemExit(f"ANCHOR FAILED on O_or world {seed}")
            if not result["pinned_routes_preserved"]:
                raise SystemExit(f"PINNING FAILED on O_or world {seed}")
            out["cells"]["O_or"][str(seed)] = result
            write(out, args.output)
            print(
                f"[O_or w{seed}] terminal median {result['terminal_median']:.4f} "
                f"end-of-task median {result['end_of_task_median']:.4f} "
                f"{'PASS' if result['passes'] else 'FAIL'}"
            )

    passes = {name: cell_passes(out["cells"][name]) for name in out["cells"]}
    out["cell_passes"] = passes
    out["classification"] = classify(passes["C_lo"], passes["O_or"], passes["L_lo"])
    out["secondary_L_hi"] = (
        "route inference feasible offline at Stage C budget"
        if passes["L_hi"]
        else "learned routing fails offline even at Stage C budget"
    )
    out["complete"] = True
    write(out, args.output)
    print("cell passes:", passes)
    print("classification:", out["classification"])
    print("L_hi:", out["secondary_L_hi"])


if __name__ == "__main__":
    main()
