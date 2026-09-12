"""J1c: length curriculum (CF3), with CF6 survival statistics.

Frozen in `J1C_LENGTH_CURRICULUM_PLAN.md` (77984dd). Three stages share only
the shared library: 60 length-1 tasks, 64 length-2 tasks (both from
`curriculum_world`, programs with replacement), then the canonical 64-task
length-3 rotated world on SO1's stream 103. Routes are the learner's own soft
task codes throughout; no oracle anywhere. RESET re-initializes the library
before stage 3; SO1's `L_b2_g131072` is the matched-compute non-staged
baseline.

Restartable: one durable hashed record per (arm, world) and a tensor-only
model per stage; relaunch resumes; timestamped run.log and status.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1_search_loop import ari
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import (
    THRESHOLD, _flat_shared, _relative_change, score, world_config,
)
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.learned_lifetime import _shared_optimizer, _training_values
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, restore_model, save_model, writer_lock,
)
from row.rotated_world import generate_rotated_world

PLAN = Path("J1C_LENGTH_CURRICULUM_PLAN.md")
SO1_REPORT = Path("reports/so1_budget_bracket_r2.json")
OUTPUT = Path("reports/j1c_curriculum.json")
ROOT = Path("artifacts/j1c_curriculum")
PROTOCOL_ID = "J1c-length-curriculum-v1"
WORLDS = (0, 1, 2)
BATCH = 2
BASELINE_KEY = "L_b2_g131072"
# (length, tasks, updates, stream root, stream tail)
STAGES = ((1, 60, 16384, 1708, 1), (2, 64, 16384, 1708, 2), (3, 64, 32768, 1702, 103))


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "worlds": list(WORLDS), "batch": BATCH,
            "stages": [{"length": L, "tasks": n, "updates": u, "stream": [r, "world", t]}
                       for L, n, u, r, t in STAGES],
            "threshold": THRESHOLD, "implementation": "batched_rotation_v1", "baseline": BASELINE_KEY,
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, SO1_REPORT)}, "environment": environment()}


def stage_setup(world_seed: int, stage: int, model_seed: int = 5000):
    """Resolved config and world for one stage; stage 3 is the canonical world."""
    base = load_config("configs/v1.yaml")
    length, tasks, updates, root, tail = STAGES[stage - 1]
    if stage == 3:
        cfg = world_config(base, world_seed)
        world = generate_rotated_world(cfg.world)
    else:
        cfg = replace(base, world=CurriculumWorldConfig.from_world(
            replace(base.world, seed=world_seed), program_length=length, tasks=tasks, stage=f"length-{length}"))
        world = generate_curriculum_world(cfg.world)
    cfg = replace(cfg, discrete_model=replace(cfg.discrete_model, task_steps=length, seed=model_seed))
    return cfg, world, updates, np.random.SeedSequence([root, world_seed, tail])


def library_sha(model) -> str:
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()


def slot_functions(model, probe: torch.Tensor) -> list[torch.Tensor]:
    with torch.no_grad():
        return [operator(probe).clone() for operator in model.library]


def train_stage(cfg, world, updates: int, seed_sequence, model=None) -> tuple:
    """One stage: learned soft task codes, shared library trained; no oracle."""
    global_lr, task_lr, weight_decay, _, _, _, _ = _training_values(cfg, "rotated_discrete")
    if model is None:
        model = build_fast(cfg)
    codes = [model.begin_task(task.task_id) for task in world.tasks]
    optimizer = _shared_optimizer(model, global_lr, weight_decay)
    optimizer.add_param_group({"params": codes, "lr": task_lr, "weight_decay": 0.0})
    initial_shared = _flat_shared(model)
    initial_codes = torch.cat([c.detach().flatten().clone() for c in codes])
    all_x = torch.tensor(np.concatenate([t.train_x for t in world.tasks]), dtype=torch.float32)
    all_y = torch.tensor(np.concatenate([t.train_y for t in world.tasks]), dtype=torch.float32)
    all_ids = [t.task_id for t in world.tasks for _ in range(len(t.train_x))]
    rng = np.random.default_rng(seed_sequence)
    trajectory = {"0": score(model, world)["median"]}
    started = time.perf_counter()
    for update in range(1, updates + 1):
        model.set_training_progress((update - 1) / max(1, updates - 1))
        model.train()
        idx = rng.integers(0, len(all_x), size=BATCH)
        indices = torch.tensor(idx, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            model.forward_tasks(all_x.index_select(0, indices), [all_ids[int(i)] for i in idx]),
            all_y.index_select(0, indices))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        optimizer.step()
        if update in {updates // 8, updates // 4, updates // 2, updates}:
            trajectory[str(update)] = score(model, world)["median"]
    final = score(model, world)
    hard = model.hard_routes()
    teacher = {t.task_id: [int(p) for p in t.program.primitive_ids] for t in world.tasks}
    ids = list(teacher)
    result = {"length": cfg.discrete_model.task_steps, "tasks": len(ids), "updates": updates,
              "terminal_median": final["median"], "below_0.05": final["below_0.05"], "trajectory": trajectory,
              "final_per_task": final["per_task"],
              "ari_by_position": [ari([hard[t][p] for t in ids], [teacher[t][p] for t in ids])
                                  for p in range(cfg.discrete_model.task_steps)],
              "slots_used": len({s for t in ids for s in hard[t]}),
              "shared_relative_change": _relative_change(initial_shared, _flat_shared(model)),
              "code_relative_change": _relative_change(
                  initial_codes, torch.cat([c.detach().flatten().clone() for c in codes])),
              "library_sha256": library_sha(model), "seconds": round(time.perf_counter() - started, 1)}
    # Which slot does each teacher operation's traffic go through (analysis only)?
    traffic = {}
    for task_id in ids:
        for position, primitive in enumerate(teacher[task_id]):
            traffic.setdefault(primitive, Counter())[hard[task_id][position]] += 1
    result["slot_by_operation"] = {str(p): int(c.most_common(1)[0][0]) for p, c in traffic.items()}
    return model, result


def run_arm(arm: str, world_seed: int, artifact: Path | None = None, scale: int = 1,
            model_seed: int = 5000) -> dict:
    torch.set_num_threads(1)
    probe_cfg, probe_world, _, _ = stage_setup(world_seed, 3, model_seed)
    probe = torch.tensor(probe_world.tasks[0].eval_x, dtype=torch.float32)
    stages, model, transfers, stage1_functions = {}, None, [], None
    for stage in (1, 2, 3):
        cfg, world, updates, stream = stage_setup(world_seed, stage, model_seed)
        carried = None
        if model is not None and not (arm == "RESET" and stage == 3):
            fresh = build_fast(cfg)
            fresh.library.load_state_dict(model.library.state_dict())
            carried = library_sha(fresh)
            if carried != library_sha(model):
                raise RuntimeError("library did not transfer bitwise")
            if fresh.task_codes:
                raise RuntimeError("task codes carried across a stage boundary")
            model = fresh
        elif model is not None:
            model = build_fast(cfg)  # RESET: fresh library before stage 3
        model, result = train_stage(cfg, world, max(1, updates // scale), stream, model)
        result["library_sha256_at_start"] = carried
        stages[str(stage)] = result
        transfers.append({"stage": stage, "carried_library_sha256": carried})
        if stage == 1:
            stage1_functions = slot_functions(model, probe)
        if artifact is not None:
            save_model(artifact / f"stage{stage}", model, cfg)
    survival = []
    final_functions = slot_functions(model, probe)
    for operation, slot in stages["1"]["slot_by_operation"].items():
        reference = stage1_functions[slot]
        survival.append({
            "operation": int(operation), "stage1_slot": slot,
            "stage3_slot": stages["3"]["slot_by_operation"].get(operation),
            "still_used": stages["3"]["slot_by_operation"].get(operation) == slot,
            "functional_distance": float(torch.norm(final_functions[slot] - reference) / torch.norm(reference)),
        })
    return {"arm": arm, "world": world_seed, "model_seed": model_seed, "stages": stages, "transfers": transfers,
            "survival": survival, "terminal_median": stages["3"]["terminal_median"],
            "persisting_pairings": sum(s["still_used"] for s in survival)}


def classify(cells: dict, baseline: dict) -> str:
    expected = {f"{a}_w{w}" for a in ("STAGED", "RESET") for w in WORLDS}
    if not expected <= set(cells):
        return "HARNESS_FAILED"
    for cell in cells.values():
        for k in ("2", "3"):
            carried = cell["stages"][k]["library_sha256_at_start"]
            expected_carry = not (cell["arm"] == "RESET" and k == "3")
            if expected_carry and carried != cell["stages"][str(int(k) - 1)]["library_sha256"]:
                return "HARNESS_FAILED"  # library did not transfer from the previous stage
            if not expected_carry and carried is not None:
                return "HARNESS_FAILED"  # RESET must start stage 3 with a fresh library
        for stage in cell["stages"].values():
            if not (math.isfinite(stage["terminal_median"]) and stage["shared_relative_change"] > 0
                    and stage["code_relative_change"] > 0):
                return "HARNESS_FAILED"
    staged = {w: cells[f"STAGED_w{w}"]["terminal_median"] for w in WORLDS}
    reset = {w: cells[f"RESET_w{w}"]["terminal_median"] for w in WORLDS}
    passing = [w for w in WORLDS if staged[w] <= THRESHOLD]
    if len(passing) >= 2 and all(staged[w] < reset[w] and staged[w] < baseline[w] for w in passing):
        return "J1C_ACQUIRES"
    if sum(staged[w] <= 0.5 * min(reset[w], baseline[w]) for w in WORLDS) >= 2:
        return "J1C_IMPROVES"
    return "J1C_FAILS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="world 1, budgets divided by 64")
    args = parser.parse_args()
    if args.dry_run:
        for arm in ("STAGED", "RESET"):
            started = time.perf_counter()
            r = run_arm(arm, 1, scale=64)
            print(arm, {k: round(v["terminal_median"], 4) for k, v in r["stages"].items()},
                  "carried", [t["carried_library_sha256"] is not None for t in r["transfers"]],
                  "persist", r["persisting_pairings"], f"{time.perf_counter() - started:.1f}s")
        return 0

    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    require_clean_code(OUTPUT)
    expected = protocol()
    sha = fingerprint(expected)
    so1 = json.loads(SO1_REPORT.read_text())["cells"]
    baseline = {w: so1[BASELINE_KEY][str(w)]["terminal_median"] for w in WORLDS}
    run_log, status_path = ROOT / "run.log", ROOT / "status.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    with writer_lock(ROOT / "launcher.lock"):
        if OUTPUT.exists():
            out = json.loads(OUTPUT.read_text())
            if out.get("protocol_sha256") != sha or out.get("git_commit") != git_commit():
                raise SystemExit("existing report has a different protocol or commit; preserve it first")
            if out.get("complete"):
                return 0
            log_line(run_log, f"RESUME at {git_commit()}")
        else:
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": git_commit(), "protocol": expected,
                   "protocol_sha256": sha, "started_utc": now(), "cells": {}, "baseline": baseline,
                   "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID} at {git_commit()} pid {os.getpid()}")
        atomic_json(OUTPUT, out)
        jobs = [(arm, w) for arm in ("STAGED", "RESET") for w in (1, 2, 0)]
        try:
            for done, (arm, w) in enumerate(jobs):
                name = f"{arm}_w{w}"
                cell_dir = ROOT / "cells" / name
                path = cell_dir / "result.json"
                stamp = {"arm": arm, "world": w, "git_commit": git_commit(), "protocol_sha256": sha}
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": git_commit(),
                                          "cells_done": done, "cells_total": len(jobs), "current": name,
                                          "updated_utc": now()})
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["result_sha256"] != fingerprint(stored["result"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    result = stored["result"]
                    log_line(run_log, f"[{name}] reused validated durable cell")
                else:
                    log_line(run_log, f"[{name}] start")
                    result = run_arm(arm, w, artifact=cell_dir)
                    atomic_json(path, {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                                       "artifact_sha256": {f"stage{s}/model.pt": digest(cell_dir / f"stage{s}" / "model.pt")
                                                           for s in (1, 2, 3)}, "finished_utc": now()})
                out["cells"][name] = result
                atomic_json(OUTPUT, out)
                per_stage = {k: round(v["terminal_median"], 4) for k, v in result["stages"].items()}
                log_line(run_log, f"[{name}] saved: stages {per_stage} below_0.05 "
                                  f"{result['stages']['3']['below_0.05']}/64 persist "
                                  f"{result['persisting_pairings']}/6 ari3 "
                                  f"{[round(a, 2) for a in result['stages']['3']['ari_by_position']]}")
            out["classification"] = classify(out["cells"], baseline)
            out["complete"] = True
            out["finished_utc"] = now()
            atomic_json(OUTPUT, out)
            atomic_json(status_path, {"state": "complete", "classification": out["classification"],
                                      "cells_done": len(jobs), "cells_total": len(jobs), "updated_utc": now()})
            log_line(run_log, f"COMPLETE classification {out['classification']}")
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(ROOT / "exit.json", {"git_commit": git_commit(), "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
