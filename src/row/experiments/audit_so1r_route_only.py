"""SO1R: route-only inference on frozen oracle-acquired SO1 libraries.

Frozen in `SO1R_ROUTE_ONLY_PLAN.md` (cf32b61). For each of the six terminal
libraries of SO1 cells O_b2_g131072 and O_b64_g262144 (worlds 0-2), with every
library parameter frozen, four route arms choose a hard route per task from
SUPPORT data only and are scored on held-out query data: ORACLE (stored pinned
route; must reproduce SO1 bitwise), ENUM (exhaustive 1,728 routes by support
MSE), OPT (the learner's own softmax relaxation, Adam 0.05, T 1.0 -> 0.1,
2,000 full-batch steps), RANDOM (seeded floor control).

Restartable: one durable hashed record per library; relaunch resumes; a
timestamped run.log and an atomic status.json live in the run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import world_config
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, restore_model,
    world_digest, writer_lock,
)
from row.metrics import nmse
from row.rotated_world import generate_rotated_world

PLAN = Path("SO1R_ROUTE_ONLY_PLAN.md")
SO1_REPORT = Path("reports/so1_budget_bracket_r2.json")
SO1_CELLS = Path("artifacts/so1_restart2/cells")
OUTPUT = Path("reports/so1r_route_only.json")
ROOT = Path("artifacts/so1r_route_only")
PROTOCOL_ID = "SO1R-route-only-v1"
LIBRARIES = ("O_b2_g131072", "O_b64_g262144")  # library_index 0, 1
WORLDS = (0, 1, 2)
THRESHOLD = 0.05
OPT_STEPS = 2000
OPT_LR = 0.05
T_START, T_END = 1.0, 0.1
RANDOM_SEED = 1704
MIN_SUPPORT_DROP = 0.10


def protocol(config) -> dict:
    return {
        "id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "libraries": list(LIBRARIES),
        "worlds": list(WORLDS), "threshold": THRESHOLD, "opt_steps": OPT_STEPS, "opt_lr": OPT_LR,
        "temperature": [T_START, T_END], "random_seed": RANDOM_SEED,
        "min_support_drop": MIN_SUPPORT_DROP, "enum_routes": 12 ** 3,
        "selection": "support (train) MSE only; query (eval) NMSE for scoring",
        "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, SO1_REPORT)},
        "environment": environment(),
    }


class FrozenLibrary:
    """The fast kind's forward with an explicit coefficient matrix.

    Reproduces `FastRotatedDiscreteLibraryLearner.forward` operation for
    operation, so a one-hot coefficient matrix gives the model's own hard-route
    prediction bitwise (checked by the ORACLE arm against the SO1 report)."""

    def __init__(self, model):
        library = list(model.library)
        with torch.no_grad():
            self.V = torch.stack([op.V for op in library]).detach()
            self.U = torch.stack([op.U for op in library]).detach()
            self.b = torch.stack([op.b for op in library]).detach()
            self.alpha = torch.stack([op.alpha if torch.is_tensor(op.alpha) else torch.tensor(float(op.alpha))
                                      for op in library]).detach()
            from row.models.learned_models import batched_householder
            self.Q = batched_householder([op.rotation.vectors for op in library]).detach()
        self.gelu = library[0].activation == "gelu"
        self.slots = len(library)
        self.steps = model.task_steps

    def candidates(self, z):
        hidden = torch.einsum("bd,srd->bsr", z, self.V) + self.b
        hidden = torch.nn.functional.gelu(hidden) if self.gelu else torch.tanh(hidden)
        residual = z.unsqueeze(1) + self.alpha.view(1, -1, 1) * torch.einsum("bsr,sdr->bsd", hidden, self.U)
        return torch.einsum("bsd,sde->bse", residual, self.Q)

    def forward(self, x, coefficients):
        z = x
        for step in range(self.steps):
            z = torch.sum(coefficients[step].view(1, -1, 1) * self.candidates(z), dim=1)
        return z

    def hard(self, x, route):
        return self.forward(x, torch.nn.functional.one_hot(torch.tensor(route), self.slots).to(x.dtype))

    def all_route_support_mse(self, x, y):
        """Support MSE of every hard route, flattened lexicographically (i, j, k)."""
        with torch.no_grad():
            n, d = x.shape
            z = x
            for _ in range(self.steps):
                z = self.candidates(z.reshape(-1, d)).reshape(n, -1, d)  # (n, 12^t, d)
            return torch.mean((z - y.unsqueeze(1)) ** 2, dim=(0, 2))


def query_nmse(library, task, route) -> float:
    with torch.no_grad():
        prediction = library.hard(torch.tensor(task.eval_x, dtype=torch.float32), route)
    return float(nmse(prediction.cpu().numpy(), task.eval_y))


def unflatten(index: int, slots: int, steps: int) -> list[int]:
    route = []
    for _ in range(steps):
        route.append(index % slots)
        index //= slots
    return route[::-1]


def optimize_route(library, x, y, steps: int):
    code = torch.zeros(library.steps, library.slots, requires_grad=True)
    optimizer = torch.optim.Adam([code], lr=OPT_LR)
    initial = final = None
    for step in range(steps):
        temperature = T_START * (T_END / T_START) ** (step / max(1, steps - 1))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((library.forward(x, torch.softmax(code / temperature, dim=-1)) - y) ** 2)
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    return torch.argmax(code.detach(), dim=-1).tolist(), initial, final, float(code.detach().abs().sum())


def run_library(key: str, world_seed: int, library_index: int, tasks_limit=None, opt_steps=OPT_STEPS) -> dict:
    torch.set_num_threads(1)
    cfg = world_config(load_config("configs/v1.yaml"), world_seed)
    world = generate_rotated_world(cfg.world)
    path = SO1_CELLS / key / f"world_{world_seed}"
    stored = json.loads((path / "result.json").read_text())
    for name, sha in stored["artifact_sha256"].items():
        if digest(path / name) != sha:
            raise ValueError(f"SO1 artifact hash mismatch: {path / name}")
    so1 = json.loads(SO1_REPORT.read_text())["cells"][key][str(world_seed)]
    if so1 != stored["result"] or so1["world_sha256"] != world_digest(world):
        raise ValueError("SO1 report, durable cell and regenerated world disagree")
    model = restore_model(path, cfg, world, build_fast)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()
    library = FrozenLibrary(model)
    rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, world_seed, library_index]))
    random_routes = [rng.integers(0, library.slots, size=library.steps).tolist() for _ in world.tasks]
    rows = {}
    tasks = world.tasks if tasks_limit is None else world.tasks[:tasks_limit]
    for index, task in enumerate(tasks):
        x = torch.tensor(task.train_x, dtype=torch.float32)
        y = torch.tensor(task.train_y, dtype=torch.float32)
        oracle = torch.argmax(model.task_codes[task.task_id].detach(), dim=-1).tolist()
        support = library.all_route_support_mse(x, y)
        flat_oracle = (oracle[0] * library.slots + oracle[1]) * library.slots + oracle[2]
        enum_index = int(torch.argmin(support))
        enum = unflatten(enum_index, library.slots, library.steps)
        opt, init_loss, final_loss, code_change = optimize_route(library, x, y, opt_steps)
        rows[task.task_id] = {
            "oracle_route": oracle, "enum_route": enum, "opt_route": opt, "random_route": random_routes[index],
            "oracle": query_nmse(library, task, oracle), "enum": query_nmse(library, task, enum),
            "opt": query_nmse(library, task, opt), "random": query_nmse(library, task, random_routes[index]),
            "k0": query_nmse(library, task, [0] * library.steps),
            "enum_support_mse": float(support[enum_index]), "oracle_support_mse": float(support[flat_oracle]),
            "opt_support_initial": init_loss, "opt_support_final": final_loss, "opt_code_abs_sum": code_change,
        }
    after = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.library.parameters())).hexdigest()
    if before != after:
        raise ValueError("library parameters changed during route inference")
    return {"key": key, "world": world_seed, "library_index": library_index, "library_frozen_sha256": before,
            "so1_terminal_median": so1["terminal_median"], "so1_final_per_task": so1["final_per_task"],
            "tasks": rows, "opt_steps": opt_steps}


def summarize(record: dict) -> dict:
    rows = record["tasks"].values()
    med = {arm: float(np.median([r[arm] for r in rows])) for arm in ("oracle", "enum", "opt", "random", "k0")}
    drops = [(r["opt_support_initial"] - r["opt_support_final"]) / r["opt_support_initial"] for r in rows]
    return {
        "medians": med, "passes": {arm: med[arm] <= THRESHOLD for arm in med},
        "eligible": med["oracle"] <= THRESHOLD,
        "oracle_bitwise": all(r["oracle"] == record["so1_final_per_task"][t] for t, r in record["tasks"].items()),
        "enum_equals_oracle_route": float(np.mean([r["enum_route"] == r["oracle_route"] for r in rows])),
        "opt_equals_oracle_route": float(np.mean([r["opt_route"] == r["oracle_route"] for r in rows])),
        "enum_support_le_oracle": all(r["enum_support_mse"] <= r["oracle_support_mse"] for r in rows),
        "opt_median_support_drop": float(np.median(drops)),
        "opt_codes_moved": all(r["opt_code_abs_sum"] > 0 for r in rows),
        "k0_differs_from_opt": med["k0"] != med["opt"],
    }


def classify(summaries: dict) -> str:
    eligible = [s for s in summaries.values() if s["eligible"]]
    vacuous = any(not (s["oracle_bitwise"] and s["enum_support_le_oracle"] and s["opt_codes_moved"]
                       and s["opt_median_support_drop"] >= MIN_SUPPORT_DROP and s["k0_differs_from_opt"])
                  for s in summaries.values())
    if vacuous or len(eligible) < 2 or any(s["passes"]["random"] for s in eligible):
        return "HARNESS_FAILED"
    if all(s["passes"]["enum"] and s["passes"]["opt"] for s in eligible):
        return "ROUTES_RECOVERABLE"
    if all(s["passes"]["enum"] for s in eligible):
        return "SEARCH_ONLY"
    return "NOT_IDENTIFIABLE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="world 1, first library, 2 tasks, 20 OPT steps")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        started = time.perf_counter()
        record = run_library(LIBRARIES[0], 1, 0, tasks_limit=2, opt_steps=20)
        for task_id, r in record["tasks"].items():
            print(task_id, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
            print("  oracle bitwise:", r["oracle"] == record["so1_final_per_task"][task_id])
        print(f"{time.perf_counter() - started:.1f}s")
        return 0

    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    require_clean_code(OUTPUT)
    config = load_config("configs/v1.yaml")
    expected = protocol(config)
    sha = fingerprint(expected)
    run_log, status_path = ROOT / "run.log", ROOT / "status.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    with writer_lock(ROOT / "launcher.lock"):
        if OUTPUT.exists():
            out = json.loads(OUTPUT.read_text())
            if out.get("protocol_sha256") != sha or out.get("git_commit") != git_commit():
                raise SystemExit("existing report has a different protocol or commit; preserve it first")
            if out.get("complete"):
                log_line(run_log, "report already complete")
                return 0
            log_line(run_log, f"RESUME at {git_commit()}")
        else:
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": git_commit(), "protocol": expected,
                   "protocol_sha256": sha, "started_utc": now(), "cells": {}, "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID} at {git_commit()} pid {os.getpid()}")
        atomic_json(OUTPUT, out)
        jobs = [(key, w, i) for i, key in enumerate(LIBRARIES) for w in WORLDS]
        try:
            for done, (key, w, i) in enumerate(jobs):
                name = f"{key}_w{w}"
                path = ROOT / "cells" / name / "result.json"
                stamp = {"key": key, "world": w, "git_commit": git_commit(), "protocol_sha256": sha}
                atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": git_commit(),
                                          "cells_done": done, "cells_total": len(jobs), "current": name,
                                          "updated_utc": now()})
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["stamp"] != stamp or stored["record_sha256"] != fingerprint(stored["record"]):
                        raise SystemExit(f"durable cell mismatch: {path}")
                    record = stored["record"]
                    log_line(run_log, f"[{name}] reused validated durable cell")
                else:
                    started = time.perf_counter()
                    log_line(run_log, f"[{name}] start")
                    record = run_library(key, w, i)
                    record["seconds"] = round(time.perf_counter() - started, 1)
                    atomic_json(path, {"stamp": stamp, "record": record, "record_sha256": fingerprint(record),
                                       "finished_utc": now()})
                summary = summarize(record)
                out["cells"][name] = summary
                atomic_json(OUTPUT, out)
                m = summary["medians"]
                log_line(run_log, f"[{name}] saved: oracle {m['oracle']:.4f} enum {m['enum']:.4f} "
                                  f"opt {m['opt']:.4f} random {m['random']:.4f} eligible {summary['eligible']} "
                                  f"({record.get('seconds')}s)")
            out["classification"] = classify(out["cells"])
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
