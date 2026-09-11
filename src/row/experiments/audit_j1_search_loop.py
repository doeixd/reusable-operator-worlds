"""J1: search-in-the-loop co-formation of library and routes (CF4).

Frozen in `J1_SEARCH_IN_THE_LOOP_PLAN.md` (7e7ff1b). The training loop is
Stage D's `offline_cell` with one hook: every `RESEARCH` updates (and before
update 1) each task's route is re-chosen by exhaustive support-only search on
the CURRENT library (J1) or re-chosen and then permuted across tasks (SHAM),
and pinned hard. With the hook off and oracle routes pinned it must reproduce
SO1's `O_b2_g131072` bitwise (checked at update 8,192 on every world).

Restartable: one durable hashed record per cell; relaunch resumes; timestamped
run.log and atomic status.json. Teacher programs are used only by the
equivalence gate (oracle routes) and the analysis-only ARI.
"""
from __future__ import annotations

import argparse
import json
import math
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
from row.experiments.audit_rotated_g5r_interference import (
    THRESHOLD, _assignment, _flat_shared, _relative_change, oracle_routes, pin_code,
    pinned_is_one_hot, score, world_config,
)
from row.experiments.audit_so1_budget_bracket import build_fast, checkpoints_for
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.learned_lifetime import _shared_optimizer, _training_values
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, restore_model, save_model, writer_lock,
)
from row.rotated_world import generate_rotated_world

PLAN = Path("J1_SEARCH_IN_THE_LOOP_PLAN.md")
SO1_REPORT = Path("reports/so1_budget_bracket_r2.json")
OUTPUT = Path("reports/j1_search_loop.json")
ROOT = Path("artifacts/j1_search_loop")
PROTOCOL_ID = "J1-search-in-the-loop-v1"
WORLDS = (0, 1, 2)
BATCH = 2
UPDATES = 65536
STREAM = 103  # SO1 ORACLE_CELL_INDEX[(2, 131072)]
RESEARCH = 1024
EQ_STOP = 8192
SHAM_SEED = 1706
ORACLE_KEY, LEARNED_KEY = "O_b2_g131072", "L_b2_g131072"


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "worlds": list(WORLDS), "batch": BATCH,
            "updates": UPDATES, "stream": STREAM, "research_every": RESEARCH, "eq_stop": EQ_STOP,
            "sham_seed": SHAM_SEED, "threshold": THRESHOLD, "implementation": "batched_rotation_v1",
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, SO1_REPORT)}, "environment": environment()}


def search_routes(model, world) -> tuple[dict, dict]:
    library = FrozenLibrary(model)
    routes, check = {}, {}
    for task in world.tasks:
        support = library.all_route_support_mse(torch.tensor(task.train_x, dtype=torch.float32),
                                                torch.tensor(task.train_y, dtype=torch.float32))
        index = int(torch.argmin(support))
        routes[task.task_id] = tuple(unflatten(index, library.slots, library.steps))
        check[task.task_id] = bool(support[index] <= support.min())
    return routes, check


def permuted(routes: dict, world_seed: int, round_index: int) -> dict:
    ids = list(routes)
    order = np.random.default_rng(np.random.SeedSequence([SHAM_SEED, world_seed, round_index])).permutation(len(ids))
    return {ids[i]: routes[ids[j]] for i, j in enumerate(order)}


def ari(labels_a, labels_b) -> float:
    """Adjusted Rand index (analysis only)."""
    n = len(labels_a)
    pairs = lambda k: k * (k - 1) / 2
    table = {}
    for a, b in zip(labels_a, labels_b):
        table[(a, b)] = table.get((a, b), 0) + 1
    rows, cols = {}, {}
    for (a, b), k in table.items():
        rows[a] = rows.get(a, 0) + k
        cols[b] = cols.get(b, 0) + k
    index = sum(pairs(k) for k in table.values())
    expected = sum(pairs(k) for k in rows.values()) * sum(pairs(k) for k in cols.values()) / pairs(n)
    maximum = (sum(pairs(k) for k in rows.values()) + sum(pairs(k) for k in cols.values())) / 2
    return 1.0 if maximum == expected else float((index - expected) / (maximum - expected))


def train_cell(world_seed: int, mode: str, stop_at: int | None = None, updates: int = UPDATES,
               research: int = RESEARCH, artifact: Path | None = None) -> dict:
    """mode: 'oracle' (hook off, oracle routes), 'j1' (search), 'sham' (permuted search)."""
    torch.set_num_threads(1)
    cfg = world_config(load_config("configs/v1.yaml"), world_seed)
    world = generate_rotated_world(cfg.world)
    global_lr, task_lr, weight_decay, _, _, _, _ = _training_values(cfg, "rotated_discrete")
    model = build_fast(cfg)
    search_seconds, rounds = 0.0, []

    def choose(round_index: int) -> dict:
        nonlocal search_seconds
        started = time.perf_counter()
        found, check = search_routes(model, world)
        search_seconds += time.perf_counter() - started
        chosen = found if mode == "j1" else permuted(found, world_seed, round_index)
        rounds.append({"round": round_index, "argmin_ok": all(check.values()),
                       "slots_used": len({s for r in chosen.values() for s in r})})
        return chosen

    routes = oracle_routes(world, _assignment(cfg, world)) if mode == "oracle" else choose(0)
    codes = {}
    for task in world.tasks:
        codes[task.task_id] = model.begin_task(task.task_id)
        pin_code(codes[task.task_id], routes[task.task_id])
    optimizer = _shared_optimizer(model, global_lr, weight_decay)
    initial_shared = _flat_shared(model)
    all_x = torch.tensor(np.concatenate([t.train_x for t in world.tasks]), dtype=torch.float32)
    all_y = torch.tensor(np.concatenate([t.train_y for t in world.tasks]), dtype=torch.float32)
    all_ids = [t.task_id for t in world.tasks for _ in range(len(t.train_x))]
    rng = np.random.default_rng(np.random.SeedSequence([1702, cfg.world.seed, STREAM]))
    last = stop_at or updates
    wanted = {c for c in checkpoints_for(updates) if c <= last} | {last}
    checkpoints = {"0": score(model, world)}
    started = time.perf_counter()
    for update in range(1, last + 1):
        if mode != "oracle" and update > 1 and (update - 1) % research == 0:
            new = choose((update - 1) // research)
            rounds[-1]["changed"] = float(np.mean([new[t] != routes[t] for t in routes]))
            routes = new
            for task_id, code in codes.items():
                pin_code(code, routes[task_id])
        model.set_training_progress((update - 1) / max(1, updates - 1))
        model.train()
        indices_np = rng.integers(0, len(all_x), size=BATCH)
        indices = torch.tensor(indices_np, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_tasks(all_x.index_select(0, indices), [all_ids[int(i)] for i in indices_np])
        loss = torch.nn.functional.mse_loss(prediction, all_y.index_select(0, indices))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        optimizer.step()
        if update in wanted:
            checkpoints[str(update)] = score(model, world)
    final = checkpoints[str(last)]
    result = {
        "mode": mode, "world": world_seed, "updates": last,
        "trajectory": {k: v["median"] for k, v in checkpoints.items()},
        "checkpoint_per_task": {k: v["per_task"] for k, v in checkpoints.items()},
        "terminal_median": final["median"], "final_per_task": final["per_task"],
        "shared_relative_change": _relative_change(initial_shared, _flat_shared(model)),
        "pinned_one_hot": all(pinned_is_one_hot(c, 1.0) for c in codes.values()),
        "finite": all(math.isfinite(v) for v in final["per_task"].values()),
        "routes": {t: list(r) for t, r in routes.items()}, "rounds": rounds,
        "search_seconds": round(search_seconds, 1), "train_seconds": round(time.perf_counter() - started, 1),
    }
    if mode != "oracle":
        teacher = {t.task_id: [int(p) for p in t.program.primitive_ids] for t in world.tasks}
        ids = list(routes)
        result["ari_by_position"] = [ari([routes[t][p] for t in ids], [teacher[t][p] for t in ids])
                                     for p in range(model.task_steps)]
    if artifact is not None:
        save_model(artifact, model, cfg)
        restored = restore_model(artifact, cfg, world, build_fast)
        result["reload_exact"] = score(restored, world)["per_task"] == final["per_task"]
    return result


def classify(cells: dict, so1: dict, gate_ok: bool) -> str:
    expected = {f"{a}_w{w}" for a in ("J1", "SHAM") for w in WORLDS}
    if not gate_ok or not expected <= set(cells):
        return "HARNESS_FAILED"
    for name in expected:
        c = cells[name]
        if not (c["finite"] and c["pinned_one_hot"] and c["shared_relative_change"] > 0
                and c.get("reload_exact") and all(r["argmin_ok"] for r in c["rounds"])):
            return "HARNESS_FAILED"
    j1 = {w: cells[f"J1_w{w}"]["terminal_median"] for w in WORLDS}
    sham = {w: cells[f"SHAM_w{w}"]["terminal_median"] for w in WORLDS}
    learned = {w: so1[LEARNED_KEY][str(w)]["terminal_median"] for w in WORLDS}
    passing = [w for w in WORLDS if j1[w] <= THRESHOLD]
    if len(passing) >= 2 and all(j1[w] < sham[w] and j1[w] < learned[w] for w in passing):
        return "J1_ACQUIRES"
    if sum(j1[w] <= 0.5 * min(sham[w], learned[w]) for w in WORLDS) >= 2:
        return "J1_IMPROVES"
    return "J1_FAILS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="world 1, 40 updates, re-search every 16")
    args = parser.parse_args()
    if args.dry_run:
        for mode in ("oracle", "j1", "sham"):
            t = time.perf_counter()
            r = train_cell(1, mode, updates=40, research=16)
            print(mode, "median", round(r["terminal_median"], 4), "rounds", len(r["rounds"]),
                  "changed", [x.get("changed") for x in r["rounds"]], "slots", [x["slots_used"] for x in r["rounds"]],
                  "argmin_ok", all(x["argmin_ok"] for x in r["rounds"]), "search_s", r["search_seconds"],
                  "ari", r.get("ari_by_position"), f"{time.perf_counter() - t:.1f}s")
        return 0

    for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
        if subprocess.run([sys.executable, tool]).returncode != 0:
            raise SystemExit(f"{tool} failed")
    require_clean_code(OUTPUT)
    expected = protocol()
    sha = fingerprint(expected)
    so1 = json.loads(SO1_REPORT.read_text())["cells"]
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
                   "protocol_sha256": sha, "started_utc": now(), "cells": {}, "equivalence": {}, "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID} at {git_commit()} pid {os.getpid()}")
        atomic_json(OUTPUT, out)
        jobs = [("EQ", w) for w in WORLDS] + [("J1", w) for w in (1, 2, 0)] + [("SHAM", w) for w in (1, 2, 0)]
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
                    for fname, fsha in stored.get("artifact_sha256", {}).items():
                        if digest(cell_dir / fname) != fsha:
                            raise SystemExit(f"artifact hash mismatch: {cell_dir / fname}")
                    result = stored["result"]
                    log_line(run_log, f"[{name}] reused validated durable cell")
                else:
                    log_line(run_log, f"[{name}] start")
                    if arm == "EQ":
                        result = train_cell(w, "oracle", stop_at=EQ_STOP)
                    else:
                        result = train_cell(w, arm.lower(), artifact=cell_dir)
                    record = {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                              "finished_utc": now()}
                    if arm != "EQ":
                        record["artifact_sha256"] = {f: digest(cell_dir / f) for f in
                                                     ("model.pt", "model_state.json", "config.yaml")}
                    atomic_json(path, record)
                if arm == "EQ":
                    reference = so1[ORACLE_KEY][str(w)]["checkpoints"][str(EQ_STOP)]["per_task"]
                    ok = result["checkpoint_per_task"][str(EQ_STOP)] == reference
                    out["equivalence"][str(w)] = ok
                    log_line(run_log, f"[{name}] equivalence at {EQ_STOP}: bitwise {ok}")
                    if not ok:
                        raise SystemExit("equivalence gate failed: J1 loop does not reproduce SO1 oracle cell")
                else:
                    out["cells"][name] = {k: v for k, v in result.items() if k not in ("checkpoint_per_task",)}
                    log_line(run_log, f"[{name}] saved: median {result['terminal_median']:.4f} trajectory "
                                      f"{ {k: round(v, 4) for k, v in result['trajectory'].items()} } "
                                      f"slots {result['rounds'][-1]['slots_used']} ari {[round(a, 2) for a in result['ari_by_position']]} "
                                      f"search {result['search_seconds']}s train {result['train_seconds']}s")
                atomic_json(OUTPUT, out)
            out["references"] = {k: {str(w): so1[k][str(w)]["terminal_median"] for w in WORLDS}
                                 for k in (ORACLE_KEY, LEARNED_KEY)}
            out["classification"] = classify(out["cells"], so1, all(out["equivalence"].values()))
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
