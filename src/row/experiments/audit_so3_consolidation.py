"""SO3: stage-3 consolidation settings on fresh development worlds (Tier 2).

Frozen in `SO3_STAGE3_CONSOLIDATION_PLAN.md` (462e5cd; protected eaf2bd6).
SO2's online staged protocol through all three stages on worlds 3-5, model seed
6000, three replay streams per arm. Stages 1-2 depend only on (world, stream)
and are computed once per pair (a "prefix"), then fanned out to the three
stage-3 arms. Gates G0 (additive replay_seed is bitwise-neutral) and G1 (prefix
sharing equals recomputation) run before any scored cell.

Restartable: durable stamped, hashed records per gate, prefix and cell; relaunch
resumes; timestamped run.log; atomic status.json listing only cells actually
running; exit.json.
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import ANCHOR_TOLERANCE, _last_task_end_of_task, score
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so2_online_gate import carry_library, export_diagnostic
from row.experiments.audit_so2p_plasticity import changed_fields
from row.experiments.census_so2_interference import slot_drift, spearman, task_summaries
from row.experiments.learned_lifetime import run
from row.experiments.score_so2_online_gate import restore_stage
from row.experiments.so1_storage import atomic_json, digest, environment, fingerprint, log_line, now, save_model, writer_lock

PLAN = Path("SO3_STAGE3_CONSOLIDATION_PLAN.md")
OUTPUT = Path("reports/so3_consolidation.json")
ROOT = Path("artifacts/so3_consolidation")
SO2_CELL = Path("artifacts/so2_online_gate/cells/STAGED_w1")
PROTOCOL_ID = "SO3-stage3-consolidation-v1"
KIND = "rotated_discrete_fast"
WORLDS = (3, 4, 5)
STREAMS = (0, 1, 2)
MODEL_SEED = 6000
SO2_MODEL_SEED = 5000
THRESHOLD = 0.05
TOL = 1e-6
WORKERS = 3
STREAM_ROOT = 7300
ARMS = {
    "BASE": None,
    "LR_HALF": ("global_learning_rate", 0.0005),
    "STORE_8": ("replay_examples_per_task", 8),
}
G1_CELL = (3, 0, "BASE")


def replay_seed_for(world: int, stream: int) -> int | None:
    """Stream 0 is the canonical `seed + 1` (passed as None); streams 1-2 are registered."""
    if stream == 0:
        return None
    return int(np.random.SeedSequence([STREAM_ROOT, world, stream]).generate_state(1)[0])


def arm_config(cfg, arm: str):
    change = ARMS[arm]
    return cfg if change is None else replace(cfg, discrete_model=replace(cfg.discrete_model, **{change[0]: change[1]}))


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "kind": KIND, "worlds": list(WORLDS),
            "streams": list(STREAMS), "model_seed": MODEL_SEED, "stream_seeds": "stream 0: seed+1 (None); "
            "else SeedSequence([7300, world, stream]).generate_state(1)[0]",
            "arms": {k: list(v) if v else None for k, v in ARMS.items()}, "threshold": THRESHOLD,
            "tolerance": TOL, "g1_cell": list(G1_CELL), "margin": "not computed (plan)",
            "input_sha256": {PLAN.as_posix(): digest(PLAN)}, "environment": environment()}


def scaled(cfg, scale: int):
    if scale == 1:
        return cfg
    return replace(cfg, world=replace(cfg.world, tasks=max(2, cfg.world.tasks // scale),
                                      examples_per_task=max(4, cfg.world.examples_per_task // scale),
                                      evaluation_examples=max(4, cfg.world.evaluation_examples // scale)))


def lifetime(cfg, world, output: Path, model, replay_seed, scale: int):
    ran = scaled(cfg, scale)
    summary = run(replace(ran, output_directory=output), KIND, world=world, model=model,
                  return_model=True, replay_seed=replay_seed)
    return summary, summary.pop("terminal_model"), ran


def stage_record(summary, model, world, output: Path, carried) -> dict:
    trained = SimpleNamespace(tasks=[t for t in world.tasks if t.task_id in model.task_codes])
    terminal = score(model, trained)
    last_id, last_end, end_median = _last_task_end_of_task(output)
    return {"terminal_median": terminal["median"], "terminal_below": terminal["below_0.05"],
            "terminal_per_task": terminal["per_task"], "end_of_task_median": end_median,
            "anchor_task_id": last_id, "anchor_abs_error": abs(terminal["per_task"][last_id] - last_end),
            "novel_probe_available": summary.get("novel_composition", {}).get("available", True),
            "prequential": summary.get("cumulative_prequential_gaussian_log_loss"),
            "library_sha256": library_sha(model), "library_sha256_at_start": carried,
            "task_ids": [t.task_id for t in trained.tasks]}


def build_prefix(world_seed: int, stream: int, directory: Path | None, scale: int = 1, root: Path = ROOT):
    """Stages 1-2 for one (world, stream). Returns (records, stage-2 model).
    `directory=None` computes in memory (G1), writing lifetime outputs under `root`/scratch."""
    torch.set_num_threads(1)
    rs = replay_seed_for(world_seed, stream)
    model, records = None, {}
    for stage in (1, 2):
        cfg, world, _, _ = stage_setup(world_seed, stage, MODEL_SEED)
        carried = None
        if model is not None:
            model = carry_library(model, cfg)
            carried = library_sha(model)
        output = (directory or root / "scratch" / f"prefix_w{world_seed}_s{stream}") / f"stage{stage}"
        summary, model, ran = lifetime(cfg, world, output, model, rs, scale)
        records[str(stage)] = stage_record(summary, model, world, output, carried)
        if directory is not None:
            save_model(output, model, ran)
    return records, model


def stage3_cell(world_seed: int, stream: int, arm: str, stage2_model, stage2_record: dict,
                output: Path, save: bool, scale: int = 1) -> dict:
    torch.set_num_threads(1)
    started = time.perf_counter()
    rs = replay_seed_for(world_seed, stream)
    base_cfg, world, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    cfg = arm_config(base_cfg, arm)
    if library_sha(stage2_model) != stage2_record["library_sha256"]:
        raise RuntimeError("stage-2 library does not match its prefix record")
    model = carry_library(stage2_model, cfg)
    carried = library_sha(model)
    summary, model, ran = lifetime(cfg, world, output, model, rs, scale)
    record = stage_record(summary, model, world, output, carried)
    rows = task_summaries(output)
    ids = [r["task_id"] for r in rows]
    end = [r["final_nmse"] for r in rows]
    term = [record["terminal_per_task"][i] for i in ids]
    ratio = [math.log10(max(t, 1e-12) / max(e, 1e-12)) for t, e in zip(term, end)]
    if save:
        save_model(output, model, ran)
    return record | {
        "world": world_seed, "stream": stream, "arm": arm, "replay_seed": rs, "scale": scale,
        "changed_fields": changed_fields(base_cfg, cfg),
        "training": {k: getattr(ran.discrete_model, k) for k in (
            "global_learning_rate", "task_learning_rate", "updates_per_example",
            "replay_examples_per_task", "replay_ratio", "weight_decay", "seed")},
        "end_of_task_below": int(sum(e <= THRESHOLD for e in end)),
        "lost_threshold": int(sum(e <= THRESHOLD < t for t, e in zip(term, end))),
        "gained_threshold": int(sum(t <= THRESHOLD < e for t, e in zip(term, end))),
        "spearman_position_vs_log_ratio": spearman(list(range(len(ids))), ratio) if len(ids) > 2 else None,
        "drift_from_stage2": slot_drift(stage2_model, model, world_seed, cfg.world.state_dim),
        "online_examples": ran.world.tasks * ran.world.examples_per_task,
        "export_diagnostic": export_diagnostic(cfg, world, model) if scale == 1 else None,
        "seconds": round(time.perf_counter() - started, 1),
    }


# ----------------------------------------------------------------- pool jobs


def job_g0(variant: str, root: Path) -> dict:
    """SO2 world-1 stage 3 from its saved stage-2 library, replay_seed omitted or explicit."""
    torch.set_num_threads(1)
    so2 = json.loads((SO2_CELL / "result.json").read_text())["result"]
    cfg2, world2, _, _ = stage_setup(1, 2, SO2_MODEL_SEED)
    stage2 = restore_stage(SO2_CELL / "stage2", cfg2, world2, so2["stages"]["2"]["novel_probe_available"])
    cfg3, world3, _, _ = stage_setup(1, 3, SO2_MODEL_SEED)
    model = carry_library(stage2, cfg3)
    output = root / "gates" / f"G0_{variant}"
    rs = None if variant == "omitted" else cfg3.discrete_model.seed + 1
    summary = run(replace(cfg3, output_directory=output), KIND, world=world3, model=model,
                  return_model=True, replay_seed=rs)
    terminal_model = summary.pop("terminal_model")
    trained = SimpleNamespace(tasks=[t for t in world3.tasks if t.task_id in terminal_model.task_codes])
    per_task = score(terminal_model, trained)["per_task"]
    rec3 = so2["stages"]["3"]
    worst = max(abs(per_task[k] - v) for k, v in rec3["terminal_per_task"].items())
    return {"variant": variant, "replay_seed": rs, "library_sha256": library_sha(terminal_model),
            "sha_matches_so2": library_sha(terminal_model) == rec3["library_sha256"],
            "worst_per_task_abs_diff": worst, "passes": library_sha(terminal_model) == rec3["library_sha256"]
            and worst <= TOL}


def job_prefix(world_seed: int, stream: int, root: Path, scale: int) -> dict:
    directory = root / "prefixes" / f"w{world_seed}_s{stream}"
    records, _ = build_prefix(world_seed, stream, directory, scale)
    return {"world": world_seed, "stream": stream, "stages": records}


def restore_prefix_stage2(path: Path, cfg, record: dict):
    """Rebuild a saved stage-2 model with exactly the task codes it trained (plus the probe
    code where the probe ran), so a scaled prefix restores as strictly as a full one."""
    model = build_fast(cfg)
    for task_id in record["task_ids"]:
        model.begin_task(task_id)
    if record["novel_probe_available"]:
        model.begin_task("task_novel_composition_0")
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True), strict=True)
    state = json.loads((path / "model_state.json").read_text())
    if set(state["requires_grad"]) != set(dict(model.named_parameters())):
        raise ValueError(f"{path}: incomplete requires_grad state")
    model.temperature = state["temperature"]
    return model


def job_cell(world_seed: int, stream: int, arm: str, prefix: dict, root: Path, scale: int) -> dict:
    directory = root / "prefixes" / f"w{world_seed}_s{stream}" / "stage2"
    cfg2, _, _, _ = stage_setup(world_seed, 2, MODEL_SEED)
    stage2 = restore_prefix_stage2(directory, cfg2, prefix["stages"]["2"])
    output = root / "cells" / f"w{world_seed}_s{stream}_{arm}" / "stage3"
    return stage3_cell(world_seed, stream, arm, stage2, prefix["stages"]["2"], output, True, scale)


def job_g1(root: Path, scale: int) -> dict:
    """G1: the G1 cell with its prefix recomputed in memory (no save/restore)."""
    w, s, arm = G1_CELL
    records, stage2 = build_prefix(w, s, None, scale, root=root)
    output = root / "gates" / "G1_fresh" / "stage3"
    cell = stage3_cell(w, s, arm, stage2, records["2"], output, False, scale)
    return {"prefix_stage2_sha256": records["2"]["library_sha256"], "library_sha256": cell["library_sha256"],
            "terminal_per_task": cell["terminal_per_task"]}


# ------------------------------------------------------------ classification


def gates(out: dict) -> dict:
    g0 = all(out["gates"].get(f"G0_{v}", {}).get("passes") is True for v in ("omitted", "explicit"))
    fresh = out["gates"].get("G1")
    shared = out["cells"].get("w{}_s{}_{}".format(*G1_CELL))
    g1 = bool(fresh and shared and fresh["library_sha256"] == shared["library_sha256"] and
              fresh["prefix_stage2_sha256"] == out["prefixes"]["w{}_s{}".format(*G1_CELL[:2])]["stages"]["2"]["library_sha256"]
              and all(abs(fresh["terminal_per_task"][k] - v) <= TOL for k, v in shared["terminal_per_task"].items()))
    cells = out["cells"]
    g2 = all(len({cells[f"w{w}_s{s}_{a}"]["library_sha256"] for a in ARMS}) == len(ARMS)
             for w in WORLDS for s in STREAMS) and \
         all(len({cells[f"w{w}_s{s}_{a}"]["library_sha256"] for s in STREAMS}) == len(STREAMS)
             for w in WORLDS for a in ARMS)
    stage_records = [st for p in out["prefixes"].values() for st in p["stages"].values()] + list(cells.values())
    g3 = all(r["anchor_abs_error"] <= ANCHOR_TOLERANCE for r in stage_records)
    g4 = True
    for (w, s), p in ((tuple(int(x[1:]) for x in k.split("_")), v) for k, v in out["prefixes"].items()):
        g4 &= p["stages"]["2"]["library_sha256_at_start"] == p["stages"]["1"]["library_sha256"]
        g4 &= not set(p["stages"]["1"]["task_ids"]) & set(p["stages"]["2"]["task_ids"])
        for a in ARMS:
            c = cells[f"w{w}_s{s}_{a}"]
            g4 &= c["library_sha256_at_start"] == p["stages"]["2"]["library_sha256"]
            g4 &= not (set(c["task_ids"]) & (set(p["stages"]["1"]["task_ids"]) | set(p["stages"]["2"]["task_ids"])))
    return {"G0_replay_seed_neutral": bool(g0), "G1_prefix_sharing": bool(g1), "G2_non_vacuity": bool(g2),
            "G3_anchor": bool(g3), "G4_transfer": bool(g4)}


def classify(medians: dict, gate: dict) -> dict:
    """`medians[arm][world][stream]` = terminal median M. The frozen ladder; returns labels."""
    complete = all(math.isfinite(medians.get(a, {}).get(w, {}).get(s, float("nan")))
                   for a in ARMS for w in WORLDS for s in STREAMS)
    if not complete or not all(gate.values()):
        return {"program": "HARNESS_FAILED"}
    wm = {a: {w: float(np.median([medians[a][w][s] for s in STREAMS])) for w in WORLDS} for a in ARMS}
    labels = {}
    for arm in ("LR_HALF", "STORE_8"):
        passes = sum(wm[arm][w] <= THRESHOLD and wm[arm][w] < wm["BASE"][w] for w in WORLDS) >= 2
        partial = sum(wm[arm][w] <= 0.5 * wm["BASE"][w] for w in WORLDS) >= 2
        if arm == "STORE_8":
            robust = sum(wm[arm][w] < min(medians["BASE"][w][s] for s in STREAMS) for w in WORLDS) >= 2
            labels["STORE_8_STREAM_ROBUST"] = robust
            if passes:
                labels[arm] = "STORE_8_PASSES" if robust else "STORE_8_STREAM_CONFOUNDED"
                continue
        labels[arm] = f"{arm}_PASSES" if passes else f"{arm}_PARTIAL" if partial else f"{arm}_FAILS"
    real = labels["LR_HALF"] == "LR_HALF_PASSES" or labels["STORE_8"] == "STORE_8_PASSES"
    soft = labels["LR_HALF"] == "LR_HALF_PARTIAL" or labels["STORE_8"] in ("STORE_8_PARTIAL",
                                                                           "STORE_8_STREAM_CONFOUNDED")
    labels["program"] = "SO3_PASSES" if real else "SO3_PARTIAL" if soft else "SO3_FAILS"
    labels["world_medians"] = wm
    labels["base_world_passes"] = int(sum(wm["BASE"][w] <= THRESHOLD for w in WORLDS))
    return labels


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="G0 at full scale, then one prefix and one cell at scale 16; no durable record")
    parser.add_argument("--smoke", action="store_true",
                        help="restart test: world 3, streams 0-1, all arms, scale 16, separate root; never a result")
    args = parser.parse_args()
    if args.dry_run:
        root = ROOT.parent / "so3_consolidation_dry"
        for variant in ("omitted", "explicit"):
            started = time.perf_counter()
            print("G0", job_g0(variant, root) | {"seconds": round(time.perf_counter() - started, 1)}, flush=True)
        started = time.perf_counter()
        prefix = job_prefix(3, 1, root, 16)
        cell = job_cell(3, 1, "STORE_8", prefix, root, 16)
        print("scaled path ok:", cell["changed_fields"], cell["replay_seed"], f"{time.perf_counter() - started:.1f}s")
        return 0

    root, output, scale, worlds, streams = ROOT, OUTPUT, 1, WORLDS, STREAMS
    if args.smoke:
        root, scale, worlds, streams = Path("artifacts/so3_consolidation_smoke"), 16, (3,), (0, 1)
        output = root / "report.json"
    else:
        for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
            if subprocess.run([sys.executable, tool]).returncode != 0:
                raise SystemExit(f"{tool} failed")
        require_clean_code(OUTPUT)
    expected = protocol() | ({"smoke": {"scale": scale, "worlds": list(worlds), "streams": list(streams)}}
                             if args.smoke else {})
    sha = fingerprint(expected)
    commit = git_commit()
    root.mkdir(parents=True, exist_ok=True)
    run_log, status_path = root / "run.log", root / "status.json"
    atomic_json(root / "run.pid", {"pid": os.getpid(), "started_utc": now()})
    with writer_lock(root / "launcher.lock"):
        if output.exists():
            out = json.loads(output.read_text())
            if out.get("protocol_sha256") != sha or out.get("git_commit") != commit:
                raise SystemExit("existing report has a different protocol or commit; preserve it first")
            if out.get("complete"):
                return 0
            log_line(run_log, f"RESUME at {commit}")
        else:
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": commit, "protocol": expected,
                   "protocol_sha256": sha, "started_utc": now(), "gates": {}, "prefixes": {}, "cells": {},
                   "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID}{' SMOKE' if args.smoke else ''} at {commit} pid {os.getpid()}")
        atomic_json(output, out)

        def durable(kind: str, name: str) -> tuple[Path, dict]:
            return root / "records" / kind / f"{name}.json", {"kind": kind, "name": name, "git_commit": commit,
                                                               "protocol_sha256": sha}

        def reuse(kind: str, name: str):
            path, stamp = durable(kind, name)
            if not path.exists():
                return None
            stored = json.loads(path.read_text())
            if stored["stamp"] != stamp or stored["result_sha256"] != fingerprint(stored["result"]):
                raise SystemExit(f"durable record mismatch: {path}")
            log_line(run_log, f"[{kind}/{name}] reused validated durable record")
            return stored["result"]

        def store(kind: str, name: str, result: dict) -> None:
            path, stamp = durable(kind, name)
            atomic_json(path, {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                               "finished_utc": now()})
            {"gate": out["gates"], "prefix": out["prefixes"], "cell": out["cells"]}[kind][name] = result
            atomic_json(output, out)

        started_at = time.time()
        # G0 (2, not in smoke) + one prefix and len(ARMS) cells per (world, stream) + G1 (1).
        total = (0 if args.smoke else 2) + len(worlds) * len(streams) * (1 + len(ARMS)) + 1

        def status(running: dict) -> None:
            done = len(out["gates"]) + len(out["prefixes"]) + len(out["cells"])
            atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": commit,
                                      "started_utc": out["started_utc"], "jobs_done": done, "jobs_total": total,
                                      "running": sorted(running.values()), "elapsed_seconds": round(time.time() - started_at),
                                      "updated_utc": now()})

        def run_phase(pool, jobs: list[tuple[str, str, tuple]]) -> None:
            """Bounded submission: at most WORKERS jobs submitted at once, so `running` is truthful."""
            queue, running = [], {}
            for kind, name, call in jobs:
                cached = reuse(kind, name)
                if cached is None:
                    queue.append((kind, name, call))
                else:
                    {"gate": out["gates"], "prefix": out["prefixes"], "cell": out["cells"]}[kind][name] = cached
            atomic_json(output, out)
            while queue or running:
                while queue and len(running) < WORKERS:
                    kind, name, call = queue.pop(0)
                    running[pool.submit(*call)] = f"{kind}/{name}"
                    log_line(run_log, f"[{kind}/{name}] start")
                status(running)
                done_now, _ = wait(set(running), return_when=FIRST_COMPLETED)
                for future in done_now:
                    label = running.pop(future)
                    kind, name = label.split("/", 1)
                    result = future.result()
                    store(kind, name, result)
                    headline = (f"terminal {result['terminal_median']:.4f} ({result['terminal_below']}/64) "
                                f"lost {result['lost_threshold']} drift {result['drift_from_stage2']['median']:.3f} "
                                f"({result['seconds']}s)") if kind == "cell" else json.dumps(
                        {k: v for k, v in result.items() if k in ("passes", "worst_per_task_abs_diff")}) if kind == "gate" \
                        else f"stage2 terminal {result['stages']['2']['terminal_median']:.4f}"
                    log_line(run_log, f"[{label}] saved: {headline}")
                status(running)

        try:
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                phase_a = [] if args.smoke else [("gate", f"G0_{v}", (job_g0, v, root)) for v in ("omitted", "explicit")]
                phase_a += [("prefix", f"w{w}_s{s}", (job_prefix, w, s, root, scale)) for w in worlds for s in streams]
                run_phase(pool, phase_a)
                if not args.smoke and not all(out["gates"][f"G0_{v}"]["passes"] for v in ("omitted", "explicit")):
                    raise SystemExit("G0 failed: replay_seed is not bitwise-neutral; no scored cell runs")
                phase_b = [("gate", "G1", (job_g1, root, scale))]
                phase_b += [("cell", f"w{w}_s{s}_{a}", (job_cell, w, s, a, out["prefixes"][f"w{w}_s{s}"], root, scale))
                            for w in worlds for s in streams for a in ARMS]
                run_phase(pool, phase_b)
            if args.smoke:
                out["classification"] = {"program": "SMOKE (not a result)"}
            else:
                out["gate_results"] = gates(out)
                medians = {a: {w: {s: out["cells"][f"w{w}_s{s}_{a}"]["terminal_median"] for s in STREAMS}
                               for w in WORLDS} for a in ARMS}
                out["classification"] = classify(medians, out["gate_results"])
            out["complete"], out["finished_utc"] = True, now()
            atomic_json(output, out)
            atomic_json(status_path, {"state": "complete", "classification": out["classification"]["program"],
                                      "gates": out.get("gate_results"), "updated_utc": now()})
            log_line(run_log, f"COMPLETE gates {out.get('gate_results')} classification {out['classification']}")
            atomic_json(root / "exit.json", {"git_commit": commit, "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(root / "exit.json", {"git_commit": commit, "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
