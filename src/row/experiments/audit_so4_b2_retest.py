"""SO4: the B2 online gate re-tested with measured stream variance (Tier 2).

Frozen in `SO4_B2_RETEST_PLAN.md` (eba13b0). SO2's online staged protocol,
unchanged, on development worlds 6-9 at model seed 7000 with three replay
streams per world, a PLAIN control, and G5R's export margin on each world's
pre-specified stream-0 terminal library. SO3's gated machinery is reused: the
generic helpers are imported; the seed/stream-bound functions are re-implemented
here with SO4's constants.

Margins are computed as one pool job per held-out program pair, constructed
exactly as `audit_so2_online_gate.export_margin` constructs them (the dry run
checks that equivalence against `export_margin` itself).

Restartable: durable stamped, hashed records per gate, prefix, cell, plain
lifetime and margin pair; relaunch resumes; run.log, status.json (running jobs
only), exit.json.
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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import audit_so2_online_gate as so2
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_STEPS, scratch_model
from row.experiments.audit_e8_length import adapt_cell
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5 import held_out_programs
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import ANCHOR_TOLERANCE
from row.experiments.audit_so2_online_gate import carry_library, export_diagnostic
from row.experiments.audit_so3_consolidation import job_g0, lifetime, restore_prefix_stage2, stage_record
from row.experiments.census_so2_interference import slot_drift, spearman, task_summaries
from row.experiments.so1_storage import atomic_json, digest, environment, fingerprint, log_line, now, save_model, writer_lock
from row.rotated_world import rotated_library
from row.support_split_world import _build_tasks

PLAN = Path("SO4_B2_RETEST_PLAN.md")
AMENDMENT = Path("SO4_AMENDMENT_1.md")
OUTPUT = Path("reports/so4_b2_retest.json")
ROOT = Path("artifacts/so4_b2_retest")
PROTOCOL_ID = "SO4-b2-retest-v1"
WORLDS = (6, 7, 8, 9)
STREAMS = (0, 1, 2)
MODEL_SEED = 7000
STREAM_ROOT = 7400
THRESHOLD = 0.05
STREAM_CAP = 0.10  # SO4 Amendment 1
MARGIN = 0.75
TOL = 1e-6
WORKERS = 3
HELD_OUT = 12
MARGIN_SEED = 1500
MARGIN_STREAM = 0
G1_CELL = (6, 0)


def replay_seed_for(world: int, stream: int) -> int | None:
    if stream == 0:
        return None
    return int(np.random.SeedSequence([STREAM_ROOT, world, stream]).generate_state(1)[0])


def protocol() -> dict:
    return {"id": PROTOCOL_ID, "frozen_plan": PLAN.as_posix(), "kind": so2.KIND, "worlds": list(WORLDS),
            "streams": list(STREAMS), "model_seed": MODEL_SEED,
            "stream_seeds": "stream 0: seed+1 (None); else SeedSequence([7400, world, stream]).generate_state(1)[0]",
            "threshold": THRESHOLD, "stream_cap": STREAM_CAP, "amendment": AMENDMENT.as_posix(), "margin": MARGIN, "held_out": HELD_OUT, "adapt_steps": ADAPT_STEPS,
            "margin_stream": MARGIN_STREAM, "margin_construction": "G5R verbatim via SO2 export_margin, one job per "
            "program pair", "tolerance": TOL, "g1_cell": list(G1_CELL),
            "input_sha256": {p.as_posix(): digest(p) for p in (PLAN, AMENDMENT)}, "environment": environment()}


# ------------------------------------------------------------------ lifetimes


def build_prefix(world_seed: int, stream: int, directory: Path | None, scale: int, root: Path):
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


def staged_cell(world_seed: int, stream: int, stage2_model, stage2_record: dict, output: Path, save: bool,
                scale: int) -> dict:
    torch.set_num_threads(1)
    started = time.perf_counter()
    rs = replay_seed_for(world_seed, stream)
    cfg, world, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
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
        "world": world_seed, "stream": stream, "replay_seed": rs, "scale": scale,
        "lost_threshold": int(sum(e <= THRESHOLD < t for t, e in zip(term, end))),
        "gained_threshold": int(sum(t <= THRESHOLD < e for t, e in zip(term, end))),
        "spearman_position_vs_log_ratio": spearman(list(range(len(ids))), ratio) if len(ids) > 2 else None,
        "drift_from_stage2": slot_drift(stage2_model, model, world_seed, cfg.world.state_dim),
        "export_diagnostic": export_diagnostic(cfg, world, model) if scale == 1 else None,
        "online_examples": ran.world.tasks * ran.world.examples_per_task,
        "seconds": round(time.perf_counter() - started, 1)}


# ---------------------------------------------------------------- pool jobs


def job_prefix(world_seed: int, stream: int, root: Path, scale: int) -> dict:
    records, _ = build_prefix(world_seed, stream, root / "prefixes" / f"w{world_seed}_s{stream}", scale, root)
    return {"world": world_seed, "stream": stream, "stages": records}


def job_cell(world_seed: int, stream: int, prefix: dict, root: Path, scale: int) -> dict:
    cfg2, _, _, _ = stage_setup(world_seed, 2, MODEL_SEED)
    stage2 = restore_prefix_stage2(root / "prefixes" / f"w{world_seed}_s{stream}" / "stage2", cfg2,
                                   prefix["stages"]["2"])
    output = root / "cells" / f"w{world_seed}_s{stream}" / "stage3"
    return staged_cell(world_seed, stream, stage2, prefix["stages"]["2"], output, True, scale)


def job_g1(root: Path, scale: int) -> dict:
    w, s = G1_CELL
    records, stage2 = build_prefix(w, s, None, scale, root)
    cell = staged_cell(w, s, stage2, records["2"], root / "gates" / "G1_fresh" / "stage3", False, scale)
    return {"prefix_stage2_sha256": records["2"]["library_sha256"], "library_sha256": cell["library_sha256"],
            "terminal_per_task": cell["terminal_per_task"]}


def job_plain(world_seed: int, root: Path, scale: int) -> dict:
    """SO2's non-staged control: one canonical length-3 lifetime from a fresh model, stream 0."""
    torch.set_num_threads(1)
    cfg, world, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    output = root / "plain" / f"w{world_seed}" / "stage3"
    summary, model, ran = lifetime(cfg, world, output, None, replay_seed_for(world_seed, 0), scale)
    record = stage_record(summary, model, world, output, None)
    save_model(output, model, ran)
    return record | {"world": world_seed, "export_diagnostic": export_diagnostic(cfg, world, model)
                     if scale == 1 else None}


def margin_task(cfg, world, world_seed: int, index: int):
    """Program `index` of G5R's held-out draw, built exactly as `so2.export_margin` builds it."""
    rng = np.random.default_rng(np.random.SeedSequence([MARGIN_SEED, world_seed]))
    programs = held_out_programs(cfg, world, rng, HELD_OUT)
    program = programs[index]
    library = rotated_library(cfg.world)
    task = _build_tasks(cfg.world, library, [program], [f"so2_{world_seed}_{index}"],
                        index_offset=95000 + index)[0]
    return program, library, task


def margin_pair(model, cfg, world, world_seed: int, index: int, steps: int) -> dict:
    program, library, task = margin_task(cfg, world, world_seed, index)
    scratch = scratch_model(cfg, "rotated_discrete", 7717)
    trained = adapt_cell(model, task, f"so2L_{world_seed}_{index}", cfg.world.program_length, False, library,
                         program, steps=steps)
    fresh = adapt_cell(scratch, task, f"so2S_{world_seed}_{index}", cfg.world.program_length, True, library,
                       program, steps=steps)
    return {"index": index, "program": list(program), "trained": trained["query_nmse"],
            "scratch": fresh["query_nmse"]}


def job_margin(world_seed: int, index: int, cell_record: dict, root: Path, steps: int) -> dict:
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, world, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    path = root / "cells" / f"w{world_seed}_s{MARGIN_STREAM}" / "stage3"
    model = restore_prefix_stage2(path, cfg, cell_record)  # generic restore: task codes + probe code, strict
    if library_sha(model) != cell_record["library_sha256"]:
        raise RuntimeError("stream-0 terminal library does not match its cell record")
    return margin_pair(model, cfg, world, world_seed, index, steps) | {
        "world": world_seed, "steps": steps, "seconds": round(time.perf_counter() - started, 1)}


# ------------------------------------------------------------ classification


def geo(values) -> float:
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values)))


def margin_from_pairs(pairs: list[dict]) -> float:
    return math.log(geo([p["scratch"] for p in pairs])) - math.log(geo([p["trained"] for p in pairs]))


def gates(out: dict, smoke: bool = False) -> dict:
    worlds = sorted({c["world"] for c in out["cells"].values()})
    g0 = smoke or all(out["gates"].get(f"G0_{v}", {}).get("passes") is True for v in ("omitted", "explicit"))
    fresh = out["gates"].get("G1")
    shared = out["cells"].get("w{}_s{}".format(*G1_CELL))
    g1 = bool(fresh and shared and fresh["library_sha256"] == shared["library_sha256"]
              and fresh["prefix_stage2_sha256"] == out["prefixes"]["w{}_s{}".format(*G1_CELL)]["stages"]["2"]["library_sha256"]
              and all(abs(fresh["terminal_per_task"][k] - v) <= TOL for k, v in shared["terminal_per_task"].items()))
    streams = sorted({c["stream"] for c in out["cells"].values()})
    g2 = all(len({out["cells"][f"w{w}_s{s}"]["library_sha256"] for s in streams}) == len(streams) for w in worlds)
    records = [st for p in out["prefixes"].values() for st in p["stages"].values()] + list(out["cells"].values()) \
        + list(out["plain"].values())
    g3 = all(r["anchor_abs_error"] <= ANCHOR_TOLERANCE for r in records)
    g4 = True
    for key, p in out["prefixes"].items():
        g4 &= p["stages"]["2"]["library_sha256_at_start"] == p["stages"]["1"]["library_sha256"]
        g4 &= not set(p["stages"]["1"]["task_ids"]) & set(p["stages"]["2"]["task_ids"])
        c = out["cells"][key]
        g4 &= c["library_sha256_at_start"] == p["stages"]["2"]["library_sha256"]
        g4 &= not set(c["task_ids"]) & (set(p["stages"]["1"]["task_ids"]) | set(p["stages"]["2"]["task_ids"]))
    return {"G0_replay_seed_neutral": bool(g0), "G1_prefix_sharing": bool(g1), "G2_streams_distinct": bool(g2),
            "G3_anchor": bool(g3), "G4_transfer": bool(g4)}


def classify(medians: dict, margins: dict, gate: dict) -> dict:
    """`medians[world][stream]` = terminal median M; `margins[world]` = G5R margin. The frozen ladder."""
    complete = all(math.isfinite(medians.get(w, {}).get(s, float("nan"))) for w in WORLDS for s in STREAMS) \
        and all(math.isfinite(margins.get(w, float("nan"))) for w in WORLDS)
    if not complete or not all(gate.values()):
        return {"program": "HARNESS_FAILED"}
    W = {w: float(np.median([medians[w][s] for s in STREAMS])) for w in WORLDS}
    terminal_worlds = [w for w in WORLDS if W[w] <= THRESHOLD]
    # Amendment 1: with three streams "median <= t" already implies ">= 2 of 3 <= t", so the
    # original sub-clause could never fail. A terminal-passing world is stream-robust only if no
    # stream exceeds STREAM_CAP (twice the threshold).
    robust = {w: max(medians[w][s] for s in STREAMS) <= STREAM_CAP for w in terminal_worlds}
    terminal = len(terminal_worlds) >= 3
    stream_ok = all(robust.values())
    margin_ok = sum(margins[w] >= MARGIN for w in WORLDS) >= 3
    if terminal and stream_ok and margin_ok:
        label = "SO4_PASSES"
    elif terminal and stream_ok:
        label = "SO4_ACQUIRES_ONLY"
    elif terminal:
        label = "SO4_STREAM_FRAGILE"
    else:
        label = "SO4_FAILS"
    return {"program": label, "world_medians": W, "terminal_worlds": terminal_worlds, "stream_robust": robust,
            "margins": margins, "margin_worlds": int(sum(margins[w] >= MARGIN for w in WORLDS))}


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="G0 at full scale; per-pair margin jobs vs so2.export_margin at 3 steps on SO2's saved "
                             "STAGED_w1 model; one scaled prefix/cell/plain path. No durable record.")
    parser.add_argument("--smoke", action="store_true",
                        help="restart test: world 6, streams 0-1, scale 16, 2 margin pairs at 2 steps; separate root")
    args = parser.parse_args()
    if args.dry_run:
        root = ROOT.parent / "so4_b2_retest_dry"
        for variant in ("omitted", "explicit"):
            t = time.perf_counter()
            print("G0", job_g0(variant, root) | {"seconds": round(time.perf_counter() - t, 1)}, flush=True)
        # Margin equivalence: one job per pair must equal export_margin's own loop, bitwise.
        from row.experiments.score_so2_online_gate import restore_stage
        rec = json.loads((so2.ROOT / "cells" / "STAGED_w1" / "result.json").read_text())["result"]
        cfg3, world3, _, _ = so2.stage_setup(1, 3, 5000)
        model = restore_stage(so2.ROOT / "cells" / "STAGED_w1" / "stage3", cfg3, world3,
                              rec["stages"]["3"]["novel_probe_available"])
        so2.ADAPT_STEPS, original = 3, so2.ADAPT_STEPS
        t = time.perf_counter()
        reference = so2.export_margin(cfg3, world3, model, 1)
        so2.ADAPT_STEPS = original
        pairs = [margin_pair(model, cfg3, world3, 1, i, 3) for i in range(HELD_OUT)]
        same = all(p["trained"] == r["trained"] and p["scratch"] == r["scratch"] and p["program"] == r["program"]
                   for p, r in zip(pairs, reference["rows"]))
        print("margin pair-jobs bitwise == export_margin (3 steps):", same,
              "margin", margin_from_pairs(pairs), "vs", reference["margin"], f"{time.perf_counter() - t:.1f}s",
              flush=True)
        t = time.perf_counter()
        prefix = job_prefix(6, 1, root, 16)
        cell = job_cell(6, 1, prefix, root, 16)
        plain = job_plain(6, root, 16)
        print("scaled path ok:", cell["replay_seed"], round(cell["terminal_median"], 4),
              round(plain["terminal_median"], 4), f"{time.perf_counter() - t:.1f}s", flush=True)
        return 0 if same else 1

    root, output, scale, worlds, streams, margin_steps, pairs_n = ROOT, OUTPUT, 1, WORLDS, STREAMS, ADAPT_STEPS, HELD_OUT
    if args.smoke:
        root, scale, worlds, streams, margin_steps, pairs_n = Path("artifacts/so4_b2_retest_smoke"), 16, (6,), (0, 1), 2, 2
        output = root / "report.json"
    else:
        for tool in ("tools/check_prereg.py", "tools/check_invalid.py"):
            if subprocess.run([sys.executable, tool]).returncode != 0:
                raise SystemExit(f"{tool} failed")
        require_clean_code(OUTPUT)
    expected = protocol() | ({"smoke": {"scale": scale, "worlds": list(worlds), "streams": list(streams),
                                        "margin_steps": margin_steps, "pairs": pairs_n}} if args.smoke else {})
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
            out = {"frozen_plan": PLAN.as_posix(), "git_commit": commit, "protocol": expected, "protocol_sha256": sha,
                   "started_utc": now(), "gates": {}, "prefixes": {}, "cells": {}, "plain": {}, "margin_pairs": {},
                   "complete": False}
            log_line(run_log, f"LAUNCH {PROTOCOL_ID}{' SMOKE' if args.smoke else ''} at {commit} pid {os.getpid()}")
        atomic_json(output, out)
        sections = {"gate": "gates", "prefix": "prefixes", "cell": "cells", "plain": "plain", "margin": "margin_pairs"}

        def durable(kind, name):
            return root / "records" / kind / f"{name}.json", {"kind": kind, "name": name, "git_commit": commit,
                                                            "protocol_sha256": sha}

        def reuse(kind, name):
            path, stamp = durable(kind, name)
            if not path.exists():
                return None
            stored = json.loads(path.read_text())
            if stored["stamp"] != stamp or stored["result_sha256"] != fingerprint(stored["result"]):
                raise SystemExit(f"durable record mismatch: {path}")
            log_line(run_log, f"[{kind}/{name}] reused validated durable record")
            return stored["result"]

        def store(kind, name, result):
            path, stamp = durable(kind, name)
            atomic_json(path, {"stamp": stamp, "result": result, "result_sha256": fingerprint(result),
                               "finished_utc": now()})
            out[sections[kind]][name] = result
            atomic_json(output, out)

        started_at = time.time()

        def status(running):
            done = sum(len(out[s]) for s in sections.values())
            atomic_json(status_path, {"state": "running", "pid": os.getpid(), "git_commit": commit,
                                      "started_utc": out["started_utc"], "jobs_done": done,
                                      "running": sorted(running.values()),
                                      "elapsed_seconds": round(time.time() - started_at), "updated_utc": now()})

        def run_phase(pool, jobs):
            queue, running = [], {}
            for kind, name, call in jobs:
                cached = reuse(kind, name)
                if cached is None:
                    queue.append((kind, name, call))
                else:
                    out[sections[kind]][name] = cached
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
                    if kind in ("cell", "plain"):
                        head = f"terminal {result['terminal_median']:.4f} ({result['terminal_below']}/64)"
                    elif kind == "margin":
                        head = f"trained {result['trained']:.4f} scratch {result['scratch']:.4f} ({result['seconds']}s)"
                    elif kind == "gate":
                        head = json.dumps({k: v for k, v in result.items() if k in ("passes", "worst_per_task_abs_diff")})
                    else:
                        head = f"stage2 terminal {result['stages']['2']['terminal_median']:.4f}"
                    log_line(run_log, f"[{label}] saved: {head}")
                status(running)

        try:
            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                phase_a = [] if args.smoke else [("gate", f"G0_{v}", (job_g0, v, root)) for v in ("omitted", "explicit")]
                phase_a += [("prefix", f"w{w}_s{s}", (job_prefix, w, s, root, scale)) for w in worlds for s in streams]
                run_phase(pool, phase_a)
                if not args.smoke and not all(out["gates"][f"G0_{v}"]["passes"] for v in ("omitted", "explicit")):
                    raise SystemExit("G0 failed: replay_seed is not bitwise-neutral; no scored cell runs")
                phase_b = [("gate", "G1", (job_g1, root, scale))]
                phase_b += [("cell", f"w{w}_s{s}", (job_cell, w, s, out["prefixes"][f"w{w}_s{s}"], root, scale))
                            for w in worlds for s in streams]
                phase_b += [("plain", f"w{w}", (job_plain, w, root, scale)) for w in worlds]
                run_phase(pool, phase_b)
                phase_c = [("margin", f"w{w}_p{i}", (job_margin, w, i, out["cells"][f"w{w}_s{MARGIN_STREAM}"], root,
                                                     margin_steps)) for w in worlds for i in range(pairs_n)]
                run_phase(pool, phase_c)
            margins = {w: margin_from_pairs([out["margin_pairs"][f"w{w}_p{i}"] for i in range(pairs_n)])
                       for w in worlds}
            out["margins"] = {str(w): v for w, v in margins.items()}
            if args.smoke:
                out["gate_results"] = gates(out, smoke=True)
                out["classification"] = {"program": "SMOKE (not a result)"}
            else:
                out["gate_results"] = gates(out)
                medians = {w: {s: out["cells"][f"w{w}_s{s}"]["terminal_median"] for s in STREAMS} for w in WORLDS}
                out["classification"] = classify(medians, margins, out["gate_results"])
            out["complete"], out["finished_utc"] = True, now()
            atomic_json(output, out)
            atomic_json(status_path, {"state": "complete", "classification": out["classification"]["program"],
                                      "gates": out["gate_results"], "updated_utc": now()})
            log_line(run_log, f"COMPLETE gates {out['gate_results']} classification {out['classification']}")
            atomic_json(root / "exit.json", {"git_commit": commit, "exit_code": 0, "finished_utc": now()})
        except BaseException as error:
            log_line(run_log, f"FAILED: {error!r}\n{traceback.format_exc()}")
            atomic_json(status_path, {"state": f"failed: {error!r}", "updated_utc": now()})
            atomic_json(root / "exit.json", {"git_commit": commit, "exit_code": 1, "finished_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
