"""E6B: does macro substitution causally reduce synthesis cost, by the predicted amount?

`E6_MACRO_PLAN.md`, frozen at Amendment 2. E5.1 measured `C_find ~ D` across a
depth SWEEP; E6B asks whether SHORTENING a program by macro substitution moves
`C_find` the same way -- i.e. whether the scaling law is causal in program length
rather than a property of the task distribution at each depth.

Three arms, PAIRED BY TASK (Amendment 2). Every arm solves the same task with the
same optimizer, budget and support set:

    P   plain      12 slots, depth D            the reference
    M   macro      13 slots (12 + M), depth D-L+1   length falls, width rises
    K   width      13 slots (12 + duplicate), depth D   width rises ALONE

so the effect decomposes as `Delta C_macro = Delta C_length + Delta C_K`. The
dummy symbol is a DUPLICATE of an existing operator: it widens the route variable
without adding capability and without costing more per use, which is what makes
it a pure width control.

Registered prediction, from E5.1's own 192 cells and with no free coefficient:

    C_hat(D) = 2.8287 D - 1.1719   ->   Delta C_pred = a (L-1) = 5.66 s at L = 3

Correctness is FUNCTIONAL: the macro arm must not be cheaper by being worse.
Fails closed; protocol-fingerprinted cache; atomic report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from row.arm_provenance import arms_differ, describe_arm
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS, nmse
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams, plant_corpus
from row.experiments.audit_e6a_macro_economics import PRIMARY_L, substitute
from row.experiments.audit_e8_length import VariableDepthDiscrete, adapt_cell
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
PAIRS = 8                       # n per arm; power computed before running (~7.1 SE)
CORPUS = 128                    # must match E6A so the cached routes apply
PLANT_FRACTION = 0.5
FIT_A = 2.8287                  # E5.1 fitted cost law, registered in Amendment 2
FIT_B = -1.1719
CACHE = Path("reports/e6b_cache")


class MacroOperator(nn.Module):
    """`M := (P_a, P_b, P_c)` -- a definitional macro: execution IS the expansion."""

    def __init__(self, operators) -> None:
        super().__init__()
        self.steps = nn.ModuleList(operators)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for operator in self.steps:
            z = operator(z)
        return z


def fingerprint() -> str:
    payload = {"depth": DEPTH, "L": MACRO_LEN, "pairs": PAIRS, "corpus": CORPUS,
               "steps": ADAPT_STEPS, "lr": ADAPT_LR, "fit": [FIT_A, FIT_B]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cached(key: str, compute):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.json"
    stamp = fingerprint()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol") != stamp:
            raise SystemExit(f"cached cell {key} under a different protocol; refuse to mix")
        return stored["value"]
    value = compute()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"protocol": stamp, "value": value}), encoding="utf-8")
    os.replace(tmp, path)
    return value


def extend(model, extra: nn.Module):
    """A copy of `model` whose library gains one slot. Nothing else changes."""
    wide = copy.deepcopy(model)
    wide.__class__ = VariableDepthDiscrete
    wide.library = nn.ModuleList(list(wide.library) + [extra])
    wide.operator_slots = len(wide.library)
    wide.task_codes = nn.ParameterDict()      # codes are per-arm, never shared
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6b_search_savings.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    predicted = FIT_A * (MACRO_LEN - 1)
    out = {"frozen_plan": "E6_MACRO_PLAN.md (Amendment 2)", "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN, "pairs": PAIRS,
                        "steps": ADAPT_STEPS, "lr": ADAPT_LR,
                        "cost_law": {"a": FIT_A, "b": FIT_B, "source": "E5.1, 192 cells"},
                        "predicted_delta_seconds": predicted,
                        "ratio_form_seconds": (MACRO_LEN - 1) / DEPTH * (FIT_A * DEPTH + FIT_B),
                        "dummy": "duplicate of library[0]: width without capability",
                        "note": "paired by task; macro arm must not be cheaper by being worse"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        teacher_library = cell["world"].library
        base = cell["model"]

        # Rebuild E6A's corpus EXACTLY, so its macro and its bearing routes apply.
        rng = np.random.default_rng(np.random.SeedSequence([970, world, DEPTH]))
        motif, programs, carries, sites = plant_corpus(
            rng, config.world.teacher_primitives, CORPUS, DEPTH, PRIMARY_L,
            int(CORPUS * PLANT_FRACTION))
        routes = []
        for index in range(CORPUS):
            path = Path("reports/e6a_cache") / f"w{world}_d{DEPTH}_{index}.json"
            fatal(path.exists(), f"missing E6A route cache {path}; run E6A first")
            routes.append(json.loads(path.read_text(encoding="utf-8"))["value"]["route"])
        tally = Counter()
        for route in routes:
            for gram in ngrams(route, MACRO_LEN):
                tally[gram] += 1
        macro, _ = tally.most_common(1)[0]

        bearing = [i for i, r in enumerate(routes) if substitute(r, macro)[1] > 0]
        fatal(len(bearing) >= PAIRS,
              f"world {world} has {len(bearing)} bearing routes, needs {PAIRS}")
        chosen = bearing[:PAIRS]

        macro_ops = [copy.deepcopy(base.library[s]) for s in macro]
        wide_macro = extend(base, MacroOperator(macro_ops))
        wide_dummy = extend(base, copy.deepcopy(base.library[0]))
        macro_slot = wide_macro.operator_slots - 1

        # Non-vacuity 1: the macro executor IS the expansion. An identity, and
        # therefore an implementation check -- reported as one, never as evidence.
        probe = torch.tensor(np.random.default_rng(
            np.random.SeedSequence([980, world])).normal(size=(64, config.world.state_dim)),
            dtype=torch.float32)
        with torch.no_grad():
            direct = probe
            for s in macro:
                direct = base.library[s](direct)
            through = wide_macro.library[macro_slot](probe)
        fatal(bool(torch.equal(direct, through)),
              "macro operator is not bitwise its own expansion")

        arms = describe_arm("P", base, init_source="trained", steps=ADAPT_STEPS,
                            trainable=[], data_seen="support only")
        arm_m = describe_arm("M", wide_macro, init_source="copy:trained",
                             steps=ADAPT_STEPS, trainable=[], data_seen="support only")
        # The arms are DELIBERATELY different constructions; record how, so the
        # difference is in the artifact rather than assumed from the labels.
        arm_delta = arms_differ(arms, arm_m)

        # One untimed throwaway adaptation, so no measured cell pays warmup.
        warm = _build_tasks(config.world, teacher_library, [programs[chosen[0]]],
                            ["task_e6b_warmup"], index_offset=79999)[0]
        adapt_cell(base, warm, f"e6bWARM_w{world}", DEPTH, False,
                   teacher_library, programs[chosen[0]], steps=25)

        rows = []
        for index in chosen:
            program = programs[index]
            task = _build_tasks(config.world, teacher_library, [program],
                                [f"task_e6a_d{DEPTH}_{index}"],
                                index_offset=70000 + DEPTH * 200 + index)[0]
            tag = f"w{world}_{index}"

            def cellwise(task=task, program=program, tag=tag, slot=len(rows)):
                res = {}
                spec = [("P", base, DEPTH),
                        ("M", wide_macro, DEPTH - MACRO_LEN + 1),
                        ("K", wide_dummy, DEPTH)]
                # ROTATE arm order per task. The dry run showed the first timed
                # call in a process carries warmup (1.10s against 0.44s), and a
                # fixed order would put that cost on P every time -- inflating
                # `P - M` toward the registered prediction. Rotation cancels any
                # residual position effect over the paired set.
                spec = spec[slot % 3:] + spec[:slot % 3]
                for name, model, depth in spec:
                    t0 = time.process_time()
                    got = adapt_cell(model, task, f"e6b{name}_{tag}", depth, False,
                                     teacher_library, program, steps=ADAPT_STEPS)
                    res[name] = {"seconds": time.process_time() - t0,
                                 "order_position": [n for n, _, _ in spec].index(name),
                                 "nmse": got["query_nmse"], "route": got["route"],
                                 "support_reduction": got["support_reduction_objective"]}
                return res

            res = cached(tag, cellwise)
            for name in ("P", "M", "K"):
                fatal(res[name]["support_reduction"] > 0.0,
                      f"arm {name} did not optimize at {tag}")
            rows.append(res)

        def col(name, field):
            return np.array([r[name][field] for r in rows], dtype=float)

        d_macro = col("P", "seconds") - col("M", "seconds")
        d_width = col("K", "seconds") - col("P", "seconds")
        d_length = d_macro + d_width          # remove the width cost from the macro effect
        quality = np.log(col("M", "nmse")) - np.log(col("P", "nmse"))

        def summarize(v):
            return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "se": float(v.std(ddof=1) / math.sqrt(len(v)))}

        entry = {
            "macro": list(macro), "macro_is_constant": len(set(macro)) == 1,
            "tasks": chosen,
            "seconds": {n: summarize(col(n, "seconds")) for n in ("P", "M", "K")},
            "nmse": {n: float(np.exp(np.mean(np.log(col(n, "nmse"))))) for n in ("P", "M", "K")},
            "delta_macro": summarize(d_macro),
            "delta_width": summarize(d_width),
            "delta_length": summarize(d_length),
            "predicted": predicted,
            "ratio_predicted": (MACRO_LEN - 1) / DEPTH * float(col("P", "seconds").mean()),
            "ratio_observed_over_predicted": float(d_macro.mean() / predicted),
            "within_factor_2": bool(0.5 <= d_macro.mean() / predicted <= 2.0),
            "quality_log_gap_M_minus_P": summarize(quality),
            # Registered: the macro arm must not buy its speed with accuracy.
            "quality_ok": bool(quality.mean() <= 0.15),
            "arm_fields_differing": arm_delta,
            "macro_used_by_M": float(np.mean(
                [macro_slot in r["M"]["route"] for r in rows])),
            "arm_records": {"P": arms, "M": arm_m},
        }
        out["worlds"][str(world)] = entry
        print(f"[w{world}] macro {macro} | P {entry['seconds']['P']['mean']:.2f}s "
              f"M {entry['seconds']['M']['mean']:.2f}s K {entry['seconds']['K']['mean']:.2f}s")
        print(f"       dC_macro {d_macro.mean():+.2f}s (SE {entry['delta_macro']['se']:.2f}) "
              f"vs predicted {predicted:.2f}s -> x{d_macro.mean()/predicted:.2f} "
              f"{'WITHIN 2x' if entry['within_factor_2'] else 'OUTSIDE 2x'}")
        print(f"       dC_width {d_width.mean():+.2f}s (SE {entry['delta_width']['se']:.2f}) | "
              f"dC_length {d_length.mean():+.2f}s | "
              f"quality M-P {quality.mean():+.3f} log | "
              f"M uses macro in {entry['macro_used_by_M']:.0%} of tasks", flush=True)
        write(out, args.output)

    flags = [out["worlds"][str(w)]["within_factor_2"] for w in WORLDS]
    widths = [out["worlds"][str(w)]["delta_width"]["mean"] for w in WORLDS]
    quality = [out["worlds"][str(w)]["quality_ok"] for w in WORLDS]
    out["verdict"] = {
        "worlds_within_factor_2": sum(1 for f in flags if f),
        "worlds_quality_ok": sum(1 for f in quality if f),
        "passes": sum(1 for f in flags if f) >= 2 and sum(1 for f in quality if f) >= 2,
        "mean_delta_width_seconds": float(np.mean(widths)),
        "width_under_1s": bool(abs(float(np.mean(widths))) < 1.0),
    }
    write(out, args.output)
    print(f"\nE6B: {out['verdict']['worlds_within_factor_2']}/3 worlds within a factor of 2 "
          f"of the predicted {predicted:.2f}s -> "
          f"{'PREDICTED SAVING CONFIRMED' if out['verdict']['passes'] else 'NOT CONFIRMED'}")
    print(f"mean width cost {out['verdict']['mean_delta_width_seconds']:+.2f}s "
          f"({'under' if out['verdict']['width_under_1s'] else 'over'} the registered 1s)")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
