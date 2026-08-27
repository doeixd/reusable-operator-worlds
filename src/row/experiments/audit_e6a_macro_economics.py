"""E6A: oracle macro economics -- does a macro pay where the code says it should?

`E6_MACRO_PLAN.md`, frozen. The candidate is GIVEN, not discovered: the most
frequent contiguous `L`-gram of the learner's own inferred corpus.

Registered coding scheme. With `N` programs of length `D` over `K` symbols the
corpus costs `N D log2 K`. Introducing one macro of expansion length `L` costs
its definition `L log2 K` PLUS an ALPHABET TAX `N D log2((K+1)/K)`, because every
symbol in every program now costs more; each use saves `(L-1) log2(K+1)`. Hence

    H* = [ L log2 K + N D log2((K+1)/K) ] / [ (L-1) log2(K+1) ]

The naive accounting (charging only the definition) gives `H* = L/(L-1)` = 1.5
uses at `L = 3` and was REJECTED IN ADVANCE in the frozen plan as vacuous.

Uses are counted in the LEARNER's realized corpus. E6 step 0 measured that only
~36% of planted teacher recurrences survive as the same learner gram, so every
sweep here is reported in EFFECTIVE uses, never planted ones.

Route inference is paid once per (world, depth); the `H`, `L` and `N` sweeps are
derived from that corpus arithmetically. Fails closed; cached; atomic report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams, plant_corpus
from row.experiments.audit_e8_length import adapt_cell
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTHS = (4, 6, 8, 10)
CORPUS = 128                 # inferred once per (world, depth)
PLANT_FRACTION = 0.5
MACRO_LENS = (2, 3, 4)
CORPUS_SIZES = (16, 32, 64, 128)
PRIMARY_DEPTH = 6
PRIMARY_L = 3
CACHE = Path("reports/e6a_cache")


def fingerprint() -> str:
    payload = {"depths": list(DEPTHS), "corpus": CORPUS, "plant": PLANT_FRACTION,
               "steps": ADAPT_STEPS, "lr": ADAPT_LR}
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


def substitute(route, macro) -> tuple[list, int]:
    """Greedy non-overlapping replacement; returns the shortened route and uses."""
    out, i, uses, length = [], 0, 0, len(macro)
    while i < len(route):
        if tuple(route[i:i + length]) == tuple(macro):
            out.append("M")
            i += length
            uses += 1
        else:
            out.append(route[i])
            i += 1
    return out, uses


def code_lengths(routes, macro, slots: int) -> dict:
    """Two-part code with and without the macro, under the registered scheme."""
    n = len(routes)
    depth_total = sum(len(r) for r in routes)
    without = depth_total * math.log2(slots)
    shortened, uses = [], 0
    for route in routes:
        new, u = substitute(route, macro)
        shortened.append(new)
        uses += u
    definition = len(macro) * math.log2(slots)
    tax_alphabet = math.log2(slots + 1)
    with_macro = definition + sum(len(r) for r in shortened) * tax_alphabet
    return {"without": without, "with": with_macro, "uses": uses,
            "saving": without - with_macro, "n": n,
            "symbols_without": depth_total,
            "symbols_with": sum(len(r) for r in shortened)}


def predicted_crossing(macro_len: int, depth: int, n: int, slots: int) -> float:
    a, b = math.log2(slots), math.log2(slots + 1)
    return (macro_len * a + n * depth * (b - a)) / ((macro_len - 1) * b)


def infer_corpus(cell, world: int, depth: int, teacher_library) -> dict:
    """Route inference, paid once per (world, depth)."""
    config = cell["config"]
    rng = np.random.default_rng(np.random.SeedSequence([970, world, depth]))
    planted = int(CORPUS * PLANT_FRACTION)
    motif, programs, carries, sites = plant_corpus(
        rng, config.world.teacher_primitives, CORPUS, depth, PRIMARY_L, planted)
    routes = []
    for index, program in enumerate(programs):
        task = _build_tasks(config.world, teacher_library, [program],
                            [f"task_e6a_d{depth}_{index}"],
                            index_offset=70000 + depth * 200 + index)[0]
        tag = f"w{world}_d{depth}_{index}"

        def cellwise(task=task, program=program, tag=tag):
            opt = adapt_cell(cell["model"], task, f"e6a_{tag}", depth, False,
                             teacher_library, program, steps=ADAPT_STEPS)
            return {"route": opt["route"],
                    "support_reduction": opt["support_reduction_objective"]}

        res = cached(tag, cellwise)
        fatal(res["support_reduction"] > 0.0, f"route inference did not optimize at {tag}")
        routes.append(res["route"])
    return {"motif": list(motif), "routes": routes, "carries": carries}


def crossing_at_n(routes_with_uses, macro, slots, depth, n_target: int) -> dict:
    """The observed/predicted crossing at a FIXED corpus size `n_target`.

    `N` is held fixed while `H` is swept, because the alphabet tax scales with
    `N D`; letting both move would confound the two variables the formula
    separates.
    """
    bearing = [r for r, u in routes_with_uses if u > 0]
    plain = [r for r, u in routes_with_uses if u == 0]
    if len(plain) < n_target:
        return {"reachable": False,
                "reason": f"needs {n_target} macro-free routes, corpus has {len(plain)}"}
    curve = []
    for h in range(0, min(len(bearing), n_target) + 1):
        corpus = bearing[:h] + plain[:n_target - h]
        lengths = code_lengths(corpus, macro, slots)
        curve.append({"h_routes": h, "uses": lengths["uses"], "saving": lengths["saving"]})
    crossing = next((c["uses"] for c in curve if c["saving"] > 0), None)
    pred = predicted_crossing(len(macro), depth, n_target, slots)
    ratio = (crossing / pred) if crossing else None
    return {"reachable": True, "n": n_target, "observed_crossing_uses": crossing,
            "predicted_H_star": pred, "ratio": ratio,
            "within_factor_2": bool(ratio is not None and 0.5 <= ratio <= 2.0),
            "max_uses": curve[-1]["uses"] if curve else 0,
            "bracketed": bool((curve[-1]["uses"] if curve else 0) >= pred)}


def observed_crossing(routes_with_uses, macro, slots, depth) -> dict:
    """Smallest sub-corpus use-count at which the macro-bearing code is shorter.

    Sub-corpora are built by adding macro-BEARING routes to a fixed base of
    non-bearing ones, holding `N` constant so the alphabet tax does not move as
    `H` is swept -- otherwise the crossing would confound two variables.
    """
    bearing = [r for r, u in routes_with_uses if u > 0]
    plain = [r for r, u in routes_with_uses if u == 0]
    n = min(len(plain), len(bearing))
    if n == 0 or not bearing:
        return {"crossing": None, "reason": "no bearing or no plain routes"}
    curve = []
    total = len(plain)                       # hold N fixed at the plain-route count
    for h in range(0, min(len(bearing), total) + 1):
        corpus = bearing[:h] + plain[:total - h]
        lengths = code_lengths(corpus, macro, slots)
        curve.append({"h_routes": h, "uses": lengths["uses"],
                      "saving": lengths["saving"]})
    crossing = next((c["uses"] for c in curve if c["saving"] > 0), None)
    return {"crossing": crossing, "curve": curve, "n_held": total,
            "max_uses": curve[-1]["uses"] if curve else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6a_macro_economics.json"))
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E6_MACRO_PLAN.md", "git_commit": git_commit(),
           "protocol": {"depths": list(args.depths), "corpus": CORPUS,
                        "plant_fraction": PLANT_FRACTION, "macro_lens": list(MACRO_LENS),
                        "corpus_sizes": list(CORPUS_SIZES), "steps": ADAPT_STEPS,
                        "coding": "uniform over current alphabet, with alphabet tax",
                        "note": "uses counted in the LEARNER corpus, never planted"},
           "depths": {}}
    cells = {w: load_cell(w, args.config) for w in WORLDS}
    slots = cells[0]["shipped"].operator_slots

    for depth in args.depths:
        per_world = {}
        for world in WORLDS:
            cell = cells[world]
            corpus = infer_corpus(cell, world, depth, cell["world"].library)
            routes = corpus["routes"]
            entry = {"teacher_motif": corpus["motif"], "by_macro_len": {}}
            for macro_len in MACRO_LENS:
                tally = Counter()
                for route in routes:
                    for gram in ngrams(route, macro_len):
                        tally[gram] += 1
                macro, _ = tally.most_common(1)[0]
                with_uses = [(r, substitute(r, macro)[1]) for r in routes]
                obs = observed_crossing(with_uses, macro, slots, depth)
                pred = predicted_crossing(macro_len, depth, obs.get("n_held") or len(routes),
                                          slots)
                bracketed = bool(obs.get("max_uses", 0) >= pred)
                ratio = (obs["crossing"] / pred) if obs.get("crossing") else None
                entry["by_macro_len"][str(macro_len)] = {
                    "macro": list(macro),
                    "macro_is_constant": len(set(macro)) == 1,
                    "total_uses_in_corpus": sum(u for _, u in with_uses),
                    "bearing_routes": sum(1 for _, u in with_uses if u > 0),
                    "observed_crossing_uses": obs.get("crossing"),
                    "predicted_H_star": pred,
                    "ratio_observed_over_predicted": ratio,
                    "within_factor_2": bool(ratio is not None and 0.5 <= ratio <= 2.0),
                    "bracketed": bracketed,
                    "n_held": obs.get("n_held"),
                    "max_uses": obs.get("max_uses"),
                    "by_corpus_size": {
                        str(n): crossing_at_n(with_uses, macro, slots, depth, n)
                        for n in CORPUS_SIZES},
                }
                r = entry["by_macro_len"][str(macro_len)]
                print(f"[d{depth} w{world} L{macro_len}] macro {macro}"
                      f"{' CONSTANT' if r['macro_is_constant'] else ''} "
                      f"uses {r['total_uses_in_corpus']} in {r['bearing_routes']} routes | "
                      f"H_obs {r['observed_crossing_uses']} vs H* {pred:.2f}"
                      + (f" (x{ratio:.2f})" if ratio else "")
                      + f" | bracketed {bracketed}", flush=True)
                ns = r["by_corpus_size"]
                print("        N sweep: " + "  ".join(
                    (f"N{n}:obs {ns[str(n)]['observed_crossing_uses']}"
                     f"/H* {ns[str(n)]['predicted_H_star']:.1f}"
                     if ns[str(n)]["reachable"] else f"N{n}:unreachable")
                    for n in CORPUS_SIZES), flush=True)
            per_world[str(world)] = entry
        out["depths"][str(depth)] = {"worlds": per_world}
        write(out, args.output)

    # Registered primary: does the observed crossing match H* within a factor of 2,
    # in >= 2 of 3 worlds, at the primary cell?
    key = str(PRIMARY_DEPTH)
    if key in out["depths"]:
        flags = [out["depths"][key]["worlds"][str(w)]["by_macro_len"][str(PRIMARY_L)]
                 ["within_factor_2"] for w in WORLDS]
        out["primary"] = {"depth": PRIMARY_DEPTH, "macro_len": PRIMARY_L,
                          "worlds_within_factor_2": sum(1 for f in flags if f),
                          "passes": sum(1 for f in flags if f) >= 2}
        print(f"\nPRIMARY (d{PRIMARY_DEPTH}, L{PRIMARY_L}): "
              f"{out['primary']['worlds_within_factor_2']}/3 worlds within a factor of 2 "
              f"-> {'PASS' if out['primary']['passes'] else 'FAIL'}")
    write(out, args.output)


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
