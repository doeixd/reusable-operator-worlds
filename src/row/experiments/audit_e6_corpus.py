"""E6 step 0: does a planted teacher subsequence survive into the LEARNER's corpus?

`E6_MACRO_PLAN.md`, frozen. Non-vacuity check 1, run before any macro is priced.

The macro is a symbol of the learner's language, defined over the learner's own
INFERRED routes, never over teacher primitive indices. The gauge results (E0.1
assignment margins 0.001-0.019; inferred routes beating the teacher's in 7 of 9
E2 cells) mean a subsequence planted in teacher space need not appear as a clean
recurring subsequence in learner space. Whether it does is measured here.

Method. Generate `N` depth-`D` teacher programs, `H` of which contain a fixed
contiguous teacher subsequence of length `L` at a random position. Infer each
task's program with the sealed/E5.1 route optimization (support only, query
labels never used). Then count contiguous `L`-grams over the inferred corpus and
compare the most frequent against a null that preserves the corpus's per-position
symbol marginals -- so a learner that simply over-uses a few slots cannot pass by
that alone.

A condition failing this gate is UNSCOREABLE for E6, not negative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from row.arm_provenance import describe_arm
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e8_length import adapt_cell
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
N_TASKS = 64
PLANTED = 32                 # H, chosen far above the predicted H* = 7.44
NULL_DRAWS = 20_000
CACHE = Path("reports/e6_corpus_cache")


def fingerprint() -> str:
    payload = {"depth": DEPTH, "L": MACRO_LEN, "n": N_TASKS, "planted": PLANTED,
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


def plant_corpus(rng, primitives: int, n: int, depth: int, length: int, planted: int):
    """`n` teacher programs, `planted` of which carry one fixed subsequence."""
    motif = tuple(int(v) for v in rng.integers(0, primitives, length))
    programs, carries, sites = [], [], []
    for index in range(n):
        program = [int(v) for v in rng.integers(0, primitives, depth)]
        carry = index < planted
        site = -1
        if carry:
            site = int(rng.integers(0, depth - length + 1))
            program[site:site + length] = list(motif)
        programs.append(tuple(program))
        carries.append(carry)
        sites.append(site)
    return motif, programs, carries, sites


def ngrams(route, length: int):
    return [tuple(route[i:i + length]) for i in range(len(route) - length + 1)]


def null_max_count(source_routes, length: int, draws: int, rng,
                   n_target: int | None = None) -> dict:
    """Max `L`-gram count for a motif-free corpus of `n_target` routes.

    CORRECTION (2026-08-27, before any E6 verdict): the first version resampled
    per-position marginals from the FULL corpus, which already contains the
    planted motif -- so the null inherited the very effect it was meant to
    baseline and could reproduce it. In world 1 that scored a gram appearing at
    14 of 32 planted sites as indistinguishable from chance. This is the standing
    rule against fitting and scoring a structure measurement on the same objects
    (V5's in-sample 0.730 versus leave-one-out 0.021), reproduced in a new place.

    `source_routes` must therefore be the UNPLANTED routes only. Position
    marginals are still preserved, so a learner that merely over-uses a few slots
    cannot pass the gate on slot skew alone.
    """
    arr = np.array(source_routes)
    n, depth = arr.shape
    n = int(n_target or n)
    columns = [arr[:, j] for j in range(depth)]
    counts = np.empty(draws, dtype=np.int32)
    for d in range(draws):
        sample = np.stack([rng.choice(col, size=n, replace=True) for col in columns], axis=1)
        best = 0
        tally = Counter()
        for row in sample:
            for gram in ngrams(list(int(v) for v in row), length):
                tally[gram] += 1
        best = max(tally.values()) if tally else 0
        counts[d] = best
    return {"mean": float(counts.mean()), "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)), "max": int(counts.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6_corpus.json"))
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E6_MACRO_PLAN.md", "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN, "n_tasks": N_TASKS,
                        "planted_H": PLANTED, "steps": ADAPT_STEPS, "lr": ADAPT_LR,
                        "null": "per-position marginal resampling from UNPLANTED routes only",
                        "note": "gate only; a failing condition is UNSCOREABLE, not negative"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        teacher_library = cell["world"].library
        rng = np.random.default_rng(np.random.SeedSequence([960, world, DEPTH]))
        motif, programs, carries, sites = plant_corpus(
            rng, config.world.teacher_primitives, N_TASKS, DEPTH, MACRO_LEN, PLANTED)
        fatal(sum(carries) == PLANTED, "planted count does not match the registered H")

        routes, qnmse, arm_record = [], [], None
        for index, program in enumerate(programs):
            task = _build_tasks(config.world, teacher_library, [program],
                                [f"task_e6_{index}"], index_offset=60000 + index)[0]
            tag = f"w{world}_{index}"

            def cellwise(task=task, program=program, tag=tag):
                opt = adapt_cell(cell["model"], task, f"e6_{tag}", DEPTH, False,
                                 teacher_library, program, steps=ADAPT_STEPS)
                return {"route": opt["route"], "query_nmse": opt["query_nmse"],
                        "support_reduction": opt["support_reduction_objective"]}

            res = cached(tag, cellwise)
            fatal(res["support_reduction"] > 0.0, f"route inference did not optimize at {tag}")
            routes.append(res["route"])
            qnmse.append(res["query_nmse"])
            if arm_record is None:
                arm_record = describe_arm("OPT", cell["model"], init_source="trained",
                                          steps=ADAPT_STEPS, trainable=[],
                                          data_seen="support only")

        tally = Counter()
        for route in routes:
            for gram in ngrams(route, MACRO_LEN):
                tally[gram] += 1
        top_gram, top_count = tally.most_common(1)[0]
        # A route that collapses to one slot produces constant grams and would
        # pass a naive recurrence gate for a degenerate reason. Report the top
        # NON-CONSTANT gram and route diversity so that cannot hide.
        nonconst = Counter({g: c for g, c in tally.items() if len(set(g)) > 1})
        nc_gram, nc_count = nonconst.most_common(1)[0] if nonconst else (None, 0)
        diversity = float(np.mean([len(set(r)) for r in routes]))
        constant_routes = float(np.mean([len(set(r)) == 1 for r in routes]))

        # How often does the SAME learner gram sit where the motif was planted?
        at_site = Counter()
        for route, carry, site in zip(routes, carries, sites):
            if carry:
                at_site[tuple(route[site:site + MACRO_LEN])] += 1
        site_gram, site_count = at_site.most_common(1)[0] if at_site else (None, 0)

        unplanted = [r for r, carry in zip(routes, carries) if not carry]
        fatal(len(unplanted) > 0, "no unplanted routes to build an uncontaminated null")
        null = null_max_count(unplanted, MACRO_LEN, args.null_draws,
                              np.random.default_rng(np.random.SeedSequence([961, world])),
                              n_target=len(routes))
        survives = bool(top_count > null["p99"])
        out["worlds"][str(world)] = {
            "teacher_motif": list(motif), "planted_H": PLANTED,
            "top_learner_gram": list(top_gram), "top_count": int(top_count),
            "gram_at_planted_site": list(site_gram) if site_gram else None,
            "count_at_planted_site": int(site_count),
            "distinct_grams": len(tally),
            "top_gram_is_constant": len(set(top_gram)) == 1,
            "top_nonconstant_gram": list(nc_gram) if nc_gram else None,
            "top_nonconstant_count": int(nc_count),
            "mean_distinct_slots_per_route": diversity,
            "fraction_constant_routes": constant_routes,
            "null_max_count": null, "survives_gate": survives,
            "null_source": "unplanted routes only (contamination fix)",
            "motif_survival_rate": float(site_count / PLANTED),
            "mean_query_nmse": float(np.mean(qnmse)),
            "arm": arm_record,
        }
        r = out["worlds"][str(world)]
        print(f"[w{world}] motif {motif} planted {PLANTED}/{N_TASKS} | "
              f"top learner gram {top_gram} x{top_count} "
              f"(null p99 {null['p99']:.1f}, mean {null['mean']:.1f}) | "
              f"at planted site {site_gram} x{site_count} | "
              f"top non-const {nc_gram} x{nc_count} | slots/route {diversity:.2f} | "
              f"{'SURVIVES' if survives else 'DOES NOT SURVIVE'}", flush=True)
        write(out, args.output)

    passed = sum(1 for w in out["worlds"].values() if w["survives_gate"])
    out["gate"] = {"worlds_surviving": passed,
                   "scoreable": passed >= 2,
                   "rule": "planted recurrence must survive inference in >= 2 of 3 worlds"}
    write(out, args.output)
    print(f"\nplanted recurrence survives in {passed}/3 worlds -> "
          f"E6 is {'SCOREABLE' if passed >= 2 else 'UNSCOREABLE as designed'}")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
