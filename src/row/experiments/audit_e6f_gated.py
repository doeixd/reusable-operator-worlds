"""E6F: nominate with a robust statistic, gate with a fragile one.

`E6_MACRO_PLAN.md`, frozen at Amendment 5.

E6E found that a trend projection of recurrence REFUSES a decaying pattern
correctly when applied to one nominated candidate, and fails completely when the
same statistic is maximised over ~150 candidate grams -- because the maximum of a
noisy trend estimate is reliably a fluke. That was a post-hoc observation. E6F
registers it as a rule and repairs the generator confound that made E6E's world 0
unscoreable while being counted as a failure.

    NOMINATE  argmax of the RETROSPECTIVE net saving over every contiguous
              L-gram of the observed corpus
    GATE      accept only if a TREND projection of that one candidate's
              recurrence clears the crossing

Case B's schedule is now `p(i) = min(1, max(0, 2 - i/32))`: ~48.5 planted against
E6E's ~36.5, decay ratio ~0.52, future empty. A world whose REALIZED
within-observed ratio exceeds 0.7 is UNSCOREABLE, not negative.

Fails closed; protocol-fingerprinted cache; atomic report.
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

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e1_export import ADAPT_LR, ADAPT_STEPS
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams, plant_corpus
from row.experiments.audit_e6a_macro_economics import (
    code_lengths, predicted_crossing, substitute)
from row.experiments.audit_e8_length import adapt_cell
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
CORPUS = 128
SPLIT = 64
LATE = 32
ELIGIBLE_RATIO = 0.7        # realized within-observed decay required for case B
CACHE = Path("reports/e6f_cache")
E6A_CACHE = Path("reports/e6a_cache")


def carry_probability(index: int) -> float:
    """Amendment 5's registered schedule: front-loaded, empty by task 64."""
    return min(1.0, max(0.0, 2.0 - index / 32.0))


def fingerprint() -> str:
    payload = {"depth": DEPTH, "L": MACRO_LEN, "corpus": CORPUS, "split": SPLIT,
               "late": LATE, "schedule": "min(1,max(0,2-i/32))",
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


def decayed_corpus(rng, primitives: int, n: int, depth: int, length: int):
    motif = tuple(int(v) for v in rng.integers(0, primitives, length))
    programs, carries, sites = [], [], []
    for index in range(n):
        program = [int(v) for v in rng.integers(0, primitives, depth)]
        carry = bool(rng.random() < carry_probability(index))
        site = -1
        if carry:
            site = int(rng.integers(0, depth - length + 1))
            program[site:site + length] = list(motif)
        programs.append(tuple(program))
        carries.append(carry)
        sites.append(site)
    return motif, programs, carries, sites


def net_saving(uses: float, gram, n: int, depth: int, slots: int) -> float:
    a, b = float(np.log2(slots)), float(np.log2(slots + 1))
    return uses * (len(gram) - 1) * b - len(gram) * a - n * depth * (b - a)


def nominate(observed, slots: int, depth: int):
    """Robust statistic over a large field: realized net saving."""
    tally = Counter()
    for route in observed:
        for gram in ngrams(route, MACRO_LEN):
            tally[gram] += 1
    best, best_score = None, -float("inf")
    for gram in tally:
        uses = sum(substitute(r, gram)[1] for r in observed)
        s = net_saving(uses, gram, len(observed), depth, slots)
        if s > best_score:
            best, best_score = gram, s
    return best, best_score, len(tally)


def gate(observed, gram, slots: int, depth: int) -> dict:
    """Fragile statistic applied to ONE candidate: extrapolate the trend."""
    n = len(observed)
    early_block, late_block = observed[:n - LATE], observed[-LATE:]
    early = sum(substitute(r, gram)[1] for r in early_block)
    late = sum(substitute(r, gram)[1] for r in late_block)
    early_rate = early / max(len(early_block), 1)
    late_rate = late / max(len(late_block), 1)
    slope = late_rate - early_rate
    projected = sum(max(0.0, late_rate + slope * (k + 1)) * LATE for k in range(n // LATE))
    saving = net_saving(projected, gram, n, depth, slots)
    return {"observed_early_uses": early, "observed_late_uses": late,
            "realized_ratio": (late / early) if early else None,
            "projected_uses": projected, "gated_net_saving_bits": saving,
            "accept": bool(saving > 0),
            "H_star": predicted_crossing(len(gram), depth, n, slots)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6f_gated.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E6_MACRO_PLAN.md (Amendment 5)", "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN, "corpus": CORPUS,
                        "split": SPLIT, "late_window": LATE,
                        "schedule": "p(i) = min(1, max(0, 2 - i/32))",
                        "eligible_ratio": ELIGIBLE_RATIO,
                        "rule": "nominate by retrospective argmax, gate by trend",
                        "note": "total planted cannot match case A while decaying; "
                                "reported as a covariate"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        slots = cell["shipped"].operator_slots
        teacher_library = cell["world"].library

        base = []
        for index in range(CORPUS):
            path = E6A_CACHE / f"w{world}_d{DEPTH}_{index}.json"
            fatal(path.exists(), f"missing E6A cache {path}; run E6A first")
            base.append(json.loads(path.read_text(encoding="utf-8"))["value"]["route"])
        perm = np.random.default_rng(np.random.SeedSequence([997, world])).permutation(CORPUS)

        rng = np.random.default_rng(np.random.SeedSequence([998, world]))
        motif_b, programs_b, carries_b, _ = decayed_corpus(
            rng, config.world.teacher_primitives, CORPUS, DEPTH, MACRO_LEN)
        routes_b = []
        for index, program in enumerate(programs_b):
            task = _build_tasks(config.world, teacher_library, [program],
                                [f"task_e6f_{index}"], index_offset=83000 + index)[0]
            tag = f"w{world}_{index}"

            def cellwise(task=task, program=program, tag=tag):
                got = adapt_cell(cell["model"], task, f"e6f_{tag}", DEPTH, False,
                                 teacher_library, program, steps=ADAPT_STEPS)
                return {"route": got["route"],
                        "support_reduction": got["support_reduction_objective"]}

            res = cached(tag, cellwise)
            fatal(res["support_reduction"] > 0.0, f"route inference did not optimize at {tag}")
            routes_b.append(res["route"])

        cases = {"A": [base[i] for i in perm], "B": routes_b, "C": base}
        planted = {"A": CORPUS // 2, "B": int(sum(carries_b)), "C": CORPUS // 2}

        entry = {}
        for name, routes in cases.items():
            observed = routes[:SPLIT]
            future = routes[SPLIT:]
            gram, retro_score, field = nominate(observed, slots, DEPTH)
            g = gate(observed, gram, slots, DEPTH)
            realized = code_lengths(future, gram, slots)
            eligible = (name != "B") or (g["realized_ratio"] is not None
                                         and g["realized_ratio"] <= ELIGIBLE_RATIO)
            want = {"A": True, "B": False, "C": None}[name]
            entry[name] = {
                "nominated": list(gram), "candidate_field": field,
                "retro_net_saving_bits": retro_score,
                "retro_creates": bool(retro_score > 0),
                "gate": g, "eligible": bool(eligible),
                "planted_total": planted[name],
                "future_uses": realized["uses"],
                "future_net_saving_bits": realized["saving"],
                "gated_correct": (None if want is None else bool(g["accept"] == want)),
            }
            r = entry[name]
            ratio = (f"{g['realized_ratio']:.2f}" if g["realized_ratio"] is not None else "n/a")
            print(f"[w{world} {name}] nominated {str(list(gram)):>14} from {field} candidates | "
                  f"planted {planted[name]:>2} | early {g['observed_early_uses']:>2} "
                  f"late {g['observed_late_uses']:>2} (ratio {ratio})")
            print(f"      ungated {'CREATE' if r['retro_creates'] else 'REFUSE':>6} "
                  f"({retro_score:+8.1f} bits) | "
                  f"gated {'CREATE' if g['accept'] else 'REFUSE':>6} "
                  f"(proj {g['projected_uses']:5.1f} uses vs H* {g['H_star']:.1f}) | "
                  f"future {realized['saving']:+8.1f} bits"
                  + ("" if want is None else
                     f" | gated {'right' if r['gated_correct'] else 'WRONG'}")
                  + ("" if eligible else "  [INELIGIBLE: decay did not take]"), flush=True)
        out["worlds"][str(world)] = entry
        write(out, args.output)

    def tally(case, field, want):
        return sum(1 for w in WORLDS
                   if out["worlds"][str(w)][case]["eligible"]
                   and out["worlds"][str(w)][case][field] == want)

    a_create = tally("A", "gated_correct", True)
    b_refuse = tally("B", "gated_correct", True)
    b_eligible = sum(1 for w in WORLDS if out["worlds"][str(w)]["B"]["eligible"])
    ungated_b = tally("B", "retro_creates", True)
    if b_eligible < 2:
        verdict = "UNSCOREABLE (fewer than 2 eligible case-B worlds)"
    elif a_create >= 2 and b_refuse >= 2 and ungated_b >= 2:
        verdict = "GATED CRITERION DEMONSTRATED"
    elif b_refuse < 2:
        verdict = "NO IMPROVEMENT"
    else:
        verdict = "INCONCLUSIVE"
    out["verdict"] = {"A_gated_creates": a_create, "B_gated_refuses": b_refuse,
                      "B_eligible_worlds": b_eligible, "B_ungated_creates": ungated_b,
                      "verdict": verdict}
    write(out, args.output)
    print(f"\nA gated-create {a_create}/3 | B gated-refuse {b_refuse}/{b_eligible} eligible | "
          f"B ungated-create {ungated_b}/{b_eligible}")
    print(f"E6F: {verdict}")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
