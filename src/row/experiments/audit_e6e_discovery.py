"""E6E: can the learner discover WHICH macro to create, and WHEN?

`E6_MACRO_PLAN.md`, frozen at Amendment 4.

E6D established two things. The registered two-part code picks the right macro
out of a supplied field of rivals (3/3), and it creates macros that stop
recurring (0/3). The correction recorded the same day showed E6D's accidental
control is a STEP FUNCTION -- uniform across the observed window, absent after --
so no past-only criterion could have refused it. E6E therefore grades its
controls by DETECTABILITY:

    A  continuing   motif spread evenly over observed and future     CREATE
    B  decaying     motif thins measurably WITHIN the observed half  REFUSE
    C  sharp step   E6D's construction                               UNSCOREABLE

Only B discriminates, because only B puts the evidence where the learner can see
it. Two criteria are compared, neither told the motif, both enumerating every
contiguous gram of the observed half:

    RETRO        create iff the code shortens on realized observed uses
    PROSPECTIVE  create iff it shortens on H_proj, projected from the LATE
                 observed window alone -- no future data

Discovery and timing are reported separately. Fails closed; cached; atomic.
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
from row.experiments.audit_e6_corpus import ngrams
from row.experiments.audit_e6a_macro_economics import (
    PRIMARY_L, code_lengths, predicted_crossing, substitute)
from row.experiments.audit_e8_length import adapt_cell
from row.support_split_world import _build_tasks

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
CORPUS = 128
SPLIT = 64                     # observed = [0:64], future = [64:128]
LATE = 32                      # the late observed window used for projection
DECAY_SCALE = 72.0             # p(i) = max(0, 1 - i/72)
CACHE = Path("reports/e6e_cache")
E6A_CACHE = Path("reports/e6a_cache")


def fingerprint() -> str:
    payload = {"depth": DEPTH, "L": MACRO_LEN, "corpus": CORPUS, "split": SPLIT,
               "late": LATE, "decay": DECAY_SCALE, "steps": ADAPT_STEPS, "lr": ADAPT_LR}
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


def decaying_corpus(rng, primitives: int, n: int, depth: int, length: int):
    """Carry probability declines across the corpus: `p(i) = max(0, 1 - i/72)`."""
    motif = tuple(int(v) for v in rng.integers(0, primitives, length))
    programs, carries, sites = [], [], []
    for index in range(n):
        program = [int(v) for v in rng.integers(0, primitives, depth)]
        carry = bool(rng.random() < max(0.0, 1.0 - index / DECAY_SCALE))
        site = -1
        if carry:
            site = int(rng.integers(0, depth - length + 1))
            program[site:site + length] = list(motif)
        programs.append(tuple(program))
        carries.append(carry)
        sites.append(site)
    return motif, programs, carries, sites


def infer(cell, world: int, tag_prefix: str, programs, teacher_library) -> list[list[int]]:
    config = cell["config"]
    routes = []
    for index, program in enumerate(programs):
        task = _build_tasks(config.world, teacher_library, [program],
                            [f"task_{tag_prefix}_{index}"],
                            index_offset=81000 + index)[0]
        tag = f"{tag_prefix}_w{world}_{index}"

        def cellwise(task=task, program=program, tag=tag):
            got = adapt_cell(cell["model"], task, f"e6e_{tag}", DEPTH, False,
                             teacher_library, program, steps=ADAPT_STEPS)
            return {"route": got["route"],
                    "support_reduction": got["support_reduction_objective"]}

        res = cached(tag, cellwise)
        fatal(res["support_reduction"] > 0.0, f"route inference did not optimize at {tag}")
        routes.append(res["route"])
    return routes


def enumerate_candidates(observed, length: int):
    """Every contiguous gram of the observed half. The learner is told nothing."""
    tally = Counter()
    for route in observed:
        for gram in ngrams(route, length):
            tally[gram] += 1
    return tally


def score(observed, gram, slots: int, depth: int, criterion: str) -> dict:
    """Net saving under realized observed uses, or under a projected count.

    Three criteria, none of which touches future data:

    `RETRO`        realized uses in the observed corpus (the E6D rule).
    `PROSP_RATE`   REGISTERED in Amendment 4: the late-window RATE scaled to the
                   corpus size.
    `PROSP_TREND`  ADDED after Amendment 4 was frozen, and disclosed as such. The
                   registered rate estimator cannot express a refusal on case B --
                   a decaying pattern still has a positive late-window rate, so
                   scaling it forward predicts continuation. Extrapolating the
                   TREND across observed blocks can. Amendment 4's prose says
                   "a trend read off the second half" while its formula is a rate;
                   both are reported and the registered one is primary.
    """
    lengths = code_lengths(observed, gram, slots)
    n = len(observed)
    early_block = observed[:n - LATE]
    late_block = observed[-LATE:]
    early_uses = sum(substitute(r, gram)[1] for r in early_block)
    late_uses = sum(substitute(r, gram)[1] for r in late_block)
    if criterion == "RETRO":
        uses = lengths["uses"]
    elif criterion == "PROSP_RATE":
        uses = late_uses * (n / LATE)
    elif criterion == "PROSP_TREND":
        # Per-block rates, extrapolated forward two blocks and clipped at zero.
        early_rate = early_uses / max(len(early_block), 1)
        late_rate = late_uses / max(len(late_block), 1)
        slope = late_rate - early_rate
        uses = sum(max(0.0, late_rate + slope * (k + 1)) * LATE for k in range(n // LATE))
    else:
        raise ValueError(criterion)
    a, b = np.log2(slots), np.log2(slots + 1)
    definition = len(gram) * a
    tax = n * depth * (b - a)
    saving = uses * (len(gram) - 1) * b - definition - tax
    return {"uses_used": float(uses), "realized_uses": lengths["uses"],
            "observed_early_uses": early_uses, "observed_late_uses": late_uses,
            "net_saving_bits": float(saving), "create": bool(saving > 0),
            "H_star": predicted_crossing(len(gram), depth, n, slots)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6e_discovery.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E6_MACRO_PLAN.md (Amendment 4)", "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN, "corpus": CORPUS,
                        "split": SPLIT, "late_window": LATE, "decay_scale": DECAY_SCALE,
                        "cases": {"A": "continuing (CREATE)", "B": "decaying (REFUSE)",
                                  "C": "sharp step (UNSCOREABLE)"},
                        "note": "discovery and timing reported separately"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        slots = cell["shipped"].operator_slots
        teacher_library = cell["world"].library

        # --- case C and case A both come from E6A's cached corpus -------------
        base_routes = []
        base_carries = []
        for index in range(CORPUS):
            path = E6A_CACHE / f"w{world}_d{DEPTH}_{index}.json"
            fatal(path.exists(), f"missing E6A cache {path}; run E6A first")
            base_routes.append(json.loads(path.read_text(encoding="utf-8"))["value"]["route"])
            base_carries.append(index < CORPUS // 2)
        # E6A's plant record, regenerated deterministically so the motif's SITES
        # are available -- discovery is scored against the motif's learner image,
        # which only the planted sites identify (the gauge is not the teacher's).
        from row.experiments.audit_e6_corpus import plant_corpus
        motif_a, _programs_a, carries_a, sites_a = plant_corpus(
            np.random.default_rng(np.random.SeedSequence([970, world, DEPTH])),
            config.world.teacher_primitives, CORPUS, DEPTH, MACRO_LEN, CORPUS // 2)
        fatal(list(carries_a) == list(base_carries), "E6A plant record did not reproduce")
        perm = np.random.default_rng(np.random.SeedSequence([995, world])).permutation(CORPUS)
        cases = {
            "A": {"routes": [base_routes[i] for i in perm],
                  "carries": [carries_a[i] for i in perm],
                  "sites": [sites_a[i] for i in perm], "motif": list(motif_a)},
            "C": {"routes": base_routes, "carries": carries_a,
                  "sites": sites_a, "motif": list(motif_a)},
        }

        # --- case B needs its own corpus --------------------------------------
        rng = np.random.default_rng(np.random.SeedSequence([996, world]))
        motif_b, programs_b, carries_b, sites_b = decaying_corpus(
            rng, config.world.teacher_primitives, CORPUS, DEPTH, MACRO_LEN)
        routes_b = infer(cell, world, "decay", programs_b, teacher_library)
        cases["B"] = {"routes": routes_b, "carries": carries_b,
                      "sites": sites_b, "motif": list(motif_b)}

        entry = {}
        for name in ("A", "B", "C"):
            routes = cases[name]["routes"]
            observed, future = routes[:SPLIT], routes[SPLIT:]
            tally = enumerate_candidates(observed, MACRO_LEN)
            fatal(len(tally) > 0, f"world {world} case {name}: no candidate grams")

            # The motif's learner image: the gram the learner actually writes
            # where the motif was planted. Discovery is scored against THIS, not
            # against the teacher's symbols.
            at_site = Counter()
            for route, carry, site in zip(observed, cases[name]["carries"][:SPLIT],
                                          cases[name]["sites"][:SPLIT]):
                if carry and site >= 0:
                    at_site[tuple(route[site:site + MACRO_LEN])] += 1
            image = list(at_site.most_common(1)[0][0]) if at_site else None
            image_share = (at_site.most_common(1)[0][1] / sum(at_site.values())
                           if at_site else None)

            picks = {}
            for criterion in ("RETRO", "PROSP_RATE", "PROSP_TREND"):
                scored = {g: score(observed, g, slots, DEPTH, criterion) for g in tally}
                best = max(scored, key=lambda g: scored[g]["net_saving_bits"])
                realized = code_lengths(future, best, slots)
                picks[criterion] = {
                    "macro": list(best), "decision": scored[best],
                    "recovers_motif_image": bool(image is not None and list(best) == image),
                    "future_uses": realized["uses"],
                    "future_net_saving_bits": realized["saving"],
                    "was_right": bool(scored[best]["create"] == (realized["saving"] > 0)),
                }

            # Detectability, measured rather than assumed: how much does the
            # chosen gram thin across the OBSERVED window?
            chosen = tuple(picks["RETRO"]["macro"])
            early = sum(substitute(r, chosen)[1] for r in observed[:SPLIT - LATE])
            late = sum(substitute(r, chosen)[1] for r in observed[-LATE:])
            entry[name] = {
                "candidates_enumerated": len(tally),
                "motif_learner_image": image, "image_share_at_planted_sites": image_share,
                "observed_early_uses": early, "observed_late_uses": late,
                "within_observed_ratio": (late / early) if early else None,
                "picks": picks,
                "planted_motif": cases[name]["motif"],
                "carries_observed": int(sum(cases[name]["carries"][:SPLIT])),
                "carries_future": int(sum(cases[name]["carries"][SPLIT:])),
            }
            r = entry[name]
            shown = (f"image {image} ({image_share:.0%} of planted sites)"
                     if image else "no planted sites in the observed half")
            ratio = (f" (ratio {r['within_observed_ratio']:.2f})"
                     if r["within_observed_ratio"] else "")
            print(f"[w{world} {name}] {len(tally)} candidates | {shown} | "
                  f"observed uses early {early} late {late}{ratio}")
            for criterion in ("RETRO", "PROSP_RATE", "PROSP_TREND"):
                p = picks[criterion]
                print(f"    {criterion:<12} picks {str(p['macro']):>14} -> "
                      f"{'CREATE' if p['decision']['create'] else 'REFUSE':>6} | "
                      f"future uses {p['future_uses']:>3}, "
                      f"future {p['future_net_saving_bits']:+8.1f} bits | "
                      f"{'right' if p['was_right'] else 'WRONG'}"
                      f"{' | found motif image' if p['recovers_motif_image'] else ''}",
                      flush=True)
        out["worlds"][str(world)] = entry
        write(out, args.output)

    def ok(w, case, criterion, want_create):
        p = out["worlds"][str(w)][case]["picks"][criterion]
        return p["decision"]["create"] == want_create

    retro_fails_b = sum(1 for w in WORLDS if ok(w, "B", "RETRO", True))
    prosp_a = sum(1 for w in WORLDS if ok(w, "A", "PROSP_RATE", True))
    prosp_b = sum(1 for w in WORLDS if ok(w, "B", "PROSP_RATE", False))
    trend_a = sum(1 for w in WORLDS if ok(w, "A", "PROSP_TREND", True))
    trend_b = sum(1 for w in WORLDS if ok(w, "B", "PROSP_TREND", False))
    if prosp_a >= 2 and prosp_b >= 2 and retro_fails_b >= 2:
        verdict = "PROSPECTIVE CRITERION DEMONSTRATED"
    elif prosp_b < 2 and retro_fails_b >= 2:
        verdict = "NO IMPROVEMENT"
    else:
        verdict = "INCONCLUSIVE"
    out["verdict"] = {"registered_estimator": "PROSP_RATE",
                      "trend_creates_on_A": trend_a, "trend_refuses_on_B": trend_b,
                      "trend_note": "added after Amendment 4 was frozen; secondary",
                      "prospective_creates_on_A": prosp_a,
                      "prospective_refuses_on_B": prosp_b,
                      "retro_creates_on_B": retro_fails_b,
                      "verdict": verdict,
                      "C_note": "limit case; reported, never counted"}
    write(out, args.output)
    print(f"\nA create (PROSPECTIVE) {prosp_a}/3 | B refuse (PROSPECTIVE) {prosp_b}/3 | "
          f"B create (RETRO, the failure being improved on) {retro_fails_b}/3")
    print(f"E6E (registered estimator): {verdict}")
    print(f"secondary trend estimator: A create {trend_a}/3, B refuse {trend_b}/3")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
