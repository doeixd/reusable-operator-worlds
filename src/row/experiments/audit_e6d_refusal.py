"""E6D: does the macro economics REFUSE, or does it just compress?

`E6_MACRO_PLAN.md`, frozen at Amendment 3. E6A showed that representing repeated
syntax by a symbol saves bits — which is close to classical compression. E6D asks
whether the same accounting is a DECISION RULE: does it CREATE when creation pays
and REFUSE when it does not?

Four controls. Amendment 3 classifies them, and the classification is enforced
here rather than left to the reader:

    1  below-crossing   IMPLEMENTATION CHECK  (H_eff < H* => saving < 0 by
                                               definition of H*)
    4  sham alias       IMPLEMENTATION CHECK  (shortens nothing => saving = -tax)
    2  accidental       EVIDENCE              (retrospective rule vs a pattern
                                               that does not continue)
    3  wrong grouping   EVIDENCE              (true macro vs plausible rivals)

The verdict rests on 2 and 3 ONLY. Controls 1 and 4 would catch a sign or
accounting error and are reported as what they are.

No new compute: every control is arithmetic over E6A's cached `D = 6` corpus.
`plant_corpus` places the motif in tasks 0..63 and not in 64..127, so the
accidental-pattern structure is already present in that corpus.

Fails closed; atomic report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams
from row.experiments.audit_e6a_macro_economics import (
    code_lengths, predicted_crossing, substitute)

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
CORPUS = 128
SPLIT = 64
E6A_CACHE = Path("reports/e6a_cache")


def load_routes(world: int) -> list[list[int]]:
    routes = []
    for index in range(CORPUS):
        path = E6A_CACHE / f"w{world}_d{DEPTH}_{index}.json"
        fatal(path.exists(), f"missing E6A route cache {path}; run E6A first")
        routes.append(json.loads(path.read_text(encoding="utf-8"))["value"]["route"])
    return routes


def top_gram(routes, length: int):
    tally = Counter()
    for route in routes:
        for gram in ngrams(route, length):
            tally[gram] += 1
    return tally.most_common(1)[0][0], tally


def decide(routes, macro, slots: int, depth: int) -> dict:
    """The registered creation rule: CREATE iff the two-part code gets shorter."""
    lengths = code_lengths(routes, macro, slots)
    star = predicted_crossing(len(macro), depth, len(routes), slots)
    return {"uses": lengths["uses"], "H_star": star,
            "net_saving_bits": lengths["saving"],
            "create": bool(lengths["saving"] > 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6d_refusal.json"))
    args = parser.parse_args()
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E6_MACRO_PLAN.md (Amendment 3)", "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN, "corpus": CORPUS,
                        "split": SPLIT,
                        "evidence_controls": ["accidental", "wrong_grouping"],
                        "implementation_checks": ["below_crossing", "sham_alias"],
                        "note": "verdict rests on the evidence controls only"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        slots = cell["shipped"].operator_slots
        routes = load_routes(world)
        macro, tally = top_gram(routes, MACRO_LEN)
        entry = {"macro": list(macro)}

        # ---- control 1: below crossing (IMPLEMENTATION CHECK) ----------------
        # A corpus holding too few uses of the macro. Built by taking macro-free
        # routes and adding bearing ones until just BELOW the crossing.
        bearing = [r for r in routes if substitute(r, macro)[1] > 0]
        plain = [r for r in routes if substitute(r, macro)[1] == 0]
        fatal(len(plain) >= 32, f"world {world}: only {len(plain)} macro-free routes")
        below = None
        for h in range(0, min(len(bearing), 32) + 1):
            corpus = bearing[:h] + plain[:32 - h] if h <= 32 else None
            if corpus is None:
                break
            d = decide(corpus, macro, slots, DEPTH)
            if not d["create"]:
                below = d
            else:
                break
        fatal(below is not None, "no below-crossing corpus exists; control 1 unbuildable")
        entry["control_1_below_crossing"] = {
            "kind": "implementation check",
            "uses": below["uses"], "H_star": below["H_star"],
            "net_saving_bits": below["net_saving_bits"],
            "refused": not below["create"], "passes": not below["create"]}

        # ---- control 4: sham alias (IMPLEMENTATION CHECK) -------------------
        # A symbol defined over a gram that does not occur: it shortens nothing
        # and still pays the alphabet tax.
        absent = None
        for a in range(slots):
            for b in range(slots):
                for c in range(slots):
                    if tally.get((a, b, c), 0) == 0:
                        absent = (a, b, c)
                        break
                if absent: break
            if absent: break
        fatal(absent is not None, "every trigram occurs; no sham alias available")
        sham = decide(routes, absent, slots, DEPTH)
        entry["control_4_sham_alias"] = {
            "kind": "implementation check", "alias": list(absent),
            "uses": sham["uses"], "net_saving_bits": sham["net_saving_bits"],
            "refused": not sham["create"], "passes": not sham["create"]}

        # ---- control 2: accidental pattern (EVIDENCE) -----------------------
        # STRUCTURED split: the motif is present in the observed half and absent
        # from the future half, so a retrospective rule is being asked to notice
        # that a real historical regularity does not continue.
        obs_s, fut_s = routes[:SPLIT], routes[SPLIT:]
        macro_s, _ = top_gram(obs_s, MACRO_LEN)
        d_obs_s = decide(obs_s, macro_s, slots, DEPTH)
        d_fut_s = decide(fut_s, macro_s, slots, DEPTH)
        # RANDOM split: the same pattern continues, so creating is correct. This
        # is the paired positive case; without it a refusal everywhere would look
        # like success.
        rng = np.random.default_rng(np.random.SeedSequence([990, world]))
        order = rng.permutation(CORPUS)
        obs_r = [routes[i] for i in order[:SPLIT]]
        fut_r = [routes[i] for i in order[SPLIT:]]
        macro_r, _ = top_gram(obs_r, MACRO_LEN)
        d_obs_r = decide(obs_r, macro_r, slots, DEPTH)
        d_fut_r = decide(fut_r, macro_r, slots, DEPTH)
        # The rule passes only if it distinguishes the two: create on the random
        # split (and be right), and NOT create-and-be-wrong on the structured one.
        fooled = bool(d_obs_s["create"] and d_fut_s["net_saving_bits"] <= 0)
        correct_positive = bool(d_obs_r["create"] and d_fut_r["net_saving_bits"] > 0)
        entry["control_2_accidental"] = {
            "kind": "evidence",
            "structured": {"macro": list(macro_s),
                           "observed": d_obs_s, "future": d_fut_s},
            "random": {"macro": list(macro_r),
                       "observed": d_obs_r, "future": d_fut_r},
            "fooled_by_accidental_pattern": fooled,
            "correct_on_continuing_pattern": correct_positive,
            "passes": bool(correct_positive and not fooled)}

        # ---- control 3: wrong grouping (EVIDENCE) ---------------------------
        a, b, c = macro
        competitors: dict[str, tuple] = {
            "nested_prefix": (a, b),
            "nested_suffix": (b, c),
            "permuted_bac": (b, a, c),
            "permuted_cba": (c, b, a),
            "non_adjacent_ac": (a, c),
        }
        rng2 = np.random.default_rng(np.random.SeedSequence([991, world]))
        drawn = 0
        while drawn < 3:
            candidate = tuple(int(v) for v in rng2.integers(0, slots, MACRO_LEN))
            # A random draw that lands ON the true macro would be scored against
            # itself and fail the control spuriously.
            if candidate == tuple(macro) or candidate in competitors.values():
                continue
            competitors[f"random_{drawn}"] = candidate
            drawn += 1
        # A competitor that COINCIDES with the true macro cannot be a rival: when
        # a macro repeats a symbol (world 1's (8,8,5)), the b-a-c permutation is
        # the macro itself, and scoring it as a rival fails the control against
        # its own object. Degenerate competitors are dropped and recorded.
        degenerate = [n for n, g in competitors.items() if tuple(g) == tuple(macro)]
        for name in degenerate:
            del competitors[name]
        true_score = decide(routes, macro, slots, DEPTH)["net_saving_bits"]
        rival = {name: decide(routes, g, slots, DEPTH)["net_saving_bits"]
                 for name, g in competitors.items()}
        beaten = {name: bool(true_score > score) for name, score in rival.items()}
        entry["control_3_wrong_grouping"] = {
            "kind": "evidence",
            "true_macro": list(macro), "true_net_saving_bits": true_score,
            "competitors": {name: {"gram": list(g), "net_saving_bits": rival[name],
                                   "beaten": beaten[name]}
                            for name, g in competitors.items()},
            "degenerate_competitors_dropped": degenerate,
            "beats_all": bool(all(beaten.values())),
            "beaten_by": [n for n, ok in beaten.items() if not ok],
            "passes": bool(all(beaten.values()))}

        out["worlds"][str(world)] = entry
        c1 = entry["control_1_below_crossing"]; c4 = entry["control_4_sham_alias"]
        c2 = entry["control_2_accidental"]; c3 = entry["control_3_wrong_grouping"]
        print(f"[w{world}] macro {macro}")
        print(f"   1 below-crossing  {'REFUSED' if c1['passes'] else 'CREATED':>8}  "
              f"(uses {c1['uses']}, H* {c1['H_star']:.1f}, {c1['net_saving_bits']:+.1f} bits)"
              "   [implementation check]")
        print(f"   4 sham alias      {'REFUSED' if c4['passes'] else 'CREATED':>8}  "
              f"({c4['net_saving_bits']:+.1f} bits)   [implementation check]")
        print(f"   2 accidental      {'PASS' if c2['passes'] else 'FAIL':>8}  "
              f"fooled={c2['fooled_by_accidental_pattern']} "
              f"correct-on-continuing={c2['correct_on_continuing_pattern']} | "
              f"structured future {c2['structured']['future']['net_saving_bits']:+.1f} bits, "
              f"random future {c2['random']['future']['net_saving_bits']:+.1f} bits")
        print(f"   3 wrong grouping  {'PASS' if c3['passes'] else 'FAIL':>8}  "
              f"true {true_score:+.1f} bits"
              + (f", beaten by {c3['beaten_by']}" if c3['beaten_by'] else ", beats all"),
              flush=True)
        write(out, args.output)

    ev2 = sum(1 for w in WORLDS if out["worlds"][str(w)]["control_2_accidental"]["passes"])
    ev3 = sum(1 for w in WORLDS if out["worlds"][str(w)]["control_3_wrong_grouping"]["passes"])
    checks = all(out["worlds"][str(w)][k]["passes"] for w in WORLDS
                 for k in ("control_1_below_crossing", "control_4_sham_alias"))
    if ev3 >= 2 and ev2 >= 2:
        verdict = "CREATION CRITERION DEMONSTRATED"
    elif ev3 >= 2:
        verdict = "RETROSPECTIVE ECONOMICS INSUFFICIENT"
    else:
        verdict = "NO CREATION CRITERION"
    out["verdict"] = {"accidental_worlds": ev2, "wrong_grouping_worlds": ev3,
                      "implementation_checks_all_pass": checks, "verdict": verdict}
    write(out, args.output)
    print(f"\nevidence: accidental {ev2}/3, wrong-grouping {ev3}/3 | "
          f"implementation checks {'all pass' if checks else 'FAILED'}")
    print(f"E6D: {verdict}")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
