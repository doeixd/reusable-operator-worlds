"""E2-feas: is the compositional test constructible at all? (E0_PHASE0_AUDIT_PLAN.md)

Pure combinatorics on the teacher program space -- no model, no artifact. The
horizon-grid rule one level up: verify that E2's held-out strata EXIST at usable
size under E2's own coverage constraints before a world is designed around them.

For K primitives and depth D there are K^D programs. A lifetime trains on at
most `budget` of them, subject to:

  * every primitive appears;
  * every primitive appears in every position;
  * every primitive appears in at least `c_min` distinct surrounding contexts;
  * balanced frequencies (max primitive count <= 2x min primitive count).

Held-out strata:

  H1  unseen triple, every adjacent pair of which was seen in training
  H2  unseen triple with at least one adjacent pair never seen in training

Registered thresholds: E2 is CONSTRUCTIBLE iff |H1| >= 16 and |H2| >= 16 are
simultaneously achievable with a training set of at most 64 programs.

Search is a seeded greedy construction plus local repair, run from several
starts; it reports the best found, which is a LOWER bound on what is achievable.
A negative therefore means "not found by this search", and the report says so.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

MIN_H1 = 16
MIN_H2 = 16


def pairs_of(program: tuple[int, ...]) -> set[tuple[int, int, int]]:
    """Adjacent pairs, tagged by their position boundary."""
    return {(i, program[i], program[i + 1]) for i in range(len(program) - 1)}


def coverage_ok(train: list[tuple[int, ...]], k: int, d: int, c_min: int) -> dict:
    counts = Counter(p for program in train for p in program)
    by_position = [Counter(program[i] for program in train) for i in range(d)]
    contexts: dict[int, set] = {p: set() for p in range(k)}
    for program in train:
        for i, p in enumerate(program):
            contexts[p].add((i, program[:i], program[i + 1:]))
    missing_positions = [
        (p, i) for p in range(k) for i in range(d) if by_position[i][p] == 0
    ]
    thin = [p for p in range(k) if len(contexts[p]) < c_min]
    balance = (max(counts.values()) / min(counts.values())) if counts and min(counts.values()) else float("inf")
    return {
        "all_primitives": len(counts) == k,
        "every_position": not missing_positions,
        "context_min": min(len(v) for v in contexts.values()) if contexts else 0,
        "context_ok": not thin,
        "balance_ratio": balance,
        "balance_ok": balance <= 2.0,
        "missing_positions": missing_positions[:5],
        "thin_primitives": thin[:5],
    }


def strata(train: list[tuple[int, ...]], all_programs: list[tuple[int, ...]]) -> tuple[list, list]:
    seen_pairs = set()
    for program in train:
        seen_pairs |= pairs_of(program)
    train_set = set(train)
    h1, h2 = [], []
    for program in all_programs:
        if program in train_set:
            continue
        (h1 if pairs_of(program) <= seen_pairs else h2).append(program)
    return h1, h2


def build(k: int, d: int, budget: int, c_min: int, seed: int, fill: str = "min_pairs") -> dict:
    """Greedy: satisfy coverage first, then ADD programs that keep H2 large.

    H2 shrinks as training covers more adjacent pairs, so the construction must
    stop covering pairs once coverage is met -- the tension the gate is testing.
    """
    rng = np.random.default_rng(np.random.SeedSequence([760, seed]))
    all_programs = [tuple(p) for p in itertools.product(range(k), repeat=d)]
    order = list(rng.permutation(len(all_programs)))
    train: list[tuple[int, ...]] = []
    # Stage 1: minimal coverage of primitive x position, cheaply.
    need = {(p, i) for p in range(k) for i in range(d)}
    for index in order:
        if not need:
            break
        program = all_programs[index]
        gain = {(p, i) for i, p in enumerate(program)} & need
        if gain:
            train.append(program)
            need -= gain
    # Stage 2: fill to EXACTLY `budget` programs (Amendment 1: the training set
    # is the lifetime, not a bound on it). Once coverage is met the remaining
    # picks minimise newly covered adjacent pairs, because the adversarial
    # question is whether a lifetime CAN preserve an unseen-pair stratum.
    while len(train) < budget:
        check = coverage_ok(train, k, d, c_min)
        covered = check["context_ok"] and check["balance_ok"] and check["all_primitives"]
        if covered and fill == "random":
            remaining = [all_programs[i] for i in order if all_programs[i] not in train]
            train.append(remaining[0])
            continue
        seen_pairs = set()
        for program in train:
            seen_pairs |= pairs_of(program)
        counts = Counter(p for program in train for p in program)
        contexts: dict[int, set] = {p: set() for p in range(k)}
        for program in train:
            for i, p in enumerate(program):
                contexts[p].add((i, program[:i], program[i + 1:]))
        best, best_key = None, None
        for index in order:
            program = all_programs[index]
            if program in train:
                continue
            new_pairs = len(pairs_of(program) - seen_pairs)
            context_gain = sum(
                1 for i, p in enumerate(program)
                if (i, program[:i], program[i + 1:]) not in contexts[p] and len(contexts[p]) < c_min
            )
            rarity = -sum(counts[p] for p in program)
            key = (-context_gain, new_pairs, -rarity)
            if best_key is None or key < best_key:
                best, best_key = program, key
        if best is None:
            break
        train.append(best)
    h1, h2 = strata(train, all_programs)
    return {"seed": seed, "fill": fill, "train_size": len(train),
            "coverage": coverage_ok(train, k, d, c_min),
            "H1": len(h1), "H2": len(h2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--context-min", type=int, default=3)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("reports/e2_feasibility.json"))
    args = parser.parse_args()
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")
    runs = [build(args.k, args.depth, args.budget, args.context_min, s, fill)
            for fill in ("min_pairs", "random") for s in range(args.starts)]
    valid_all = [r for r in runs if r["coverage"]["all_primitives"] and r["coverage"]["every_position"]
                 and r["coverage"]["context_ok"] and r["coverage"]["balance_ok"]
                 and r["train_size"] == args.budget]
    valid = [r for r in valid_all if r["fill"] == "min_pairs"]
    valid_random = [r for r in valid_all if r["fill"] == "random"]
    best_h2 = max(valid, key=lambda r: r["H2"]) if valid else None
    best_h1 = max(valid, key=lambda r: r["H1"]) if valid else None
    both = [r for r in valid if r["H1"] >= MIN_H1 and r["H2"] >= MIN_H2]
    both_random = [r for r in valid_random if r["H1"] >= MIN_H1 and r["H2"] >= MIN_H2]
    best_random = max(valid_random, key=lambda r: r["H2"]) if valid_random else None
    report = {
        "frozen_plan": "E0_PHASE0_AUDIT_PLAN.md",
        "grid": {"K": args.k, "depth": args.depth, "budget": args.budget,
                 "context_min": args.context_min, "starts": args.starts,
                 "program_space": args.k ** args.depth},
        "thresholds": {"min_H1": MIN_H1, "min_H2": MIN_H2},
        "valid_constructions": {"min_pairs": len(valid), "random": len(valid_random)},
        "best_H1": best_h1, "best_H2": best_h2,
        "best_random": best_random,
        # The registered question is EXISTENCE of a valid training set, so the
        # verdict is over both fill objectives; the per-arm rows show what each
        # objective trades away (pair-minimising fill maximises H2 by collapsing
        # H1: concentrating on few pairs pushes almost every unseen triple into
        # the unseen-pair stratum).
        "constructible": bool(both or both_random),
        "constructible_designed_fill": bool(both),
        "constructible_random_fill": bool(both_random),
        "designed_schedule_required": bool(both) and not bool(both_random),
        "example_satisfying_both": (both or both_random)[0] if (both or both_random) else None,
        "note": ("greedy seeded search reports a LOWER bound; a negative means "
                 "not found by this search, not proven impossible"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"K={args.k} D={args.depth} train_size={args.budget}: "
          f"min_pairs {len(valid)}/{args.starts} valid, random {len(valid_random)}/{args.starts}")
    if best_h1:
        print(f"  designed fill: best |H1| = {best_h1['H1']}, best |H2| = {best_h2['H2']}")
    if best_random:
        print(f"  random fill:   best |H1| = {best_random['H1']}, best |H2| = {best_random['H2']}")
    print(f"  CONSTRUCTIBLE at |H1|>={MIN_H1} and |H2|>={MIN_H2}: {report['constructible']}"
          f" (random fill: {report['constructible_random_fill']}; "
          f"designed schedule required: {report['designed_schedule_required']})")


if __name__ == "__main__":
    main()
