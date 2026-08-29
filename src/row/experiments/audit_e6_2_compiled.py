"""E6.2: can a recurring program fragment be compiled into a reusable operator?

`E6_2_COMPILED_MACRO_PLAN.md`, frozen at Amendment 1.

E6C asked whether a macro stays substitutable for its expansion in unseen
contexts and was retired: a DEFINITIONAL macro IS its expansion, bitwise. A
COMPILED macro is an approximation, so the question becomes empirical. We distil

    P_M(x) ~ P_c(P_b(P_a(x)))

into a single operator of the SAME architecture and parameter count as one
library slot (matched budget), fitted only on states the fragment actually sees
in the observed corpus, then substitute it in contexts graded by how far they
move its input distribution:

    C0  held-out programs, positions seen in fitting   non-vacuity check
    C1  new immediate neighbours
    C2  new prefixes and suffixes
    C3  positions the fragment never occupied
    C4  program depths the fragment never saw

The evidence lives in C2-C4: on-distribution agreement is near-guaranteed for a
smooth composition at matched capacity, so C0 is a check that the fit works, not
a result. An untrained `P_M` must fail. Fails closed; cached; atomic report.
"""
from __future__ import annotations

import argparse
import copy
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
from row.experiments.audit_e3_program_economy import (
    behavioural_rate, operator_scalars, quantize_operator)
from row.experiments.audit_e5_synthesizer import execute, fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams
from row.experiments.audit_e6a_macro_economics import predicted_crossing, substitute
from row.models.torch_oracle import LearnedOperator

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
CORPUS = 128
SPLIT = 64
FIT_STATES = 2048
FIT_STEPS = 3000
FIT_LR = 3e-3
PROBES = 256
PROGRAMS_PER_CLASS = 32
TOLERANCE = 0.15            # E2's log tolerance
CACHE = Path("reports/e6_2_cache")
E6A_CACHE = Path("reports/e6a_cache")


def fingerprint() -> str:
    payload = {"depth": DEPTH, "L": MACRO_LEN, "fit_states": FIT_STATES,
               "fit_steps": FIT_STEPS, "fit_lr": FIT_LR, "probes": PROBES,
               "per_class": PROGRAMS_PER_CLASS, "tolerance": TOLERANCE}
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


def agreement_nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """How far the substituted program's output is from the expansion's.

    Substitutability is agreement WITH THE EXPANSION -- that is what the macro is
    defined to reproduce -- so the expansion supplies the reference.
    """
    return float(torch.mean((a - b) ** 2) / torch.mean(b ** 2))


@torch.no_grad()
def run_program(library, x, program, macro=None, replacement=None):
    """Execute a symbol sequence; `macro` occurrences use `replacement` if given."""
    z, i, L = x, 0, MACRO_LEN
    while i < len(program):
        if replacement is not None and tuple(program[i:i + L]) == tuple(macro):
            z = replacement(z)
            i += L
        else:
            z = library[program[i]](z)
            i += 1
    return z


@torch.no_grad()
def states_before(library, x, program, position):
    z = x
    for step in range(position):
        z = library[program[step]](z)
    return z


def fit_states(library, routes, macro, rng, d: int):
    """Input states the fragment actually sees in the OBSERVED corpus."""
    sites = []
    for route in routes:
        for j in range(len(route) - MACRO_LEN + 1):
            if tuple(route[j:j + MACRO_LEN]) == tuple(macro):
                sites.append((route, j))
    fatal(len(sites) > 0, "macro does not occur in the observed corpus")
    per = max(1, FIT_STATES // len(sites))
    states = []
    for route, j in sites:
        x = torch.tensor(rng.normal(size=(per, d)), dtype=torch.float32)
        states.append(states_before(library, x, route, j))
    positions = sorted({j for _, j in sites})
    return torch.cat(states, dim=0), positions


def distil(library, macro, states, d: int, rank: int, alpha: float,
           activation: str, seed: int):
    """One operator, matched to a library slot, fitted to the composition."""
    with torch.no_grad():
        target = states
        for s in macro:
            target = library[s](target)
    model = LearnedOperator(d, rank, alpha, seed, learnable_alpha=True,
                            activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=FIT_LR)
    first = last = None
    for step in range(FIT_STEPS):
        optimizer.zero_grad()
        loss = torch.mean((model(states) - target) ** 2)
        if not bool(torch.isfinite(loss)):
            raise SystemExit("non-finite distillation loss")
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    return model, {"first_loss": first, "last_loss": last,
                   "scalars": operator_scalars(model)}


MAX_ATTEMPTS = 4000


def make_programs(rng, slots: int, macro, cls: str, seen_positions, depth: int,
                  seen_neighbours):
    """Programs containing the macro, in contexts graded by distribution shift.

    Amendment 2: C3 ("positions never occupied") is EMPTY at depth 6 -- a length-3
    fragment has only four possible starts and the macro occupies all of them --
    so C3 is built in LONGER programs at start index >= 4, which confounds
    position with depth. The confound is disclosed, and every class reports
    whether it was constructible instead of spinning forever looking.
    """
    out = []
    L = MACRO_LEN
    attempts = 0
    while len(out) < PROGRAMS_PER_CLASS and attempts < MAX_ATTEMPTS:
        attempts += 1
        if cls == "C4":
            d = depth + int(rng.integers(2, 5))          # depths never seen
        elif cls == "C3":
            d = depth + int(rng.integers(1, 4))          # long enough for a late start
        else:
            d = depth
        span = d - L
        if span < 1:
            continue
        if cls in ("C0", "C1", "C2"):
            choices = [p for p in seen_positions if p <= span]
            if not choices:
                continue
            j = int(rng.choice(choices))
        else:                                            # C3, C4: unseen positions
            choices = [p for p in range(span + 1) if p not in seen_positions]
            if not choices:
                continue
            j = int(rng.choice(choices))
        program = [int(v) for v in rng.integers(0, slots, d)]
        program[j:j + L] = list(macro)
        if cls == "C1":
            # force at least one immediate neighbour never seen adjacent
            if j > 0:
                unseen = [s for s in range(slots) if (s, "left") not in seen_neighbours]
                if unseen:
                    program[j - 1] = int(rng.choice(unseen))
            if j + L < d:
                unseen = [s for s in range(slots) if (s, "right") not in seen_neighbours]
                if unseen:
                    program[j + L] = int(rng.choice(unseen))
        if tuple(program[j:j + L]) != tuple(macro):
            continue
        out.append((tuple(program), j))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e6_2_compiled.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if subprocess.run(["python", "tools/check_prereg.py"]).returncode != 0:
        raise SystemExit("prereg check failed")

    out = {"frozen_plan": "E6_2_COMPILED_MACRO_PLAN.md (Amendment 1)",
           "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN,
                        "fit_states": FIT_STATES, "fit_steps": FIT_STEPS,
                        "tolerance": TOLERANCE, "per_class": PROGRAMS_PER_CLASS,
                        "reference": "agreement with the expansion",
                        "note": "C0 is a non-vacuity check; evidence is C2-C4"},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        library = cell["shipped"].library
        slots = cell["shipped"].operator_slots
        d = config.world.state_dim

        routes = []
        for index in range(CORPUS):
            path = E6A_CACHE / f"w{world}_d{DEPTH}_{index}.json"
            fatal(path.exists(), f"missing E6A cache {path}; run E6A first")
            routes.append(json.loads(path.read_text(encoding="utf-8"))["value"]["route"])
        observed = routes[:SPLIT]
        tally = Counter()
        for route in observed:
            for gram in ngrams(route, MACRO_LEN):
                tally[gram] += 1
        macro = tally.most_common(1)[0][0]

        rng = np.random.default_rng(np.random.SeedSequence([1001, world]))
        states, positions = fit_states(library, observed, macro, rng, d)
        neighbours = set()
        for route in observed:
            for j in range(len(route) - MACRO_LEN + 1):
                if tuple(route[j:j + MACRO_LEN]) == tuple(macro):
                    if j > 0:
                        neighbours.add((route[j - 1], "left"))
                    if j + MACRO_LEN < len(route):
                        neighbours.add((route[j + MACRO_LEN], "right"))

        # P_M is built from the SAME hyperparameters as a library slot, read off
        # the frozen config rather than hard-coded, so "matched budget" is a fact
        # about the artifact and not an assumption.
        proto = library[0]
        rank = config.discrete_model.operator_rank
        alpha = config.discrete_model.operator_alpha_init
        activation = config.discrete_model.operator_activation

        trained, fitinfo = distil(library, macro, states, d, rank, alpha,
                                  activation, seed=4200 + world)
        fatal(fitinfo["last_loss"] < fitinfo["first_loss"],
              "distillation loss did not fall")
        untrained = LearnedOperator(d, rank, alpha, 9900 + world,
                                    learnable_alpha=True, activation=activation)

        entry = {"macro": list(macro), "fit": fitinfo,
                 "seen_positions": positions,
                 "matched_budget": {"P_M_scalars": operator_scalars(trained),
                                    "library_slot_scalars": operator_scalars(proto)},
                 "classes": {}}
        fatal(operator_scalars(trained) == operator_scalars(proto),
              "P_M is not matched to a library slot")

        fit_mean = states.mean(dim=0)
        fit_std = states.std(dim=0).mean()

        for cls in ("C0", "C1", "C2", "C3", "C4"):
            progs = make_programs(rng, slots, macro, cls, positions, DEPTH, neighbours)
            if len(progs) < PROGRAMS_PER_CLASS:
                entry["classes"][cls] = {
                    "constructible": False, "programs": len(progs),
                    "reason": "no admissible context exists for this class",
                    "within_tolerance": None, "untrained_fails": None}
                print(f"[w{world} {cls}] UNCONSTRUCTIBLE "
                      f"({len(progs)}/{PROGRAMS_PER_CLASS} programs) -- excluded",
                      flush=True)
                continue
            gaps, wrong, shifts = [], [], []
            for program, j in progs:
                x = torch.tensor(rng.normal(size=(PROBES, d)), dtype=torch.float32)
                ref = run_program(library, x, program)
                got = run_program(library, x, program, macro, trained)
                bad = run_program(library, x, program, macro, untrained)
                gaps.append(agreement_nmse(got, ref))
                wrong.append(agreement_nmse(bad, ref))
                seen = states_before(library, x, program, j)
                shifts.append(float(torch.norm(seen.mean(dim=0) - fit_mean) /
                                    (torch.norm(fit_mean) + 1e-12)))
            g = float(np.exp(np.mean(np.log(np.maximum(gaps, 1e-12)))))
            w = float(np.exp(np.mean(np.log(np.maximum(wrong, 1e-12)))))
            entry["classes"][cls] = {
                "constructible": True,
                "programs": len(progs), "substitution_nmse": g,
                "untrained_nmse": w, "input_shift": float(np.mean(shifts)),
                "within_tolerance": bool(g <= TOLERANCE),
                "untrained_fails": bool(w > TOLERANCE)}
            r = entry["classes"][cls]
            print(f"[w{world} {cls}] substitution NMSE {g:.5f} "
                  f"({'within' if r['within_tolerance'] else 'OUTSIDE'} {TOLERANCE}) | "
                  f"untrained {w:.4f} ({'fails' if r['untrained_fails'] else 'PASSES?!'}) | "
                  f"input shift {r['input_shift']:.3f}", flush=True)

        # --- economics: D*(P_M) measured behaviourally, bits only (Amendment 1)
        probe_progs = make_programs(rng, slots, macro, "C0", positions, DEPTH, neighbours)
        x = torch.tensor(rng.normal(size=(PROBES, d)), dtype=torch.float32)
        refs = [run_program(library, x, p) for p, _ in probe_progs]
        base = float(np.mean([agreement_nmse(run_program(library, x, p, macro, trained), r)
                              for (p, _), r in zip(probe_progs, refs)]))

        def evaluate(bits: int) -> float:
            q = quantize_operator(trained, bits)
            got = float(np.mean([agreement_nmse(run_program(library, x, p, macro, q), r)
                                 for (p, _), r in zip(probe_progs, refs)]))
            return got / max(base, 1e-12)

        rate = behavioural_rate(evaluate)
        d_star = rate["bits_per_scalar"] * operator_scalars(trained)
        a, b = math.log2(slots), math.log2(slots + 1)
        crossing = (d_star + SPLIT * DEPTH * (b - a)) / ((MACRO_LEN - 1) * b)
        realized = sum(substitute(r, macro)[1] for r in observed)
        entry["economics"] = {
            "D_star_bits_per_scalar": rate["bits_per_scalar"],
            "D_star_bits": d_star, "saturated": rate["saturated"],
            "definitional_crossing": predicted_crossing(MACRO_LEN, DEPTH, SPLIT, slots),
            "compiled_crossing": crossing,
            "realized_uses": realized,
            "pays": bool(realized >= crossing),
            "execution_saving_per_use_ops": MACRO_LEN - 1,
            "units_note": "crossing in description bits only; execution saving "
                          "reported separately and never summed with bits"}
        e = entry["economics"]
        print(f"      economics: D*(P_M) {d_star:.0f} bits ({rate['bits_per_scalar']:.2f}"
              f"/scalar over {operator_scalars(trained)}) | crossing {crossing:.1f} uses "
              f"vs definitional {e['definitional_crossing']:.1f} | realized {realized} -> "
              f"{'PAYS' if e['pays'] else 'DOES NOT PAY'}", flush=True)
        out["worlds"][str(world)] = entry
        write(out, args.output)

    def passes(cls):
        return sum(1 for w in WORLDS
                   if out["worlds"][str(w)]["classes"][cls].get("within_tolerance") is True)

    c0 = passes("C0")
    control = sum(1 for w in WORLDS
                  if all(out["worlds"][str(w)]["classes"][c].get("untrained_fails") is True
                         for c in ("C0", "C1", "C2", "C3", "C4")
                         if out["worlds"][str(w)]["classes"][c].get("constructible")))
    if c0 < 2 or control < 2:
        verdict = "UNSCOREABLE (non-vacuity failed)"
    elif passes("C3") >= 2 and passes("C2") >= 2 and passes("C1") >= 2:
        verdict = "COMPILATION HOLDS"
    elif passes("C1") >= 2:
        verdict = "COMPILATION IS CONTEXT-BOUND"
    else:
        verdict = "COMPILATION FAILS"
    pays = sum(1 for w in WORLDS if out["worlds"][str(w)]["economics"]["pays"])
    out["verdict"] = {"per_class": {c: passes(c) for c in ("C0", "C1", "C2", "C3", "C4")},
                      "untrained_control_worlds": control,
                      "semantic": verdict,
                      "economics": "COMPILATION PAYS" if pays >= 2 else "COMPILATION DOES NOT PAY",
                      "economics_worlds": pays}
    write(out, args.output)
    print("\nper class within tolerance: " +
          "  ".join(f"{c} {passes(c)}/3" for c in ("C0", "C1", "C2", "C3", "C4")))
    print(f"untrained control fails everywhere in {control}/3 worlds")
    print(f"E6.2 semantic: {verdict}")
    print(f"E6.2 economics: {out['verdict']['economics']} ({pays}/3 worlds)")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
