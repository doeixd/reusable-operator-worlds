"""E7 census: are the NON-MATCHING recurrences a low-dimensional family?

`notes/e7-sketch.txt`, section 3. This is a CENSUS, not a scored rung: it decides
whether a parameterized-macro plan is worth writing, and it produces no verdict
about any hypothesis. Any E7 plan built on it is designed AFTER seeing this data
and must disclose that.

E6 step 0 measured that only ~36% of planted recurrences produce the SAME learner
gram. The other ~64% are the same COMPUTATION written as DIFFERENT SYMBOL
SEQUENCES. A rigid macro `M` collects only the matching ones; a parameterized
`M(alpha)` could collect the family -- IF the non-matching grams are actually a
family in function space rather than scatter.

Method, with three of this project's standing rules built in:

  * COMMON PROBE. Every gram's effective operator is evaluated at the SAME
    inputs. Two functions can only be compared at the same coordinates (the V5
    effective-operator correction).
  * LEAVE-ONE-OUT. Subspace capture is fit on all-but-one and scored on the held
    out vector. Never fit and score a shared-structure measurement on the same
    objects (V5 read 0.730 in-sample where the truth was 0.021).
  * AN UNCONTAMINATED NULL. Random grams of the same count, scored identically.
    The space of 3-step compositions is itself restricted, so "the family lies in
    a subspace" is only informative RELATIVE to what random grams do.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_e5_synthesizer import fatal, load_cell
from row.experiments.audit_e6_corpus import ngrams, plant_corpus

WORLDS = (0, 1, 2)
DEPTH = 6
MACRO_LEN = 3
CORPUS = 128
PROBE = 512
DIMS = (1, 2, 3, 4, 6, 8)
E6A_CACHE = Path("reports/e6a_cache")


@torch.no_grad()
def contribution(library, gram, probe: torch.Tensor) -> np.ndarray:
    """The fragment's own effect at COMMON inputs, flattened to one vector."""
    z = probe
    for s in gram:
        z = library[s](z)
    return (z - probe).flatten().numpy().astype(np.float64)


def loo_capture(vectors: np.ndarray, k: int) -> float:
    """Mean fraction of a held-out vector captured by a k-dim fit of the rest."""
    n = len(vectors)
    if n <= k + 1:
        return float("nan")
    scores = []
    for i in range(n):
        rest = np.delete(vectors, i, axis=0)
        centre = rest.mean(axis=0)
        centred = rest - centre
        # right singular vectors span the subspace of the remaining points
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = vt[:k]
        held = vectors[i] - centre
        norm = np.linalg.norm(held)
        if norm < 1e-12:
            continue
        projected = basis.T @ (basis @ held)
        scores.append(1.0 - np.linalg.norm(held - projected) / norm)
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/e7_census.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    out = {"kind": "CENSUS -- decides whether to write a plan; produces no verdict",
           "source": "notes/e7-sketch.txt section 3", "git_commit": git_commit(),
           "protocol": {"depth": DEPTH, "macro_len": MACRO_LEN, "probe": PROBE,
                        "dims": list(DIMS),
                        "guards": ["common probe", "leave-one-out",
                                   "uncontaminated random-gram null"]},
           "worlds": {}}

    for world in WORLDS:
        cell = load_cell(world, args.config)
        config = cell["config"]
        library = cell["shipped"].library
        slots = cell["shipped"].operator_slots

        routes = []
        for index in range(CORPUS):
            path = E6A_CACHE / f"w{world}_d{DEPTH}_{index}.json"
            fatal(path.exists(), f"missing E6A cache {path}; run E6A first")
            routes.append(json.loads(path.read_text(encoding="utf-8"))["value"]["route"])

        # Recover E6A's plant record so the PLANTED SITES are known.
        rng = np.random.default_rng(np.random.SeedSequence([970, world, DEPTH]))
        _motif, _programs, carries, sites = plant_corpus(
            rng, config.world.teacher_primitives, CORPUS, DEPTH, MACRO_LEN, CORPUS // 2)

        at_site = Counter()
        for route, carry, site in zip(routes, carries, sites):
            if carry and site >= 0:
                at_site[tuple(route[site:site + MACRO_LEN])] += 1
        fatal(len(at_site) > 0, "no planted sites recovered")
        modal, modal_count = at_site.most_common(1)[0]
        planted = sum(at_site.values())

        # The family under test: grams the learner wrote AT planted sites that
        # are NOT the modal image. These are the ~64% a rigid macro misses.
        family = [g for g in at_site if g != modal]
        if len(family) < 6:
            # Not a failure: a world whose gauge is TIGHT has few non-matching
            # grams, so there is no family to collect and the census is
            # uninformative there. Reported, not crashed on.
            out["worlds"][str(world)] = {
                "modal_gram": list(modal), "modal_count": modal_count,
                "planted_sites": planted, "modal_share": modal_count / planted,
                "distinct_non_matching_grams": len(family),
                "informative": False,
                "reason": "too few non-matching grams; the gauge is tight in this "
                          "world, so a parameterized macro would have little to collect"}
            print(f"[w{world}] modal {modal} x{modal_count}/{planted} planted sites "
                  f"({modal_count / planted:.0%}) | only {len(family)} non-matching "
                  f"grams -- UNINFORMATIVE for this census", flush=True)
            write(out, args.output)
            continue

        probe = torch.tensor(np.random.default_rng(
            np.random.SeedSequence([1100, world])).normal(size=(PROBE, config.world.state_dim)),
            dtype=torch.float32)

        fam = np.stack([contribution(library, g, probe) for g in family])
        nrng = np.random.default_rng(np.random.SeedSequence([1101, world]))
        null_grams = []
        while len(null_grams) < len(family):
            g = tuple(int(v) for v in nrng.integers(0, slots, MACRO_LEN))
            if g in at_site or g in null_grams:
                continue
            null_grams.append(g)
        nul = np.stack([contribution(library, g, probe) for g in null_grams])

        # A SECOND, HARDER NULL. Random grams control for "is this a subspace at
        # all", but the family is made of LEARNER-WRITTEN grams, which may be
        # structured relative to random ones for reasons unrelated to the planted
        # motif. The null that isolates the motif is grams the learner wrote at
        # UNPLANTED sites: same author, same corpus, no planted structure.
        unplanted = Counter()
        for route, carry in zip(routes, carries):
            if not carry:
                for g in ngrams(route, MACRO_LEN):
                    if g != modal and g not in at_site:
                        unplanted[g] += 1
        pool = [g for g in unplanted]
        lrng = np.random.default_rng(np.random.SeedSequence([1102, world]))
        if len(pool) >= len(family):
            picked = [pool[i] for i in lrng.choice(len(pool), len(family), replace=False)]
            learner_null = np.stack([contribution(library, g, probe) for g in picked])
        else:
            learner_null = None

        # Where does the MODAL image sit relative to its own near-misses? If the
        # family is real, the matching gram should be inside it.
        modal_vec = contribution(library, modal, probe)
        centre = fam.mean(axis=0)
        _, _, vt = np.linalg.svd(fam - centre, full_matrices=False)

        entry = {"modal_gram": list(modal), "modal_count": modal_count,
                 "planted_sites": planted,
                 "modal_share": modal_count / planted,
                 "distinct_non_matching_grams": len(family),
                 "informative": True,
                 "capture": {}, "modal_in_family_subspace": {}}
        for k in DIMS:
            if k >= len(family) - 1:
                continue
            f = loo_capture(fam, k)
            n = loo_capture(nul, k)
            ln = (loo_capture(learner_null, k) if learner_null is not None
                  else float("nan"))
            held = modal_vec - centre
            basis = vt[:k]
            proj = basis.T @ (basis @ held)
            inside = 1.0 - np.linalg.norm(held - proj) / max(np.linalg.norm(held), 1e-12)
            entry["capture"][str(k)] = {
                "family": f, "random_null": n,
                "learner_null": None if not np.isfinite(ln) else ln,
                "excess_over_random": (f - n) if np.isfinite(f) and np.isfinite(n) else None,
                "excess_over_learner": ((f - ln) if np.isfinite(f) and np.isfinite(ln)
                                        else None)}
            entry["modal_in_family_subspace"][str(k)] = float(inside)

        out["worlds"][str(world)] = entry
        print(f"[w{world}] modal {modal} x{modal_count}/{planted} planted sites "
              f"({entry['modal_share']:.0%}) | {len(family)} distinct non-matching grams")
        for k in sorted(entry["capture"], key=int):
            c = entry["capture"][k]
            er, el = c["excess_over_random"], c["excess_over_learner"]
            er_text = "n/a" if er is None else f"{er:+.3f}"
            el_text = "n/a" if el is None else f"{el:+.3f}"
            ln_text = "n/a" if c["learner_null"] is None else f"{c['learner_null']:.3f}"
            # The MOTIF-SPECIFIC claim needs the learner null, not the random one.
            mark = "  <-- motif-specific" if (el is not None and el > 0.15) else ""
            print(f"    dim {k:>2}: family {c['family']:.3f} | random {c['random_null']:.3f} "
                  f"({er_text}) | learner-written {ln_text} ({el_text}){mark}")
        write(out, args.output)

    write(out, args.output)
    print("\nCENSUS ONLY. No verdict is recorded from this; it decides whether an")
    print("E7 plan is worth writing, and any such plan is designed AFTER this data.")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
