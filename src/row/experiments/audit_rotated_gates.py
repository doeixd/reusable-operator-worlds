"""Rotated-substrate gates G1, G2 and G4 -- teacher-side, no learner, no lifetimes.

`ROTATED_SUBSTRATE_SPEC.md`, frozen at Amendment 1.

The existing substrate's operators have the form `tanh(z + small)` and contract to
a fixed point, so iterates converge and iteration count is nearly unidentifiable:
one fixed repeat count approximates any distribution of counts to NMSE <= 0.12.
The rotated family

    P(z) = Q ( z + a . U tanh(V z + b) ),   Q orthogonal

has no fixed point, so `P^k` traverses distinct states. This module measures
whether that makes a world in which fixed-length programs provably lose.

    G1 NECESSITY      min_d NMSE(P^d) >= 0.25 at the largest sigma, monotone
    G2 ACHIEVABILITY  NMSE(loop oracle) <= 0.02   IMPLEMENTATION CHECK, not evidence
    G4 BALANCE        mean k, output variance, output norm within 10% across sigma

G3 was withdrawn at Amendment 1: its denominator is exactly zero by construction.
G5 (learnability) requires lifetimes and is separately authorized.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from row.experiments.audit_e0_export import git_commit

WORLDS = (0, 1, 2)
PRIMITIVES = 6
STATE_DIM = 16
RANK = 8
ALPHA = 0.35
MU = 4.0
SIGMAS = (0.0, 0.5, 1.0, 1.5)
K_MIN, K_MAX = 2, 6
PROBE = 8192
TASKS = 12
NECESSITY = 0.25
ACHIEVABILITY = 0.02
BALANCE = 0.10


def spectral(m: np.ndarray) -> np.ndarray:
    return m / max(np.linalg.norm(m, ord=2), 1e-12)


def make_primitive(seed: int, rotated: bool):
    """`world.py`'s Primitive conventions, plus an orthogonal Q when rotated."""
    g = np.random.default_rng(seed)
    U = spectral(g.normal(size=(STATE_DIM, RANK)))
    V = spectral(g.normal(size=(RANK, STATE_DIM)))
    b = g.normal(scale=0.2, size=RANK)
    Q = np.linalg.qr(g.normal(size=(STATE_DIM, STATE_DIM)))[0] if rotated else None
    return {"U": U, "V": V, "b": b, "Q": Q}


def apply(p, z: np.ndarray) -> np.ndarray:
    inner = z + ALPHA * (np.tanh(z @ p["V"].T + p["b"]) @ p["U"].T)
    return inner @ p["Q"].T if p["Q"] is not None else np.tanh(inner)


def shaping(u: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance under a standard normal projection."""
    t = np.tanh(u)
    return (t - t.mean()) / max(t.std(), 1e-12)


def counts(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    k = np.round(MU + sigma * shaping(x @ w))
    return np.clip(k, K_MIN, K_MAX).astype(int)


def nmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred - y) ** 2) / max(np.mean(y ** 2), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/rotated_gates.json"))
    args = parser.parse_args()

    out = {"frozen_spec": "ROTATED_SUBSTRATE_SPEC.md (Amendment 1)",
           "git_commit": git_commit(),
           "protocol": {"mu": MU, "sigmas": list(SIGMAS), "k_range": [K_MIN, K_MAX],
                        "probe": PROBE, "tasks": TASKS, "alpha": ALPHA,
                        "gates": {"G1_necessity": NECESSITY,
                                  "G2_achievability": ACHIEVABILITY,
                                  "G4_balance": BALANCE},
                        "note": "G2 is an implementation check; G3 withdrawn as degenerate"},
           "families": {}}

    for rotated in (False, True):
        name = "rotated" if rotated else "current"
        per_world = {}
        for world in WORLDS:
            lib = [make_primitive(1000 * (world + 1) + i, rotated) for i in range(PRIMITIVES)]
            rng = np.random.default_rng(np.random.SeedSequence([1400, world, int(rotated)]))
            cells = {}
            for sigma in SIGMAS:
                best_fixed, loop_err, ks, out_var, out_norm = [], [], [], [], []
                for t in range(TASKS):
                    p = lib[t % PRIMITIVES]
                    w = rng.normal(size=STATE_DIM)
                    w = w / np.linalg.norm(w)
                    x = rng.normal(size=(PROBE, STATE_DIM))
                    k = counts(x, w, sigma)
                    # iterate once, keep every prefix, then gather each row's own k
                    states = {0: x.copy()}
                    z = x.copy()
                    for step in range(1, K_MAX + 1):
                        z = apply(p, z)
                        states[step] = z.copy()
                    y = np.stack([states[k[i]][i] for i in range(PROBE)])
                    best_fixed.append(min(nmse(states[d], y) for d in range(K_MIN, K_MAX + 1)))
                    # G2 must be an INDEPENDENT recomputation, not the gather that
                    # built `y` -- comparing an array to itself is not a check.
                    # Iterate each row its own k times and compare.
                    loop = np.empty_like(x)
                    for kk in range(K_MIN, K_MAX + 1):
                        rows = np.flatnonzero(k == kk)
                        if rows.size == 0:
                            continue
                        zz = x[rows]
                        for _ in range(kk):
                            zz = apply(p, zz)
                        loop[rows] = zz
                    loop_err.append(nmse(loop, y))
                    ks.append(float(k.mean()))
                    out_var.append(float(np.var(y)))
                    out_norm.append(float(np.sqrt(np.mean(y ** 2))))
                cells[str(sigma)] = {
                    "best_fixed_nmse": float(np.mean(best_fixed)),
                    "loop_oracle_nmse": float(np.mean(loop_err)),
                    "mean_k": float(np.mean(ks)),
                    "output_variance": float(np.mean(out_var)),
                    "output_norm": float(np.mean(out_norm))}
                c = cells[str(sigma)]
                print(f"[{name} w{world} sigma={sigma}] best-fixed {c['best_fixed_nmse']:.4f} "
                      f"| loop {c['loop_oracle_nmse']:.2e} | mean k {c['mean_k']:.2f} "
                      f"| out var {c['output_variance']:.4f}", flush=True)
            per_world[str(world)] = cells
        out["families"][name] = per_world
        write(out, args.output)

    def cell(fam, w, s):
        return out["families"][fam][str(w)][str(s)]

    verdict = {}
    for fam in ("current", "rotated"):
        largest = max(SIGMAS)
        g1 = sum(1 for w in WORLDS if cell(fam, w, largest)["best_fixed_nmse"] >= NECESSITY)
        mono = sum(1 for w in WORLDS
                   if all(cell(fam, w, SIGMAS[i])["best_fixed_nmse"]
                          <= cell(fam, w, SIGMAS[i + 1])["best_fixed_nmse"] + 1e-9
                          for i in range(len(SIGMAS) - 1)))
        g2 = sum(1 for w in WORLDS for s in SIGMAS
                 if cell(fam, w, s)["loop_oracle_nmse"] <= ACHIEVABILITY)
        bal = {}
        for field in ("mean_k", "output_variance", "output_norm"):
            spreads = []
            for w in WORLDS:
                vals = [cell(fam, w, s)[field] for s in SIGMAS]
                spreads.append((max(vals) - min(vals)) / max(abs(np.mean(vals)), 1e-12))
            bal[field] = float(np.mean(spreads))
        verdict[fam] = {
            "G1_worlds_passing": g1, "G1_monotone_worlds": mono,
            "G2_cells_passing": g2, "G2_cells_total": len(WORLDS) * len(SIGMAS),
            "G4_relative_spread": bal,
            "G4_passes": all(v <= BALANCE for v in bal.values()),
            "usable": bool(g1 >= 2 and mono >= 2
                           and g2 == len(WORLDS) * len(SIGMAS)
                           and all(v <= BALANCE for v in bal.values()))}
    out["verdict"] = verdict
    write(out, args.output)

    print()
    for fam in ("current", "rotated"):
        v = verdict[fam]
        print(f"{fam:>8}: G1 {v['G1_worlds_passing']}/3 (monotone {v['G1_monotone_worlds']}/3) "
              f"| G2 {v['G2_cells_passing']}/{v['G2_cells_total']} "
              f"| G4 spreads " + " ".join(f"{k.split('_')[-1]} {x:.3f}"
                                          for k, x in v["G4_relative_spread"].items())
              + f" -> {'USABLE' if v['usable'] else 'not usable'}")


def write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
