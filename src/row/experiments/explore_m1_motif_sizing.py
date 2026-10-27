"""EXPLORATORY M1 sizing (no verdict; decision 23 after M0): does a LEARNED library formed on a motif stream let
macros mined from its own routes solve held-out depth-9 motif tasks that the primitive-level incumbent cannot?

One (world, stream) per call, development band 6. Stream `motif_stream.build_stream` (316 tasks). Formation =
D2's confirmed end-of-stream protocol: wake (`o2`/`learned_lifetime` construction, `dn_stream.planned_model`),
then `deep_reroute.reroute` of every stream task on its reservoir, then O3's sleep (8,192 updates). Shallow motif
tasks are scored before and after. Deep test on the FROZEN formed library, 128 support / 256 query per task:
- MAPPED: the true program with each primitive replaced by the slot its length-1 anchors route to (majority);
  the learner-side reference (it uses the teacher program, so it is not an arm a learner could run);
- MINED: 3-grams of the learner's own shallow-motif routes, count >= 3, top 12; every triple enumerated, then a
  one/two-position slot polish; RANDOM: as many random slot grams, same procedure;
- INCUMBENT: SY1's OPT (SO1R `optimize_route`, 2,000 steps) then the same polish.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import deep_reroute as dr
from row.experiments import dn_stream as dn
from row.experiments import learned_lifetime as ll
from row.experiments import motif_stream as ms
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary, optimize_route
from row.metrics import nmse

MIN_COUNT, MAX_ALPHABET = 3, 12
POLISH_CAP = 30000
OUT = Path('reports/m1_sizing')


def slot_apply(lib, z, k):
    h = z @ lib.V[k].T + lib.b[k]
    h = torch.nn.functional.gelu(h) if lib.gelu else torch.tanh(h)
    return (z + lib.alpha[k] * h @ lib.U[k].T) @ lib.Q[k]


def run_route(lib, route, x):
    z = x
    for k in route:
        z = slot_apply(lib, z, int(k))
    return z


def support_mse(lib, route, x, y):
    return float(torch.mean((run_route(lib, route, x) - y) ** 2))


def polish(lib, seq, x, y):
    seq, cur, evals = list(seq), support_mse(lib, seq, x, y), 0
    for width in (1, 2):
        improved = True
        while improved and evals < POLISH_CAP:
            improved = False
            for pos in itertools.combinations(range(len(seq)), width):
                for vals in itertools.product(range(lib.slots), repeat=width):
                    if all(seq[p] == v for p, v in zip(pos, vals)) or evals >= POLISH_CAP:
                        continue
                    cand = list(seq)
                    for p, v in zip(pos, vals):
                        cand[p] = v
                    s = support_mse(lib, cand, x, y)
                    evals += 1
                    if s < cur:
                        seq, cur, improved = cand, s, True
            if improved and width == 2:
                break
    return seq, cur, evals


def macro_arm(lib, alphabet, x, y):
    t0 = time.perf_counter()
    best = (np.inf, None)
    if alphabet:
        def rec(z, seq, k):
            nonlocal best
            if k == ms.DEEP_DEPTH // ms.MOTIF_LEN:
                v = float(torch.mean((z - y) ** 2))
                if v < best[0]:
                    best = (v, seq)
                return
            for g in alphabet:
                rec(run_route(lib, g, z), seq + list(g), k + 1)
        with torch.no_grad():
            rec(x, [], 0)
    start = best[1] if best[1] is not None else [0] * ms.DEEP_DEPTH
    enum_score = best[0]
    with torch.no_grad():
        seq, score_, evals = polish(lib, start, x, y)
    return {'enum_route': start, 'enum_support_mse': float(enum_score), 'route': seq, 'support_mse': score_,
            'polish_evaluations': evals, 'enum_evaluations': len(alphabet) ** 3, 'seconds': time.perf_counter() - t0}


def incumbent_arm(lib, x, y):
    t0 = time.perf_counter()
    lib.steps = ms.DEEP_DEPTH
    route, _, _, _ = optimize_route(lib, x, y, 2000)
    with torch.no_grad():
        opt_score = support_mse(lib, route, x, y)
        seq, score_, evals = polish(lib, route, x, y)
    return {'opt_route': route, 'opt_support_mse': opt_score, 'route': seq, 'support_mse': score_,
            'polish_evaluations': evals, 'seconds': time.perf_counter() - t0}


def deep_test(model, st, plan, shallow, deep, seed):
    lib = FrozenLibrary(model)
    probe = torch.randn(5, lib.V.shape[2], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        cand = lib.candidates(probe)
        if max(float((slot_apply(lib, probe, k) - cand[:, k]).abs().max()) for k in range(lib.slots)) > 1e-5:
            raise RuntimeError('slot_apply differs from FrozenLibrary.candidates')
    routes = model.hard_routes()
    votes = {}
    for t in st:
        if plan[t.task_id] == 1:
            votes.setdefault(t.program.primitive_ids[0], Counter())[int(routes[t.task_id][0])] += 1
    slot_of = {p: c.most_common(1)[0][0] for p, c in votes.items()}
    shallow_routes = [tuple(int(v) for v in routes[t.task_id][:plan[t.task_id]]) for t in shallow]
    mapped_shallow = [tuple(slot_of[p] for p in t.program.primitive_ids) for t in shallow]
    counts = Counter(r[j:j + 3] for r in shallow_routes for j in range(len(r) - 2))
    alphabet = sorted((g for g, c in counts.items() if c >= MIN_COUNT), key=lambda g: (-counts[g], g))[:MAX_ALPHABET]
    motif_images = [tuple(slot_of[p] for p in m) for m in ms.motifs_of(seed[0], 6)]
    rng = np.random.default_rng(np.random.SeedSequence([6405, *seed]))
    rand = []
    while len(rand) < len(alphabet):
        g = tuple(int(v) for v in rng.integers(0, lib.slots, 3))
        if len(set(g)) > 1 and g not in rand:
            rand.append(g)
    rows = []
    for t in deep:
        x = torch.tensor(t.train_x, dtype=torch.float32)
        y = torch.tensor(t.train_y, dtype=torch.float32)
        xq = torch.tensor(t.eval_x, dtype=torch.float32)
        mapped = [slot_of[p] for p in t.program.primitive_ids]
        row = {'program': list(t.program.primitive_ids), 'mapped': mapped}
        with torch.no_grad():
            row['MAPPED'] = {'route': mapped, 'support_mse': support_mse(lib, mapped, x, y)}
        row['MINED'] = macro_arm(lib, alphabet, x, y)
        row['RANDOM'] = macro_arm(lib, rand, x, y)
        row['INCUMBENT'] = incumbent_arm(lib, x, y)
        for arm in ('MAPPED', 'MINED', 'RANDOM', 'INCUMBENT'):
            with torch.no_grad():
                row[arm]['query_nmse'] = float(nmse(run_route(lib, row[arm]['route'], xq).numpy(), t.eval_y))
            row[arm]['equals_mapped'] = [int(v) for v in row[arm]['route']] == mapped
        rows.append(row)
        print(f"deep {len(rows)}: MAPPED {row['MAPPED']['query_nmse']:.4f} MINED {row['MINED']['query_nmse']:.4f} "
              f"RANDOM {row['RANDOM']['query_nmse']:.4f} INCUMBENT {row['INCUMBENT']['query_nmse']:.4f} "
              f"({row['INCUMBENT']['seconds']:.0f}s)", flush=True)
    return {'slot_of': {str(k): v for k, v in slot_of.items()}, 'slots_distinct': len(set(slot_of.values())),
            'anchor_vote_purity': {str(p): c.most_common(1)[0][1] / sum(c.values()) for p, c in votes.items()},
            'shallow_routes_equal_mapped': sum(a == b for a, b in zip(shallow_routes, mapped_shallow)),
            'motif_images': [list(m) for m in motif_images], 'alphabet': [list(g) for g in alphabet],
            'motif_images_in_alphabet': sum(m in alphabet for m in motif_images),
            'top_grams': {','.join(map(str, g)): c for g, c in counts.most_common(20)},
            'random_alphabet': [list(g) for g in rand], 'deep': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--world', type=int, required=True)
    ap.add_argument('--stream', type=int, default=0)
    ap.add_argument('--scale', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(1)
    w, s = args.world, args.stream
    t_start = time.perf_counter()
    cfg, world, st, plan, shallow, deep, motifs = ms.build_stream(w, s)
    for t in st[:20]:   # anchor tasks share the teacher library
        if not np.allclose(t.program.execute(world.library, t.train_x), t.train_y):
            raise RuntimeError(f'{t.task_id}: teacher mismatch')
    work = Path('artifacts/m1_sizing') / f'w{w}_s{s}'
    ran = o2.run_cfg(cfg, args.scale)
    summary = ll.run(dataclasses.replace(ran, output_directory=work / 'lifetime'), o2.KIND, world=world,
                     model=dn.planned_model(ms.DEPTH, cfg, plan), return_model=True,
                     replay_seed=dn.replay_seed_for(w, s))
    model = summary.pop('terminal_model')
    rec = {'world': w, 'stream': s, 'scale': args.scale, 'motifs': [list(m) for m in motifs],
           'wake_seconds': time.perf_counter() - t_start,
           'wake_shallow_median': score(model, SimpleNamespace(tasks=shallow))['median']}
    print('wake', rec['wake_seconds'], rec['wake_shallow_median'], flush=True)
    pool = o2d.reservoir(st, w, s, o3.MEMORY)
    t0 = time.perf_counter()
    rec['routes_changed'] = dr.reroute(model, st, plan, pool)
    rec['reroute_seconds'] = time.perf_counter() - t0
    rec['reroute_shallow_median'] = score(model, SimpleNamespace(tasks=shallow))['median']
    o2c.consolidate(cfg, model, pool, [t.task_id for t in st], [1941, w, s, o3.MEMORY], updates=o3.EXTRA_UPDATES)
    sc = score(model, SimpleNamespace(tasks=shallow))
    rec.update({'formed_shallow_median': sc['median'], 'formed_shallow_below_0.05': sc['below_0.05'],
                'formation_seconds': time.perf_counter() - t_start})
    print('formed', rec['formed_shallow_median'], sc['below_0.05'], '/', len(shallow), flush=True)
    model.eval()
    rec['deep_test'] = deep_test(model, st, plan, shallow, deep[:4] if args.scale != 1 else deep, (w, s))
    rec['seconds'] = time.perf_counter() - t_start
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'w{w}_s{s}{"_scale" + str(args.scale) if args.scale != 1 else ""}.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    os.replace(tmp, path)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
