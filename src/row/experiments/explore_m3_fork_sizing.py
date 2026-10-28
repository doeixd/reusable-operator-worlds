"""EXPLORATORY Tier 1 sizing (decision 24a; no verdict): does a learner-side slot FORK repair, applied before mining,
rescue the M2 streams whose formation failed, without harming the ones that formed?

Substrate: the saved M2 wake terminals (`explore_m3_slot_census.load_wake`, gated on the cell's library sha256).
Pipeline = M2's `CYC` branch with one step inserted after the common re-route:
- FORK (learner-computable; no teacher identity): anchors = stream tasks of length 1. While some anchor's best-slot
  reservoir NMSE exceeds TAU (at most MAX_FORKS rounds): take a FREE slot (no anchor's best slot; least used by routes of tasks of length <= 5), copy into
  it the slot most of those misfit anchors currently choose, fit only the new slot (Adam, FIT_STEPS) on the misfit
  anchors' reservoir examples, and re-evaluate. Then each fork is refit on the anchors that now choose it. Every task
  of length <= 5 is then re-routed exhaustively again.
- then M2's mining, CYC re-route of the 48 deep tasks, and O3 sleep, unchanged.
When no anchor is misfit the repair is a no-op and the result must equal the M2 `CYC` branch exactly (checked).
Output `reports/m3_fork_sizing/w{w}_s{s}.json`.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from row.experiments import deep_reroute as dr
from row.experiments import m2_motif_cycle as m2
from row.experiments import motif_stream as ms
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.explore_m3_slot_census import CELLS, load_wake, nmse_t
from row.models.learned_models import batched_householder

TAU = 0.5
MAX_FORKS = 3
FIT_STEPS = 600
FIT_LR = 0.01
OUT = Path('reports/m3_fork_sizing')


def op_apply(op, z):
    h = torch.tanh(z @ op.V.T + op.b)
    return (z + op.alpha * h @ op.U.T) @ batched_householder([op.rotation.vectors])[0]


def fit_slot(op, xs, ys):
    opt = torch.optim.Adam(op.parameters(), lr=FIT_LR)
    with torch.enable_grad():
        for _ in range(FIT_STEPS):
            opt.zero_grad()
            loss = sum(torch.mean((op_apply(op, x) - y) ** 2) for x, y in zip(xs, ys)) / len(xs)
            loss.backward()
            opt.step()
    return float(loss.detach())


def anchor_fit(model, anchors, xy):
    lib = FrozenLibrary(model)
    with torch.no_grad():
        return {t: sorted((nmse_t(m2.slot_apply(lib, xy[t][0], k), xy[t][1]), k) for k in range(lib.slots))[0]
                for t in anchors}


def reroute_short(model, st, plan, xy):
    lib = FrozenLibrary(model)
    changed = 0
    with torch.no_grad():
        for t in st:
            d = plan[t.task_id]
            if d <= ms.DEPTH:
                new = dr.exhaustive_route(lib, *xy[t.task_id], d)
                changed += [int(v) for v in model.hard_routes()[t.task_id][:d]] != new
                m2.set_route(model, t.task_id, new)
    return changed


def fork(model, st, plan, xy):
    anchors = [t.task_id for t in st if plan[t.task_id] == 1]
    info = {'rounds': []}
    forks = []
    for _ in range(MAX_FORKS):
        fit = anchor_fit(model, anchors, xy)
        misfit = [t for t in anchors if fit[t][0] > TAU]
        if not misfit:
            break
        routes = model.hard_routes()
        claimed = {fit[t][1] for t in anchors} | set(forks)
        usage = Counter(int(r) for t in st if plan[t.task_id] <= ms.DEPTH for r in routes[t.task_id][:plan[t.task_id]])
        src = Counter(fit[t][1] for t in misfit).most_common(1)[0][0]
        shape = model.library[src].rotation.vectors.shape   # reflection count fixes the orientation class
        free = sorted((usage[k], k) for k in range(len(model.library))
                      if k not in claimed and model.library[k].rotation.vectors.shape == shape)
        if not free:
            info['no_free_slot'] = True
            break
        new = free[0][1]
        model.library[new].load_state_dict(model.library[src].state_dict())
        loss = fit_slot(model.library[new], [xy[t][0] for t in misfit], [xy[t][1] for t in misfit])
        forks.append(new)
        info['rounds'].append({'misfit': len(misfit), 'src': src, 'new': new, 'fit_loss': loss})
    if forks:
        fit = anchor_fit(model, anchors, xy)
        for k in forks:
            mine = [t for t in anchors if fit[t][1] == k]
            if mine:
                fit_slot(model.library[k], [xy[t][0] for t in mine], [xy[t][1] for t in mine])
        fit = anchor_fit(model, anchors, xy)
        info['after'] = {'misfit': sum(fit[t][0] > TAU for t in anchors),
                         'fork_occupancy': {str(k): sum(fit[t][1] == k for t in anchors) for k in forks}}
    info['forks'] = forks
    return info


def run(w, s):
    torch.set_num_threads(1)
    t_start = time.perf_counter()
    rec = json.loads((CELLS / f'w{w}_s{s}.json').read_text())['record']
    cfg, model, st, plan, shallow, deep, motifs = load_wake(w, s)
    if library_sha(model) != rec['wake_library_sha256']:
        raise RuntimeError('library sha differs')
    m2.scores(model, shallow)   # the M2 cell scored WAKE (eval mode) before the common re-route
    pool = o2d.reservoir(st, w, s, o3.MEMORY)
    xy = m2.pool_xy(pool)
    common = copy.deepcopy(model)
    out = {'world': w, 'stream': s, 'm2_cyc_solved': m2.solved_fraction(rec, 'CYC')}
    out['reroute_changed'] = reroute_short(common, st, plan, xy)
    if out['reroute_changed'] != rec['shallow_reroute']['changed']:
        raise RuntimeError('common re-route differs from M2')
    out['fork'] = fork(common, st, plan, xy)
    if out['fork']['forks']:
        out['reroute2_changed'] = reroute_short(common, st, plan, xy)
    alphabet, top = m2.mine(common, st, plan)
    routes = common.hard_routes()
    votes = {}
    for t in st:
        if plan[t.task_id] == 1:
            votes.setdefault(t.program.primitive_ids[0], Counter())[int(routes[t.task_id][0])] += 1
    slot_of = {p: c.most_common(1)[0][0] for p, c in votes.items()}   # diagnostic only
    images = [tuple(slot_of.get(p, -1) for p in mo) for mo in motifs]
    out['mining'] = {'alphabet': [list(g) for g in alphabet], 'images_in_alphabet': sum(i in alphabet for i in images),
                     'distinct_anchor_slots': len(set(slot_of.values()))}
    lib = FrozenLibrary(common)
    m = copy.deepcopy(common)
    with torch.no_grad():
        for t in deep:
            x, y = xy[t.task_id]
            cur = [int(v) for v in routes[t.task_id][:ms.DEEP_DEPTH]]
            cand, _ = m2.choose_cyc(lib, alphabet, x, y)
            m2.set_route(m, t.task_id, cur if m2.support_mse(lib, cur, x, y) <= m2.support_mse(lib, cand, x, y) else cand)
    o2c.consolidate(cfg, m, pool, [t.task_id for t in st], [1941, w, s, o3.MEMORY], updates=o3.EXTRA_UPDATES)
    deep_s, sh_s = m2.scores(m, deep), m2.scores(m, shallow)
    out['deep'] = deep_s
    out['solved'] = sum(v < m2.SOLVED for v in deep_s.values()) / len(deep_s)
    out['shallow_median'] = statistics.median(sh_s.values())
    out['library_sha256'] = library_sha(m)
    if not out['fork']['forks']:
        out['noop_equals_m2'] = (deep_s == rec['branches']['CYC']['deep']
                                 and out['library_sha256'] == rec['branches']['CYC']['library_sha256'])
    out['seconds'] = time.perf_counter() - t_start
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'w{w}_s{s}.json'
    path.with_suffix('.tmp').write_text(json.dumps(out, indent=1), encoding='utf-8')
    os.replace(path.with_suffix('.tmp'), path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', nargs='+', default=['w84_s1', 'w85_s2', 'w86_s0', 'w86_s2', 'w84_s0', 'w85_s1'])
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    keys = [(int(k[1:3]), int(k[-1])) for k in args.keys]
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run, w, s): (w, s) for w, s in keys}
        for f in as_completed(futs):
            o = f.result()
            print(f"w{o['world']}_s{o['stream']} m2 {o['m2_cyc_solved']:.2f} -> {o['solved']:.2f} shallow "
                  f"{o['shallow_median']:.4f} forks {o['fork']['forks']} after {o['fork'].get('after')} "
                  f"images {o['mining']['images_in_alphabet']} noop_eq {o.get('noop_equals_m2')} {o['seconds']:.0f}s",
                  flush=True)


if __name__ == '__main__':
    main()
