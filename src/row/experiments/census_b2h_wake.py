"""B2-hard wake census (Tier 0, descriptive, post hoc): is the candidate arms' broken WAKE vocabulary stale routes
or the library, and are the roots already there at the end of wake?

In `reports/b2_hard.json` the wake canonical median is ~1.9 in both candidate arms (ALLLOOP, ALLCOUNT) even in cells
that pass after the end of stream, against 0.046 for GIVEN. Per saved wake model (all three arms, worlds 53-55,
streams 0-2), behind gate G0 (reload reproduces the recorded wake canonical, loop and anchor medians exactly):
- REROUTE: the registered exhaustive re-route of every non-loop-plan task on the retained examples
  (`deep_reroute.reroute`, as the runner does first at end of stream), with NO sleep and nothing else changed;
  canonical, anchor and loop medians after it. If REROUTE alone brings the canonical median near GIVEN's, the wake
  damage is stale routes; if it stays high, it is the library.
- ROOTS: on the wake library, for every straight length-1 anchor, the best fixed count (body slot, L in 0..K) on
  its retained examples; per teacher operation, whether all its anchors are best fit at L >= 2 (a rooted operation).
  GIVEN's anchors have no iteration machinery but the same library-level fit is computed for comparison.
Teacher programs enter only this analysis. Questions written before it ran. Tier 0 on the cells where the pattern was
found: a sizing for a plan, not a verdict.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b2_hard as bh
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import deep_reroute as dr
from row.experiments import loop_stream as ls
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/b2h_wake_census.json')


def best_count(lib, X, Y):
    best = None
    for s in range(lib.slots):
        U = c2.unrolled(lib, s, X)
        for L in range(bh.K + 1):
            loss = float(np.mean((U[L] - Y) ** 2))
            if best is None or loss < best[0]:
                best = (loss, s, L)
    return best


def one(args):
    arm, w, s = args
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, bh.N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    cands = [(i, t) for i, t in enumerate(st) if plan[t.task_id] == 1]
    cand_ids = {t.task_id for _, t in cands}
    anchors = [t for _, t in cands if t.task_id not in loop_ids]
    model = bh.build(arm, cfg, plan, cand_ids, loop_ids)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(bh.ROOT / 'work' / f'{arm}_w{w}_s{s}' / 'lifetime' / 'model.pt', weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=False)   # only the unused probe code is unexpected
    model.eval()
    ref = json.loads((bh.ROOT / 'cells' / f'{arm}_w{w}_s{s}.json').read_text())['record']

    def med(ts):
        return score(model, SimpleNamespace(tasks=ts))['median']

    g0 = (med(canonical), med(ltasks), med(anchors)) == (ref['wake_canonical_median'], ref['wake_loop_median'],
                                                        ref['wake_anchor_median'])
    pool = o2d.reservoir(st, w, s, bh.MEMORY)
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    per_op = collections.defaultdict(list)
    for t in anchors:
        X = np.stack([a for a, _ in by[t.task_id]])
        Y = np.stack([b for _, b in by[t.task_id]])
        per_op[int(t.program.primitive_ids[0])].append(best_count(lib, X, Y)[2])
    rooted = sorted(op for op, Ls in per_op.items() if all(L >= 2 for L in Ls))
    changed = dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    canon = score(model, SimpleNamespace(tasks=canonical))
    prog = {t.task_id: set(t.program.primitive_ids) for t in canonical}
    with_root = [e for t, e in canon['per_task'].items() if prog[t] & set(rooted)]
    return {'arm': arm, 'world': w, 'stream': s, 'G0': g0, 'routes_changed': changed,
            'routes_changed_matches': changed == ref['routes_changed'],
            'wake_canonical_median': ref['wake_canonical_median'], 'reroute_canonical_median': canon['median'],
            'reroute_anchor_median': med(anchors), 'reroute_loop_median': med(ltasks),
            'terminal_canonical_median': ref['canonical_median'],
            'anchor_best_L_by_op': {str(k): v for k, v in sorted(per_op.items())}, 'rooted_ops': rooted,
            'reroute_canonical_median_with_rooted_op': float(np.median(with_root)) if with_root else None,
            'seconds': time.perf_counter() - started}


def main(out=OUTPUT):
    jobs = [(a, w, s) for a in bh.ARMS for w in bh.WORLDS for s in bh.STREAMS]
    recs = []
    with ProcessPoolExecutor(max_workers=bh.JOBS) as ex:
        for r in ex.map(one, jobs):
            recs.append(r)
            print(now(), r['arm'], r['world'], r['stream'], f"G0={r['G0']}", f"wake={r['wake_canonical_median']:.4g}",
                  f"reroute={r['reroute_canonical_median']:.4g}", f"terminal={r['terminal_canonical_median']:.4g}",
                  f"rooted={r['rooted_ops']}", f"{r['seconds']:.0f}s", flush=True)
    summary = {}
    for a in bh.ARMS:
        rs = [r for r in recs if r['arm'] == a]
        summary[a] = {'G0': f"{sum(r['G0'] for r in rs)}/{len(rs)}",
                      'wake_canonical_median': float(np.median([r['wake_canonical_median'] for r in rs])),
                      'reroute_canonical_median': float(np.median([r['reroute_canonical_median'] for r in rs])),
                      'reroute_below_0.05': sum(r['reroute_canonical_median'] < bh.THRESHOLD for r in rs),
                      'cells_with_rooted_op': sum(bool(r['rooted_ops']) for r in rs),
                      'terminal_fail_and_rooted': sum(r['terminal_canonical_median'] >= bh.THRESHOLD and bool(r['rooted_ops'])
                                                      for r in rs),
                      'terminal_fail': sum(r['terminal_canonical_median'] >= bh.THRESHOLD for r in rs)}
    atomic_json(out, {'tier': 0, 'descriptive': True, 'post_hoc': True, 'source': 'reports/b2_hard.json',
                      'summary': summary, 'cells': recs, 'finished_utc': now()})
    print(json.dumps(summary, indent=1))
    return 0 if all(r['G0'] for r in recs) else 1


if __name__ == '__main__':
    sys.exit(main())
