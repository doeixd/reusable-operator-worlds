"""B2-hard follow-up census (Tier 0, descriptive, post hoc): is B2-hard's formation failure a ROOT impostor, and
does a one-step count form at the end of stream repair it?

Found post hoc in `reports/b2_hard.json` (B2-hard, FORMATION_BROKEN 6/9): in every cell that failed formation, in
both candidate arms, the end-of-stream best COUNT form for the straight length-1 tasks of one to three teacher
operations is L = 2 or 3 iterations of a slot, for every anchor of that operation, so the library holds a slot Q
with Q^L ~ P (a root of the primitive) instead of P; depth-3 tasks using that operation fail (~1.4) while the
others pass (~0.04). Iteration machinery on the anchors makes the root representable; GIVEN (no machinery on
anchors) forms 9/9. Teacher programs enter only this analysis, never a learner.

Arms, per saved B2-hard wake model (ALLLOOP and ALLCOUNT, worlds 53-55, streams 0-2):
- REPLAY: the registered end of stream (`b2_hard`: re-route, `end_of_stream`, the arm's sleep), behind gates G0
  (reload reproduces the recorded wake canonical, loop and anchor medians exactly) and G1 (REPLAY reproduces the
  recorded terminal canonical per-task errors exactly), run on GATE_CELLS; the recorded terminals are the baseline;
- ONE_STEP: identical except that every COUNT form is restricted to L in {0, 1} (an anchor's planned depth is
  one application; L = 0 keeps loop tasks' usual best count). The loop form and data test are unchanged.
Questions, written before it ran: per cell, terminal canonical, anchor and loop medians and decisions under each
arm; how many failing cells ONE_STEP repairs (< 0.05) and how many passing cells it breaks. If a root slot is
coherent error (A1), sleep may reinforce it rather than repair it, so the census can come out either way.
Tier 0 on the same cells where the pattern was found: a sizing for a plan, not a verdict.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import b2_hard as bh
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import deep_reroute as dr
from row.experiments import loop_stream as ls
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, now

OUTPUT = Path('reports/b2h_root_census.json')
ARMS = ('ALLLOOP', 'ALLCOUNT')
VARIANTS = ('REPLAY', 'ONE_STEP')
ONE_STEP = (0, 1)
# REPLAY is bitwise (G1); it runs as a gate on a passing and a failing cell per arm, and the RECORDED terminals
# are the baseline for every cell (performance pass: a full replay would double the cost and change nothing).
GATE_CELLS = ((53, 0), (54, 2))


def one_step_end_of_stream(model, cands, pool, w, arm):
    """`b2_hard.end_of_stream` with every COUNT form restricted to L in ONE_STEP."""
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    out = {}
    for i, t in cands:
        X = np.stack([a for a, _ in by[t.task_id]])
        Y = np.stack([b for _, b in by[t.task_id]])
        seed = int(np.random.SeedSequence([5820, w, i]).generate_state(1)[0]) % 2**31
        _, bl = bh.fit_forms(lib, X, Y, seed)
        bc = None
        for s in range(lib.slots):
            U = c2.unrolled(lib, s, X)
            for L in ONE_STEP:
                loss = float(np.mean((U[L] - Y) ** 2))
                if bc is None or loss < bc[0]:
                    bc = (loss, s, L)
        keep = arm == 'ALLLOOP' and bl[0] < bh.LOOP_RATIO * bc[0]
        if keep:
            bh.install_loop(model, t.task_id, bl[1], bl[2])
        else:
            bh.install_count(model, t.task_id, bc[1], bc[2])
        out[t.task_id] = {'loop': bool(keep), 'count_support': bc[0], 'count_L': bc[2], 'loop_support': bl[0]}
    return out


def finish(arm, variant, w, s):
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
    changed = dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    eos = bh.end_of_stream if variant == 'REPLAY' else one_step_end_of_stream
    dec = eos(model, cands, pool, w, arm)
    if arm == 'ALLLOOP':
        bh.sleep_decay(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY])
    else:
        b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY], model.loop_parameters(cand_ids))
    canon = score(model, SimpleNamespace(tasks=canonical))
    rec = {'arm': arm, 'variant': variant, 'world': w, 'stream': s, 'G0': g0, 'routes_changed': changed,
           'routes_changed_matches': changed == ref['routes_changed'],
           'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
           'loop_median': med(ltasks), 'anchor_median': med(anchors),
           'loops_kept': sum(dec[t]['loop'] for t in loop_ids),
           'straight_kept_count': sum(not v['loop'] for t, v in dec.items() if t not in loop_ids),
           'straight_count_L': np.bincount([v['count_L'] for t, v in dec.items() if t not in loop_ids],
                                           minlength=bh.K + 1).tolist(),
           'recorded_canonical_median': ref['canonical_median']}
    if variant == 'REPLAY':
        rec['G1'] = canon['per_task'] == ref['canonical_per_task']
    return rec


def job(args):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = finish(*args)
    rec['seconds'] = time.perf_counter() - started
    return rec


def summarize(recs, worlds):
    by = {(r['arm'], r['variant'], r['world'], r['stream']): r for r in recs}
    summary = {}
    for a in ARMS:
        gates = [r for r in recs if r['arm'] == a and r['variant'] == 'REPLAY']
        rep = [json.loads((bh.ROOT / 'cells' / f'{a}_w{w}_s{s}.json').read_text())['record']
               for w in worlds for s in bh.STREAMS]
        one = [by[(a, 'ONE_STEP', w, s)] for w in worlds for s in bh.STREAMS]

        def medians(key):
            return {'RECORDED': float(np.median([r[key] for r in rep])),
                    'ONE_STEP': float(np.median([r[key] for r in one]))}

        summary[a] = {'G0': f"{sum(r['G0'] for r in gates + one)}/{len(gates + one)}",
                      'G1': f"{sum(r['G1'] for r in gates)}/{len(gates)}",
                      'recorded_passes': sum(r['canonical_median'] < bh.THRESHOLD for r in rep),
                      'one_step_passes': sum(r['canonical_median'] < bh.THRESHOLD for r in one),
                      'repaired': sum(r['canonical_median'] >= bh.THRESHOLD > o['canonical_median']
                                      for r, o in zip(rep, one)),
                      'broken': sum(r['canonical_median'] < bh.THRESHOLD <= o['canonical_median']
                                    for r, o in zip(rep, one)),
                      'canonical_median': medians('canonical_median'), 'loop_median': medians('loop_median'),
                      'anchor_median': medians('anchor_median')}
    return summary


def main(worlds=bh.WORLDS, out=OUTPUT):
    jobs = ([(a, 'REPLAY', w, s) for a in ARMS for w, s in GATE_CELLS if w in worlds] +
            [(a, 'ONE_STEP', w, s) for a in ARMS for w in worlds for s in bh.STREAMS])
    recs = []
    with ProcessPoolExecutor(max_workers=bh.JOBS) as ex:
        for rec in ex.map(job, jobs):
            recs.append(rec)
            print(now(), rec['arm'], rec['variant'], rec['world'], rec['stream'], f"G0={rec['G0']}",
                  f"G1={rec.get('G1')}", f"can={rec['canonical_median']:.4g}", f"loop={rec['loop_median']:.4g}",
                  f"anchor={rec['anchor_median']:.4g}", f"L={rec['straight_count_L']}", f"{rec['seconds']:.0f}s",
                  flush=True)
    summary = summarize(recs, worlds)
    atomic_json(out, {'tier': 0, 'descriptive': True, 'post_hoc': True, 'source': 'reports/b2_hard.json',
                      'summary': summary, 'cells': recs, 'finished_utc': now()})
    print(json.dumps(summary, indent=1))
    return 0 if all(r['G0'] for r in recs) and all(r['G1'] for r in recs if r['variant'] == 'REPLAY') else 1


if __name__ == '__main__':
    sys.exit(main())
