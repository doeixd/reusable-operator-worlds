"""B2-online localization census (Tier 0, descriptive): which end-of-stream step breaks LOOP's vocabulary?

B2-online (`reports/b2_online.json`, FORMATION_BROKEN) left LOOP's terminal canonical median at 0.036-0.095 while
the same re-route + sleep took both controls to 0.008-0.021, and LOOP's vocabulary was no worse after wake. The
re-fit does not touch the library (it writes loop task codes and halting only), so the canonical damage must enter
in sleep; the question is through which path.

On each saved LOOP wake model (`artifacts/b2_online/work/LOOP_w{w}_s{s}/lifetime/model.pt`), behind gate G0 (the
reloaded model reproduces the recorded wake canonical and loop medians exactly), re-route once (deterministic,
shared by all variants), then rebuild the end of stream five ways:
- FULL: B2-online's construction (re-fit, sleep training library + all task codes + halting). Gate G1: reproduces
  the recorded terminal per-task canonical and loop values exactly.
- NO_REFIT: no loop re-fit; sleep as FULL (wake's halting).
- FROZEN_HALT: re-fit; halting parameters left out of sleep's optimizer (gradient still flows through them).
- DETACH_HALT: re-fit; during sleep the halting logit reads the state DETACHED, so no gradient reaches the library
  (or earlier states) through the halting predicate; everything else as FULL.
- NO_LOOP_SLEEP: re-fit; sleep on the 188 straight-line tasks only (loop examples and codes excluded), same seed
  (a shorter pool, so its minibatch draws differ from FULL's: sampling is confounded with the exclusion).
Also recorded: canonical and loop medians after re-route + re-fit, before sleep (PRE_SLEEP).

Readings, written before any variant ran:
- DETACH_HALT restores formation (canonical < 0.05 in most cells, near the controls' ~0.015) while FROZEN_HALT
  does not: the damage is gradient through the state-dependent halting predicate into the shared library (the
  difference between LOOP and CONSTCOUNT, whose halting cannot read the state).
- Only NO_LOOP_SLEEP restores: loop tasks' multi-application objective itself, not the halting path.
- NO_REFIT restores: the re-fit's installed predicates (e.g. their sharpness) drive the damage.
- Nothing restores except NO_LOOP_SLEEP partially: interference is not localized to one path.
Tier 0 on 9 reused development cells; descriptive, no verdict.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from row.experiments import b1_online as b1o
from row.experiments import b2_online as b2
from row.experiments import deep_reroute as dr
from row.experiments import loop_stream as ls
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_rotated_g5r_interference import score
from row.models.loop_gated import LoopLearner

ROOT = Path('artifacts/b2o_localize')
OUTPUT = Path('reports/b2o_localize_census.json')
VARIANTS = ('FULL', 'NO_REFIT', 'FROZEN_HALT', 'DETACH_HALT', 'NO_LOOP_SLEEP')


def detached_halting(model):
    orig = LoopLearner.halt_logit.__get__(model)
    model.halt_logit = types.MethodType(lambda self, z, t, step: orig(z.detach(), t, step), model)


def one(w, s):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, b2.N_LOOP)
    loop_ids = {t.task_id for t in ltasks}
    model = b1o.build(LoopLearner, cfg, plan, loop_plan={t: 'state' for t in loop_ids}, max_iterations=b2.K,
                      state_halt=True)
    for t in st:
        model.begin_task(t.task_id)
    state = torch.load(f'artifacts/b2_online/work/LOOP_w{w}_s{s}/lifetime/model.pt', weights_only=True)['model_state_dict']
    model.load_state_dict(state, strict=False)   # only the unused novel-composition probe code is unexpected
    model.eval()
    ref = json.loads(Path(f'artifacts/b2_online/cells/LOOP_w{w}_s{s}.json').read_text())['record']
    canon = SimpleNamespace(tasks=canonical)
    loops = SimpleNamespace(tasks=ltasks)
    g0 = (score(model, canon)['median'] == ref['wake_canonical_median']
          and score(model, loops)['median'] == ref['wake_loop_median'])
    pool = o2d.reservoir(st, w, s, b2.MEMORY)
    dr.reroute(model, [t for t in st if t.task_id not in loop_ids], plan, pool)
    rerouted = copy.deepcopy(model)
    out = {'world': w, 'stream': s, 'G0_reload_reproduces': g0, 'variants': {}}
    for v in VARIANTS:
        m = copy.deepcopy(rerouted)
        installed = None if v == 'NO_REFIT' else b2.refit_loops(m, ltasks, pool, w, 'LOOP')
        if v == 'FULL':
            out['PRE_SLEEP'] = {'canonical_median': score(m, canon)['median'], 'loop_median': score(m, loops)['median']}
        sleep_pool, ids, extra = pool, [t.task_id for t in st], m.loop_parameters(loop_ids)
        if v == 'FROZEN_HALT':
            extra = []
        if v == 'NO_LOOP_SLEEP':
            sleep_pool = [p for p in pool if p[2] not in loop_ids]
            ids = [t for t in ids if t not in loop_ids]
            extra = []
        if v == 'DETACH_HALT':
            detached_halting(m)
        b1o.sleep(cfg, m, sleep_pool, ids, [1941, w, s, b2.MEMORY], extra)
        if v == 'DETACH_HALT':
            del m.halt_logit
        c, lp = score(m, canon), score(m, loops)
        out['variants'][v] = {'refits_installed': installed, 'canonical_median': c['median'], 'loop_median': lp['median'],
                              'canonical_per_task': c['per_task'], 'loop_per_task': lp['per_task']}
        if v == 'FULL':
            out['G1_full_reproduces'] = (c['per_task'] == ref['canonical_per_task']
                                         and lp['per_task'] == ref['loop_per_task'])
    out['seconds'] = time.perf_counter() - started
    return out


def summarize(cells):
    s = {'cells': len(cells), 'G0': all(c['G0_reload_reproduces'] for c in cells),
         'G1': all(c['G1_full_reproduces'] for c in cells),
         'pre_sleep_canonical_median': float(np.median([c['PRE_SLEEP']['canonical_median'] for c in cells])),
         'pre_sleep_loop_median': float(np.median([c['PRE_SLEEP']['loop_median'] for c in cells]))}
    for v in VARIANTS:
        cm = [c['variants'][v]['canonical_median'] for c in cells]
        lm = [c['variants'][v]['loop_median'] for c in cells]
        s[v] = {'canonical_passes': int(sum(x < b2.THRESHOLD for x in cm)), 'canonical_median': float(np.median(cm)),
                'canonical_range': [float(min(cm)), float(max(cm))], 'loop_median': float(np.median(lm)),
                'loop_range': [float(min(lm)), float(max(lm))]}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells', default='all', help="'all' or e.g. 44:1,46:1")
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    todo = ([(w, s) for w in b2.WORLDS for s in b2.STREAMS] if args.cells == 'all'
            else [tuple(int(v) for v in c.split(':')) for c in args.cells.split(',')])
    (ROOT / 'cells').mkdir(parents=True, exist_ok=True)
    log = (ROOT / 'run.log').open('a', encoding='utf-8')
    log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} start {todo}\n'); log.flush()
    pending = [c for c in todo if not (ROOT / 'cells' / f'w{c[0]}_s{c[1]}.json').exists()]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one, w, s): (w, s) for w, s in pending}
        for f in as_completed(futs):
            w, s = futs[f]
            r = f.result()
            tmp = ROOT / 'cells' / f'w{w}_s{s}.json.tmp'
            tmp.write_text(json.dumps(r))
            tmp.replace(ROOT / 'cells' / f'w{w}_s{s}.json')
            log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} done w{w}_s{s} {r["seconds"]:.0f}s G0={r["G0_reload_reproduces"]} '
                      f'G1={r["G1_full_reproduces"]} ' + ' '.join(f'{v}={r["variants"][v]["canonical_median"]:.4f}/'
                                                                   f'{r["variants"][v]["loop_median"]:.3f}' for v in VARIANTS) + '\n')
            log.flush()
    cells = [json.loads((ROOT / 'cells' / f'w{w}_s{s}.json').read_text()) for w in b2.WORLDS for s in b2.STREAMS
             if (ROOT / 'cells' / f'w{w}_s{s}.json').exists()]
    report = {'census': 'b2o-localize', 'tier': 0, 'summary': summarize(cells), 'cells': cells}
    tmp = OUTPUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(OUTPUT)
    log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} exit cells={len(cells)}\n')
    print(json.dumps(report['summary'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
