"""B2-depth2 Tier 1 (exploratory): with loops whose BODY is two steps, does the one-step-start learner discover which
tasks iterate, without being told, while forming the vocabulary, and what does not being told cost?

Plan: `B2_DEPTH2_PLAN.md`. Development band 5 (fresh): worlds 61, 62, 63, streams 0-2; dry-run world 66. Stream:
`loop_stream.stream(w, s, 48, body_depth=2)` (O2's SHUFFLED depth-3 stream plus 48 loops with body P2(P1(z)), 236
tasks); candidates are the 112 depth-2 tasks (48 loop, 64 straight length-2).

Why: the census (`reports/b2d2_necessity_census.json`) found two-step-body loops necessary and their body trivially
findable on formed vocabularies; the sizing (`reports/b2d2_sizing.json`, world 60) found both arms forming and wake
separating loops (AUC 0.979), with a larger cost of discovery than at depth 1.

Arms (constructions; helpers from `b2d2_sizing`, unchanged):
- ONESTEP2: `LoopLearner(body_depth=2)` with every candidate in `loop_plan`, B2-hard v2's ONESTEP halting (state_halt,
  step_bias, detached input, h decay 0.1, halt_init 4). End of stream: re-route the other tasks; per candidate the
  ONE_STEP data test over two-step bodies (`b2d2_sizing.end_of_stream_onestep`); then `b2_hard.sleep_decay`.
- GIVEN2: loop identity given (the 48 loops only, B2-online v2's halting, `body_depth=2`); end of stream: re-route the
  other tasks, re-fit each loop (`b2d2_sizing.refit_given`), `b1_online.sleep` with the loop parameters.
Every arm records the length-2 anchors' best fixed count over all two-step bodies at the end of wake (a root
analogue; teacher programs used only in that record).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import b1_online as b1o
from row.experiments import b2_hard as bh
from row.experiments import b2d2_sizing as sz
from row.experiments import census_b2_iteration_necessity as c2
from row.experiments import census_b2d2_necessity as cd
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import loop_stream as ls
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models.loop_gated import LoopLearner

PLAN = Path('B2_DEPTH2_PLAN.md')
ROOT = Path('artifacts/b2_depth2')
DRY_ROOT = Path('artifacts/b2_depth2_dry')
OUTPUT = Path('reports/b2_depth2.json')
WORLDS = (61, 62, 63)
STREAMS = (0, 1, 2)
ARMS = ('ONESTEP2', 'GIVEN2')   # decisive arm first
N_LOOP = 48
N_CAND = 112
K = sz.K
THRESHOLD = 0.05
LOOP_BAR = 0.5   # below the input-picked-count impostor at 128 support (census 0.508); registered in the plan
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 66
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def run_arm(arm, w, s, work: Path, scale=1):
    cfg, mixed, st, plan, canonical, ltasks, meta = ls.stream(w, s, N_LOOP, sz.BODY)
    loop_ids = {t.task_id for t in ltasks}
    cands = [(i, t) for i, t in enumerate(st) if plan[t.task_id] == sz.BODY]
    cand_ids = {t.task_id for _, t in cands}
    anchors2 = [t for _, t in cands if t.task_id not in loop_ids]
    anchors1 = [t for t in st if plan[t.task_id] == 1]
    model = sz.build(arm, cfg, plan, cand_ids, loop_ids)
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')

    def scored(ts):
        return score(model, SimpleNamespace(tasks=ts))

    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'halt_init': model.halt_init,
           'body_depth': model.body_depth, 'stream_tasks': len(st), 'candidates': len(cands),
           'loop_task_ids': sorted(loop_ids),
           'wake_canonical_median': scored(canonical)['median'], 'wake_loop_median': scored(ltasks)['median'],
           'wake_anchor2_median': scored(anchors2)['median'], 'wake_anchor1_median': scored(anchors1)['median'],
           'wake_count_accuracy_median': float(np.median(sz.count_accuracy(model, ltasks, meta)))}
    halted = [t.task_id for t in st if t.task_id in model.halt_b]

    def moved(t):
        b0 = model._initial_biases() if model.halt_b[t].numel() == K else torch.zeros(1)
        d = [model.halt_b[t].detach() - b0] + ([model.halt_w[t].detach()] if t in model.halt_w else [])
        return float(torch.cat([x.flatten() for x in d]).norm())

    rec['wake_halt_moved_median'] = float(np.median([moved(t) for t in halted]))
    if arm == 'ONESTEP2':
        norms = {t: float(model.halt_w[t].detach().norm()) for t in cand_ids}
        rec['wake_state_norms'] = norms
        rec['wake_state_auc'] = bh.auc([norms[t] for t in loop_ids], [v for t, v in norms.items() if t not in loop_ids])
        rec['wake_state_norm_median'] = {'loop': float(np.median([norms[t] for t in loop_ids])),
                                         'straight': float(np.median([v for t, v in norms.items() if t not in loop_ids]))}
    pool = o2d.reservoir(st, w, s, bh.MEMORY)
    data = sz.by_task(pool)
    best = sz.anchor_best_counts(model, anchors2, data)
    rec['wake_anchor2_best_L'] = {t: v['L'] for t, v in best.items()}
    rec['wake_anchor2_rooted'] = sum(v['L'] >= 2 for v in best.values())
    rec['routes_changed'] = dr.reroute(model, [t for t in st if t.task_id not in model.loop_plan], plan, pool)
    rec['reroute_canonical_median'] = scored(canonical)['median']
    before = library_sha(model)
    if arm == 'ONESTEP2':
        dec = sz.end_of_stream_onestep(model, cands, data, w)
        rec['decisions'] = dec
        rec['loops_kept'] = sum(dec[t]['loop'] for t in loop_ids)
        rec['straight_kept_count'] = sum(not v['loop'] for t, v in dec.items() if t not in loop_ids)
        bh.sleep_decay(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY])
    else:
        rec['loop_refits_installed'] = sz.refit_given(model, ltasks, data, w)
        b1o.sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, bh.MEMORY], model.loop_parameters(loop_ids))
    rec['library_sha256_before_sleep'], rec['library_sha256'] = before, library_sha(model)
    canon, lp, a2, a1 = scored(canonical), scored(ltasks), scored(anchors2), scored(anchors1)
    acc = sz.count_accuracy(model, ltasks, meta)
    rec.update({'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
                'scored_canonical': len(canon['per_task']), 'loop_median': lp['median'], 'loop_per_task': lp['per_task'],
                'loop_below_threshold': sum(v < THRESHOLD for v in lp['per_task'].values()),
                'anchor2_median': a2['median'], 'anchor2_per_task': a2['per_task'],
                'anchor1_median': a1['median'],
                'count_accuracy_median': float(np.median(acc)),
                'count_accuracy_per_task': dict(zip([t.task_id for t in ltasks], acc))})
    return rec


def run_cell(arm, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_arm(arm, w, s, Path(root) / 'work' / f'{arm}_w{w}_s{s}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if (rec['scored_canonical'] != 64 or rec['stream_tasks'] != 188 + N_LOOP or len(rec['loop_per_task']) != N_LOOP
            or rec['candidates'] != N_CAND or len(rec['anchor2_per_task']) != N_CAND - N_LOOP
            or len(rec['wake_anchor2_best_L']) != N_CAND - N_LOOP or rec['body_depth'] != 2):
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before_sleep']:
        raise ValueError(f'{key}: sleep left the library unchanged')
    if not rec['wake_halt_moved_median'] > 0:
        raise ValueError(f'{key}: wake never moved the halting parameters')
    if rec['halt_init'] != {'ONESTEP2': sz.HALT_INIT, 'GIVEN2': None}[rec['arm']]:
        raise ValueError(f'{key}: halt_init')
    if rec['arm'] == 'ONESTEP2' and len(rec['decisions']) != N_CAND:
        raise ValueError(f'{key}: decisions')


def passing(m, bar=THRESHOLD):
    return m is not None and math.isfinite(m) and m < bar


def label(k_formation, auc_median, recall, specificity, n_terminal):
    if k_formation <= 6:
        return 'FORMATION_BROKEN'
    if auc_median >= 0.9 and recall >= 0.8 and specificity >= 0.95 and n_terminal >= 8:
        return 'DISCOVERS'
    return 'NOT_DISCOVERED'


def summarize(records, worlds=WORLDS):
    pairs = [(w, s) for w in worlds for s in STREAMS]
    arms = {a: {p: records[f'{a}_w{p[0]}_s{p[1]}'] for p in pairs} for a in ARMS}
    O, G = arms['ONESTEP2'], arms['GIVEN2']
    k_formation = sum(passing(O[p]['canonical_median']) for p in pairs)
    auc_median = float(np.median([O[p]['wake_state_auc'] for p in pairs]))
    recall = sum(O[p]['loops_kept'] for p in pairs) / (N_LOOP * len(pairs))
    specificity = sum(O[p]['straight_kept_count'] for p in pairs) / ((N_CAND - N_LOOP) * len(pairs))
    n_terminal = sum(passing(O[p]['loop_median'], LOOP_BAR) for p in pairs)

    def med(D, key):
        return float(np.median([D[p][key] for p in pairs]))

    keys = ('canonical_median', 'loop_median', 'anchor2_median', 'anchor1_median', 'wake_canonical_median',
            'wake_loop_median', 'wake_anchor2_median', 'reroute_canonical_median', 'count_accuracy_median',
            'wake_count_accuracy_median')
    return {'cells': len(pairs), 'k_formation': k_formation, 'wake_state_auc_median': auc_median,
            'loop_recall': recall, 'straight_specificity': specificity, 'n_terminal': n_terminal,
            'label': label(k_formation, auc_median, recall, specificity, n_terminal),
            'canonical_passes': {a: sum(passing(D[p]['canonical_median']) for p in pairs) for a, D in arms.items()},
            'rooted_anchor_cells': {a: sum(D[p]['wake_anchor2_rooted'] > 0 for p in pairs) for a, D in arms.items()},
            'loop_ratio_to_given_median': float(np.median([O[p]['loop_median'] / G[p]['loop_median'] for p in pairs])),
            'n_loop_at_or_below_given': sum(O[p]['loop_median'] <= G[p]['loop_median'] for p in pairs),
            'n_canonical_at_or_below_given': sum(O[p]['canonical_median'] <= G[p]['canonical_median'] for p in pairs),
            **{k: {a: med(D, k) for a, D in arms.items()} for k in keys},
            'wake_state_norm_median': {k: float(np.median([O[p]['wake_state_norm_median'][k] for p in pairs]))
                                       for k in ('loop', 'straight')}}


def protocol(root=ROOT):
    mods = {'b2d2_sizing': sz, 'b2_hard': bh, 'b1_online': b1o, 'b2_census': c2, 'b2d2_census': cd,
            'deep_reroute': dr, 'lifetime': ll, 'loop_stream': ls, 'o2': o2, 'o2d': o2d}
    return {'id': 'b2-depth2', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1, 'exploratory': True,
            'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS), 'arms': list(ARMS),
            'n_loop': N_LOOP, 'n_candidates': N_CAND, 'k': K, 'body_depth': sz.BODY, 'memory': bh.MEMORY,
            'sleep_updates': bh.SLEEP_UPDATES, 'halt_init': sz.HALT_INIT, 'halt_decay': sz.HALT_DECAY,
            'loop_ratio': sz.LOOP_RATIO, 'one_step': list(sz.ONE_STEP), 'threshold': THRESHOLD, 'loop_bar': LOOP_BAR,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)),
            'loop_learner_sha256': digest(Path(sys.modules[LoopLearner.__module__].__file__)),
            **{f'{k}_sha256': digest(Path(m.__file__)) for k, m in mods.items()}}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(root / 'manifest.json', manifest)
        free = psutil.virtual_memory().available / 2 ** 30
        atomic_json(root / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': free >= MIN_FREE_GIB, 'jobs': jobs, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, todo = {}, []
        try:
            for arm, w, s in todo_cells:
                key = f'{arm}_w{w}_s{s}'
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, w, s))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s, str(root), scale): (a, w, s) for a, w, s in todo}
                while futures:
                    done = next(as_completed(futures))
                    a, w, s = futures.pop(done)
                    key = f'{a}_w{w}_s{s}'
                    record = done.result()
                    validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    extra = (f" auc={record['wake_state_auc']:.3f} kept={record['loops_kept']}"
                             f" count={record['straight_kept_count']}" if 'wake_state_auc' in record else '')
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"canonical={record['canonical_median']:.4g} loop={record['loop_median']:.4g} "
                                               f"anchor2={record['anchor2_median']:.4g} wake={record['wake_canonical_median']:.4g} "
                                               f"rooted_anchors={record['wake_anchor2_rooted']}{extra}")
                    _status(root, 'running', len(records), total, [f'{x}_w{y}_s{z}' for x, y, z in futures.values()][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records, worlds)
            atomic_json(output, report)
            _status(root, 'complete', len(records), total, [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            _status(root, 'failed', len(records), len(todo_cells), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(root / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        run([(a, DRY_WORLD, 0) for a in ARMS], root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print('B2-depth2 dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'B2-depth2 report: {OUTPUT}')


if __name__ == '__main__':
    main()
