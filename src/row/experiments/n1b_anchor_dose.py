"""N1b: how many anchors, and of which length?

Plan: `N1B_ANCHOR_DOSE_PLAN.md`. Offline, worlds 0-2, development evidence only.

Same harness as N1 - 65,536 updates, one pooled stream of 188 tasks, the same
minibatch seed sequence, scored on the canonical 64 length-3 tasks - with the
124 non-canonical slots split between anchors and N1's own length-3 fillers.
Pool order is `anchors(k) + canonical + fillers(124 - k)`, so k = 124 is N1's
INTERLEAVED pool and k = 0 is N1's SHAM pool, element for element.

`n1_anchor_supply` is imported, never edited: its hash is in N1's protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
import torch

from row.experiments import n1_anchor_supply as n1
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('N1B_ANCHOR_DOSE_PLAN.md')
ROOT = Path('artifacts/n1b_anchor_dose')
OUTPUT = Path('reports/n1b_anchor_dose.json')

WORLDS = (0, 1, 2)
ARMS = ('L2_ONLY', 'DOSE_8', 'L1_ONLY', 'DOSE_32')   # most decisive first
DOSE_STREAM = 1913
FILLERS = 124
NEAR_LOW, NEAR_HIGH = 0.03, 0.08
JOBS = 3
MIN_FREE_GIB = 8.0                                     # pool of 3, fails closed

# Committed endpoints from N1 (reports/n1_anchor_supply.json, 63e63f6).
ENDPOINT_REPORT = Path('reports/n1_anchor_supply.json')


def protocol():
    return {'id': 'n1b-anchor-dose-v1', 'git_commit': git_commit(), 'plan': PLAN.as_posix(),
            'worlds': list(WORLDS), 'arms': list(ARMS), 'updates': n1.UPDATES,
            'pool_stream': n1.POOL_STREAM, 'dose_stream': DOSE_STREAM, 'fillers': FILLERS,
            'threshold': n1.THRESHOLD, 'near_threshold': [NEAR_LOW, NEAR_HIGH], 'jobs': JOBS,
            'pool_order': 'anchors(k) + canonical + fillers(124-k)',
            'input_sha256': {p.as_posix(): digest(p) for p in
                             (PLAN, Path('configs/v1.yaml'), ENDPOINT_REPORT)},
            'implementation_sha256': digest(Path(__file__)),
            'n1_implementation_sha256': digest(Path(n1.__file__))}


def select_anchors(arm, anchors, world):
    """The registered anchor subset for an arm, in N1's original order."""
    if arm == 'L1_ONLY':
        return [t for t in anchors if t.depth == 1]
    if arm == 'L2_ONLY':
        return [t for t in anchors if t.depth == 2]
    if arm.startswith('DOSE_'):
        k = int(arm.split('_')[1])
        perm = np.random.default_rng(np.random.SeedSequence([DOSE_STREAM, world])).permutation(len(anchors))
        return [anchors[i] for i in sorted(int(j) for j in perm[:k])]
    if arm == 'ENDPOINT_ALL':
        return list(anchors)
    if arm == 'ENDPOINT_NONE':
        return []
    raise ValueError(f'unknown arm {arm}')


def build_pool(arm, world):
    cfg, world_obj, canonical = n1.canonical_tasks(world)
    anchors = n1.anchor_tasks(world)
    fillers = n1.extra_length3_tasks(cfg, world_obj, FILLERS, world)
    chosen = select_anchors(arm, anchors, world)
    pool = chosen + canonical + fillers[:FILLERS - len(chosen)]
    return cfg, canonical, chosen, pool


def run_cell(arm, world):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, canonical, chosen, pool = build_pool(arm, world)
    model, drawn = n1.train_pooled(cfg, pool, canonical, n1.UPDATES,
                                   np.random.SeedSequence([n1.POOL_STREAM, world]))
    final = n1.score(model, n1._ScoredWorld(canonical))
    return {
        'arm': arm, 'world': world, 'updates': n1.UPDATES,
        'anchors': len(chosen),
        'anchor_depths': {str(d): sum(1 for t in chosen if t.depth == d) for d in (1, 2)},
        'pool_tasks': len(pool), 'pool_examples': sum(len(t.train_x) for t in pool),
        'scored_tasks': len(canonical),
        'terminal_median': final['median'], 'below_0.05': final['below_0.05'],
        'per_task': final['per_task'], 'library_sha256': n1.library_sha(model),
        'first_64_draws': drawn,
        'near_threshold': NEAR_LOW <= final['median'] <= NEAR_HIGH,
        'seconds': time.perf_counter() - started,
    }


EXPECTED_ANCHORS = {'L1_ONLY': (60, {'1': 60, '2': 0}), 'L2_ONLY': (64, {'1': 0, '2': 64}),
                    'DOSE_8': (8, None), 'DOSE_32': (32, None)}


def validate_cell(record):
    if record['arm'] not in ARMS or record['world'] not in WORLDS:
        raise ValueError('cell identity')
    if not math.isfinite(record['terminal_median']) or record['terminal_median'] < 0:
        raise ValueError('bad terminal median')
    if record['pool_tasks'] != 188 or record['pool_examples'] != 24064:
        raise ValueError('pool is not the registered 188 tasks / 24,064 examples')
    if record['scored_tasks'] != 64:
        raise ValueError('every arm is scored on the canonical 64 tasks')
    count, depths = EXPECTED_ANCHORS[record['arm']]
    if record['anchors'] != count:
        raise ValueError(f"{record['arm']} must carry {count} anchors, has {record['anchors']}")
    if depths is not None and record['anchor_depths'] != depths:
        raise ValueError(f"{record['arm']} anchor depths wrong")


def endpoints():
    report = json.loads(ENDPOINT_REPORT.read_text())
    cells = report['cells']
    return {w: {'k0': cells[f'SHAM_w{w}']['terminal_median'],
                'k124': cells[f'INTERLEAVED_w{w}']['terminal_median']} for w in WORLDS}


def _passes(values):
    return sum(1 for v in values if v < n1.THRESHOLD) >= 2


def triage(records):
    by = {}
    for r in records.values():
        by.setdefault(r['arm'], {})[r['world']] = r['terminal_median']
    if any(len(by.get(a, {})) != len(WORLDS) for a in ARMS):
        return {'verdict': 'INCOMPLETE'}
    passes = {a: _passes(by[a].values()) for a in ARMS}
    l1, l2 = passes['L1_ONLY'], passes['L2_ONLY']
    length = ('BOTH_SUFFICE' if l1 and l2 else 'L1_SUFFICES_ONLY' if l1
              else 'L2_SUFFICES_ONLY' if l2 else 'NEITHER')
    ends = endpoints()
    dose_pass = {0: _passes([ends[w]['k0'] for w in WORLDS]), 8: passes['DOSE_8'],
                 32: passes['DOSE_32'], 124: _passes([ends[w]['k124'] for w in WORLDS])}
    doses = sorted(dose_pass)
    k_star = next((k for k in doses if all(dose_pass[j] for j in doses if j >= k)), None)
    non_monotone = any(dose_pass[a] and not dose_pass[b] for a in doses for b in doses if b > a)
    near = sorted(f"{r['arm']}_w{r['world']}" for r in records.values() if r['near_threshold'])
    return {'length_verdict': length, 'dose_k_star': k_star, 'dose_non_monotone': non_monotone,
            'passes': passes, 'dose_pass': {str(k): v for k, v in dose_pass.items()},
            'by_arm': {a: {str(w): v for w, v in by[a].items()} for a in ARMS},
            'endpoints': {str(w): v for w, v in ends.items()},
            'near_threshold_cells': near}


def host_ok():
    free = psutil.virtual_memory().available / 2 ** 30
    return free, free >= MIN_FREE_GIB


def _status(state, done, total, running):
    atomic_json(ROOT / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                       'running': running, 'pid': os.getpid(),
                                       'updated_utc': now()})


def run(cells, jobs=JOBS):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'cells').mkdir(exist_ok=True)
    expected = protocol()
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    with writer_lock(ROOT / 'launcher.lock'):
        if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(ROOT / 'manifest.json', manifest)
        atomic_json(ROOT / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
        free, ok = host_ok()
        atomic_json(ROOT / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': ok, 'jobs': jobs, 'checked_utc': now()})
        if not ok:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free, {MIN_FREE_GIB} required')
        records, todo = {}, []
        try:
            for arm, world in cells:
                key = f'{arm}_w{world}'
                path = ROOT / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(ROOT / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, world))
            log_line(ROOT / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} '
                                       f'jobs={jobs} free={free:.1f}GiB')
            _status('running', len(records), len(cells), [f'{a}_w{w}' for a, w in todo[:jobs]])
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, arm, world): (arm, world) for arm, world in todo}
                for future in as_completed(futures):
                    arm, world = futures[future]
                    key = f'{arm}_w{world}'
                    record = future.result()
                    validate_cell(record)
                    atomic_json(ROOT / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True,
                                 'record': record, 'record_sha256': fingerprint(record),
                                 'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g}")
                    pending = [f'{a}_w{w}' for a, w in todo if f'{a}_w{w}' not in records]
                    _status('running', len(records), len(cells), pending[:jobs])
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'cells': records,
                                 'summary': triage(records), 'finished_utc': now()})
            _status('complete', len(records), len(cells), [])
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            _status('failed', len(records), len(cells), [])
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=JOBS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    cells = [(a, w) for a in ARMS for w in WORLDS]
    run(cells, jobs=args.jobs)
    print(f'N1b report: {OUTPUT}')


if __name__ == '__main__':
    main()
