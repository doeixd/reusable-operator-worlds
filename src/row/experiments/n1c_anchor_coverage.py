"""N1c: is it operation COVERAGE, or anchor COUNT?

Plan: `N1C_ANCHOR_COVERAGE_PLAN.md`. Offline, worlds 0-2, development only.

Same harness as N1 and N1b: 65,536 updates, one pooled stream of 188 tasks, the
same minibatch seed sequence, scored on the canonical 64 length-3 tasks. K
length-1 anchors plus 124 - K of N1's length-3 fillers; pool order
`anchors + canonical + fillers(124 - K)`.

`n1_anchor_supply` is imported, never edited: N1's protocol hashes it.
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

PLAN = Path('N1C_ANCHOR_COVERAGE_PLAN.md')
ROOT = Path('artifacts/n1c_anchor_coverage')
OUTPUT = Path('reports/n1c_anchor_coverage.json')

WORLDS = (0, 1, 2)
ARMS = ('COVER6_K6', 'COVER5_K6', 'COVER5_K18')
COVER_STREAM = 1914
OPERATIONS = 6
FILLERS = 124
NEAR_LOW, NEAR_HIGH = 0.03, 0.08
JOBS = 3
MIN_FREE_GIB = 8.0
ENDPOINT_REPORT = Path('reports/n1_anchor_supply.json')


def protocol():
    return {'id': 'n1c-anchor-coverage-v1', 'git_commit': git_commit(), 'plan': PLAN.as_posix(),
            'worlds': list(WORLDS), 'arms': list(ARMS), 'updates': n1.UPDATES,
            'pool_stream': n1.POOL_STREAM, 'cover_stream': COVER_STREAM, 'fillers': FILLERS,
            'threshold': n1.THRESHOLD, 'near_threshold': [NEAR_LOW, NEAR_HIGH], 'jobs': JOBS,
            'pool_order': 'anchors + canonical + fillers(124-K)',
            'input_sha256': {p.as_posix(): digest(p) for p in
                             (PLAN, Path('configs/v1.yaml'), ENDPOINT_REPORT)},
            'implementation_sha256': digest(Path(__file__)),
            'n1_implementation_sha256': digest(Path(n1.__file__))}


def excluded_and_rng(world):
    rng = np.random.default_rng(np.random.SeedSequence([COVER_STREAM, world]))
    excluded = int(rng.integers(0, OPERATIONS))
    return excluded, rng


def select_anchors(arm, world, length1):
    """The registered length-1 anchor subset for an arm, in N1's original order.

    Both incomplete arms use the SAME excluded operation, drawn first from the
    world's seed so it does not depend on which arm is being built.
    """
    excluded, rng = excluded_and_rng(world)
    by_op = {op: [i for i, t in enumerate(length1) if t.program[0] == op] for op in range(OPERATIONS)}
    # One registered draw per operation, made in a fixed order so every arm
    # sees the same choices regardless of which it keeps.
    first = {op: by_op[op][int(rng.integers(0, len(by_op[op])))] for op in range(OPERATIONS)}
    others = [op for op in range(OPERATIONS) if op != excluded]
    doubled = others[int(rng.integers(0, len(others)))]
    second = [i for i in by_op[doubled] if i != first[doubled]]
    extra = second[int(rng.integers(0, len(second)))]
    pool5 = [i for op in others for i in by_op[op]]
    perm = rng.permutation(len(pool5))
    if arm == 'COVER6_K6':
        picked = [first[op] for op in range(OPERATIONS)]
    elif arm == 'COVER5_K6':
        picked = [first[op] for op in others] + [extra]
    elif arm == 'COVER5_K18':
        picked = [pool5[int(j)] for j in perm[:18]]
    else:
        raise ValueError(f'unknown arm {arm}')
    return excluded, [length1[i] for i in sorted(picked)]


def build_pool(arm, world):
    cfg, world_obj, canonical = n1.canonical_tasks(world)
    length1 = [t for t in n1.anchor_tasks(world) if t.depth == 1]
    fillers = n1.extra_length3_tasks(cfg, world_obj, FILLERS, world)
    excluded, chosen = select_anchors(arm, world, length1)
    pool = chosen + canonical + fillers[:FILLERS - len(chosen)]
    return cfg, canonical, chosen, pool, excluded


def split_by_excluded(per_task, canonical, excluded):
    """Registered descriptive split: tasks whose program uses the excluded op or not."""
    uses = [per_task[t.task_id] for t in canonical if excluded in t.program]
    free = [per_task[t.task_id] for t in canonical if excluded not in t.program]
    return {'tasks_using_excluded': len(uses), 'tasks_not_using_excluded': len(free),
            'median_using_excluded': float(np.median(uses)) if uses else None,
            'median_not_using_excluded': float(np.median(free)) if free else None}


def run_cell(arm, world):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, canonical, chosen, pool, excluded = build_pool(arm, world)
    model, drawn = n1.train_pooled(cfg, pool, canonical, n1.UPDATES,
                                   np.random.SeedSequence([n1.POOL_STREAM, world]))
    final = n1.score(model, n1._ScoredWorld(canonical))
    covered = sorted({t.program[0] for t in chosen})
    return {
        'arm': arm, 'world': world, 'updates': n1.UPDATES,
        'anchors': len(chosen), 'operations_covered': covered, 'excluded_operation': excluded,
        'pool_tasks': len(pool), 'pool_examples': sum(len(t.train_x) for t in pool),
        'scored_tasks': len(canonical),
        'terminal_median': final['median'], 'below_0.05': final['below_0.05'],
        'per_task': final['per_task'],
        'split': split_by_excluded(final['per_task'], canonical, excluded),
        'library_sha256': n1.library_sha(model), 'first_64_draws': drawn,
        'near_threshold': NEAR_LOW <= final['median'] <= NEAR_HIGH,
        'seconds': time.perf_counter() - started,
    }


EXPECTED = {'COVER6_K6': (6, 6), 'COVER5_K6': (6, 5), 'COVER5_K18': (18, 5)}


def validate_cell(record):
    if record['arm'] not in ARMS or record['world'] not in WORLDS:
        raise ValueError('cell identity')
    if not math.isfinite(record['terminal_median']) or record['terminal_median'] < 0:
        raise ValueError('bad terminal median')
    if record['pool_tasks'] != 188 or record['pool_examples'] != 24064:
        raise ValueError('pool is not the registered 188 tasks / 24,064 examples')
    if record['scored_tasks'] != 64:
        raise ValueError('every arm is scored on the canonical 64 tasks')
    count, coverage = EXPECTED[record['arm']]
    if record['anchors'] != count or len(record['operations_covered']) != coverage:
        raise ValueError(f"{record['arm']} must carry {count} anchors covering {coverage} operations")
    if coverage == 5 and record['excluded_operation'] in record['operations_covered']:
        raise ValueError('an incomplete arm covers its own excluded operation')


def _suffices(values):
    return sum(1 for v in values if v < n1.THRESHOLD) >= 2


PRIMARY = {(True, False): 'COVERAGE_MATTERS', (True, True): 'COVERAGE_NOT_NEEDED_AT_K6',
           (False, False): 'K6_INSUFFICIENT', (False, True): 'ANOMALOUS'}


def triage(records):
    by = {}
    for r in records.values():
        by.setdefault(r['arm'], {})[r['world']] = r
    if any(len(by.get(a, {})) != len(WORLDS) for a in ARMS):
        return {'verdict': 'INCOMPLETE'}
    suff = {a: _suffices([by[a][w]['terminal_median'] for w in WORLDS]) for a in ARMS}
    primary = PRIMARY[(suff['COVER6_K6'], suff['COVER5_K6'])]
    secondary = 'COUNT_COMPENSATES' if suff['COVER5_K18'] else 'COUNT_DOES_NOT_COMPENSATE'
    descriptive = {}
    for arm in ('COVER5_K6', 'COVER5_K18'):
        for w in WORLDS:
            r = by[arm][w]
            if r['terminal_median'] >= n1.THRESHOLD:
                free = r['split']['median_not_using_excluded']
                descriptive[f'{arm}_w{w}'] = 'LOCAL' if free is not None and free < n1.THRESHOLD else 'GLOBAL'
    return {'primary': primary, 'secondary': secondary, 'descriptive_split': descriptive,
            'suffices': suff,
            'by_arm': {a: {str(w): by[a][w]['terminal_median'] for w in WORLDS} for a in ARMS},
            'excluded_by_world': {str(w): by['COVER5_K6'][w]['excluded_operation'] for w in WORLDS},
            'near_threshold_cells': sorted(f'{a}_w{w}' for a in ARMS for w in WORLDS
                                           if by[a][w]['near_threshold'])}


def host_ok():
    free = psutil.virtual_memory().available / 2 ** 30
    return free, free >= MIN_FREE_GIB


def _status(state, done, total, running):
    atomic_json(ROOT / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                       'running': running, 'pid': os.getpid(), 'updated_utc': now()})


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
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
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
    print(f'N1c report: {OUTPUT}')


if __name__ == '__main__':
    main()
