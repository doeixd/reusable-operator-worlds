"""SC0 Tier 1 (exploratory): how many single-operation anchors does the CONFIRMED online baseline (wake + end-of-stream
exhaustive re-route + sleep, O6) need? The answer is the opportunity gate for an adaptive task sampler (Track S, SC1).

Plan: `SC0_ANCHOR_DOSE_PLAN.md`. Development band 5: world 64, streams 0-4; dry-run world 66.
Stream (dose n): O2's SHUFFLED stream for (w, s) (60 length-1, 64 length-2, 64 canonical length-3, one random order)
with all but n of the 60 length-1 tasks removed, the rest of the order unchanged. The n kept anchors are the first n
of a permutation of the 60 (sorted by task id) seeded [6100, w, s], so doses are NESTED within a stream. Dose 60 is
O2's stream exactly.
Protocol (O6's REROUTE_SLEEP construction): lifetime on the stream (O2's planned-depth learner, the stream's replay
seed); reservoir of 64 examples per task (`o2d.reservoir`); exhaustive re-route of every task (`o5.reroute`);
`o2c.consolidate` for 8,192 updates seeded [1941, w, s, 64]. Scored on the canonical 64 (terminal median NMSE).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('SC0_ANCHOR_DOSE_PLAN.md')
ROOT = Path('artifacts/sc0_anchor_dose')
DRY_ROOT = Path('artifacts/sc0_anchor_dose_dry')
OUTPUT = Path('reports/sc0_anchor_dose.json')
WORLD = 64
STREAMS = (0, 1, 2, 3, 4)
DOSES = (4, 8, 16)            # the decision doses, five streams each
CONTROLS = ((60, 0), (0, 0))  # (dose, stream): the confirmed construction and the floor, one stream each
ANCHOR_SEED = 6100
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 66
DRY_SCALE = 16


def cells(world=WORLD):
    """Largest streams first (cost grows with stream length), the floor last."""
    dec = [(n, world, s) for n in sorted(DOSES, reverse=True) for s in STREAMS]
    return [(60, world, 0)] + dec + [(0, world, 0)]


def build_dose_stream(n, w, s):
    cfg3, world3, st, plan, canonical = o2.build_stream('SHUFFLED', w, s)
    anchors = sorted(t.task_id for t in st if plan[t.task_id] == 1)
    if len(anchors) != 60:
        raise ValueError('expected 60 length-1 anchors')
    perm = np.random.default_rng(np.random.SeedSequence([ANCHOR_SEED, w, s])).permutation(len(anchors))
    keep = {anchors[int(i)] for i in perm[:n]}
    tasks = [t for t in st if plan[t.task_id] != 1 or t.task_id in keep]
    return cfg3, world3, tasks, {t.task_id: plan[t.task_id] for t in tasks}, canonical, sorted(keep)


def run_dose(n, w, s, work: Path, scale=1):
    cfg3, world3, st, plan, canonical, kept = build_dose_stream(n, w, s)
    mixed = dataclasses.replace(world3, tasks=tuple(st))
    _, model = o2.lifetime(cfg3, mixed, work / 'lifetime', o2.planned_model(cfg3, plan), o2.replay_seed_for(w, s),
                           scale)

    def scored(ts):
        return score(model, SimpleNamespace(tasks=ts))['median'] if ts else None

    by_len = {d: [t for t in st if plan[t.task_id] == d] for d in (1, 2)}
    rec = {'dose': n, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st), 'anchor_ids': kept,
           'depth_histogram': {str(d): sum(1 for v in plan.values() if v == d) for d in (1, 2, 3)},
           'length1_positions': [i for i, t in enumerate(st) if plan[t.task_id] == 1],
           'covered_operations': len({t.program.primitive_ids[0] for t in by_len[1]}),
           'wake_canonical_median': scored(canonical), 'wake_anchor1_median': scored(by_len[1]),
           'wake_anchor2_median': scored(by_len[2])}
    pool = o2d.reservoir(st, w, s, o3.MEMORY)
    before = library_sha(model)
    rec['routes_changed'] = o5.reroute(model, st, plan, pool)
    rec['reroute_canonical_median'] = scored(canonical)
    o2c.consolidate(cfg3, model, pool, [t.task_id for t in st], [1941, w, s, o3.MEMORY], updates=o3.EXTRA_UPDATES)
    term = score(model, SimpleNamespace(tasks=canonical))
    rec.update({'pool_size': len(pool), 'extra_updates': o3.EXTRA_UPDATES,
                'library_sha256_before_sleep': before, 'library_sha256': library_sha(model),
                'canonical_median': term['median'], 'canonical_per_task': term['per_task'],
                'scored_canonical': len(term['per_task']),
                'anchor2_median': scored(by_len[2]), 'anchor1_median': scored(by_len[1])})
    return rec


def run_cell(n, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_dose(n, w, s, Path(root) / 'work' / f'D{n}_w{w}_s{s}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"D{rec['dose']}_w{rec['world']}_s{rec['stream']}"
    n = rec['dose']
    if (rec['scored_canonical'] != 64 or rec['stream_tasks'] != 128 + n
            or rec['depth_histogram'] != {'1': n, '2': 64, '3': 64} or len(rec['anchor_ids']) != n
            or rec['pool_size'] != (128 + n) * o3.MEMORY or rec['extra_updates'] != o3.EXTRA_UPDATES):
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before_sleep']:
        raise ValueError(f'{key}: sleep left the library unchanged')


def passing(m):
    return m is not None and math.isfinite(m) and m < THRESHOLD


def label(control_ok, k):
    """k: dose -> passing cells of 5."""
    if not control_ok:
        return 'UNINTERPRETABLE'
    if k[4] == 5:
        return 'NO_OPPORTUNITY'
    if k[4] <= 1 and k[16] >= 4:
        return 'OPPORTUNITY'
    return 'GRADED'


def summarize(records):
    rec = {(r['dose'], r['stream']): r for r in records.values()}
    k = {n: sum(passing(rec[(n, s)]['canonical_median']) for s in STREAMS) for n in DOSES}
    top_passes = passing(rec[(60, 0)]['canonical_median'])
    floor_fails = not passing(rec[(0, 0)]['canonical_median'])

    def med(n, key):
        return float(np.median([rec[(n, s)][key] for s in STREAMS]))

    return {'k': {str(n): k[n] for n in DOSES}, 'top_passes': top_passes, 'floor_fails': floor_fails,
            'label': label(top_passes and floor_fails, k),
            **{key: {str(n): med(n, key) for n in DOSES}
               for key in ('canonical_median', 'wake_canonical_median', 'reroute_canonical_median')},
            'controls': {str(n): rec[(n, 0)]['canonical_median'] for n, _ in CONTROLS},
            'monotone_streams': sum(rec[(4, s)]['canonical_median'] >= rec[(8, s)]['canonical_median']
                                    >= rec[(16, s)]['canonical_median'] for s in STREAMS)}


def protocol(root=ROOT):
    mods = {'lifetime': ll, 'o2': o2, 'o2c': o2c, 'o2d': o2d, 'o3': o3, 'o5': o5}
    return {'id': 'sc0-anchor-dose', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'root': Path(root).as_posix(), 'world': WORLD, 'streams': list(STREAMS),
            'doses': list(DOSES), 'controls': [list(c) for c in CONTROLS], 'anchor_seed': ANCHOR_SEED,
            'memory': o3.MEMORY, 'sleep_updates': o3.EXTRA_UPDATES, 'threshold': THRESHOLD,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)),
            **{f'{k}_sha256': digest(Path(m.__file__)) for k, m in mods.items()}}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None):
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
            for n, w, s in todo_cells:
                key = f'D{n}_w{w}_s{s}'
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((n, w, s))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, n, w, s, str(root), scale): (n, w, s) for n, w, s in todo}
                while futures:
                    done = next(as_completed(futures))
                    n, w, s = futures.pop(done)
                    key = f'D{n}_w{w}_s{s}'
                    record = done.result()
                    validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"canonical={record['canonical_median']:.4g} "
                                               f"wake={record['wake_canonical_median']:.4g} "
                                               f"reroute={record['reroute_canonical_median']:.4g} "
                                               f"changed={record['routes_changed']}")
                    _status(root, 'running', len(records), total,
                            [f'D{x}_w{y}_s{z}' for x, y, z in futures.values()][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records)
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
        run([(60, DRY_WORLD, 0), (4, DRY_WORLD, 0), (0, DRY_WORLD, 0)], root=DRY_ROOT,
            output=DRY_ROOT / 'report.json', scale=DRY_SCALE, stop_after=args.stop_after)
        print('SC0 dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'SC0 report: {OUTPUT}')


if __name__ == '__main__':
    main()
