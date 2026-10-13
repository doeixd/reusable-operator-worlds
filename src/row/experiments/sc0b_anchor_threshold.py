"""SC0b Tier 1 (exploratory): where between 16 and 60 single-operation anchors does the CONFIRMED online baseline
(wake + end-of-stream exhaustive re-route + sleep, O6) start forming reliably? Follows SC0 (`GRADED`: 4 and 8 fail
5/5, 16 passes 2/5, 60 passes).

Plan: `SC0B_ANCHOR_THRESHOLD_PLAN.md`. World 64, streams 0-4 (the SC0 streams), doses 24, 32, 40, built and run by
the frozen SC0 functions `build_dose_stream` and `run_dose` (doses nest with the SC0 doses within each stream).
Anchor cell: the SC0 cell D16_w64_s0 rerun, which must reproduce the SC0 record bitwise (library and per-task
terminal NMSE). Dry-run world 66.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
import torch

from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments import sc0_anchor_dose as sc0
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('SC0B_ANCHOR_THRESHOLD_PLAN.md')
ROOT = Path('artifacts/sc0b_anchor_threshold')
DRY_ROOT = Path('artifacts/sc0b_anchor_threshold_dry')
OUTPUT = Path('reports/sc0b_anchor_threshold.json')
SC0_REPORT = Path('reports/sc0_anchor_dose.json')
SC0_DRY_CELL = Path('artifacts/sc0_anchor_dose_dry/cells/D4_w66_s0.json')
WORLD = 64
STREAMS = (0, 1, 2, 3, 4)
DOSES = (24, 32, 40)
ANCHOR = (16, 64, 0)    # SC0 cell rerun; must reproduce bitwise
THRESHOLD = sc0.THRESHOLD
JOBS = 3
MIN_FREE_GIB = 2.0      # PI 2026-10-03 instruction
DRY_WORLD = 66
DRY_SCALE = 16


def key_of(n, w, s):
    return f'D{n}_w{w}_s{s}'


def cells(world=WORLD):
    """The anchor cell first (it decides interpretability), then the largest streams first."""
    return [ANCHOR] + [(n, world, s) for n in sorted(DOSES, reverse=True) for s in STREAMS]


def run_cell(n, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = sc0.run_dose(n, w, s, Path(root) / 'work' / key_of(n, w, s), scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    sc0.validate(rec)


def reproduces(new, old):
    """Bitwise: same library after sleep and identical per-task terminal NMSE."""
    return (new['library_sha256'] == old['library_sha256']
            and new['canonical_per_task'] == old['canonical_per_task'])


def passing(m):
    return sc0.passing(m)


def label(anchor_ok, k):
    """k: dose -> passing cells of 5."""
    if not anchor_ok:
        return 'UNINTERPRETABLE'
    if k[32] == 5 and k[40] >= 4:
        return 'LENGTHS_SUFFICE'
    if k[40] <= 1:
        return 'LENGTHS_INSUFFICIENT'
    return 'GRADED'


def stream_threshold(curve):
    """Smallest dose that passes with every larger tested dose also passing (persistence), or None."""
    doses = sorted(curve)
    for i, d in enumerate(doses):
        if all(curve[x] for x in doses[i:]):
            return d
    return None


def summarize(records, sc0_cells):
    rec = {(r['dose'], r['stream']): r for k, r in records.items() if k != key_of(*ANCHOR)}
    k = {n: sum(passing(rec[(n, s)]['canonical_median']) for s in STREAMS) for n in DOSES}
    anchor_ok = reproduces(records[key_of(*ANCHOR)], sc0_cells[key_of(*ANCHOR)])
    curves = {}
    for s in STREAMS:
        c = {n: passing(sc0_cells[key_of(n, WORLD, s)]['canonical_median']) for n in sc0.DOSES}
        c.update({n: passing(rec[(n, s)]['canonical_median']) for n in DOSES})
        curves[s] = c
    return {'k': {str(n): k[n] for n in DOSES}, 'anchor_reproduces': anchor_ok, 'label': label(anchor_ok, k),
            **{key: {str(n): float(np.median([rec[(n, s)][key] for s in STREAMS])) for n in DOSES}
               for key in ('canonical_median', 'wake_canonical_median', 'reroute_canonical_median')},
            'per_stream_pass': {str(s): {str(n): v for n, v in sorted(c.items())} for s, c in curves.items()},
            'per_stream_threshold': {str(s): stream_threshold(c) for s, c in curves.items()},
            'pass_fail_monotone_streams': sum(
                all(c[a] <= c[b] for a, b in zip(sorted(c), sorted(c)[1:])) for c in curves.values())}


def protocol(root=ROOT):
    mods = {'lifetime': ll, 'o2': o2, 'o2c': o2c, 'o2d': o2d, 'o3': o3, 'o5': o5, 'sc0': sc0}
    return {'id': 'sc0b-anchor-threshold', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'root': Path(root).as_posix(), 'world': WORLD, 'streams': list(STREAMS),
            'doses': list(DOSES), 'anchor_cell': list(ANCHOR), 'anchor_seed': sc0.ANCHOR_SEED,
            'memory': o3.MEMORY, 'sleep_updates': o3.EXTRA_UPDATES, 'threshold': THRESHOLD,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml', SC0_REPORT.as_posix())},
            'implementation_sha256': digest(Path(__file__)),
            **{f'{k}_sha256': digest(Path(m.__file__)) for k, m in mods.items()}}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, reference=None):
    """reference: callable(records) -> extra report fields, applied once every cell is done."""
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
                key = key_of(n, w, s)
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} '
                                       f'free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, n, w, s, str(root), scale): (n, w, s) for n, w, s in todo}
                while futures:
                    done = next(as_completed(futures))
                    n, w, s = futures.pop(done)
                    key = key_of(n, w, s)
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
                            [key_of(x, y, z) for x, y, z in futures.values()][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if reference is not None:
                report.update(reference(records))
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
        # The SC0 dry cell D4_w66_s0 (scale 16) must be reproduced bitwise: the anchor gate, exercised before launch.
        old = json.loads(SC0_DRY_CELL.read_text())['record']

        def dry_reference(records):
            return {'dry_anchor_reproduces': reproduces(records['D4_w66_s0'], old)}

        run([(4, DRY_WORLD, 0), (40, DRY_WORLD, 0)], root=DRY_ROOT, output=DRY_ROOT / 'report.json',
            scale=DRY_SCALE, stop_after=args.stop_after, reference=dry_reference)
        print('SC0b dry run complete; anchor reproduces:',
              json.loads((DRY_ROOT / 'report.json').read_text())['dry_anchor_reproduces'])
        return
    require_clean_code(OUTPUT)
    sc0_cells = json.loads(SC0_REPORT.read_text())['cells']
    run(cells(), reference=lambda records: {'summary': summarize(records, sc0_cells)})
    print(f'SC0b report: {OUTPUT}')


if __name__ == '__main__':
    main()
