"""C-seal (Tier 2, SEALED): does the B2-hard v2 construction discover which tasks iterate, without being told, while
forming the vocabulary, on sealed worlds?

Plan: `B2_SEALED_CONFIRMATION_PLAN.md`. Sealed band 1100-1129 (decision 19): worlds 1100-1114; 1115-1129 stay sealed.
Every cell is B2-hard v2's frozen cell function (`b2_hard_v2.run_cell`, arms ONESTEP / ZERO / GIVEN), unchanged.
- ONESTEP: 15 worlds x streams 0-1 (30 cells), the confirmatory arm.
- ZERO and GIVEN: 15 worlds x stream 0 (15 cells each): the paired control for the start (mechanism) and the
  identity-given reference (instrument gate and cost of discovery).
- Anchor: B2-hard v2's development cell ONESTEP_w57_s0, rerun first, must reproduce its committed record bitwise.
Registered early stops: an anchor that does not reproduce stops the run; once ONESTEP has 4 formation failures
(FORMATION_BROKEN determined), its pending cells are cancelled and ZERO and GIVEN still run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
import torch

from row.experiments import b2_hard_v2 as v2
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('B2_SEALED_CONFIRMATION_PLAN.md')
ROOT = Path('artifacts/b2_sealed')
DRY_ROOT = Path('artifacts/b2_sealed_dry')
OUTPUT = Path('reports/b2_sealed.json')
DEV_REPORT = Path('reports/b2_hard_v2.json')
DEV_DRY_CELLS = Path('artifacts/b2_hard_v2_dry/cells')
WORLDS = tuple(range(1100, 1115))
ONESTEP_STREAMS = (0, 1)
PAIRED_STREAM = 0
ANCHOR = ('ONESTEP', 57, 0)
REPRO_FIELDS = ('library_sha256', 'canonical_per_task', 'loop_per_task', 'anchor_per_task', 'decisions',
                'wake_rooted_ops')
N_LOOP, N_CAND = v2.N_LOOP, v2.N_CAND
THRESHOLD, LOOP_BAR = v2.THRESHOLD, v2.LOOP_BAR
K_FORMATION_MIN = 27     # of 30
N_TERMINAL_MIN = 27      # of 30
GIVEN_GATE_MIN = 12      # of 15
FAILURES_TO_STOP = 30 - K_FORMATION_MIN + 1
JOBS = 4
MIN_FREE_GIB = 2.0       # PI 2026-10-03 instruction
DRY_WORLD = 67           # B2-hard v2's dry-run world (development band 5)
DRY_SCALE = 16


def key_of(arm, w, s):
    return f'{arm}_w{w}_s{s}'


def cells(worlds=WORLDS):
    """Anchor first, then the longest cells first: ONESTEP (~42 min), ZERO (~40), GIVEN (~25)."""
    return ([ANCHOR] + [('ONESTEP', w, s) for s in ONESTEP_STREAMS for w in worlds]
            + [('ZERO', w, PAIRED_STREAM) for w in worlds] + [('GIVEN', w, PAIRED_STREAM) for w in worlds])


def reproduces(new, old):
    return all(new.get(f) == old.get(f) for f in REPRO_FIELDS)


def passing(m, bar=THRESHOLD):
    return m is not None and math.isfinite(m) and m < bar


def label(gate_ok, k_formation, auc_median, recall, specificity, n_terminal):
    if not gate_ok:
        return 'UNINTERPRETABLE'
    if k_formation < K_FORMATION_MIN:
        return 'FORMATION_BROKEN'
    if auc_median >= 0.9 and recall >= 0.8 and specificity >= 0.95 and n_terminal >= N_TERMINAL_MIN:
        return 'CONFIRMED'
    return 'NOT_DISCOVERED'


def roots_label(r_one, r_zero):
    if r_zero <= 1:
        return 'NO_ROOTS_IN_CONTROL'
    if r_one <= 1:
        return 'ROOTS_PREVENTED'
    return 'ROOTS_NOT_PREVENTED'


def summarize(records, dev_anchor, worlds=WORLDS):
    """Registered labels from complete cells. ONESTEP cells cancelled by the early stop count as not formed."""
    anchor_ok = key_of(*ANCHOR) in records and reproduces(records[key_of(*ANCHOR)], dev_anchor)
    one = [records.get(key_of('ONESTEP', w, s)) for s in ONESTEP_STREAMS for w in worlds]
    zero = [records.get(key_of('ZERO', w, PAIRED_STREAM)) for w in worlds]
    given = [records.get(key_of('GIVEN', w, PAIRED_STREAM)) for w in worlds]
    done = [r for r in one if r is not None]
    k_formation = sum(passing(r['canonical_median']) for r in done)
    k_given = sum(passing(r['canonical_median']) for r in given if r is not None)
    gate_ok = anchor_ok and all(r is not None for r in given) and k_given >= GIVEN_GATE_MIN
    complete = len(done) == len(one)
    auc_median = float(np.median([r['wake_state_auc'] for r in done])) if done else float('nan')
    recall = sum(r['loops_kept'] for r in done) / (N_LOOP * len(one))
    specificity = sum(r['straight_kept_count'] for r in done) / ((N_CAND - N_LOOP) * len(one))
    n_terminal = sum(passing(r['loop_median'], LOOP_BAR) for r in done)
    r_one = sum(bool(r['wake_rooted_ops']) for r in done)
    r_zero = sum(bool(r['wake_rooted_ops']) for r in zero if r is not None)
    lab = label(gate_ok, k_formation, auc_median, recall, specificity, n_terminal)
    if lab == 'CONFIRMED' and not complete:
        raise ValueError('CONFIRMED requires every ONESTEP cell')

    def med(rs, key):
        rs = [r for r in rs if r is not None]
        return float(np.median([r[key] for r in rs])) if rs else None

    pairs = [(o, g) for o, g in zip(one[:len(worlds)], given) if o is not None and g is not None]
    keys = ('canonical_median', 'loop_median', 'anchor_median', 'wake_canonical_median', 'wake_loop_median',
            'reroute_canonical_median', 'count_accuracy_median')
    return {'onestep_cells_complete': len(done), 'onestep_cells': len(one), 'anchor_reproduces': anchor_ok,
            'given_formation': k_given, 'gate_ok': gate_ok, 'k_formation': k_formation,
            'wake_state_auc_median': auc_median, 'loop_recall': recall, 'straight_specificity': specificity,
            'n_terminal': n_terminal, 'label': lab, 'rooted_cells': {'ONESTEP': r_one, 'ZERO': r_zero},
            'roots_label': roots_label(r_one, r_zero),
            'zero_formation': sum(passing(r['canonical_median']) for r in zero if r is not None),
            'zero_failures': sum(not passing(r['canonical_median']) for r in zero if r is not None),
            'zero_failures_rooted': sum(not passing(r['canonical_median']) and bool(r['wake_rooted_ops'])
                                        for r in zero if r is not None),
            'n_loop_at_or_below_given': sum(o['loop_median'] <= g['loop_median'] for o, g in pairs),
            'paired_loop_ratio_median': (float(np.median([o['loop_median'] / g['loop_median'] for o, g in pairs]))
                                         if pairs else None),
            **{k: {'ONESTEP': med(one, k), 'ZERO': med(zero, k), 'GIVEN': med(given, k)} for k in keys}}


def protocol(root=ROOT):
    return {'id': 'b2-sealed', 'git_commit': v2.o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 2,
            'exploratory': False, 'root': Path(root).as_posix(), 'worlds': list(WORLDS),
            'onestep_streams': list(ONESTEP_STREAMS), 'paired_stream': PAIRED_STREAM, 'anchor': list(ANCHOR),
            'repro_fields': list(REPRO_FIELDS), 'k_formation_min': K_FORMATION_MIN, 'n_terminal_min': N_TERMINAL_MIN,
            'given_gate_min': GIVEN_GATE_MIN, 'failures_to_stop': FAILURES_TO_STOP, 'jobs': JOBS,
            'v2_protocol': {k: v for k, v in v2.protocol().items()
                            if k not in ('git_commit', 'plan', 'root', 'worlds', 'input_sha256', 'tier',
                                         'exploratory', 'id')},
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml', DEV_REPORT.as_posix())},
            'implementation_sha256': digest(Path(__file__))}


def _status(root, state, done, total, running, started, note=None):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta), 'note': note})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, anchor_record=None,
        reference=None):
    """anchor_record: the development record the anchor cell must reproduce (None: no anchor rule, dry run).
    reference: callable(records, stopped) -> extra report fields."""
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
        records, todo, stopped = {}, [], []
        try:
            for arm, w, s in todo_cells:
                key = key_of(arm, w, s)
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} '
                                       f'jobs={jobs} free={free:.1f}GiB')

            def onestep_failures():
                return sum(not passing(r['canonical_median']) for k, r in records.items()
                           if r['arm'] == 'ONESTEP' and r['world'] in WORLDS)

            def check_stops(futures):
                if anchor_record is not None and key_of(*ANCHOR) in records and 'anchor' not in stopped:
                    if not reproduces(records[key_of(*ANCHOR)], anchor_record):
                        stopped.append('anchor')
                        for f in futures:
                            f.cancel()
                        log_line(root / 'run.log', 'EARLY STOP: anchor did not reproduce; pending cells cancelled')
                if 'formation' not in stopped and onestep_failures() >= FAILURES_TO_STOP:
                    stopped.append('formation')
                    n = sum(f.cancel() for f, c in futures.items() if c[0] == 'ONESTEP')
                    log_line(root / 'run.log', f'EARLY STOP: {FAILURES_TO_STOP} ONESTEP formation failures '
                                               f'(FORMATION_BROKEN determined); {n} pending ONESTEP cells cancelled')

            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(v2.run_cell, a, w, s, str(root), scale): (a, w, s) for a, w, s in todo}
                check_stops(futures)
                while futures:
                    live = {f: c for f, c in futures.items() if not f.cancelled()}
                    if not live:
                        break
                    done = next(as_completed(live))
                    a, w, s = futures.pop(done)
                    key = key_of(a, w, s)
                    record = done.result()
                    v2.validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    extra = (f" auc={record['wake_state_auc']:.3f} kept={record['loops_kept']}"
                             f" count={record['straight_kept_count']}" if 'wake_state_auc' in record else '')
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"canonical={record['canonical_median']:.4g} "
                                               f"loop={record['loop_median']:.4g} "
                                               f"wake={record['wake_canonical_median']:.4g} "
                                               f"rooted={record['wake_rooted_ops']}{extra}")
                    check_stops(futures)
                    _status(root, 'running', len(records), total,
                            [key_of(*c) for f, c in futures.items() if f.running()][:jobs], started,
                            note=('stopped: ' + ','.join(stopped)) if stopped else None)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': not stopped, 'early_stop': stopped, 'cells': records,
                      'finished_utc': now()}
            if reference is not None:
                report.update(reference(records, stopped))
            atomic_json(output, report)
            _status(root, 'complete', len(records), total, [], started,
                    note=('stopped: ' + ','.join(stopped)) if stopped else None)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'early_stop': stopped,
                                             'finished_utc': now()})
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
        # every arm on B2-hard v2's dry world at scale 16; each must reproduce B2-hard v2's dry record bitwise
        olds = {a: json.loads((DEV_DRY_CELLS / f'{key_of(a, DRY_WORLD, 0)}.json').read_text())['record']
                for a in v2.ARMS}

        def dry_reference(records, stopped):
            return {'dry_reproduces': {a: reproduces(records[key_of(a, DRY_WORLD, 0)], olds[a]) for a in v2.ARMS}}

        run([(a, DRY_WORLD, 0) for a in v2.ARMS], root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, reference=dry_reference)
        print('B2 sealed dry run complete:', json.loads((DRY_ROOT / 'report.json').read_text())['dry_reproduces'])
        return
    require_clean_code(OUTPUT)
    dev_anchor = json.loads(DEV_REPORT.read_text())['cells'][key_of(*ANCHOR)]
    run(cells(), anchor_record=dev_anchor,
        reference=lambda records, stopped: {'summary': summarize(records, dev_anchor)})
    print(f'B2 sealed report: {OUTPUT}')


if __name__ == '__main__':
    sys.exit(main())
