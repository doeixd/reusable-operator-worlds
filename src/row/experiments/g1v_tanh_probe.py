"""G1V (Tier 1, EXPLORATORY): does the `tanh` learner form online with the full pipeline, on worlds that chose nothing?

Plan: `G1V_TANH_PROBE_PLAN.md`. Decision 22 (the PI: "Use your best judgement to continue" to options (a) probe /
(b) stop; Claude chose (a)). Settings fixed by G1S / G1S2 / G1R on world 70 and NOT re-tuned here: lr 0.003, initial
gain 1, 2 wake updates per example (G1R: the better vocabulary once routes are repaired), end-of-stream exhaustive
re-routing, then 65,536 sleep updates. Worlds 71, 72 (paired with G1T's cells there) and 73, 74, 75 (fresh band 6).

- WAKE_w{w}: `o2.run_single('SHUFFLED')` verbatim under `g1s2.apply_budget(2)`;
- RR_SLEEP_w{w}: on that wake terminal, `o5.reroute` verbatim (tanh search view, G1R gate R1), then `g1s2.sleep_curve`
  for 65,536 updates scored at 8,192 / 16,384 / 32,768 / 65,536; the registered point is 65,536; the formed
  vocabulary's S1 (`g1t.formed_s1`) is measured there.

At exit the runner archives its logs and cells into `reports/g1v_tanh_probe_<date>/`."""
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
from types import SimpleNamespace

import numpy as np
import psutil
import torch
import yaml

from row.experiments import g0_generator_census as g0
from row.experiments import g1_formation_probe as g1
from row.experiments import g1r_tanh_reroute as gr
from row.experiments import g1s2_tanh_budget as g2
from row.experiments import g1s_tanh_sizing as g1s
from row.experiments import g1t_tanh_formation as g1t
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models import tanh_form
from row.models.tanh_form import FrozenTanhLibrary, TanhFormLearner

PLAN = Path('G1V_TANH_PROBE_PLAN.md')
G1T_REPORT = Path('reports/g1t_tanh_formation.json')
G1R_REPORT = Path('reports/g1r_tanh_reroute.json')
ROOT = Path('artifacts/g1v_tanh_probe')
OUTPUT = Path('reports/g1v_tanh_probe.json')
ARCHIVE_PREFIX = 'reports/g1v_tanh_probe_'
WORLDS = (71, 72, 73, 74, 75)
DRY_WORLD = 99
STREAM = 0
WAKE_UPDATES = 2
ARMS = ('WAKE', 'RR_SLEEP')
CHECKPOINTS = g2.CHECKPOINTS
REGISTERED = 65536
THRESHOLD, LOOSE, IDENTITY_ARI = 0.05, 0.1, 0.8
FORMS_K, FAILS_K = 4, 1
JOBS = 3
MIN_FREE_GIB = 2.0
DRY_SCALE = o3.DRY_SCALE


def key_of(arm, w):
    return f'{arm}_w{w}'


def cells(worlds=WORLDS):
    return [(a, w) for a in ARMS for w in worlds]


def work_of(root, w):
    return Path(root) / 'work' / f'w{w}'


def run_cell(arm, w, root, scale=1, checkpoints=CHECKPOINTS):
    torch.set_num_threads(1)
    gr.use_reroute_form(WAKE_UPDATES)
    started = time.perf_counter()
    work = work_of(root, w)
    terminal_path = work / 'lifetime' / 'model.pt'
    if arm == 'WAKE':
        rec = o2.run_single('SHUFFLED', w, STREAM, work, scale)
        rec['d1_ari'] = g1.formed_diagnostics_d1(terminal_path, w)
        resolved = yaml.safe_load((work / 'lifetime' / 'config.yaml').read_text(encoding='utf-8'))
        rec['lifetime_lr'] = resolved[f'{o2.KIND}_model']['global_learning_rate']
        rec['lifetime_updates_per_example'] = resolved[f'{o2.KIND}_model']['updates_per_example']
    elif arm == 'RR_SLEEP':
        cfg3, model, st, canonical = o3.load_shuffled_terminal(terminal_path, w, STREAM)
        _, _, _, plan, _ = o2.build_stream('SHUFFLED', w, STREAM)
        pool = o2d.reservoir(st, w, STREAM, o3.MEMORY)
        ids = [t.task_id for t in st]
        rec = {'wake': gr.point(model, canonical, st, plan, started), 'pool_size': len(pool)}
        rec['routes_changed'] = o5.reroute(model, st, plan, pool)
        rec['rerouted'] = gr.point(model, canonical, st, plan, started)
        curve = {}

        def at_checkpoint(n):
            curve[str(n)] = gr.point(model, canonical, st, plan, started)

        rec['sleep_lr'] = g2.sleep_curve(cfg3, model, pool, ids, [1941, w, STREAM, o3.MEMORY], checkpoints,
                                         at_checkpoint)
        rec['curve'] = curve
        final = curve[str(max(checkpoints))]
        rec['terminal_median'] = final['terminal_median']
        rec['terminal_per_task'] = final['terminal_per_task']
        rec['scored_tasks'] = len(final['terminal_per_task'])
        s1 = g1t.formed_s1(model, st, plan, w)
        rec['formed_s1'] = {'mapping': s1['mapping'], 'distinct_slots': s1['distinct_slots'], 'median': s1['median'],
                            'n_hyperplanes': len(s1['records']['1']['accuracy'])}
        rec['learner_class'] = type(model).__name__
        rec['lr'] = ll._training_values(cfg3, o2.KIND)[0]
        rec['init_gain'] = tanh_form.INIT_GAIN
    else:
        raise ValueError(arm)
    rec.update({'arm': arm, 'world': w, 'stream': STREAM, 'scale': scale, 'wake_updates': WAKE_UPDATES,
                'checkpoints': list(checkpoints), 'learner_implementation': TanhFormLearner.implementation,
                'seconds': time.perf_counter() - started})
    return rec


def validate(rec, scale=1):
    key = key_of(rec['arm'], rec['world'])
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if rec['arm'] == 'WAKE':
        o2.validate_cell({**rec, 'arm': 'SHUFFLED'}, scale)
        if rec['lifetime_lr'] != g2.SETTING[0] or rec['lifetime_updates_per_example'] != WAKE_UPDATES:
            raise ValueError(f'{key}: lifetime lr or updates')
        return
    if rec['pool_size'] != 188 * o3.MEMORY or rec['learner_class'] != 'TanhFormLearner':
        raise ValueError(f'{key}: pool or learner')
    if (rec['lr'], rec['sleep_lr'], rec['init_gain']) != (g2.SETTING[0], g2.SETTING[0], g2.SETTING[1]):
        raise ValueError(f'{key}: lr or gain')
    points = [rec['wake'], rec['rerouted']] + [rec['curve'][str(n)] for n in rec['checkpoints']]
    for p in points:
        if len(p['terminal_per_task']) != 64 or not all(math.isfinite(v) for v in p['terminal_per_task'].values()):
            raise ValueError(f'{key}: terminal')
        if scale == 1 and [p['depth'][d]['n'] for d in '123'] != [60, 64, 64]:
            raise ValueError(f'{key}: depth denominators')
    shas = [rec['rerouted']['library_sha256']] + [rec['curve'][str(n)]['library_sha256'] for n in rec['checkpoints']]
    if len(set(shas)) != len(shas):
        raise ValueError(f'{key}: sleep did not change the library between checkpoints')
    if rec['formed_s1']['n_hyperplanes'] != g0.HYPERPLANES:
        raise ValueError(f'{key}: formed S1 denominator')


# ------------------------------------------------------------------ reading (exploratory; no verdict)

def forms(rec):
    c = rec['curve'][str(REGISTERED)]
    return c['terminal_median'] < THRESHOLD and c['d1_ari'] >= IDENTITY_ARI


def reading(records, worlds=WORLDS):
    """Per world: FORMS iff the RR_SLEEP terminal at 65,536 is < 0.05 with d1_ari >= 0.8. Label over the 5 worlds:
    FORMS iff >= 4, FAILS iff <= 1, else MIXED. Denominator: every world (no anchor filter)."""
    sleep = {w: records[key_of('RR_SLEEP', w)] for w in worlds}
    k = sum(forms(sleep[w]) for w in worlds)
    label = 'FORMS' if k >= FORMS_K else ('FAILS' if k <= FAILS_K else 'MIXED')
    final = {w: sleep[w]['curve'][str(REGISTERED)] for w in worlds}
    g1t_cells = json.loads(G1T_REPORT.read_text())['cells']
    return {'label': label, 'forming_worlds': k, 'denominator': len(worlds),
            'below_loose': sum(final[w]['terminal_median'] < LOOSE for w in worlds),
            'terminal': {str(w): final[w]['terminal_median'] for w in worlds},
            'collapsed_tasks': {str(w): sum(v > 0.5 for v in final[w]['terminal_per_task'].values()) for w in worlds},
            'depth': {str(w): {d: final[w]['depth'][d]['median'] for d in '123'} for w in worlds},
            'd1_ari': {str(w): final[w]['d1_ari'] for w in worlds},
            'wake_terminal': {str(w): records[key_of('WAKE', w)]['terminal_median'] for w in worlds},
            'routes_changed': {str(w): sleep[w]['routes_changed'] for w in worlds},
            'curve': {str(w): {n: c['terminal_median'] for n, c in sleep[w]['curve'].items()} for w in worlds},
            'formed_s1_median': {str(w): sleep[w]['formed_s1']['median'] for w in worlds},
            'g1t_paired_rw_sleep': {str(w): g1t_cells[f'RW_SLEEP_w{w}']['terminal_median']
                                    for w in worlds if f'RW_SLEEP_w{w}' in g1t_cells}}


# ------------------------------------------------------------------ gates and protocol

def gates(worlds=WORLDS):
    """V1: the binding builds G1S2's u = 2 configuration (lr 0.003, gain 1, 2 updates per example) for every stage of
    every world, and `o5`'s search view is the tanh view. V2: G1R's gate R1 (tanh search view reproduces the learner
    bitwise at depths 1-3) on a freshly built learner. V3: `restore` returns the defaults."""
    from row.config import load_config
    out = {}
    gr.use_reroute_form(WAKE_UPDATES)
    ok = o5.FrozenLibrary is FrozenTanhLibrary and tanh_form.INIT_GAIN == g2.SETTING[1]
    for w in worlds:
        for k in (1, 2, 3):
            d = o2.stage_setup(w, k, 5000)[0].discrete_model
            ok = ok and d.global_learning_rate == g2.SETTING[0] and d.updates_per_example == WAKE_UPDATES
    out['V1'] = bool(ok)
    cfg = load_config('configs/v1.yaml')
    model = o2.planned_model(cfg, {'d1': 1, 'd2': 2, 'd3': 3})
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    v2 = []
    for t in ('d1', 'd2', 'd3'):
        model.begin_task(t)
        with torch.no_grad():
            model.task_codes[t].normal_()
    model.eval()
    for t, d in (('d1', 1), ('d2', 2), ('d3', 3)):
        library = o5.FrozenLibrary(model)
        library.steps = d
        with torch.no_grad():
            v2.append(torch.equal(library.hard(x, list(model.hard_routes()[t])), model(x, t)))
    out['V2'] = all(v2)
    gr.restore()
    from row.experiments.audit_so1r_route_only import FrozenLibrary as Rotated
    out['V3'] = o5.FrozenLibrary is Rotated and tanh_form.INIT_GAIN == 1.0
    out['all_pass'] = out['V1'] and out['V2'] and out['V3']
    return out


def protocol(root=ROOT, worlds=WORLDS, checkpoints=CHECKPOINTS):
    modules = (gr, g2, g1s, g1t, g1, g0, o2, o2c, o2d, o3, o5, ll, tanh_form, sys.modules[score.__module__])
    return {'id': 'g1v-tanh-probe-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'worlds': list(worlds), 'stream': STREAM, 'setting': list(g2.SETTING),
            'wake_updates': WAKE_UPDATES, 'checkpoints': list(checkpoints), 'registered': REGISTERED,
            'threshold': THRESHOLD, 'loose': LOOSE, 'identity_ari': IDENTITY_ARI, 'forms_k': FORMS_K,
            'fails_k': FAILS_K, 'jobs': JOBS, 'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G1T_REPORT, G1R_REPORT)},
            'implementation_sha256': digest(Path(__file__)),
            'module_sha256': {m.__name__: digest(Path(m.__file__)) for m in modules}}


def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    eta = elapsed / done * (total - done) if done else None
    atomic_json(Path(root) / 'status.json', {
        'state': state, 'cells_done': done, 'cells_total': total, 'running': running, 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)), 'updated_utc': now(),
        'eta_seconds': eta})


def _headline(rec):
    if rec['arm'] == 'WAKE':
        return f"{rec['seconds']:.1f}s terminal={rec['terminal_median']:.6g} d1_ari={rec['d1_ari']:.3f}"
    curve = ' '.join(f"{n}:{rec['curve'][str(n)]['terminal_median']:.4g}" for n in rec['checkpoints'])
    return (f"{rec['seconds']:.1f}s changed={rec['routes_changed']} rerouted={rec['rerouted']['terminal_median']:.4g}"
            f" curve {curve} s1={rec['formed_s1']['median']['1']:.3f}")


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, worlds=WORLDS,
        checkpoints=CHECKPOINTS, archive_to=None):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root, worlds, checkpoints) | {'scale': scale, 'cells': [list(c) for c in todo_cells]}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
        free = psutil.virtual_memory().available / 2 ** 30
        atomic_json(root / 'precondition.json', {'free_gib': free, 'required_gib': MIN_FREE_GIB,
                                                 'passes': free >= MIN_FREE_GIB, 'jobs': jobs, 'checked_utc': now()})
        log_line(root / 'run.log', f'PRECONDITION free={free:.2f}GiB required={MIN_FREE_GIB}')
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, todo = {}, []
        try:
            gpath = root / 'gates.json'
            g = json.loads(gpath.read_text()) if gpath.exists() else None
            if not g or g.get('protocol_sha256') != sha:
                g = gates(worlds) | {'protocol_sha256': sha}
                atomic_json(gpath, g)
            log_line(root / 'run.log', f"GATES V1={g['V1']} V2={g['V2']} V3={g['V3']}")
            if not g['all_pass']:
                raise RuntimeError(f'gates failed: {g}')
            for c in todo_cells:
                key = key_of(*c)
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append(c)
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            ready = [c for c in todo if c[0] == 'WAKE' or key_of('WAKE', c[1]) in records]
            waiting = [c for c in todo if c not in ready]
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, str(root), scale, checkpoints): (a, w) for a, w in ready}
                for a, w in ready[:jobs]:
                    log_line(root / 'run.log', f'submitted {key_of(a, w)}')
                _status(root, 'running', len(records), total, [key_of(*c) for c in ready[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, w = futures.pop(done_future)
                    key = key_of(a, w)
                    record = done_future.result()
                    validate(record, scale)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f'cell finished {key} {_headline(record)}')
                    for dep in [c for c in waiting if c[1] == w and a == 'WAKE']:
                        waiting.remove(dep)
                        futures[pool.submit(run_cell, dep[0], w, str(root), scale, checkpoints)] = dep
                        log_line(root / 'run.log', f'submitted {key_of(*dep)}')
                    _status(root, 'running', len(records), total, [key_of(*c) for c in futures.values()], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            if waiting:
                raise RuntimeError(f'dependent cells never submitted: {waiting}')
            report = {**manifest, 'complete': True, 'gates': g, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['reading'] = reading(records, worlds)
            atomic_json(output, report)
            log_line(root / 'run.log', f"EXIT 0 {report.get('reading', {}).get('label')}")
            _status(root, 'complete', len(records), total, [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            log_line(root / 'run.log', 'FAILED ' + traceback.format_exc().replace('\n', ' | '))
            _status(root, 'failed', len(records), len(todo_cells), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(root / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise
        finally:
            if archive_to is not None:
                g2.archive(root, archive_to)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gates', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.gates:
        print(json.dumps(gates(), indent=1))
        return
    if args.dry_run:
        dry = Path('artifacts/g1v_tanh_probe_dry')
        run(cells((DRY_WORLD,)), root=dry, output=dry / 'report.json', scale=DRY_SCALE, stop_after=args.stop_after,
            worlds=(DRY_WORLD,), checkpoints=(64, 128), archive_to=dry / 'archive')
        print(f'dry run complete: {dry}')
        return
    require_clean_code(OUTPUT)
    run(cells(), archive_to=ARCHIVE_PREFIX + time.strftime('%Y%m%d', time.gmtime()))
    print(f'G1V report: {OUTPUT}')


if __name__ == '__main__':
    main()
