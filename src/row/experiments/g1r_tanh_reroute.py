"""G1R (Tier 0, re-route census): are G1S2's collapsed depth-3 tasks stale routes?

Plan: `G1R_TANH_REROUTE_PLAN.md`. G1S2 (`reports/g1s2_tanh_budget.json`) found, post hoc, a bimodal depth-3 failure on
world 70 at lr 0.003 / gain 1: after sleep, 20 of 64 depth-3 tasks (wake u = 1; 34 at u = 2) sit at 0.71-1.31 while
the rest sit at 0.06-0.11. Neither sizing rung re-routed. Two cells per saved G1S2 wake terminal (u in {1, 2}):

- RR_SLEEP_u{u}: O5's end-of-stream construction (O6's confirmed order): `o5.reroute` (exhaustive search per stream
  task on its 64-per-task reservoir, minimal logit swap), then `o2c.consolidate`'s loop for 65,536 updates scored at
  8,192 / 16,384 / 32,768 / 65,536 (`g1s2.sleep_curve`, seed `[1941, w, stream, 64]`, lr 0.003);
- SLEEP_RR_u{u}: the same 65,536-update sleep WITHOUT re-routing (it must reproduce G1S2's `SLEEP_u{u}` curve exactly
  at all four checkpoints: anchor A0), then `o5.reroute` on the slept model, scored (`rr`), then one more O3 sleep of
  8,192 updates, scored (`rr_sleep`).

Only reuse: no learner is trained beyond what these constructions do. `o5.FrozenLibrary` is rebound to the `tanh`
search view (gate R1). At exit the runner archives logs and cells into `reports/g1r_tanh_reroute_<date>/`."""
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

from row.experiments import g1s2_tanh_budget as g2
from row.experiments import g1s_tanh_sizing as g1s
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

PLAN = Path('G1R_TANH_REROUTE_PLAN.md')
G1S2_REPORT = Path('reports/g1s2_tanh_budget.json')
G1S2_WORK = Path('artifacts/g1s2_tanh_budget/work')
ROOT = Path('artifacts/g1r_tanh_reroute')
OUTPUT = Path('reports/g1r_tanh_reroute.json')
ARCHIVE_PREFIX = 'reports/g1r_tanh_reroute_'
WORLD, STREAM = 70, 0
WAKES = (1, 2)
ARMS = ('RR_SLEEP', 'SLEEP_RR')
CHECKPOINTS = g2.CHECKPOINTS
COLLAPSED = 0.5      # G1S2's two modes: 0.058-0.107 and 0.706-1.313
RESCUED = 0.2
FITS = 0.05
IDENTITY_ARI = 0.8
JOBS = 4
MIN_FREE_GIB = 2.0


def key_of(arm, u):
    return f'{arm}_u{u}'


def cells():
    return [(a, u) for a in ARMS for u in WAKES]


def wake_terminal(u, work=G1S2_WORK):
    return Path(work) / f'u{u}' / 'lifetime' / 'model.pt'


def use_reroute_form(u):
    g2.apply_budget(u)
    o5.FrozenLibrary = FrozenTanhLibrary


def restore():
    o5.FrozenLibrary = g1s.g1t.FrozenLibrary
    g2.restore()


def point(model, canonical, st, plan, started):
    terminal = score(model, SimpleNamespace(tasks=canonical))
    return {'terminal_median': terminal['median'], 'terminal_per_task': terminal['per_task'],
            'depth': g1s.depth_profile(model, st, plan), 'd1_ari': g1s.d1_ari(model, st, plan),
            'library_sha256': o3.library_sha(model), 'seconds': time.perf_counter() - started}


def run_cell(arm, u, root, work=str(G1S2_WORK), checkpoints=CHECKPOINTS):
    torch.set_num_threads(1)
    use_reroute_form(u)
    started = time.perf_counter()
    cfg3, model, st, canonical = o3.load_shuffled_terminal(wake_terminal(u, work), WORLD, STREAM)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', WORLD, STREAM)
    pool = o2d.reservoir(st, WORLD, STREAM, o3.MEMORY)
    ids = [t.task_id for t in st]
    seed = [1941, WORLD, STREAM, o3.MEMORY]
    rec = {'wake': point(model, canonical, st, plan, started), 'pool_size': len(pool)}
    curve = {}

    def at_checkpoint(n):
        curve[str(n)] = point(model, canonical, st, plan, started)

    if arm == 'RR_SLEEP':
        rec['routes_changed'] = o5.reroute(model, st, plan, pool)
        rec['rerouted'] = point(model, canonical, st, plan, started)
        rec['sleep_lr'] = g2.sleep_curve(cfg3, model, pool, ids, seed, checkpoints, at_checkpoint)
        rec['curve'] = curve
    elif arm == 'SLEEP_RR':
        rec['sleep_lr'] = g2.sleep_curve(cfg3, model, pool, ids, seed, checkpoints, at_checkpoint)
        rec['curve'] = curve
        rec['routes_changed'] = o5.reroute(model, st, plan, pool)
        rec['rr'] = point(model, canonical, st, plan, started)
        o2c.consolidate(cfg3, model, pool, ids, seed, updates=o3.EXTRA_UPDATES)
        rec['rr_sleep'] = point(model, canonical, st, plan, started)
    else:
        raise ValueError(arm)
    final = rec['curve'][str(max(checkpoints))] if arm == 'RR_SLEEP' else rec['rr_sleep']
    rec.update({'arm': arm, 'wake_updates': u, 'world': WORLD, 'stream': STREAM, 'checkpoints': list(checkpoints),
                'terminal_median': final['terminal_median'], 'learner_class': type(model).__name__,
                'learner_implementation': TanhFormLearner.implementation, 'lr': ll._training_values(cfg3, o2.KIND)[0],
                'init_gain': tanh_form.INIT_GAIN, 'seconds': time.perf_counter() - started})
    return rec


def _points(rec):
    out = [rec['wake']] + [rec['curve'][str(n)] for n in rec['checkpoints']]
    out += [rec['rerouted']] if rec['arm'] == 'RR_SLEEP' else [rec['rr'], rec['rr_sleep']]
    return out


def validate(rec, full=True):
    key = key_of(rec['arm'], rec['wake_updates'])
    if rec['pool_size'] != 188 * o3.MEMORY:
        raise ValueError(f'{key}: pool')
    if rec['learner_class'] != 'TanhFormLearner' or rec['lr'] != g2.SETTING[0] or rec['init_gain'] != g2.SETTING[1]:
        raise ValueError(f'{key}: learner, lr or gain')
    if rec['sleep_lr'] != g2.SETTING[0]:
        raise ValueError(f'{key}: sleep lr')
    for p in _points(rec):
        if len(p['terminal_per_task']) != 64 or not all(math.isfinite(v) for v in p['terminal_per_task'].values()):
            raise ValueError(f'{key}: terminal')
        if full and [p['depth'][d]['n'] for d in '123'] != [60, 64, 64]:
            raise ValueError(f'{key}: depth denominators')
    shas = [p['library_sha256'] for p in [rec['wake']] + [rec['curve'][str(n)] for n in rec['checkpoints']]]
    if len(set(shas)) != len(shas):
        raise ValueError(f'{key}: sleep did not change the library between checkpoints')


# ------------------------------------------------------------------ reading (Tier 0; pre-stated, diagnostic)

def g1s2_reference():
    cells_ = json.loads(G1S2_REPORT.read_text())['cells']
    return {u: {'wake': cells_[f'WAKE_u{u}']['terminal_per_task'],
                'curve': {n: c['terminal_per_task'] for n, c in cells_[f'SLEEP_u{u}']['curve'].items()}}
            for u in WAKES}


def reading(records, ref):
    """A0: every reload reproduces G1S2's wake terminal, and SLEEP_RR's sleep reproduces G1S2's curve, exactly.
    STALE fraction: of the tasks G1S2 left collapsed (> 0.5 at 65,536), the share below 0.2 after re-routing the slept
    model (SLEEP_RR `rr`), pooled over u. FITS iff RR_SLEEP_u1's best checkpoint is < 0.05 with identity kept."""
    a0 = all(records[key_of(a, u)]['wake']['terminal_per_task'] == ref[u]['wake'] for a in ARMS for u in WAKES)
    a0 = a0 and all(records[key_of('SLEEP_RR', u)]['curve'][str(n)]['terminal_per_task'] == ref[u]['curve'][str(n)]
                    for u in WAKES for n in CHECKPOINTS)
    collapsed = {u: sorted(t for t, v in ref[u]['curve'][str(CHECKPOINTS[-1])].items() if v > COLLAPSED)
                 for u in WAKES}
    rescued = {u: [t for t in collapsed[u] if records[key_of('SLEEP_RR', u)]['rr']['terminal_per_task'][t] < RESCUED]
               for u in WAKES}
    n_c = sum(len(v) for v in collapsed.values())
    stale_fraction = sum(len(v) for v in rescued.values()) / n_c if n_c else None
    rr = records[key_of('RR_SLEEP', 1)]['curve']
    eligible = {n: c['terminal_median'] for n, c in rr.items() if c['d1_ari'] >= IDENTITY_ARI}
    best_n = min(eligible, key=eligible.get) if eligible else None
    best = eligible[best_n] if best_n else None
    if not a0:
        label = 'ANCHOR_MISS'
    elif best is not None and best < FITS:
        label = 'FITS'
    elif stale_fraction is not None and stale_fraction >= 0.5:
        label = 'STALE_ONLY'
    else:
        label = 'NOT_STALE'

    def summary(p):
        per = np.array(list(p['terminal_per_task'].values()))
        return {'median': p['terminal_median'], 'collapsed': int((per > COLLAPSED).sum()),
                'depth': {d: p['depth'][d]['median'] for d in '123'}, 'd1_ari': p['d1_ari']}

    table = {}
    for (a, u), rec in ((c, records[key_of(*c)]) for c in cells()):
        row = {'wake': summary(rec['wake'])}
        row.update({f's{n}': summary(rec['curve'][str(n)]) for n in CHECKPOINTS})
        if a == 'RR_SLEEP':
            row['rerouted'] = summary(rec['rerouted'])
        else:
            row['rr'], row['rr_sleep'] = summary(rec['rr']), summary(rec['rr_sleep'])
        row['routes_changed'] = rec['routes_changed']
        table[key_of(a, u)] = row
    return {'label': label, 'anchor_A0': a0, 'collapsed_tasks': {f'u{u}': len(v) for u, v in collapsed.items()},
            'rescued_by_reroute': {f'u{u}': len(v) for u, v in rescued.items()}, 'stale_fraction': stale_fraction,
            'rr_sleep_u1_best_checkpoint': best_n, 'rr_sleep_u1_best': best, 'table': table}


# ------------------------------------------------------------------ gates and protocol

def gates(work=G1S2_WORK):
    """R1: under the binding, `o5.reroute`'s search view is the tanh view and reproduces the reloaded learner's own hard
    prediction bitwise on one stream task of each depth (1, 2, 3). R2: `restore` returns the rotated view and the
    defaults. R3: both saved wake terminals exist (their sha256 is part of the protocol)."""
    out = {}
    use_reroute_form(1)
    cfg3, model, st, _ = o3.load_shuffled_terminal(wake_terminal(1, work), WORLD, STREAM)
    _, _, _, plan, _ = o2.build_stream('SHUFFLED', WORLD, STREAM)
    library = o5.FrozenLibrary(model)
    ok = isinstance(library, FrozenTanhLibrary)
    routes = model.hard_routes()
    model.eval()
    seen = set()
    for t in st:
        d = plan[t.task_id]
        if d in seen:
            continue
        seen.add(d)
        library.steps = d
        x = torch.tensor(t.eval_x[:32], dtype=torch.float32)
        with torch.no_grad():
            ok = ok and torch.equal(library.hard(x, list(routes[t.task_id][:d])), model(x, t.task_id))
    out['R1'] = bool(ok and seen == {1, 2, 3})
    restore()
    from row.experiments.audit_so1r_route_only import FrozenLibrary as Rotated
    out['R2'] = o5.FrozenLibrary is Rotated and tanh_form.INIT_GAIN == 1.0
    out['R3'] = all(wake_terminal(u, work).exists() for u in WAKES)
    out['all_pass'] = out['R1'] and out['R2'] and out['R3']
    return out


def protocol(root=ROOT, work=G1S2_WORK):
    modules = (g2, g1s, o2, o2c, o2d, o3, o5, ll, tanh_form, sys.modules[score.__module__])
    return {'id': 'g1r-tanh-reroute-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 0,
            'world': WORLD, 'stream': STREAM, 'wakes': list(WAKES), 'arms': list(ARMS),
            'checkpoints': list(CHECKPOINTS), 'collapsed': COLLAPSED, 'rescued': RESCUED, 'fits': FITS,
            'identity_ari': IDENTITY_ARI, 'jobs': JOBS, 'root': Path(root).as_posix(),
            'input_sha256': {p.as_posix(): digest(p) for p in
                             (PLAN, Path('configs/v1.yaml'), G1S2_REPORT) + tuple(wake_terminal(u, work) for u in WAKES)},
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
    curve = ' '.join(f"{n}:{rec['curve'][str(n)]['terminal_median']:.4g}" for n in rec['checkpoints'])
    extra = (f" rerouted={rec['rerouted']['terminal_median']:.4g}" if rec['arm'] == 'RR_SLEEP'
             else f" rr={rec['rr']['terminal_median']:.4g} rr_sleep={rec['rr_sleep']['terminal_median']:.4g}")
    return f"{rec['seconds']:.1f}s changed={rec['routes_changed']} curve {curve}{extra}"


def run(root=ROOT, output=OUTPUT, jobs=JOBS, stop_after=None, work=G1S2_WORK, checkpoints=CHECKPOINTS,
        archive_to=None, full=True):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    todo_cells = cells()
    expected = protocol(root, work) | {'checkpoints': list(checkpoints), 'full': full,
                                       'cells': [list(c) for c in todo_cells]}
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
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, todo = {}, []
        try:
            gpath = root / 'gates.json'
            g = json.loads(gpath.read_text()) if gpath.exists() else None
            if not g or g.get('protocol_sha256') != sha:
                g = gates(work) | {'protocol_sha256': sha}
                atomic_json(gpath, g)
            log_line(root / 'run.log', f"GATES R1={g['R1']} R2={g['R2']} R3={g['R3']}")
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, u, str(root), str(work), checkpoints): (a, u) for a, u in todo}
                for a, u in todo[:jobs]:
                    log_line(root / 'run.log', f'submitted {key_of(a, u)}')
                _status(root, 'running', len(records), len(todo_cells), [key_of(*c) for c in todo[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, u = futures.pop(done_future)
                    key = key_of(a, u)
                    record = done_future.result()
                    validate(record, full)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f'cell finished {key} {_headline(record)}')
                    _status(root, 'running', len(records), len(todo_cells), [key_of(*c) for c in futures.values()],
                            started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'gates': g, 'cells': records, 'finished_utc': now()}
            if full and tuple(checkpoints) == tuple(CHECKPOINTS):
                report['reading'] = reading(records, g1s2_reference())
            atomic_json(output, report)
            log_line(root / 'run.log', f"EXIT 0 {report.get('reading', {}).get('label')}")
            _status(root, 'complete', len(records), len(todo_cells), [], started)
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
        dry = Path('artifacts/g1r_tanh_reroute_dry')
        run(root=dry, output=dry / 'report.json', stop_after=args.stop_after, checkpoints=(64, 128),
            archive_to=dry / 'archive', full=True)
        print(f'dry run complete: {dry}')
        return
    require_clean_code(OUTPUT)
    run(archive_to=ARCHIVE_PREFIX + time.strftime('%Y%m%d', time.gmtime()))
    print(f'G1R report: {OUTPUT}')


if __name__ == '__main__':
    main()
