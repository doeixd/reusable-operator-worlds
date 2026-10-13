"""G3b: at what observation noise does route choice bind on the formed vocabularies? (Tier 0 census, no new lifetime.)

Plan: `G3B_NOISE_CENSUS_PLAN.md`. G3a's six rebuilt vocabularies (G1's RW_SLEEP, anchor A1 bitwise) and G3a's 64
held-out depth-3 programs per world. Gaussian noise is added to the SUPPORT targets only, with variance `v * var(y)`
per program (`v` in NOISE, the noise-to-signal ratio in NMSE units); the query stays clean, so regret is the true query
cost of choosing from noisy support. Everything else is G3a's estimand verbatim (SG0's argmin, split, bootstrap floor
and near-tie eps, seeded per program exactly as G3a). Anchor A3: at `v = 0` every program's row reproduces G3a's
committed row bitwise.
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

from row.experiments import g1_formation_probe as g1
from row.experiments import g3a_inference_census as g3a
from row.experiments import o2_online_reliability as o2
from row.experiments import sg0_headroom_gate as sg0
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('G3B_NOISE_CENSUS_PLAN.md')
G1_REPORT = g3a.G1_REPORT
G3A_REPORT = Path('reports/g3a_inference_census.json')
ROOT = Path('artifacts/g3b_noise_census')
DRY_ROOT = Path('artifacts/g3b_noise_census_dry')
OUTPUT = Path('reports/g3b_noise_census.json')
WORLDS, SHARED, SUPPORTS, DEPTH, PROGRAMS = g3a.WORLDS, g3a.SHARED, g3a.SUPPORTS, g3a.DEPTH, g3a.PROGRAMS
NOISE = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
NOISE_SEED = 3201
PLAUSIBLE_V = 0.1      # judgement: the largest noise level counted as one a formation run could plausibly face
BIND_MIN, BIND_MARGIN = 0.10, 0.10   # window margin 0.10: per-cell false-fire <= 0.03 when both arms bind equally
FAIL = g3a.FAIL
JOBS = 3
MIN_FREE_GIB = 2.0


def jobs():
    return g3a.jobs()


def predictions(library, x):
    """(examples, routes, d) outputs of every route: the first half of `sg0.per_example_squared_error`, verbatim."""
    n, dimension = x.shape
    with torch.no_grad():
        z = x
        route_count = 1
        for _ in range(DEPTH):
            z = library.candidates(z.reshape(-1, dimension))
            z = z.reshape(n, route_count * library.slots, dimension)
            route_count *= library.slots
    return z


def errors_of(z, y):
    """The second half of `sg0.per_example_squared_error`, verbatim: (routes, examples)."""
    with torch.no_grad():
        return torch.mean((z - y.unsqueeze(1)) ** 2, dim=2).transpose(0, 1).contiguous()


def measure_task(library, task, world, index):
    x = torch.tensor(task['train_x'], dtype=torch.float32)
    y_clean = np.asarray(task['train_y'])
    qx = torch.tensor(task['eval_x'], dtype=torch.float32)
    qy = torch.tensor(task['eval_y'], dtype=torch.float32)
    zs = predictions(library, x)
    errors_q = errors_of(predictions(library, qx), qy)
    target = np.asarray(task['eval_y'], dtype=np.float64)
    denominator = float(np.mean(np.square(target - np.mean(target, axis=0, keepdims=True))))
    split_rng = np.random.default_rng(np.random.SeedSequence([sg0.SPLIT_SEED, world, DEPTH, index]))
    permutation = split_rng.permutation(errors_q.shape[1])
    q_a = torch.tensor(np.sort(permutation[: errors_q.shape[1] // 2]), dtype=torch.long)
    q_b = torch.tensor(np.sort(permutation[errors_q.shape[1] // 2:]), dtype=torch.long)
    r_star, _ = sg0.argmin_route(errors_q, q_a)
    nmse_star_b = sg0.half_nmse(errors_q, r_star, q_b, denominator)
    signal = float(np.var(y_clean))
    out = {}
    for v in NOISE:
        if v == 0.0:
            y = torch.tensor(task['train_y'], dtype=torch.float32)
        else:
            eps = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, world, index])).normal(size=y_clean.shape)
            y = torch.tensor(y_clean + math.sqrt(v * signal) * eps, dtype=torch.float32)
        errors_s = errors_of(zs, y)
        if not (bool(torch.isfinite(errors_s).all()) and bool(torch.isfinite(errors_q).all())):
            raise ValueError('nonfinite squared-error matrix')
        by = {}
        for support in SUPPORTS:
            boot_rng = np.random.default_rng(
                np.random.SeedSequence([sg0.BOOTSTRAP_SEED, world, DEPTH, support, index]))
            r_hat, values = sg0.argmin_route(errors_s, torch.arange(support, dtype=torch.long))
            nmse_hat_b = sg0.half_nmse(errors_q, r_hat, q_b, denominator)
            ordered = torch.sort(values).values
            best, second = float(ordered[0]), float(ordered[1])
            near = torch.nonzero(values <= (1.0 + sg0.NEAR_TIE_EPS) * best, as_tuple=False).flatten()
            tie_nmse = [sg0.half_nmse(errors_q, int(r), q_b, denominator) for r in near]
            deltas = []
            for _ in range(sg0.BOOTSTRAP_DRAWS):
                draw = torch.tensor(boot_rng.integers(0, support, size=support), dtype=torch.long)
                resampled, _ = sg0.argmin_route(errors_s, draw)
                deltas.append(abs(sg0.half_nmse(errors_q, resampled, q_b, denominator) - nmse_hat_b))
            floor = float(np.percentile(deltas, 95))
            regret = nmse_hat_b - nmse_star_b
            by[str(support)] = {
                'r_hat': unflatten(r_hat, library.slots, DEPTH), 'r_hat_index': r_hat, 'r_star_index': r_star,
                'routes_differ': r_hat != r_star, 'nmse_hat_qb': nmse_hat_b, 'nmse_star_qb': nmse_star_b,
                'regret': regret, 'null_floor': floor, 'binds': bool(regret > floor),
                'inference_failure': bool(nmse_hat_b >= FAIL and nmse_star_b < FAIL),
                'identifiability': (second - best) / max(best, 1e-12), 'best_support_mse': best,
                'near_tie_size': int(near.numel()), 'near_tie_disagreement': float(max(tie_nmse) - min(tie_nmse))}
        out[f'{v:g}'] = by
    return out


def run_job(m, w, programs=PROGRAMS):
    torch.set_num_threads(1)
    started = time.perf_counter()
    recorded = json.loads(G1_REPORT.read_text())['cells'][g1.key_of('RW_SLEEP', m, w)]
    g3a_rows = json.loads(G3A_REPORT.read_text())['cells'][g3a.key_of(m, w)]['rows']
    model, per_task, held_out, terminal_sha = g3a.rebuild(m, w)
    anchor_terminal = per_task == recorded['terminal_per_task']
    library = FrozenLibrary(model)
    library.steps = DEPTH
    before = library_sha(model)
    rows = []
    for index, task in enumerate(held_out[:programs]):
        if list(task['program']) != g3a_rows[index]['program']:
            raise ValueError('program order differs from G3a')
        rows.append({'task': index, 'program': list(task['program']), 'by_noise': measure_task(library, task, w, index)})
    if library_sha(model) != before:
        raise ValueError('library changed during the census')
    anchor_g3a = all(row['by_noise']['0'] == g3a_rows[i]['by_support'] for i, row in enumerate(rows))
    summary = {}
    for v in map(lambda v: f'{v:g}', NOISE):
        summary[v] = {}
        for s in map(str, SUPPORTS):
            rs = [r['by_noise'][v][s] for r in rows]
            summary[v][s] = {'binds': sum(r['binds'] for r in rs),
                             'inference_failures': sum(r['inference_failure'] for r in rs),
                             'routes_differ': sum(r['routes_differ'] for r in rs),
                             'median_regret': float(np.median([r['regret'] for r in rs])),
                             'median_floor': float(np.median([r['null_floor'] for r in rs])),
                             'median_nmse_hat_qb': float(np.median([r['nmse_hat_qb'] for r in rs])),
                             'median_identifiability': float(np.median([r['identifiability'] for r in rs]))}
    return {'shared': m, 'world': w, 'programs': len(rows), 'terminal_sha256': terminal_sha, 'library_sha256': before,
            'anchor_terminal': anchor_terminal, 'anchor_g3a': anchor_g3a, 'summary': summary, 'rows': rows,
            'seconds': time.perf_counter() - started}


def validate(rec, programs=PROGRAMS):
    k = g3a.key_of(rec['shared'], rec['world'])
    if not rec['anchor_terminal']:
        raise ValueError(f'{k}: rebuilt vocabulary does not reproduce G1 (anchor A1)')
    if not rec['anchor_g3a']:
        raise ValueError(f'{k}: noise 0 does not reproduce G3a (anchor A3)')
    if rec['programs'] != programs:
        raise ValueError(f'{k}: program count')
    for row in rec['rows']:
        for v, by in row['by_noise'].items():
            for s, r in by.items():
                for f in ('nmse_hat_qb', 'nmse_star_qb', 'regret', 'null_floor', 'identifiability'):
                    if not math.isfinite(r[f]):
                        raise ValueError(f'{k}: nonfinite {f}')
                if not r['routes_differ'] and r['regret'] != 0.0:
                    raise ValueError(f'{k}: identical routes with nonzero regret')


# ------------------------------------------------------------------ reading (registered in the plan)

def fractions(records):
    n = len(WORLDS) * PROGRAMS
    return {f'm{m:g}': {f'{v:g}': {str(s): sum(records[g3a.key_of(m, w)]['summary'][f'{v:g}'][str(s)]['binds']
                                                for w in WORLDS) / n for s in SUPPORTS} for v in NOISE}
            for m in SHARED}


def reading(records):
    f = fractions(records)
    window, both, high = [], [], []
    for v in NOISE:
        for s in map(str, SUPPORTS):
            a, b = f['m4'][f'{v:g}'][s], f['m0'][f'{v:g}'][s]
            if a >= BIND_MIN and v <= PLAUSIBLE_V:
                (window if a - b >= BIND_MARGIN else both).append([v, int(s)])
            elif a >= BIND_MIN:
                high.append([v, int(s)])
    if window:
        label = 'NOISE_WINDOW'
    elif both:
        label = 'NOISE_ONLY'
    elif high:
        label = 'HIGH_NOISE_ONLY'
    else:
        label = 'NO_BIND'
    onset = {arm: {str(s): next((v for v in NOISE if f[arm][f'{v:g}'][str(s)] >= BIND_MIN), None) for s in SUPPORTS}
             for arm in f}
    return {'label': label, 'window': window, 'both_bind': both, 'high_noise': high, 'bind_fraction': f,
            'onset_noise': onset, 'programs_per_cell': len(WORLDS) * PROGRAMS}


# ------------------------------------------------------------------ protocol and run

def protocol(root, programs):
    return {'id': 'g3b-noise-census-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 0,
            'worlds': list(WORLDS), 'shared': list(SHARED), 'supports': list(SUPPORTS), 'noise': list(NOISE),
            'noise_seed': NOISE_SEED, 'plausible_v': PLAUSIBLE_V, 'bind': [BIND_MIN, BIND_MARGIN], 'depth': DEPTH,
            'programs': programs, 'root': Path(root).as_posix(),
            'terminals_sha256': {g3a.key_of(m, w): digest(g3a.terminal_path(m, w)) for m, w in jobs()},
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G1_REPORT, G3A_REPORT)},
            'implementation_sha256': digest(Path(__file__)), 'g3a_sha256': digest(Path(g3a.__file__)),
            'sg0_sha256': digest(Path(sg0.__file__)), 'g1_sha256': digest(Path(g1.__file__))}


def run(todo, root=ROOT, output=OUTPUT, programs=PROGRAMS, stop_after=None, jobs_n=JOBS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root, programs) | {'jobs': [list(j) for j in todo]}
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
                                                 'passes': free >= MIN_FREE_GIB, 'checked_utc': now()})
        if free < MIN_FREE_GIB:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free')
        records, pending = {}, []
        try:
            for m, w in todo:
                path = root / 'cells' / f'{g3a.key_of(m, w)}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {g3a.key_of(m, w)}')
                    records[g3a.key_of(m, w)] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {g3a.key_of(m, w)}')
                else:
                    pending.append((m, w))
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(pending)} reused={len(records)} programs={programs} '
                                       f'free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs_n) as pool:
                futures = {pool.submit(run_job, m, w, programs): (m, w) for m, w in pending}
                g3a._status(root, 'running', len(records), len(todo), [g3a.key_of(*j) for j in pending[:jobs_n]], started)
                for future in as_completed(futures):
                    m, w = futures[future]
                    record = future.result()
                    validate(record, programs)
                    atomic_json(root / 'cells' / f'{g3a.key_of(m, w)}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[g3a.key_of(m, w)] = record
                    finished += 1
                    sm = record['summary']
                    log_line(root / 'run.log', f"cell finished {g3a.key_of(m, w)} {record['seconds']:.1f}s binds(v:s128/s8/s2) "
                             + ' '.join(f"{v}:{sm[v]['128']['binds']}/{sm[v]['8']['binds']}/{sm[v]['2']['binds']}" for v in sm))
                    g3a._status(root, 'running', len(records), len(todo), [], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if programs == PROGRAMS and len(records) == len(jobs()):
                report['reading'] = reading(records)
            atomic_json(output, report)
            log_line(root / 'run.log', f"EXIT 0 {report.get('reading', {}).get('label')}")
            g3a._status(root, 'complete', len(records), len(todo), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            log_line(root / 'run.log', 'FAILED ' + traceback.format_exc().replace('\n', ' | '))
            g3a._status(root, 'failed', len(records), len(todo), [], started)
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
        run([(4.0, 70), (0.0, 70)], root=DRY_ROOT, output=DRY_ROOT / 'report.json', programs=4,
            stop_after=args.stop_after)
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(jobs())
    print(f'G3b report: {OUTPUT}')


if __name__ == '__main__':
    main()
