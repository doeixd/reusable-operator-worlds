"""Bounded L0d sparse-evidence opportunity gate; no PX7 verdict."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from row.experiments import audit_j2a_staged_library as j2a
from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.preflight_l0d import load_source
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('L0D_AMBIGUITY_GATE_PLAN.md')
ROOT = Path('artifacts/l0d_ambiguity_gate')
OUTPUT = Path('reports/l0d_ambiguity_gate.json')
NAME, WORLD, MODEL_SEED = 'STAGED5000', 0, 5000
SUPPORTS = (128, 4, 2, 1)
TASKS = 16


def protocol() -> dict:
    return {'id': 'l0d-ambiguity-gate-v1', 'git_commit': git_commit(), 'name': NAME,
            'world': WORLD, 'model_seed': MODEL_SEED, 'tasks': TASKS,
            'supports': list(SUPPORTS), 'selection': 'support-only exhaustive ENUM; query labels never used',
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), j2a.OUTPUT)},
            'implementation_sha256': digest(Path(__file__))}


def query_nmse(library, task, route):
    return j2a.query_error(library, route, task['eval_x'], task['eval_y'])


def measure() -> dict:
    started = time.perf_counter()
    cfg, world, model, _, anchor = load_source(NAME, WORLD)
    if cfg.world.seed != WORLD or cfg.world.program_length != 3:
        raise ValueError('unexpected source world')
    library = FrozenLibrary(model)
    before = library_sha(model)
    tasks = j2a.held_out_tasks(cfg, world)[:TASKS]
    rows = []
    for index, task in enumerate(tasks):
        historical = anchor['held_out'][str(index)]
        if list(task['program']) != historical['program']:
            raise ValueError('held-out order differs from J2A')
        row = {'task': index, 'program': list(task['program']), 'supports': {}}
        for support in SUPPORTS:
            x = torch.tensor(task['train_x'][:support], dtype=torch.float32)
            y = torch.tensor(task['train_y'][:support], dtype=torch.float32)
            started = time.perf_counter()
            mse = library.all_route_support_mse(x, y)
            seconds = time.perf_counter() - started
            if not bool(torch.isfinite(mse).all()):
                raise ValueError('nonfinite support losses')
            best, second = torch.topk(mse, 2, largest=False).values.tolist()
            route = unflatten(int(torch.argmin(mse)), library.slots, library.steps)
            error = query_nmse(library, task, route)
            if support == 128 and (route != historical['enum_route'] or error != historical['enum']):
                raise ValueError(f'128-example J2A anchor failed at task {index}')
            row['supports'][str(support)] = {
                'route': route, 'query_nmse': error, 'best_mse': best,
                'gap_mse': second - best, 'relative_gap': (second - best) / max(best, 1e-12),
                'seconds': seconds,
            }
        base = row['supports']['128']
        for support in (4, 2, 1):
            sparse = row['supports'][str(support)]
            sparse['route_changed_from_128'] = sparse['route'] != base['route']
            sparse['query_delta_from_128'] = sparse['query_nmse'] - base['query_nmse']
        rows.append(row)
    if library_sha(model) != before:
        raise ValueError('library changed during inference')
    return {'name': NAME, 'world': WORLD, 'model_seed': MODEL_SEED, 'library_sha256': before,
            'rows': rows, 'seconds': time.perf_counter() - started}


def validate(record):
    if record['name'] != NAME or record['world'] != WORLD or record['model_seed'] != MODEL_SEED:
        raise ValueError('cell identity mismatch')
    if len(record['rows']) != TASKS or [r['task'] for r in record['rows']] != list(range(TASKS)):
        raise ValueError('missing or reordered tasks')
    for row in record['rows']:
        if set(row['supports']) != {str(s) for s in SUPPORTS}:
            raise ValueError('support grid incomplete')
        for support in SUPPORTS:
            result = row['supports'][str(support)]
            for field in ('query_nmse', 'best_mse', 'gap_mse', 'relative_gap', 'seconds'):
                if not math.isfinite(result[field]) or result[field] < 0:
                    raise ValueError(f'invalid {field}')
            if len(result['route']) != 3 or any(type(s) is not int or not 0 <= s < 12 for s in result['route']):
                raise ValueError('invalid route')
        for support in (4, 2, 1):
            result = row['supports'][str(support)]
            if result['route_changed_from_128'] != (result['route'] != row['supports']['128']['route']):
                raise ValueError('route-change mismatch')
            if result['query_delta_from_128'] != result['query_nmse'] - row['supports']['128']['query_nmse']:
                raise ValueError('query-delta mismatch')


def summary(record):
    return {'tasks': len(record['rows']), 'support_cells': len(record['rows']) * len(SUPPORTS),
            'route_changes': {str(s): sum(r['supports'][str(s)]['route_changed_from_128'] for r in record['rows']) for s in (4, 2, 1)},
            'query_worsens': {str(s): sum(r['supports'][str(s)]['query_delta_from_128'] > 0 for r in record['rows']) for s in (4, 2, 1)},
            'query_improves': {str(s): sum(r['supports'][str(s)]['query_delta_from_128'] < 0 for r in record['rows']) for s in (4, 2, 1)},
            'median_query_delta': {str(s): float(np.median([r['supports'][str(s)]['query_delta_from_128'] for r in record['rows']])) for s in (4, 2, 1)},
            'interpretation': 'opportunity gate only; no PX7 verdict'}


def run(stop_after=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    expected = protocol()
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    with writer_lock(ROOT / 'launcher.lock'):
        if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; preserve prior run')
        atomic_json(ROOT / 'manifest.json', manifest)
        path = ROOT / 'cell.json'
        try:
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('stamp') != {'protocol_sha256': sha} or saved.get('complete') is not True:
                    raise ValueError('incomplete or mismatched cell')
                record = saved['record']
                if fingerprint(record) != saved['record_sha256']:
                    raise ValueError('cell record hash mismatch')
                log_line(ROOT / 'run.log', 'reused validated cell')
            else:
                log_line(ROOT / 'run.log', f'LAUNCH {sha}')
                record = measure()
                validate(record)
                atomic_json(path, {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                    'record_sha256': fingerprint(record), 'finished_utc': now()})
                log_line(ROOT / 'run.log', 'cell finished')
            validate(record)
            output = {**manifest, 'complete': True, 'cell': record, 'summary': summary(record), 'finished_utc': now()}
            atomic_json(OUTPUT, output)
            atomic_json(ROOT / 'status.json', {'state': 'complete', 'cells_done': 1, 'cells_total': 1, 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            atomic_json(ROOT / 'status.json', {'state': 'failed', 'cells_done': 0, 'cells_total': 1, 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not args.dry_run:
        require_clean_code(OUTPUT)
        for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
            subprocess.run([sys.executable, check], check=True)
    run()
    print(f'Opportunity gate report: {OUTPUT}')


if __name__ == '__main__':
    main()
