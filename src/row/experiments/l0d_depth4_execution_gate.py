"""Tier 0 depth-four execution gate on one frozen J2A library."""
from __future__ import annotations

import argparse
import itertools
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
from row.experiments.audit_j1c_curriculum import library_sha, stage_setup
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary, unflatten
from row.experiments.preflight_l0d import load_source
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('L0D_DEPTH4_EXECUTION_GATE_PLAN.md')
ROOT = Path('artifacts/l0d_depth4_execution_gate')
OUTPUT = Path('reports/l0d_depth4_execution_gate.json')
NAME, WORLD, MODEL_SEED, TASKS, DEPTH = 'STAGED5000', 0, 5000, 16, 4
SEED = 2704


class DepthLibrary(FrozenLibrary):
    def forward(self, x, coefficients):
        if coefficients.ndim != 2 or coefficients.shape[1] != self.slots or len(coefficients) == 0:
            raise ValueError('invalid coefficient matrix')
        z = x
        for coefficient in coefficients:
            z = torch.sum(coefficient.view(1, -1, 1) * self.candidates(z), dim=1)
        return z

    def all_route_support_mse_depth(self, x, y, depth):
        n, dimension = x.shape
        z = x
        route_count = 1
        for _ in range(depth):
            z = self.candidates(z.reshape(-1, dimension))
            z = z.reshape(n, route_count * self.slots, dimension)
            route_count *= self.slots
        return torch.mean((z - y.unsqueeze(1)) ** 2, dim=(0, 2))


def protocol():
    return {'id': 'l0d-depth4-execution-gate-v1', 'git_commit': git_commit(), 'name': NAME,
            'world': WORLD, 'model_seed': MODEL_SEED, 'tasks': TASKS, 'depth': DEPTH, 'seed': SEED,
            'slots': 12, 'support': 128, 'selection': 'support-only exhaustive ENUM',
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), j2a.OUTPUT)},
            'implementation_sha256': digest(Path(__file__))}


def depth4_tasks(cfg, world):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, WORLD, DEPTH]))
    choices = list(itertools.product(range(cfg.world.teacher_primitives), repeat=DEPTH))
    order = rng.permutation(len(choices))[:TASKS]
    teacher_library = world.tasks[0].teacher_library
    tasks = []
    for index, choice in enumerate(order):
        support = np.random.default_rng(np.random.SeedSequence([SEED, WORLD, DEPTH, 1 + index])).normal(
            size=(cfg.world.examples_per_task, cfg.world.state_dim))
        query = np.random.default_rng(np.random.SeedSequence([SEED, WORLD, DEPTH, 10_000 + index])).normal(
            size=(cfg.world.evaluation_examples, cfg.world.state_dim))
        value = support.copy()
        query_value = query.copy()
        for primitive in choices[int(choice)]:
            value = teacher_library[primitive](value)
            query_value = teacher_library[primitive](query_value)
        tasks.append({'program': tuple(choices[int(choice)]), 'train_x': support, 'train_y': value,
                      'eval_x': query, 'eval_y': query_value})
    if len({t['program'] for t in tasks}) != TASKS:
        raise ValueError('depth-four programs are not distinct')
    return tasks


def query_error(library, task, route):
    with torch.no_grad():
        prediction = library.hard(torch.tensor(task['eval_x'], dtype=torch.float32), route)
    return float(j2a.nmse(prediction.numpy(), task['eval_y']))


def measure():
    started = time.perf_counter()
    cfg, world, model, stage, _ = load_source(NAME, WORLD)
    library = DepthLibrary(model)
    before = library_sha(model)
    tasks = depth4_tasks(cfg, world)
    rows = []
    mapped = stage['slot_by_operation']
    for index, task in enumerate(tasks):
        x = torch.tensor(task['train_x'], dtype=torch.float32)
        y = torch.tensor(task['train_y'], dtype=torch.float32)
        t0 = time.perf_counter()
        support_mse = library.all_route_support_mse_depth(x, y, DEPTH)
        seconds = time.perf_counter() - t0
        if not bool(torch.isfinite(support_mse).all()):
            raise ValueError('nonfinite support losses')
        best = int(torch.argmin(support_mse))
        route = unflatten(best, library.slots, DEPTH)
        diagnostic_route = [mapped[str(p)] for p in task['program']]
        rows.append({'task': index, 'program': list(task['program']), 'enum_route': route,
                     'enum_support_mse': float(support_mse[best]), 'enum_query_nmse': query_error(library, task, route),
                     'diagnostic_route': diagnostic_route, 'diagnostic_query_nmse': query_error(library, task, diagnostic_route),
                     'seconds': seconds})
    if library_sha(model) != before:
        raise ValueError('library changed during depth-four gate')
    return {'name': NAME, 'world': WORLD, 'model_seed': MODEL_SEED, 'depth': DEPTH,
            'library_sha256': before, 'rows': rows, 'seconds': time.perf_counter() - started,
            'terminal_tensor_bytes_lower_bound': 128 * 12 ** DEPTH * cfg.world.state_dim * 4}


def validate(record):
    if record['name'] != NAME or record['world'] != WORLD or record['depth'] != DEPTH or len(record['rows']) != TASKS:
        raise ValueError('record identity/grid mismatch')
    if len({tuple(r['program']) for r in record['rows']}) != TASKS:
        raise ValueError('programs not distinct')
    for row in record['rows']:
        for field in ('enum_support_mse', 'enum_query_nmse', 'diagnostic_query_nmse', 'seconds'):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError('nonfinite metric')
        if len(row['enum_route']) != DEPTH or len(row['diagnostic_route']) != DEPTH:
            raise ValueError('route depth mismatch')


def summary(record):
    values = [r['enum_query_nmse'] for r in record['rows']]
    if all(v <= .05 for v in values):
        classification = 'SEARCH_USABLE'
    elif any(v <= .05 for v in values):
        classification = 'MIXED'
    else:
        classification = 'SEARCH_NOT_USABLE'
    return {'tasks': len(values), 'enum_passes_0_05': sum(v <= .05 for v in values),
            'enum_median_query_nmse': float(np.median(values)),
            'diagnostic_median_query_nmse': float(np.median([r['diagnostic_query_nmse'] for r in record['rows']])),
            'classification': classification, 'interpretation': 'depth-four execution gate only; no PX7 verdict'}


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    expected = protocol(); sha = fingerprint(expected); manifest = {'protocol': expected, 'protocol_sha256': sha}
    with writer_lock(ROOT / 'launcher.lock'):
        if (ROOT / 'manifest.json').exists() and json.loads((ROOT / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; preserve prior run')
        atomic_json(ROOT / 'manifest.json', manifest)
        cell_path = ROOT / 'cell.json'
        try:
            if cell_path.exists():
                saved = json.loads(cell_path.read_text())
                if saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete') or fingerprint(saved['record']) != saved['record_sha256']:
                    raise ValueError('cell integrity failure')
                record = saved['record']; log_line(ROOT / 'run.log', 'reused validated cell')
            else:
                log_line(ROOT / 'run.log', f'LAUNCH {sha}')
                record = measure(); validate(record)
                atomic_json(cell_path, {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                        'record_sha256': fingerprint(record), 'finished_utc': now()})
                log_line(ROOT / 'run.log', f'cell finished {record["seconds"]:.3f}s')
            validate(record)
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'cell': record, 'summary': summary(record), 'finished_utc': now()})
            atomic_json(ROOT / 'status.json', {'state': 'complete', 'cells_done': 1, 'cells_total': 1, 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            atomic_json(ROOT / 'status.json', {'state': 'failed', 'cells_done': 0, 'cells_total': 1, 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    torch.set_num_threads(1)
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    run(); print(f'Depth-four gate report: {OUTPUT}')


if __name__ == '__main__':
    main()
