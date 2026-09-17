"""Tier 0 L0d preflight; see L0D_PREFLIGHT_PLAN.md. No full census verdict."""
from __future__ import annotations

import argparse
import json
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
from row.experiments.audit_so1r_route_only import FrozenLibrary, optimize_route, unflatten
from row.experiments.so1_storage import (
    atomic_json, digest, environment, fingerprint, log_line, now, restore_model, writer_lock,
)

PLAN = Path('L0D_PREFLIGHT_PLAN.md')
ROOT = Path('artifacts/l0d_preflight')
OUTPUT = Path('reports/l0d_preflight.json')
SUPPORTS = (128, 32, 8)
OPT_STEPS = 20


class VariableDepthLibrary(FrozenLibrary):
    def forward(self, x, coefficients):
        if coefficients.ndim != 2 or coefficients.shape[1] != self.slots or len(coefficients) == 0:
            raise ValueError('expected nonempty depth by slots coefficients')
        z = x
        for coefficient in coefficients:
            z = torch.sum(coefficient.view(1, -1, 1) * self.candidates(z), dim=1)
        return z


def terminal_tensor_bytes(support, slots, depth, dimension):
    return support * slots ** depth * dimension * 4


def checked_record(path, payload_key):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get(payload_key + '_sha256') != fingerprint(value[payload_key]):
        raise ValueError(f'record hash mismatch: {path}')
    return value


def source_paths(name, world):
    template, _, report_path, _ = j2a.SOURCES[name]
    directory = Path(template.format(w=world))
    anchor = j2a.ROOT / 'cells' / f'{name}_w{world}' / 'result.json'
    return directory, report_path, anchor


def inputs(jobs):
    paths = {PLAN, Path('configs/v1.yaml'), j2a.OUTPUT}
    for name, world in jobs:
        directory, report, anchor = source_paths(name, world)
        paths.update((report, anchor, directory / 'result.json'))
        # Hash all saved stage-three reconstruction files, not only model.pt.
        paths.update(p for p in (directory / 'stage3').iterdir() if p.is_file())
        stored = checked_record(directory / 'result.json', 'result')
        paths.update(directory / p for p in stored['artifact_sha256'])
    return {p.as_posix(): digest(p) for p in sorted(paths)}


def load_source(name, world_seed):
    directory, report_path, anchor_path = source_paths(name, world_seed)
    _, model_seed, _, cell_template = j2a.SOURCES[name]
    source = checked_record(directory / 'result.json', 'result')
    report = json.loads(report_path.read_text())
    if report.get('complete') is not True or report['cells'][cell_template.format(w=world_seed)] != source['result']:
        raise ValueError('source report/cell disagreement or incomplete parent')
    for artifact, sha in source['artifact_sha256'].items():
        if digest(directory / artifact) != sha:
            raise ValueError(f'source artifact hash mismatch: {directory / artifact}')
    anchor = checked_record(anchor_path, 'record')['record']
    anchor_report = json.loads(j2a.OUTPUT.read_text())
    if anchor_report.get('complete') is not True or anchor_report['cells'][f'{name}_w{world_seed}'] != j2a.summarize(anchor):
        raise ValueError('J2A report/cell disagreement or incomplete parent')
    cfg, world, _, _ = stage_setup(world_seed, 3, model_seed)
    model = restore_model(directory / 'stage3', cfg, world, build_fast)
    expected = source['result']['stages']['3']['library_sha256']
    if library_sha(model) != expected or expected != anchor['library_sha256']:
        raise ValueError('restored library disagrees with source/J2A hash')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return cfg, world, model, source['result']['stages']['3'], anchor


def measure(name, world_seed, task_count):
    started = time.perf_counter()
    cfg, world, model, stage, anchor = load_source(name, world_seed)
    before = library_sha(model)
    reference, variable = FrozenLibrary(model), VariableDepthLibrary(model)
    tasks = j2a.held_out_tasks(cfg, world)[:task_count]
    rows = []
    coefficients = torch.softmax(torch.arange(36, dtype=torch.float32).reshape(3, 12) / 9, dim=-1)
    with torch.no_grad():
        for index, task in enumerate(tasks):
            original = anchor['held_out'][str(index)]
            if list(task['program']) != original['program']:
                raise ValueError('held-out task order differs from J2A')
            x = torch.tensor(task['train_x'], dtype=torch.float32)
            if len(x) != SUPPORTS[0]:
                raise ValueError('unexpected canonical support size')
            if not torch.equal(reference.forward(x, coefficients), variable.forward(x, coefficients)):
                raise ValueError('soft executor equivalence failed')
            if not torch.equal(reference.hard(x, original['enum_route']), variable.hard(x, original['enum_route'])):
                raise ValueError('hard executor equivalence failed')
            mapped = [stage['slot_by_operation'][str(p)] for p in task['program']]
            mapped_error = j2a.query_error(variable, mapped, task['eval_x'], task['eval_y'])
            # J2A stores the random error, but not its route. Preserve that
            # historical control as reported; do not invent a reconstruction.
            row = {'task': index, 'program': list(task['program']), 'supports': {},
                   'j2a_random_nmse': original['random'], 'mapped_route': mapped,
                   'mapped_nmse': mapped_error, 'executor_bitwise': True}
            for support in SUPPORTS:
                t0 = time.perf_counter()
                mse = reference.all_route_support_mse(x[:support], torch.tensor(task['train_y'][:support], dtype=torch.float32))
                seconds = time.perf_counter() - t0
                if not bool(torch.isfinite(mse).all()):
                    raise ValueError('nonfinite support errors')
                route = unflatten(int(torch.argmin(mse)), reference.slots, reference.steps)
                best, second = torch.topk(mse, 2, largest=False).values.tolist()
                error = j2a.query_error(variable, route, task['eval_x'], task['eval_y'])
                if support == 128 and (route != original['enum_route'] or error != original['enum']):
                    raise ValueError(f'J2A ENUM anchor failed: {name} w{world_seed} task {index}')
                row['supports'][str(support)] = {'route': route, 'query_nmse': error, 'best_mse': best,
                    'gap_mse': second - best, 'relative_gap': (second - best) / max(best, 1e-12),
                    'seconds': seconds}
            rows.append(row)
    x = torch.tensor(tasks[0]['train_x'], dtype=torch.float32)
    y = torch.tensor(tasks[0]['train_y'], dtype=torch.float32)
    timings = {}
    with torch.no_grad():
        for label, lib in [('reference', reference), ('variable', variable)]:
            lib.forward(x, coefficients)
            t0 = time.perf_counter()
            for _ in range(20):
                lib.forward(x, coefficients)
            timings[label + '_20_forwards_seconds'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, initial, final, moved = optimize_route(variable, x, y, OPT_STEPS)
    timings.update(opt_20_updates_seconds=time.perf_counter() - t0,
                   opt_initial_mse=initial, opt_final_preupdate_mse=final, opt_code_abs_sum=moved)
    if library_sha(model) != before:
        raise ValueError('frozen library mutated')
    return {'name': name, 'world': world_seed, 'library_sha256': before, 'rows': rows,
            'anchor_bitwise': True, 'library_unchanged': True, 'timings': timings,
            'terminal_tensor_bytes': {str(d): terminal_tensor_bytes(128, variable.slots, d, cfg.world.state_dim)
                                      for d in (3, 4, 5)}, 'seconds': time.perf_counter() - started}


def validate_record(record, task_count):
    if record.get('anchor_bitwise') is not True or record.get('library_unchanged') is not True:
        raise ValueError('failed anchor/freeze gate')
    rows = record['rows']
    if len(rows) != task_count or [r['task'] for r in rows] != list(range(task_count)):
        raise ValueError('incomplete or reordered tasks')
    for row in rows:
        if row['executor_bitwise'] is not True or set(row['supports']) != {str(s) for s in SUPPORTS}:
            raise ValueError('missing supports or failed equivalence')
    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError('nonfinite record')
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        if isinstance(value, list):
            for v in value:
                finite(v)
    finite(record)


def summarize(records):
    rows = [row for record in records.values() for row in record['rows']]
    if not rows:
        raise ValueError('empty preflight')
    return {'libraries': len(records), 'tasks': len(rows), 'support_cells': len(rows) * len(SUPPORTS),
            'gap_monotone_tasks': sum(r['supports']['8']['gap_mse'] <= r['supports']['32']['gap_mse']
                                      <= r['supports']['128']['gap_mse'] for r in rows),
            'mapped_fails_enum_passes': sum(r['mapped_nmse'] > .05 and r['supports']['128']['query_nmse'] <= .05 for r in rows),
            'random_worse_than_enum_128': sum(r['j2a_random_nmse'] > r['supports']['128']['query_nmse'] for r in rows),
            'interpretation': 'instrument preflight only; no PX7 verdict'}


def run(root, output, protocol, jobs, task_count, compute=measure, stop_after=None):
    root.mkdir(parents=True, exist_ok=True)
    sha = fingerprint(protocol)
    manifest = {'protocol': protocol, 'protocol_sha256': sha}
    with writer_lock(root / 'launcher.lock'):
        manifest_path = root / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            atomic_json(root / 'rejected_launch.json', {'attempted_manifest': manifest, 'utc': now(),
                        'error': 'launch commit, inputs or protocol mismatch'})
            raise ValueError('launch commit, inputs or protocol mismatch; preserve the original run')
        atomic_json(manifest_path, manifest)
        (root / 'run.pid').write_text(str(os.getpid()))
        began, started = now(), time.perf_counter()
        records, new = {}, 0
        def status(state, current=None):
            elapsed = time.perf_counter() - started
            atomic_json(root / 'status.json', {'state': state, 'pid': os.getpid(), 'git_commit': protocol['git_commit'],
                'started_utc': began, 'updated_utc': now(), 'cells_done': len(records), 'cells_total': len(jobs),
                'current': current, 'eta_seconds': elapsed / new * (len(jobs) - len(records)) if new else None})
        try:
            status('running')
            log_line(root / 'run.log', f'LAUNCH/RESUME {sha} pid {os.getpid()}')
            for name, world in jobs:
                key = f'{name}_w{world}'
                path = root / 'cells' / key / 'result.json'
                stamp = {'protocol_sha256': sha, 'name': name, 'world': world}
                status('running', key)
                if path.exists():
                    saved = checked_record(path, 'record')
                    if saved.get('complete') is not True or saved['stamp'] != stamp:
                        raise ValueError(f'incomplete or mismatched cell: {key}')
                    record = saved['record']
                    log_line(root / 'run.log', f'{key} reused validated cell')
                else:
                    log_line(root / 'run.log', f'{key} start')
                    record = compute(name, world, task_count)
                    validate_record(record, task_count)
                    atomic_json(path, {'stamp': stamp, 'complete': True, 'record': record,
                                       'record_sha256': fingerprint(record), 'finished_utc': now()})
                    new += 1
                    log_line(root / 'run.log', f'{key} finish anchors pass, {record["seconds"]:.3f}s')
                validate_record(record, task_count)
                if (record['name'], record['world']) != (name, world):
                    raise ValueError('cell identity mismatch')
                records[key] = record
                complete = len(records) == len(jobs)
                atomic_json(output, {**manifest, 'complete': complete, 'cells': records,
                                    'summary': summarize(records), 'updated_utc': now()})
                status('complete' if complete else 'running')
                if stop_after is not None and new >= stop_after and not complete:
                    status('paused')
                    log_line(root / 'run.log', 'PAUSED by stop-after; relaunch to resume')
                    atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': False, 'finished_utc': now()})
                    return
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
            log_line(root / 'run.log', 'EXIT 0 complete; no PX7 verdict')
        except BaseException:
            error = traceback.format_exc()
            status('failed')
            atomic_json(root / 'error.json', {'traceback': error, 'finished_utc': now()})
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            log_line(root / 'run.log', f'EXIT 1 {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dry-run-dir', type=Path, default=Path('artifacts/l0d_preflight_restart_smoke'))
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    torch.set_num_threads(1)
    jobs = [('STAGED5000', 0), ('STAGED5000', 1)] if args.dry_run else [(n, w) for n in j2a.SOURCES for w in j2a.WORLDS]
    task_count = 2 if args.dry_run else 16
    root = args.dry_run_dir if args.dry_run else ROOT
    output = root / 'report.json' if args.dry_run else OUTPUT
    try:
        if not args.dry_run:
            require_clean_code(output)
            for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
                subprocess.run([sys.executable, check], check=True)
        protocol = {'id': 'l0d-preflight-v1', 'git_commit': git_commit(), 'dry_run': args.dry_run,
                    'jobs': jobs, 'tasks': task_count, 'supports': SUPPORTS, 'opt_steps': OPT_STEPS,
                    'environment': environment(), 'input_sha256': inputs(jobs),
                    'implementation_sha256': digest(Path(__file__))}
        # JSON normalization makes tuples and loaded lists identical on resume.
        protocol = json.loads(json.dumps(protocol))
    except BaseException:
        # A rejected startup must not overwrite another active writer's status.
        # Give every failed attempt its own durable error, even before a manifest.
        atomic_json(root / f'startup_error_{os.getpid()}.json',
                    {'exit_code': 1, 'pid': os.getpid(), 'finished_utc': now(), 'traceback': traceback.format_exc()})
        raise
    run(root, output, protocol, jobs, task_count, stop_after=args.stop_after)
    print(f'Preflight records: {output}')


if __name__ == '__main__':
    main()
