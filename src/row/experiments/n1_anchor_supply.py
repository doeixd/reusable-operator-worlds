"""N1: is it the ORDER, or the supply of anchors?

Frozen plan: `N1_ANCHOR_SUPPLY_PLAN.md` with Amendments 1-3. Offline, worlds
0-2, development evidence only and never upgradable to confirmatory.

Four arms at a matched budget of 65,536 updates:

    STAGED        three sequential stages (J1c verbatim), library carried
    INTERLEAVED   the same 188 tasks in ONE pooled stream, no ordering
    SHAM          188 pooled tasks, the 124 non-canonical ones length-3
    NONE          the published 64-task length-3 baseline

STAGED vs INTERLEAVED isolates ORDER. INTERLEAVED vs SHAM isolates ANCHOR
EASINESS at identical pool size and identical sampling. Every arm is scored the
same way: hard-route query NMSE on the CANONICAL 64 length-3 tasks only, so the
extra programs are training stream and never enter the estimand.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

from row.experiments.audit_e0_export import git_commit
from row.experiments.audit_j1c_curriculum import (
    BATCH, STAGES, run_arm as j1c_run_arm, _flat_shared, _shared_optimizer, _training_values,
    library_sha, score, stage_setup, train_stage)
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.models.variable_depth import VariableDepthRotatedLearner
from row.experiments.so1_storage import (atomic_json, digest, fingerprint, log_line, now,
                                         writer_lock)

PLAN = Path('N1_ANCHOR_SUPPLY_PLAN.md')
ROOT = Path('artifacts/n1_anchor_supply')
OUTPUT = Path('reports/n1_anchor_supply.json')

WORLDS = (0, 1, 2)
ARMS = ('STAGED', 'INTERLEAVED', 'SHAM', 'NONE')
MODEL_SEED = 5000
UPDATES = sum(s[2] for s in STAGES)          # 65,536, the matched invariant
POOL_STREAM = 1911                           # registered seed root for pooled arms
SHAM_STREAM = 1912                           # registered seed root for SHAM's extra programs
THRESHOLD = 0.05
MIN_FREE_GIB = 6.0                           # launch precondition, fails closed


def protocol():
    return {'id': 'n1-anchor-supply-v1', 'git_commit': git_commit(), 'plan': PLAN.as_posix(),
            'worlds': list(WORLDS), 'arms': list(ARMS), 'updates': UPDATES, 'batch': BATCH,
            'model_seed': MODEL_SEED, 'threshold': THRESHOLD,
            'stages': [{'length': L, 'tasks': n, 'updates': u} for L, n, u, _, _ in STAGES],
            'pool_stream': POOL_STREAM, 'sham_stream': SHAM_STREAM,
            'implementation': 'variable_depth_rotation_v1',
            'scored_on': 'canonical 64 length-3 tasks only',
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'))},
            'implementation_sha256': digest(Path(__file__))}


# --------------------------------------------------------------------- tasks

class PooledTask:
    """A task carrying its own depth, for the pooled arms."""

    __slots__ = ('task_id', 'depth', 'train_x', 'train_y', 'eval_x', 'eval_y', 'program')

    def __init__(self, task_id, depth, train_x, train_y, eval_x, eval_y, program):
        self.task_id, self.depth = task_id, depth
        self.train_x, self.train_y = train_x, train_y
        self.eval_x, self.eval_y = eval_x, eval_y
        self.program = program


def _as_pooled(task, depth):
    return PooledTask(task.task_id, depth, task.train_x, task.train_y,
                      task.eval_x, task.eval_y, tuple(task.program.primitive_ids))


def canonical_tasks(world_seed):
    cfg, world, _, _ = stage_setup(world_seed, 3, MODEL_SEED)
    return cfg, world, [_as_pooled(t, 3) for t in world.tasks]


def anchor_tasks(world_seed):
    out = []
    for stage, length in ((1, 1), (2, 2)):
        _, world, _, _ = stage_setup(world_seed, stage, MODEL_SEED)
        out.extend(_as_pooled(t, length) for t in world.tasks)
    return out


def extra_length3_tasks(cfg, world, count, world_seed):
    """`count` DISTINCT length-3 programs not used by the canonical world.

    The world admits 6**3 = 216 distinct length-3 programs and the canonical set
    uses 64, so 152 remain against the 124 SHAM needs.
    """
    used = {tuple(t.program.primitive_ids) for t in world.tasks}
    pool = [p for p in itertools.product(range(cfg.world.teacher_primitives), repeat=3)
            if p not in used]
    if len(pool) < count:
        raise ValueError(f'only {len(pool)} spare length-3 programs, need {count}')
    rng = np.random.default_rng(np.random.SeedSequence([SHAM_STREAM, world_seed]))
    chosen = [pool[i] for i in rng.permutation(len(pool))[:count]]
    library = world.tasks[0].teacher_library
    tasks = []
    for index, program in enumerate(chosen):
        seeds = np.random.SeedSequence([SHAM_STREAM, world_seed, 1 + index])
        train_x = np.random.default_rng(seeds).normal(
            size=(cfg.world.examples_per_task, cfg.world.state_dim))
        eval_x = np.random.default_rng(np.random.SeedSequence(
            [SHAM_STREAM, world_seed, 10_000 + index])).normal(
            size=(cfg.world.evaluation_examples, cfg.world.state_dim))
        train_y, eval_y = train_x.copy(), eval_x.copy()
        for primitive in program:
            train_y = library[primitive](train_y)
            eval_y = library[primitive](eval_y)
        tasks.append(PooledTask(f'sham_{world_seed}_{index:03d}', 3,
                                train_x, train_y, eval_x, eval_y, program))
    if len({t.program for t in tasks}) != count:
        raise ValueError('extra length-3 programs are not distinct')
    return tasks


def arm_tasks(arm, world_seed):
    cfg, world, canonical = canonical_tasks(world_seed)
    if arm == 'NONE':
        return cfg, world, canonical, list(canonical)
    if arm == 'SHAM':
        pool = canonical + extra_length3_tasks(cfg, world, 124, world_seed)
    else:
        pool = anchor_tasks(world_seed) + canonical
    return cfg, world, canonical, pool


# ------------------------------------------------------------------ training

def train_pooled(cfg, tasks, scored, updates, seed_sequence):
    """One undifferentiated stream over `tasks`; scored on `scored` only."""
    global_lr, task_lr, weight_decay, *_ = _training_values(cfg, 'rotated_discrete')
    selected = cfg.discrete_model
    # Same construction as `build_fast`, with depth 3 as the maximum; a task of
    # depth d uses the first d rows of its code. At uniform depth this class
    # reduces bitwise to the committed learner.
    model = VariableDepthRotatedLearner(
        d=cfg.world.state_dim,
        operator_slots=selected.operator_slots,
        operator_rank=selected.operator_rank,
        task_steps=3,
        alpha=selected.operator_alpha_init,
        initial_temperature=selected.initial_temperature,
        final_temperature=selected.final_temperature,
        seed=selected.seed,
        learnable_alpha=selected.learnable_alpha,
        activation=selected.operator_activation,
    )
    codes = [model.begin_task(t.task_id, depth=t.depth) for t in tasks]
    optimizer = _shared_optimizer(model, global_lr, weight_decay)
    optimizer.add_param_group({'params': codes, 'lr': task_lr, 'weight_decay': 0.0})

    all_x = torch.tensor(np.concatenate([t.train_x for t in tasks]), dtype=torch.float32)
    all_y = torch.tensor(np.concatenate([t.train_y for t in tasks]), dtype=torch.float32)
    all_ids = [t.task_id for t in tasks for _ in range(len(t.train_x))]
    rng = np.random.default_rng(seed_sequence)
    drawn = []
    for update in range(1, updates + 1):
        model.set_training_progress((update - 1) / max(1, updates - 1))
        model.train()
        idx = rng.integers(0, len(all_x), size=BATCH)
        if update <= 64:
            drawn.extend(int(i) for i in idx)
        indices = torch.tensor(idx, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            model.forward_tasks(all_x.index_select(0, indices), [all_ids[int(i)] for i in idx]),
            all_y.index_select(0, indices))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'non-finite loss at update {update}')
        loss.backward()
        optimizer.step()
    return model, drawn


class _ScoredWorld:
    def __init__(self, tasks):
        self.tasks = tasks


def run_cell(arm, world_seed):
    """One cell. STAGED calls J1c's committed `run_arm` verbatim, which is what
    makes its published terminal median a usable anchor rather than a rerun."""
    started = time.perf_counter()
    if arm == 'STAGED':
        result = j1c_run_arm('STAGED', world_seed, model_seed=MODEL_SEED)
        anchor = json.loads(Path('reports/j1c_curriculum.json').read_text())
        published = anchor['cells'][f'STAGED_w{world_seed}']['terminal_median']
        record = {
            'arm': arm, 'world': world_seed, 'updates': UPDATES,
            'pool_tasks': sum(n for _, n, _, _, _ in STAGES),
            'pool_examples': None, 'scored_tasks': STAGES[2][1],
            'depth_histogram': {str(L): n for L, n, _, _, _ in STAGES},
            'terminal_median': result['terminal_median'],
            'below_0.05': result['stages']['3']['below_0.05'],
            'per_task': result['stages']['3']['final_per_task'],
            'library_sha256': result['stages']['3']['library_sha256'],
            'first_64_draws': None,
            'anchor_published': published,
            'anchor_matches': result['terminal_median'] == published,
            'seconds': time.perf_counter() - started,
        }
        if not record['anchor_matches']:
            raise RuntimeError(
                f'STAGED does not reproduce the committed J1c cell for world {world_seed}: '
                f"{result['terminal_median']} against {published}; the harness has drifted "
                'and nothing else in this run is readable')
        return record

    cfg, world, canonical, pool = arm_tasks(arm, world_seed)
    seed_sequence = np.random.SeedSequence([POOL_STREAM, world_seed])
    model, drawn = train_pooled(cfg, pool, canonical, UPDATES, seed_sequence)
    final = score(model, _ScoredWorld(canonical))
    return {
        'arm': arm, 'world': world_seed, 'updates': UPDATES,
        'pool_tasks': len(pool), 'pool_examples': sum(len(t.train_x) for t in pool),
        'scored_tasks': len(canonical),
        'depth_histogram': {str(d): sum(1 for t in pool if t.depth == d) for d in (1, 2, 3)},
        'terminal_median': final['median'], 'below_0.05': final['below_0.05'],
        'per_task': final['per_task'], 'library_sha256': library_sha(model),
        'first_64_draws': drawn, 'anchor_published': None, 'anchor_matches': None,
        'seconds': time.perf_counter() - started,
    }


def validate_cell(record):
    import math
    if record['arm'] not in ARMS or record['world'] not in WORLDS:
        raise ValueError('cell identity')
    for field in ('terminal_median',):
        if not math.isfinite(record[field]) or record[field] < 0:
            raise ValueError(f'bad {field}')
    if record['arm'] != 'NONE' and record['pool_tasks'] != 188:
        raise ValueError(f"{record['arm']} must pool 188 tasks, has {record['pool_tasks']}")
    if record['arm'] == 'NONE' and record['pool_tasks'] != 64:
        raise ValueError('NONE must hold the canonical 64 tasks')
    if record['arm'] == 'SHAM' and record['depth_histogram'].get('3') != 188:
        raise ValueError('SHAM must be entirely length-3')
    if record['arm'] == 'INTERLEAVED' and record['depth_histogram'].get('3') != 64:
        raise ValueError('INTERLEAVED must hold exactly the canonical 64 length-3 tasks')
    if record['scored_tasks'] != 64:
        raise ValueError('every arm is scored on the canonical 64 tasks')


def cell_key(arm, world):
    return f'{arm}_w{world}'


def triage(records):
    """The registered three-way rule, with Amendment 1's UNINTERPRETABLE guard."""
    def median_for(arm):
        return {r['world']: r['terminal_median'] for r in records.values() if r['arm'] == arm}
    m, sham = median_for('INTERLEAVED'), median_for('SHAM')
    if not m or not sham:
        return {'verdict': 'INCOMPLETE', 'reason': 'INTERLEAVED or SHAM cells missing'}
    suffice = sum(1 for v in m.values() if v < THRESHOLD)
    uninterpretable = sorted(w for w, v in sham.items() if v < 0.45)
    ratio = float(np.median(list(sham.values()))) / max(float(np.median(list(m.values()))), 1e-12)
    if suffice >= 2:
        verdict = 'ANCHORS_SUFFICE'
    elif uninterpretable:
        verdict = 'PARTIAL_UNINTERPRETABLE'
    elif ratio >= 5.0:
        verdict = 'ANCHORS_PARTIAL'
    else:
        verdict = 'ANCHORS_INSUFFICIENT'
    return {'verdict': verdict, 'interleaved_by_world': m, 'sham_by_world': sham,
            'worlds_under_threshold': suffice, 'sham_ratio': ratio,
            'uninterpretable_worlds': uninterpretable,
            'guard': 'SHAM below 0.45 makes the PARTIAL clause uninterpretable, never INSUFFICIENT'}


def host_ok():
    free = psutil.virtual_memory().available / 2 ** 30
    return free, free >= MIN_FREE_GIB


def run(cells):
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
        atomic_json(ROOT / 'precondition.json',
                    {'free_gib': free, 'required_gib': MIN_FREE_GIB, 'passes': ok,
                     'checked_utc': now()})
        if not ok:
            raise RuntimeError(f'host precondition failed: {free:.1f} GiB free, '
                               f'{MIN_FREE_GIB} required; the reserve is never lowered to fit a run')
        records = {}
        try:
            log_line(ROOT / 'run.log', f'LAUNCH {sha} cells={len(cells)} free={free:.1f}GiB')
            for position, (arm, world) in enumerate(cells):
                key = cell_key(arm, world)
                path = ROOT / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(ROOT / 'run.log', f'reused validated cell {key}')
                else:
                    log_line(ROOT / 'run.log', f'cell start {key}')
                    record = run_cell(arm, world)
                    validate_cell(record)
                    atomic_json(path, {'stamp': {'protocol_sha256': sha}, 'complete': True,
                                       'record': record, 'record_sha256': fingerprint(record),
                                       'finished_utc': now()})
                    records[key] = record
                    log_line(ROOT / 'run.log',
                             f"cell finished {key} {record['seconds']:.1f}s "
                             f"terminal={record['terminal_median']:.6g}")
                atomic_json(ROOT / 'status.json',
                            {'state': 'running', 'cells_done': position + 1,
                             'cells_total': len(cells), 'updated_utc': now()})
            atomic_json(OUTPUT, {**manifest, 'complete': True, 'cells': records,
                                 'summary': triage(records), 'finished_utc': now()})
            atomic_json(ROOT / 'status.json', {'state': 'complete', 'cells_done': len(cells),
                                               'cells_total': len(cells), 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            atomic_json(ROOT / 'status.json', {'state': 'failed', 'cells_done': len(records),
                                               'cells_total': len(cells), 'updated_utc': now()})
            atomic_json(ROOT / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(ROOT / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='two pooled arms on world 0 at a tiny budget, no anchor')
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--worlds', type=int, nargs='+', default=list(WORLDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    require_clean_code(OUTPUT)
    for check in ('tools/check_prereg.py', 'tools/check_invalid.py'):
        subprocess.run([sys.executable, check], check=True)
    cells = [(a, w) for w in args.worlds for a in args.arms]
    run(cells)
    print(f'N1 report: {OUTPUT}')


if __name__ == '__main__':
    main()
