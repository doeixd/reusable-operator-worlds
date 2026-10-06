"""B1-online Tier 1 (exploratory): does a learner form the vocabulary AND learn branch decisions online?

Plan: `B1_ONLINE_PLAN.md`. Development worlds 44, 45, 46 (band 30-49), streams 0-2.
Stream: O2's SHUFFLED depth-3 stream (60 length-1, 64 length-2, 64 canonical length-3) plus 48 branch tasks
`IF(w.x > 0, A, B)` (A != B length-2 programs, decision on the input), one random order (`branch_stream`).
Arms (constructions):
- GATED: `BranchGatedLearner` (branch tasks get a second route code and a linear gate on x, trained by wake);
  end of stream: `deep_reroute.reroute` of the STRAIGHT-LINE tasks only (O6's re-route), then O3-style sleep on the
  64-per-task reservoir of ALL stream tasks, training the library, every task code, and the branch codes and gates
  (8,192 updates, sampling [1941, w, s, 64]).
- REFUSAL: `PlannedDepthRotatedLearner` on the same stream, branch tasks as ordinary single-route tasks; end of
  stream: `deep_reroute.reroute` of ALL tasks, then the same sleep (task codes and library).
- NOBRANCH: the same learner on the stream WITHOUT branch tasks (O2's SHUFFLED stream), re-route + sleep: O6's
  confirmed protocol on these worlds (the formation reference).
Scored on the terminal: canonical 64 straight-line tasks (formation) and the 48 branch tasks (branching), query NMSE
with hard routes and hard gates; gate accuracy against the true predicate on branch query inputs.
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
import sys
import torch

from row.experiments import b1g_gradient_branch as b1g
from row.experiments import branch_stream as bs
from row.experiments import deep_reroute as dr
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.models.branch_gated import BranchGatedLearner
from row.models.online_variable_depth import PlannedDepthRotatedLearner
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('B1_ONLINE_PLAN.md')
ROOT = Path('artifacts/b1_online')
DRY_ROOT = Path('artifacts/b1_online_dry')
OUTPUT = Path('reports/b1_online.json')
WORLDS = (44, 45, 46)
STREAMS = (0, 1, 2)
ARMS = ('GATED', 'REFUSAL', 'NOBRANCH')
N_BRANCH = 48
MEMORY = 64
SLEEP_UPDATES = 8192
BRANCH_SEED = 5700
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 49
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def build(cls, cfg, plan, **kw):
    sel = cfg.discrete_model
    return cls(d=cfg.world.state_dim, operator_slots=sel.operator_slots, operator_rank=sel.operator_rank,
               task_steps=3, alpha=sel.operator_alpha_init, initial_temperature=sel.initial_temperature,
               final_temperature=sel.final_temperature, seed=sel.seed, learnable_alpha=sel.learnable_alpha,
               activation=sel.operator_activation, depth_plan=plan, **kw)


def sleep(cfg, model, pool, task_ids, seed_entropy, extra=()):
    """o2c.consolidate's construction, with the branch codes and gates added to the task group."""
    global_lr, task_lr, weight_decay, _, _, _, _ = ll._training_values(cfg, o2.KIND)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = ll._shared_optimizer(model, global_lr, weight_decay)
    optimizer.add_param_group({'params': [model.task_codes[t] for t in task_ids] + list(extra), 'lr': task_lr,
                               'weight_decay': 0.0})
    model.set_training_progress(1.0)
    model.train()
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    for _ in range(SLEEP_UPDATES):
        idx = rng.choice(len(pool), size=2, replace=False)
        x = torch.as_tensor(np.stack([pool[i][0] for i in idx]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([pool[i][1] for i in idx]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model.forward_tasks(x, [pool[i][2] for i in idx]), y).backward()
        optimizer.step()
    model.eval()


def refit_branches(model, btasks, pool, w):
    """The branch analogue of end-of-stream re-routing: on the frozen current library, B1g's gradient fit (two soft
    codes + linear gate, 4 restarts, chosen on support) on each branch task's retained examples; installed only if its
    hardened support loss beats the current hardened setting. Returns the number of tasks whose fit was installed."""
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    accepted = 0
    model.eval()
    for t in btasks:
        xs = torch.tensor(np.stack([a for a, _ in by[t.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[t.task_id]]), dtype=torch.float32)
        with torch.no_grad():
            current = float(torch.mean((model(xs, t.task_id) - ys) ** 2))
        d = model.depth_of(t.task_id)
        fits = [b1g.fit(lib, xs, ys, d, False, [5800, w, i, r]) for i, r in [(btasks.index(t), r) for r in range(b1g.RESTARTS)]]
        loss, c1, c2, gw, gb = min(fits, key=lambda f: f[0])
        if loss < current:
            with torch.no_grad():
                model.task_codes[t.task_id][:d] = c1
                model.branch_codes[t.task_id][:d] = c2
                model.gate_w[t.task_id].copy_(gw)
                model.gate_b[t.task_id].copy_(gb)
            accepted += 1
    return accepted


def gate_accuracy(model, btasks, meta):
    accs = []
    with torch.no_grad():
        for t in btasks:
            x = torch.tensor(t.eval_x, dtype=torch.float32)
            truth = (t.eval_x @ np.array(meta[t.task_id]['w'])) > 0
            pred = model.gate(x, t.task_id).numpy() > 0.5
            accs.append(float(np.mean(pred == truth)))
    return accs


def run_arm(arm, w, s, work: Path, scale=1):
    if arm == 'NOBRANCH':
        cfg, world, st, plan, canonical = o2.build_stream('SHUFFLED', w, s)
        mixed, btasks, meta = dataclasses.replace(world, tasks=tuple(st)), [], {}
        model = build(PlannedDepthRotatedLearner, cfg, plan)
    else:
        cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(w, s, N_BRANCH)
        if arm == 'GATED':
            model = build(BranchGatedLearner, cfg, plan, branch_plan={t.task_id: 'input' for t in btasks},
                          branch_seed=BRANCH_SEED)
        else:
            model = build(PlannedDepthRotatedLearner, cfg, plan)
    out = work / 'lifetime'
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=out), o2.KIND, world=mixed,
                     model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')
    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st)}
    rec['wake_canonical_median'] = score(model, SimpleNamespace(tasks=canonical))['median']
    if btasks:
        rec['wake_branch_median'] = score(model, SimpleNamespace(tasks=btasks))['median']
    pool = o2d.reservoir(st, w, s, MEMORY)
    branch_ids = {t.task_id for t in btasks}
    reroute_tasks = [t for t in st if not (arm == 'GATED' and t.task_id in branch_ids)]
    rec['routes_changed'] = dr.reroute(model, reroute_tasks, plan, pool)
    if arm == 'GATED':
        rec['branch_refits_accepted'] = refit_branches(model, btasks, pool, w)
    extra = model.branch_parameters(branch_ids) if arm == 'GATED' else []
    before = library_sha(model)
    sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, MEMORY], extra)
    rec['library_sha256_before_sleep'], rec['library_sha256'] = before, library_sha(model)
    canon = score(model, SimpleNamespace(tasks=canonical))
    rec.update({'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
                'scored_canonical': len(canon['per_task'])})
    if btasks:
        br = score(model, SimpleNamespace(tasks=btasks))
        rec.update({'branch_median': br['median'], 'branch_per_task': br['per_task'],
                    'branch_below_threshold': sum(v < THRESHOLD for v in br['per_task'].values())})
        if arm == 'GATED':
            acc = gate_accuracy(model, btasks, meta)
            rec['gate_accuracy_median'] = float(np.median(acc))
            rec['branch_routes_distinct'] = sum(a != b for a, b in model.branch_routes().values())
    return rec


def run_cell(arm, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_arm(arm, w, s, Path(root) / 'work' / f'{arm}_w{w}_s{s}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if rec['scored_canonical'] != 64:
        raise ValueError(f'{key}: canonical scoring')
    expected = 188 if rec['arm'] == 'NOBRANCH' else 188 + N_BRANCH
    if rec['stream_tasks'] != expected:
        raise ValueError(f'{key}: stream size {rec["stream_tasks"]}')
    if rec['arm'] != 'NOBRANCH' and len(rec['branch_per_task']) != N_BRANCH:
        raise ValueError(f'{key}: branch scoring')
    if rec['library_sha256'] == rec['library_sha256_before_sleep']:
        raise ValueError(f'{key}: sleep left the library unchanged')


def passing(m):
    return m is not None and math.isfinite(m) and m < THRESHOLD


def label(n_better, k_formation, n):
    if k_formation < math.ceil(7 * n / 9):
        return 'FORMATION_BROKEN'
    return 'ONLINE_BRANCHES' if n_better >= math.ceil(8 * n / 9) else 'NO_BRANCH_GAIN'


def summarize(records, worlds=WORLDS):
    pairs = [(w, s) for w in worlds for s in STREAMS]
    R = {k: records[k] for k in records}
    better = sum(R[f'GATED_w{w}_s{s}']['branch_median'] < R[f'REFUSAL_w{w}_s{s}']['branch_median'] for w, s in pairs)
    formation = sum(passing(R[f'GATED_w{w}_s{s}']['canonical_median']) for w, s in pairs)
    med = lambda arm, key: float(np.median([R[f'{arm}_w{w}_s{s}'][key] for w, s in pairs]))   # noqa: E731
    return {'cells': len(pairs), 'n_better': better, 'k_formation': formation,
            'label': label(better, formation, len(pairs)),
            'canonical_passes': {a: sum(passing(R[f'{a}_w{w}_s{s}']['canonical_median']) for w, s in pairs) for a in ARMS},
            'canonical_median': {a: med(a, 'canonical_median') for a in ARMS},
            'branch_median': {a: med(a, 'branch_median') for a in ('GATED', 'REFUSAL')},
            'branch_below_threshold_median': {a: med(a, 'branch_below_threshold') for a in ('GATED', 'REFUSAL')},
            'gate_accuracy_median': med('GATED', 'gate_accuracy_median')}


def protocol(root=ROOT):
    return {'id': 'b1-online-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1, 'exploratory': True,
            'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS), 'arms': list(ARMS),
            'n_branch': N_BRANCH, 'memory': MEMORY, 'sleep_updates': SLEEP_UPDATES, 'branch_seed': BRANCH_SEED,
            'threshold': THRESHOLD,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'branch_stream_sha256': digest(Path(bs.__file__)),
            'branch_learner_sha256': digest(Path(sys.modules[BranchGatedLearner.__module__].__file__)),
            'b1g_sha256': digest(Path(b1g.__file__)),
            'deep_reroute_sha256': digest(Path(dr.__file__)), 'o2_sha256': digest(Path(o2.__file__)),
            'o2d_sha256': digest(Path(o2d.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, worlds=WORLDS):
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
            for arm, w, s in todo_cells:
                key = f'{arm}_w{w}_s{s}'
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s, str(root), scale): (a, w, s) for a, w, s in todo}
                while futures:
                    done = next(as_completed(futures))
                    a, w, s = futures.pop(done)
                    key = f'{a}_w{w}_s{s}'
                    record = done.result()
                    validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    extra = f" branch={record['branch_median']:.4g}" if 'branch_median' in record else ''
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"canonical={record['canonical_median']:.4g}{extra}")
                    _status(root, 'running', len(records), total, [f'{x}_w{y}_s{z}' for x, y, z in futures.values()][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records, worlds)
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
        run([(a, DRY_WORLD, 0) for a in ARMS], root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print('B1-online dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'B1-online report: {OUTPUT}')


if __name__ == '__main__':
    main()
