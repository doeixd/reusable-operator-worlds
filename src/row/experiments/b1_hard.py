"""B1-hard Tier 1 (exploratory): does the learner DISCOVER which tasks branch, and does state-conditioning (not capacity)
carry the gain? (PI critique of B1-online, "Too easy?", 2026-10-06.)

Plan: `B1_HARD_PLAN.md`. Same worlds, streams and branch tasks as B1-online (development worlds 44-46, streams 0-2;
`branch_stream`, 48 branch tasks in the 236-task stream), so every cell is paired with B1-online's committed GATED
(branch identity GIVEN) and REFUSAL cells.

Arms (constructions):
- ALLGATE: `BranchGatedLearner` with EVERY stream task in `branch_plan` (the learner is not told which tasks branch):
  second route code and a state gate on x for all 236 tasks; the gate's state weights carry weight decay
  GATE_DECAY (a usage cost) during wake and sleep. End of stream, per task, on its 64 retained examples with the
  library frozen: the best single route (`deep_reroute.exhaustive_route`) and the best branch (B1g's gradient fit,
  4 restarts, seed [5900, w, i, r]); the branch is kept iff its hardened support MSE is below BRANCH_RATIO x the
  single route's; otherwise the task becomes single-route (route installed by O5's swap; gate pinned constant to
  route 1). Then sleep (O3's construction, 8,192 updates, sampling [1941, w, s, 64]) on all codes, gates, library.
- CONSTMIX: the same learner and gates on every task, but state-independent (`state_gate=False`: a learned bias
  only, the capacity-matched control). End of stream: every task single-route (exhaustive), then the same sleep.
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
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import b1_online as b1o
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
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models.branch_gated import BranchGatedLearner

PLAN = Path('B1_HARD_PLAN.md')
ROOT = Path('artifacts/b1_hard')
DRY_ROOT = Path('artifacts/b1_hard_dry')
OUTPUT = Path('reports/b1_hard.json')
B1O_CELLS = Path('artifacts/b1_online/cells')
WORLDS = b1o.WORLDS
STREAMS = b1o.STREAMS
ARMS = ('ALLGATE', 'CONSTMIX')
GATE_DECAY = 0.1
BRANCH_RATIO = 0.5
BRANCH_SEED = 5700
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 2.0   # PI 2026-10-03 instruction
DRY_WORLD = 49
DRY_SCALE = 16


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def auc(pos, neg):
    """Probability a random positive outscores a random negative (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def install_single(model, task_id, route):
    with torch.no_grad():
        code = model.task_codes[task_id]
        for step, new in enumerate(route):
            old = int(torch.argmax(code[step]))
            if old != new:
                a, b = code[step, old].clone(), code[step, new].clone()
                code[step, old], code[step, new] = b, a
        if task_id in model.gate_w:
            model.gate_w[task_id].zero_()
        model.gate_b[task_id].fill_(10.0)   # constant gate on route 1


def end_of_stream(model, st, plan, pool, w, arm):
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    lib = FrozenLibrary(model)
    decisions = {}
    model.eval()
    for i, t in enumerate(st):
        xs = torch.tensor(np.stack([a for a, _ in by[t.task_id]]), dtype=torch.float32)
        ys = torch.tensor(np.stack([b for _, b in by[t.task_id]]), dtype=torch.float32)
        d = plan[t.task_id]
        single = dr.exhaustive_route(lib, xs, ys, d)
        lib.steps = d
        with torch.no_grad():
            single_loss = float(torch.mean((lib.hard(xs, single) - ys) ** 2))
        branch = False
        if arm == 'ALLGATE':
            fits = [b1g.fit(lib, xs, ys, d, False, [5900, w, i, r]) for r in range(b1g.RESTARTS)]
            loss, c1, c2, gw, gb = min(fits, key=lambda f: f[0])
            branch = loss < BRANCH_RATIO * single_loss
            if branch:
                with torch.no_grad():
                    model.task_codes[t.task_id][:d] = c1
                    model.branch_codes[t.task_id][:d] = c2
                    model.gate_w[t.task_id].copy_(gw)
                    model.gate_b[t.task_id].copy_(gb)
        if not branch:
            install_single(model, t.task_id, single)
        decisions[t.task_id] = branch
    return decisions


def sleep(cfg, model, pool, task_ids, seed_entropy):
    global_lr, task_lr, weight_decay, _, _, _, _ = ll._training_values(cfg, o2.KIND)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = ll._shared_optimizer(model, global_lr, weight_decay)
    plain = [model.task_codes[t] for t in task_ids] + [model.branch_codes[t] for t in task_ids] + \
            [model.gate_b[t] for t in task_ids]
    optimizer.add_param_group({'params': plain, 'lr': task_lr, 'weight_decay': 0.0})
    ws = [model.gate_w[t] for t in task_ids if t in model.gate_w]
    if ws:
        optimizer.add_param_group({'params': ws, 'lr': task_lr, 'weight_decay': GATE_DECAY})
    model.set_training_progress(1.0)
    model.train()
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    for _ in range(b1o.SLEEP_UPDATES):
        idx = rng.choice(len(pool), size=2, replace=False)
        x = torch.as_tensor(np.stack([pool[i][0] for i in idx]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([pool[i][1] for i in idx]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model.forward_tasks(x, [pool[i][2] for i in idx]), y).backward()
        optimizer.step()
    model.eval()


def run_arm(arm, w, s, work: Path, scale=1):
    import dataclasses
    cfg, mixed, st, plan, canonical, btasks, meta = bs.stream(w, s, b1o.N_BRANCH)
    allplan = {t.task_id: 'input' for t in st}
    model = b1o.build(BranchGatedLearner, cfg, plan, branch_plan=allplan, branch_seed=BRANCH_SEED,
                      state_gate=(arm == 'ALLGATE'), gate_decay=(GATE_DECAY if arm == 'ALLGATE' else None))
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=work / 'lifetime'), o2.KIND,
                     world=mixed, model=model, return_model=True, replay_seed=o2.replay_seed_for(w, s))
    model = summary.pop('terminal_model')
    branch_ids = {t.task_id for t in btasks}
    rec = {'arm': arm, 'world': w, 'stream': s, 'scale': scale, 'stream_tasks': len(st),
           'wake_canonical_median': score(model, SimpleNamespace(tasks=canonical))['median'],
           'wake_branch_median': score(model, SimpleNamespace(tasks=btasks))['median']}
    if arm == 'ALLGATE':
        norms = {t.task_id: float(model.gate_w[t.task_id].detach().norm()) for t in st}
        rec['wake_gate_norms'] = norms
        rec['wake_gate_auc'] = auc([norms[t] for t in branch_ids], [v for t, v in norms.items() if t not in branch_ids])
        rec['wake_gate_norm_median'] = {'branch': float(np.median([norms[t] for t in branch_ids])),
                                        'straight': float(np.median([v for t, v in norms.items() if t not in branch_ids]))}
    pool = o2d.reservoir(st, w, s, b1o.MEMORY)
    decisions = end_of_stream(model, st, plan, pool, w, arm)
    rec['branch_task_ids'] = sorted(branch_ids)
    rec['decisions'] = decisions
    rec['branch_recall'] = sum(decisions[t] for t in branch_ids)
    rec['straight_kept_single'] = sum(not v for t, v in decisions.items() if t not in branch_ids)
    before = library_sha(model)
    sleep(cfg, model, pool, [t.task_id for t in st], [1941, w, s, b1o.MEMORY])
    rec['library_sha256_before_sleep'], rec['library_sha256'] = before, library_sha(model)
    canon = score(model, SimpleNamespace(tasks=canonical))
    br = score(model, SimpleNamespace(tasks=btasks))
    rec.update({'canonical_median': canon['median'], 'canonical_per_task': canon['per_task'],
                'scored_canonical': len(canon['per_task']), 'branch_median': br['median'],
                'branch_per_task': br['per_task'], 'branch_below_threshold': sum(v < THRESHOLD for v in br['per_task'].values())})
    return rec


def run_cell(arm, w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_arm(arm, w, s, Path(root) / 'work' / f'{arm}_w{w}_s{s}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if rec['scored_canonical'] != 64 or len(rec['branch_per_task']) != b1o.N_BRANCH or rec['stream_tasks'] != 236:
        raise ValueError(f'{key}: construction')
    if rec['library_sha256'] == rec['library_sha256_before_sleep']:
        raise ValueError(f'{key}: sleep left the library unchanged')
    if rec['arm'] == 'CONSTMIX' and rec['branch_recall'] != 0:
        raise ValueError(f'{key}: the control kept a branch')


def passing(m):
    return m is not None and math.isfinite(m) and m < THRESHOLD


def label(k_formation, auc_median, recall, specificity, n_control):
    if k_formation <= 6:
        return 'FORMATION_BROKEN'
    if auc_median >= 0.9 and recall >= 0.8 and specificity >= 0.95 and n_control >= 8:
        return 'DISCOVERS'
    return 'NOT_DISCOVERED'


def summarize(records, worlds=WORLDS, b1o_cells=B1O_CELLS):
    pairs = [(w, s) for w in worlds for s in STREAMS]
    A = {k: records[f'ALLGATE_{k}'] for k in (f'w{w}_s{s}' for w, s in pairs)}
    C = {k: records[f'CONSTMIX_{k}'] for k in A}
    ref = {}
    for arm in ('GATED', 'REFUSAL'):
        for k in A:
            path = Path(b1o_cells) / f'{arm}_{k}.json'
            if path.exists():
                ref[f'{arm}_{k}'] = json.loads(path.read_text())['record']
    k_formation = sum(passing(a['canonical_median']) for a in A.values())
    auc_median = float(np.median([a['wake_gate_auc'] for a in A.values()]))
    n_branch_total = b1o.N_BRANCH * len(A)
    n_straight_total = (236 - b1o.N_BRANCH) * len(A)
    recall = sum(a['branch_recall'] for a in A.values()) / n_branch_total
    specificity = sum(a['straight_kept_single'] for a in A.values()) / n_straight_total
    n_control = sum(A[k]['branch_median'] < C[k]['branch_median'] for k in A)
    med = lambda D, key: float(np.median([d[key] for d in D.values()]))   # noqa: E731
    out = {'cells': len(A), 'k_formation': k_formation, 'wake_gate_auc_median': auc_median,
           'branch_recall': recall, 'straight_specificity': specificity, 'n_better_than_control': n_control,
           'label': label(k_formation, auc_median, recall, specificity, n_control),
           'canonical_median': {'ALLGATE': med(A, 'canonical_median'), 'CONSTMIX': med(C, 'canonical_median')},
           'branch_median': {'ALLGATE': med(A, 'branch_median'), 'CONSTMIX': med(C, 'branch_median')},
           'canonical_passes': {'ALLGATE': k_formation, 'CONSTMIX': sum(passing(c['canonical_median']) for c in C.values())}}
    if len(ref) == 2 * len(A):
        out['b1_online_reference'] = {
            'GATED_canonical_median': float(np.median([ref[f'GATED_{k}']['canonical_median'] for k in A])),
            'GATED_branch_median': float(np.median([ref[f'GATED_{k}']['branch_median'] for k in A])),
            'REFUSAL_canonical_median': float(np.median([ref[f'REFUSAL_{k}']['canonical_median'] for k in A])),
            'REFUSAL_branch_median': float(np.median([ref[f'REFUSAL_{k}']['branch_median'] for k in A]))}
    return out


def protocol(root=ROOT):
    return {'id': 'b1-hard-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1, 'exploratory': True,
            'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS), 'arms': list(ARMS),
            'gate_decay': GATE_DECAY, 'branch_ratio': BRANCH_RATIO, 'threshold': THRESHOLD,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': digest(Path(__file__)), 'b1_online_sha256': digest(Path(b1o.__file__)),
            'branch_learner_sha256': digest(Path(sys.modules[BranchGatedLearner.__module__].__file__)),
            'b1g_sha256': digest(Path(b1g.__file__)), 'branch_stream_sha256': digest(Path(bs.__file__)),
            'deep_reroute_sha256': digest(Path(dr.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


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
                    extra = f" auc={record['wake_gate_auc']:.3f}" if 'wake_gate_auc' in record else ''
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s canonical={record['canonical_median']:.4g} "
                                               f"branch={record['branch_median']:.4g} recall={record['branch_recall']} "
                                               f"single={record['straight_kept_single']}{extra}")
                    atomic_json(root / 'status.json', {'state': 'running', 'cells_done': len(records), 'cells_total': total,
                                                       'running': [f'{x}_w{y}_s{z}' for x, y, z in futures.values()][:jobs],
                                                       'pid': os.getpid(), 'updated_utc': now()})
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records, worlds)
            atomic_json(output, report)
            atomic_json(root / 'status.json', {'state': 'complete', 'cells_done': total, 'cells_total': total,
                                               'pid': os.getpid(), 'updated_utc': now()})
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
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
        print('B1-hard dry run complete')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'B1-hard report: {OUTPUT}')


if __name__ == '__main__':
    main()
