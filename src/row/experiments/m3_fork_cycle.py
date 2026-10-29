"""M3 Tier 1 (`M3_FORK_CYCLE_PLAN.md`, decision 24a): does a learner-side slot FORK repair before mining make M2's
end-of-stream motif cycle reliable across streams?

Per (world, stream), one cell on M2's stream construction (`motif_stream.build_stream_with_deep`, 364 tasks, 48 deep):
one wake (D2's construction, `task_steps=9`), the common exhaustive re-route of every task of length <= 5 (as M2),
then three PAIRED branches from deep copies of the common model, all on the same reservoir:
- INC: SY1's incumbent re-route of the 48 deep tasks (OPT 2,000 steps + one/two-position polish), as M2;
- CYC: M2's cycle unchanged (3-grams of length-3..5 routes, count >= 3, top 12; triple enumeration + polish);
- FCYC: FORK first, then the same cycle on the forked model. FORK (learner-computable; anchors = stream tasks of
  length 1, whose program length the learner is told, never their primitive): while some anchor's best-slot reservoir
  NMSE exceeds TAU (at most MAX_FORKS rounds), copy the slot most misfit anchors choose into a FREE slot (no anchor's
  best slot, same reflection count, least used by routes of length <= 5), fit only that slot on the misfit anchors'
  reservoir examples (Adam, FIT_STEPS, FIT_LR), re-evaluate; then refit each fork on the anchors that now choose it;
  then re-route every task of length <= 5 exhaustively again. With no misfit anchor FCYC is CYC exactly.
INC, CYC and FCYC keep the wake route when its reservoir MSE is no higher; each then runs O3's sleep (8,192 updates,
sampling [1941, w, s, 64]). Scores: terminal query NMSE of every deep and shallow task (256 evaluation examples).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import shutil
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
import torch

from row.experiments import deep_reroute as dr
from row.experiments import dn_stream as dn
from row.experiments import learned_lifetime as ll
from row.experiments import m2_motif_cycle as m2
from row.experiments import motif_stream as ms
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, fingerprint, log_line, now, writer_lock
from row.models.learned_models import batched_householder

PLAN = Path('M3_FORK_CYCLE_PLAN.md')
ROOT = Path('artifacts/m3_fork_cycle')
DRY_ROOT = Path('artifacts/m3_fork_cycle_dry')
OUTPUT = Path('reports/m3_fork_cycle.json')
REPORT_DIR = Path('reports/m3_fork_cycle_20261029')
WORLDS = (87, 88, 89)
STREAMS = (0, 1, 2)
BRANCHES = ('WAKE', 'INC', 'CYC', 'FCYC')
TAU = 0.5
MAX_FORKS = 3
FIT_STEPS = 600
FIT_LR = 0.01
SOLVED, FORMED = 0.20, 0.05
NEED_FORMED, NEED_INC, NEED_SOLVE, HARM_MAX = 7, 7, 8, 2   # streams out of 9
JOBS = 3
MIN_FREE_GIB = 2.0
DRY_WORLD, DRY_SCALE = 99, 16


# ---------------------------------------------------------------- the fork repair
def nmse_t(pred, y):
    return float(torch.mean((pred - y) ** 2) / torch.mean((y - y.mean(0)) ** 2))


def op_apply(op, z):
    h = torch.tanh(z @ op.V.T + op.b)
    return (z + op.alpha * h @ op.U.T) @ batched_householder([op.rotation.vectors])[0]


def fit_slot(op, xs, ys):
    opt = torch.optim.Adam(op.parameters(), lr=FIT_LR)
    with torch.enable_grad():
        for _ in range(FIT_STEPS):
            opt.zero_grad()
            loss = sum(torch.mean((op_apply(op, x) - y) ** 2) for x, y in zip(xs, ys)) / len(xs)
            loss.backward()
            opt.step()
    return float(loss.detach())


def anchor_fit(model, anchors, xy):
    lib = FrozenLibrary(model)
    with torch.no_grad():
        return {t: sorted((nmse_t(m2.slot_apply(lib, xy[t][0], k), xy[t][1]), k) for k in range(lib.slots))[0]
                for t in anchors}


def reroute_short(model, st, plan, xy):
    lib = FrozenLibrary(model)
    changed = 0
    with torch.no_grad():
        for t in st:
            d = plan[t.task_id]
            if d <= ms.DEPTH:
                new = dr.exhaustive_route(lib, *xy[t.task_id], d)
                changed += [int(v) for v in model.hard_routes()[t.task_id][:d]] != new
                m2.set_route(model, t.task_id, new)
    return changed


def fork(model, st, plan, xy):
    anchors = [t.task_id for t in st if plan[t.task_id] == 1]
    info = {'rounds': []}
    forks = []
    fit = anchor_fit(model, anchors, xy)
    info['before'] = {'misfit': sum(fit[t][0] > TAU for t in anchors),
                      'median_nmse': float(np.median([fit[t][0] for t in anchors]))}
    for _ in range(MAX_FORKS):
        fit = anchor_fit(model, anchors, xy)
        misfit = [t for t in anchors if fit[t][0] > TAU]
        if not misfit:
            break
        routes = model.hard_routes()
        claimed = {fit[t][1] for t in anchors} | set(forks)
        usage = Counter(int(r) for t in st if plan[t.task_id] <= ms.DEPTH for r in routes[t.task_id][:plan[t.task_id]])
        src = Counter(fit[t][1] for t in misfit).most_common(1)[0][0]
        shape = model.library[src].rotation.vectors.shape   # reflection count fixes the orientation class
        free = sorted((usage[k], k) for k in range(len(model.library))
                      if k not in claimed and model.library[k].rotation.vectors.shape == shape)
        if not free:
            info['no_free_slot'] = True
            break
        new = free[0][1]
        model.library[new].load_state_dict(model.library[src].state_dict())
        loss = fit_slot(model.library[new], [xy[t][0] for t in misfit], [xy[t][1] for t in misfit])
        forks.append(new)
        info['rounds'].append({'misfit': len(misfit), 'src': src, 'new': new, 'fit_loss': loss})
    if forks:
        fit = anchor_fit(model, anchors, xy)
        for k in forks:
            mine = [t for t in anchors if fit[t][1] == k]
            if mine:
                fit_slot(model.library[k], [xy[t][0] for t in mine], [xy[t][1] for t in mine])
        fit = anchor_fit(model, anchors, xy)
        info['after'] = {'misfit': sum(fit[t][0] > TAU for t in anchors),
                         'fork_occupancy': {str(k): sum(fit[t][1] == k for t in anchors) for k in forks}}
        info['reroute_changed'] = reroute_short(model, st, plan, xy)
    info['forks'] = forks
    return info


# ---------------------------------------------------------------- one cell
def mining_record(model, st, plan, motifs, alphabet, top):
    routes = model.hard_routes()
    votes = {}
    for t in st:
        if plan[t.task_id] == 1:
            votes.setdefault(t.program.primitive_ids[0], Counter())[int(routes[t.task_id][0])] += 1
    slot_of = {p: c.most_common(1)[0][0] for p, c in votes.items()}   # teacher-side diagnostic only
    images = [tuple(slot_of.get(p, -1) for p in m) for m in motifs]
    return {'alphabet': [list(g) for g in alphabet], 'top_grams': top, 'motif_images': [list(m) for m in images],
            'images_in_alphabet': sum(m in alphabet for m in images), 'slot_of': {str(k): v for k, v in slot_of.items()},
            'distinct_anchor_slots': len(set(slot_of.values()))}


def run_cell(w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, world, st, plan, shallow, deep, motifs = ms.build_stream_with_deep(w, s, m2.N_DEEP)
    for t in st:
        if plan[t.task_id] <= 2 and not np.allclose(t.program.execute(world.library, t.train_x), t.train_y):
            raise RuntimeError(f'{t.task_id}: anchor teacher differs from the motif library')
    output = Path(root) / 'work' / f'w{w}_s{s}' / 'lifetime'
    summary = ll.run(dataclasses.replace(o2.run_cfg(cfg, scale), output_directory=output), o2.KIND, world=world,
                     model=dn.planned_model(ms.DEEP_DEPTH, cfg, plan), return_model=True,
                     replay_seed=dn.replay_seed_for(w, s))
    wake_model = summary.pop('terminal_model')
    rows = o2.task_summaries(output)
    rec = {'world': w, 'stream': s, 'scale': scale, 'motifs': [list(m) for m in motifs],
           'stream_tasks': len(st), 'trained_tasks': len(rows), 'wake_seconds': time.perf_counter() - started,
           'depth_histogram': {str(d): sum(v == d for v in plan.values()) for d in sorted(set(plan.values()))},
           'deep_ids': [t.task_id for t in deep], 'deep_programs': [list(t.program.primitive_ids) for t in deep],
           'prequential': summary.get('cumulative_prequential_gaussian_log_loss'),
           'wake_library_sha256': library_sha(wake_model), 'branches': {}}
    rec['branches']['WAKE'] = {'shallow': m2.scores(wake_model, shallow), 'deep': m2.scores(wake_model, deep)}
    pool = o2d.reservoir(st, w, s, o3.MEMORY)
    xy = m2.pool_xy(pool)
    common = copy.deepcopy(wake_model)
    t0 = time.perf_counter()
    rec['shallow_reroute'] = {'changed': reroute_short(common, st, plan, xy), 'seconds': time.perf_counter() - t0}
    current = {t.task_id: [int(v) for v in common.hard_routes()[t.task_id][:ms.DEEP_DEPTH]] for t in deep}
    for branch in ('INC', 'CYC', 'FCYC'):
        m = copy.deepcopy(common)
        t0 = time.perf_counter()
        extra = {}
        if branch == 'FCYC':
            extra['fork'] = fork(m, st, plan, xy)
        if branch != 'INC':
            alphabet, top = m2.mine(m, st, plan)
            extra['mining'] = mining_record(m, st, plan, motifs, alphabet, top)
        lib = FrozenLibrary(m)
        info = {}
        with torch.no_grad():
            for t in deep:
                x, y = xy[t.task_id]
                cand, detail = m2.choose_cyc(lib, alphabet, x, y) if branch != 'INC' else m2.choose_inc(lib, x, y)
                keep = m2.support_mse(lib, current[t.task_id], x, y) <= m2.support_mse(lib, cand, x, y)
                new = current[t.task_id] if keep else cand
                m2.set_route(m, t.task_id, new)
                info[t.task_id] = {**detail, 'route': new, 'kept_wake_route': keep,
                                   'reservoir_mse': m2.support_mse(lib, new, x, y)}
        search_seconds = time.perf_counter() - t0
        before = library_sha(m)
        o2c.consolidate(cfg, m, pool, [t.task_id for t in st], [1941, w, s, o3.MEMORY], updates=o3.EXTRA_UPDATES)
        rec['branches'][branch] = {'shallow': m2.scores(m, shallow), 'deep': m2.scores(m, deep),
                                   'search_seconds': search_seconds, 'deep_routes': info,
                                   'library_sha256_before_sleep': before, 'library_sha256': library_sha(m), **extra}
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"w{rec['world']}_s{rec['stream']}"
    if rec['stream_tasks'] != m2.N_STREAM or rec['trained_tasks'] != m2.N_STREAM:
        raise ValueError(f'{key}: stream size')
    hist = Counter(len(p) for p in ms.programs(rec['world'], 6)[1])
    want = {'1': 60, '2': 64, **{str(k): v for k, v in sorted(hist.items())}, '9': m2.N_DEEP}
    if rec['depth_histogram'] != want:
        raise ValueError(f'{key}: depth histogram')
    for b in BRANCHES:
        br = rec['branches'][b]
        if len(br['deep']) != m2.N_DEEP or len(br['shallow']) != ms.N_SHALLOW:
            raise ValueError(f'{key}: {b} not scored on every task')
        if not all(math.isfinite(v) for v in list(br['deep'].values()) + list(br['shallow'].values())):
            raise ValueError(f'{key}: {b} non-finite score')
        if b != 'WAKE':
            if br['library_sha256'] == br['library_sha256_before_sleep']:
                raise ValueError(f'{key}: {b} sleep left the library unchanged')
            if len(br['deep_routes']) != m2.N_DEEP:
                raise ValueError(f'{key}: {b} did not route every deep task')
    f = rec['branches']['FCYC']['fork']
    if not f['forks'] and (rec['branches']['FCYC']['deep'] != rec['branches']['CYC']['deep']
                           or rec['branches']['FCYC']['library_sha256'] != rec['branches']['CYC']['library_sha256']):
        raise ValueError(f'{key}: FCYC without forks differs from CYC')


# ---------------------------------------------------------------- scoring (independent scorer: score_m3_fork_cycle)
def solved_fraction(rec, branch):
    v = list(rec['branches'][branch]['deep'].values())
    return sum(x < SOLVED for x in v) / len(v)


def summarize(records):
    streams = [records[k] for k in sorted(records)]
    f = {b: [solved_fraction(r, b) for r in streams] for b in BRANCHES}
    shallow = {b: [float(np.median(list(r['branches'][b]['shallow'].values()))) for r in streams] for b in BRANCHES}
    formed = sum(x < FORMED for x in shallow['FCYC'])
    inc_solves = sum(x >= 0.8 for x in f['INC'])
    fcyc_solves = sum(x >= 0.8 for x in f['FCYC'])
    gain = sum(a - b >= 0.5 for a, b in zip(f['FCYC'], f['INC']))
    harm = sum(c >= 0.8 and fc < 0.8 for c, fc in zip(f['CYC'], f['FCYC']))
    rescue = sum(c < 0.8 and fc >= 0.8 for c, fc in zip(f['CYC'], f['FCYC']))
    if formed < NEED_FORMED:
        label = 'FORMATION_BROKEN'
    elif inc_solves >= NEED_INC:
        label = 'NO_NECESSITY'
    elif harm >= HARM_MAX:
        label = 'REPAIR_HARMS'
    elif fcyc_solves >= NEED_SOLVE and gain >= NEED_SOLVE:
        label = 'REPAIRED_CYCLE_SOLVES'
    else:
        label = 'PARTIAL'
    return {'label': label, 'denominator_streams': len(streams), 'formed_fcyc': formed, 'incumbent_solves': inc_solves,
            'fcyc_solves': fcyc_solves, 'cyc_solves': sum(x >= 0.8 for x in f['CYC']), 'paired_gain': gain,
            'harm': harm, 'rescue': rescue, 'forked_streams': sum(bool(r['branches']['FCYC']['fork']['forks'])
                                                                  for r in streams),
            'solved_fraction': f, 'shallow_median': shallow}


def _synthetic(rng, n, p_formed, p_work, q_work, q_inc, p_cyc_given_work, p_harm):
    """Per stream: formed w.p. p_formed; FCYC works w.p. p_work (rate q_work, else 0); CYC works w.p.
    p_cyc_given_work when FCYC works, and, when FCYC fails, CYC works w.p. p_harm (a harmed stream); INC rate q_inc."""
    recs = {}
    for i in range(n):
        work = rng.random() < p_work
        cyc_work = rng.random() < (p_cyc_given_work if work else p_harm)
        sh = 0.02 if rng.random() < p_formed else 0.5
        rates = {'WAKE': 0.0, 'INC': q_inc, 'CYC': q_work if cyc_work else 0.0, 'FCYC': q_work if work else 0.0}
        recs[i] = {'branches': {b: {'deep': {j: (0.05 if rng.random() < q else 1.5) for j in range(m2.N_DEEP)},
                                    'shallow': {0: sh}, 'fork': {'forks': []}} for b, q in rates.items()}}
    return recs


DISCRIMINATION_SCENARIOS = {
    # name: (p_formed, p_work, q_work, q_inc, p_cyc_given_work, p_harm)
    'null_half_streams': (1.0, 0.5, 0.97, 0.0, 0.7, 0.0),
    'null_m2_rate': (1.0, 2 / 3, 0.97, 0.0, 1.0, 0.0),
    'null_incumbent_also': (1.0, 1.0, 0.97, 0.9, 1.0, 0.0),
    'null_harm_0.3': (1.0, 0.7, 0.97, 0.0, 0.7, 1.0),
    'effect_sizing': (1.0, 1.0, 1.0, 0.0, 0.7, 0.0),
    'effect_0.95_streams': (1.0, 0.95, 0.97, 0.0, 0.7, 0.0),
    'effect_0.9_streams': (1.0, 0.9, 0.97, 0.0, 0.7, 0.0),
    'effect_harm_rare': (1.0, 0.95, 0.97, 0.0, 0.7, 0.5),
}


def discrimination(draws=20000, seed=2712):
    rng = np.random.default_rng(seed)
    out = {}
    for name, args in DISCRIMINATION_SCENARIOS.items():
        c = Counter(summarize(_synthetic(rng, len(WORLDS) * len(STREAMS), *args))['label'] for _ in range(draws))
        out[name] = {k: c[k] / draws for k in ('FORMATION_BROKEN', 'NO_NECESSITY', 'REPAIR_HARMS',
                                               'REPAIRED_CYCLE_SOLVES', 'PARTIAL')}
    return out


# ---------------------------------------------------------------- orchestration
def protocol(root, scale, worlds):
    from row.experiments.so1_storage import digest
    return {'id': 'm3-fork-cycle-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'root': Path(root).as_posix(), 'worlds': list(worlds), 'streams': list(STREAMS), 'n_deep': m2.N_DEEP,
            'branches': list(BRANCHES), 'min_count': m2.MIN_COUNT, 'max_alphabet': m2.MAX_ALPHABET,
            'opt_steps': m2.OPT_STEPS, 'polish_cap': m2.POLISH_CAP, 'tau': TAU, 'max_forks': MAX_FORKS,
            'fit_steps': FIT_STEPS, 'fit_lr': FIT_LR, 'solved': SOLVED, 'formed': FORMED,
            'need_streams': [NEED_FORMED, NEED_INC, NEED_SOLVE, HARM_MAX], 'memory': o3.MEMORY,
            'extra_updates': o3.EXTRA_UPDATES, 'scale': scale,
            'input_sha256': {p: (digest(Path(p)) if Path(p).exists() else None)
                             for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': {mod.__name__: digest(Path(mod.__file__))
                                      for mod in (ms, dn, dr, o2, o2c, o2d, o3, ll, m2)} | {'m3': digest(Path(__file__))}}


def _status(root, state, done, total, running, started):
    eta = None if not done else (time.time() - started) / done * (total - done)
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'updated_utc': now(),
                                             'eta_seconds': None if eta is None else round(eta)})


def run(keys, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root, scale, worlds)
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
            for w, s in keys:
                key = f'w{w}_s{s}'
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((w, s))
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} scale={scale} '
                                       f'jobs={jobs} free={free:.1f}GiB')
            _status(root, 'running', len(records), len(keys), [f'w{w}_s{s}' for w, s in todo][:jobs], started)
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = {}
                for w, s in todo:
                    futures[ex.submit(run_cell, w, s, str(root), scale)] = (w, s)
                    log_line(root / 'run.log', f'cell submitted w{w}_s{s}')
                for fut in as_completed(futures):
                    w, s = futures[fut]
                    key = f'w{w}_s{s}'
                    record = fut.result()
                    validate(record)
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.0f}s " + ' '.join(
                        f"{b}: deep {solved_fraction(record, b):.2f} shallow "
                        f"{np.median(list(record['branches'][b]['shallow'].values())):.4f}" for b in BRANCHES)
                        + f" forks {record['branches']['FCYC']['fork']['forks']}")
                    _status(root, 'running', len(records), len(keys),
                            [f'w{a}_s{b}' for (a, b) in futures.values() if f'w{a}_s{b}' not in records][:jobs], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        ex.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            report = {**manifest, 'complete': True, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['summary'] = summarize(records)
                log_line(root / 'run.log', f"SUMMARY {report['summary']['label']}")
            atomic_json(output, report)
            _status(root, 'complete', len(records), len(keys), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
            if scale == 1 and root == ROOT:
                REPORT_DIR.mkdir(parents=True, exist_ok=True)
                for name in ('run.log', 'status.json', 'manifest.json', 'precondition.json', 'exit.json'):
                    shutil.copy2(root / name, REPORT_DIR / name)
        except BaseException:
            _status(root, 'failed', len(records), len(keys), [], started)
            atomic_json(root / 'exit.json', {'exit_code': 1, 'complete': False, 'finished_utc': now()})
            atomic_json(root / 'error.json', {'traceback': traceback.format_exc(), 'finished_utc': now()})
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stop-after', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=JOBS, help='pool size (memory-bound; does not change results)')
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        run([(DRY_WORLD, 0), (DRY_WORLD, 1)], root=DRY_ROOT, output=DRY_ROOT / 'report.json', jobs=2,
            scale=DRY_SCALE, stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print('M3 dry run complete')
        return
    require_clean_code(OUTPUT)
    run([(w, s) for w in WORLDS for s in STREAMS], jobs=args.jobs, stop_after=args.stop_after)
    print(f'M3 report: {OUTPUT}')


if __name__ == '__main__':
    main()
