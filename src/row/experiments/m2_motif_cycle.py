"""M2 Tier 1 (`M2_MOTIF_CYCLE_PLAN.md`): with deep (depth-9) motif tasks IN the online stream, does an end-of-stream
library-growth cycle (macros mined from the learner's own shallow routes) let the formed library solve them where the
primitive-level incumbent re-router cannot?

Per (world, stream), one cell: the motif stream with 48 deep tasks (`motif_stream.build_stream_with_deep`, 364
tasks), one wake (D2's construction, learner `task_steps=9`), then four PAIRED branches from deep copies of the same
wake terminal, all on the same reservoir (`o2d.reservoir`, 64 retained examples per task):
- WAKE: the wake terminal itself (record only);
- every other branch first re-routes every task of length <= 5 exhaustively (`deep_reroute.exhaustive_route`, D2's
  re-route), identically; then, for the 48 deep tasks:
  - KEEP: keeps the wake's routes (the confirmed pipeline with the infeasible depth-9 exhaustive re-route skipped);
  - INC: SY1's incumbent - SO1R `optimize_route` (2,000 steps) then a one/two-position slot polish;
  - CYC: macros mined from the re-routed hard routes of the stream tasks of length 3-5 (3-grams at every offset,
    count >= 3, most frequent first, at most 12); every triple enumerated, then the same polish;
  INC and CYC keep the wake's route when it has lower reservoir MSE than their candidate. Then O3's sleep
  (8,192 updates, sampling [1941, w, s, 64]) on each of KEEP, INC, CYC.
Recorded: terminal query NMSE of every deep and shallow task per branch, mined alphabet, the mapped-teacher reference
(learner-side reference only: it uses the teacher program).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import math
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import deep_reroute as dr
from row.experiments import dn_stream as dn
from row.experiments import learned_lifetime as ll
from row.experiments import motif_stream as ms
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary, optimize_route
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('M2_MOTIF_CYCLE_PLAN.md')
ROOT = Path('artifacts/m2_motif_cycle')
DRY_ROOT = Path('artifacts/m2_motif_cycle_dry')
OUTPUT = Path('reports/m2_motif_cycle.json')
WORLDS = (84, 85, 86)
STREAMS = (0, 1, 2)
N_DEEP = 48
N_STREAM = 60 + 64 + ms.N_SHALLOW + N_DEEP   # 364
BRANCHES = ('WAKE', 'KEEP', 'INC', 'CYC')
MIN_COUNT, MAX_ALPHABET = 3, 12
OPT_STEPS = 2000
POLISH_CAP = 30000
SOLVED = 0.10
FORMED = 0.05
JOBS = 3
MIN_FREE_GIB = 2.0
DRY_WORLD = 99
DRY_SCALE = 16
REPORT_DIR = Path('reports/m2_motif_cycle_20261027')


# ---------------------------------------------------------------- routes on a frozen library
def slot_apply(lib, z, k):
    h = z @ lib.V[k].T + lib.b[k]
    h = torch.nn.functional.gelu(h) if lib.gelu else torch.tanh(h)
    return (z + lib.alpha[k] * h @ lib.U[k].T) @ lib.Q[k]


def run_route(lib, route, x):
    for k in route:
        x = slot_apply(lib, x, int(k))
    return x


def support_mse(lib, route, x, y):
    return float(torch.mean((run_route(lib, route, x) - y) ** 2))


def polish(lib, seq, x, y):
    seq, cur, evals = [int(v) for v in seq], support_mse(lib, seq, x, y), 0
    for width in (1, 2):
        improved = True
        while improved and evals < POLISH_CAP:
            improved = False
            for pos in itertools.combinations(range(len(seq)), width):
                for vals in itertools.product(range(lib.slots), repeat=width):
                    if all(seq[p] == v for p, v in zip(pos, vals)) or evals >= POLISH_CAP:
                        continue
                    cand = list(seq)
                    for p, v in zip(pos, vals):
                        cand[p] = v
                    s = support_mse(lib, cand, x, y)
                    evals += 1
                    if s < cur:
                        seq, cur, improved = cand, s, True
            if improved and width == 2:
                break
    return seq, cur, evals


def choose_cyc(lib, alphabet, x, y):
    best = (math.inf, [0] * ms.DEEP_DEPTH)

    def rec(z, seq, k):
        nonlocal best
        if k == ms.DEEP_DEPTH // ms.MOTIF_LEN:
            v = float(torch.mean((z - y) ** 2))
            if v < best[0]:
                best = (v, seq)
            return
        for g in alphabet:
            rec(run_route(lib, g, z), seq + list(g), k + 1)
    if alphabet:
        rec(x, [], 0)
    seq, s, evals = polish(lib, best[1], x, y)
    return seq, {'enum_mse': best[0], 'enum_evaluations': len(alphabet) ** 3, 'polish_evaluations': evals}


def choose_inc(lib, x, y):
    lib.steps = ms.DEEP_DEPTH
    with torch.enable_grad():
        route, _, _, _ = optimize_route(lib, x, y, OPT_STEPS)
    seq, s, evals = polish(lib, route, x, y)
    return seq, {'opt_route': [int(v) for v in route], 'opt_mse': support_mse(lib, route, x, y),
                 'polish_evaluations': evals}


def set_route(model, task_id, new):
    code = model.task_codes[task_id]
    for step, r in enumerate(new):
        old = int(torch.argmax(code[step]))
        if old != r:
            a, b = code[step, old].clone(), code[step, r].clone()
            code[step, old], code[step, r] = b, a
    if [int(v) for v in model.hard_routes()[task_id][:len(new)]] != [int(v) for v in new]:
        raise RuntimeError(f'{task_id}: route swap did not take')


def pool_xy(pool):
    by = {}
    for x, y, t in pool:
        by.setdefault(t, []).append((x, y))
    return {t: (torch.tensor(np.stack([a for a, _ in v]), dtype=torch.float32),
                torch.tensor(np.stack([b for _, b in v]), dtype=torch.float32)) for t, v in by.items()}


def mine(model, st, plan):
    routes = model.hard_routes()
    rs = [tuple(int(v) for v in routes[t.task_id][:plan[t.task_id]]) for t in st if 3 <= plan[t.task_id] <= 5]
    counts = Counter(r[j:j + 3] for r in rs for j in range(len(r) - 2))
    alphabet = sorted((g for g, c in counts.items() if c >= MIN_COUNT), key=lambda g: (-counts[g], g))[:MAX_ALPHABET]
    return alphabet, {','.join(map(str, g)): c for g, c in counts.most_common(20)}


# ---------------------------------------------------------------- one cell
def scores(model, tasks):
    model.eval()
    return score(model, SimpleNamespace(tasks=tasks))['per_task']


def run_cell(w, s, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    cfg, world, st, plan, shallow, deep, motifs = ms.build_stream_with_deep(w, s, N_DEEP)
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
           'shallow_ids': [t.task_id for t in shallow], 'deep_ids': [t.task_id for t in deep],
           'deep_programs': [list(t.program.primitive_ids) for t in deep],
           'prequential': summary.get('cumulative_prequential_gaussian_log_loss'),
           'wake_library_sha256': library_sha(wake_model), 'branches': {}}
    rec['branches']['WAKE'] = {'shallow': scores(wake_model, shallow), 'deep': scores(wake_model, deep)}
    pool = o2d.reservoir(st, w, s, o3.MEMORY)
    xy = pool_xy(pool)
    # common re-route of every task of length <= 5 (D2's chooser), done once and copied into every branch
    common = copy.deepcopy(wake_model)
    lib = FrozenLibrary(common)
    t0 = time.perf_counter()
    changed = 0
    with torch.no_grad():
        for t in st:
            d = plan[t.task_id]
            if d <= ms.DEPTH:
                new = dr.exhaustive_route(lib, *xy[t.task_id], d)
                changed += [int(v) for v in common.hard_routes()[t.task_id][:d]] != new
                set_route(common, t.task_id, new)
    rec['shallow_reroute'] = {'changed': changed, 'seconds': time.perf_counter() - t0}
    alphabet, top = mine(common, st, plan)
    slot_of = {}
    votes = {}
    routes = common.hard_routes()
    for t in st:
        if plan[t.task_id] == 1:
            votes.setdefault(t.program.primitive_ids[0], Counter())[int(routes[t.task_id][0])] += 1
    slot_of = {p: c.most_common(1)[0][0] for p, c in votes.items()}
    images = [tuple(slot_of.get(p, -1) for p in m) for m in motifs]
    rec['mining'] = {'alphabet': [list(g) for g in alphabet], 'top_grams': top,
                     'motif_images': [list(m) for m in images], 'images_in_alphabet': sum(m in alphabet for m in images),
                     'slot_of': {str(k): v for k, v in slot_of.items()}}
    lib = FrozenLibrary(common)
    with torch.no_grad():
        current = {t.task_id: [int(v) for v in routes[t.task_id][:ms.DEEP_DEPTH]] for t in deep}
        mapped = {t.task_id: [slot_of.get(p, -1) for p in t.program.primitive_ids] for t in deep}
        rec['mapped_reference'] = {
            t.task_id: (None if -1 in mapped[t.task_id] else
                        float(torch.mean((run_route(lib, mapped[t.task_id], torch.tensor(t.eval_x, dtype=torch.float32))
                                          - torch.tensor(t.eval_y, dtype=torch.float32)) ** 2)
                              / np.mean((t.eval_y - t.eval_y.mean(0)) ** 2)))
            for t in deep}
    for branch in ('KEEP', 'INC', 'CYC'):
        m = copy.deepcopy(common)
        t0 = time.perf_counter()
        info = {}
        with torch.no_grad():
            for t in deep:
                x, y = xy[t.task_id]
                if branch == 'KEEP':
                    continue
                cand, detail = choose_cyc(lib, alphabet, x, y) if branch == 'CYC' else choose_inc(lib, x, y)
                keep = support_mse(lib, current[t.task_id], x, y) <= support_mse(lib, cand, x, y)
                new = current[t.task_id] if keep else cand
                set_route(m, t.task_id, new)
                info[t.task_id] = {**detail, 'route': new, 'kept_wake_route': keep,
                                   'equals_mapped': new == mapped[t.task_id],
                                   'reservoir_mse': support_mse(lib, new, x, y)}
        search_seconds = time.perf_counter() - t0
        pre = {'shallow': scores(m, shallow), 'deep': scores(m, deep)}
        before = library_sha(m)
        o2c.consolidate(cfg, m, pool, [t.task_id for t in st], [1941, w, s, o3.MEMORY], updates=o3.EXTRA_UPDATES)
        rec['branches'][branch] = {'shallow': scores(m, shallow), 'deep': scores(m, deep),
                                   'pre_sleep_shallow_median': float(np.median(list(pre['shallow'].values()))),
                                   'pre_sleep_deep_median': float(np.median(list(pre['deep'].values()))),
                                   'search_seconds': search_seconds, 'deep_routes': info,
                                   'library_sha256_before_sleep': before, 'library_sha256': library_sha(m)}
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"w{rec['world']}_s{rec['stream']}"
    if rec['stream_tasks'] != N_STREAM or rec['trained_tasks'] != N_STREAM:
        raise ValueError(f'{key}: stream size')
    hist = Counter(len(p) for p in ms.programs(rec['world'], 6)[1])
    want = {'1': 60, '2': 64, **{str(k): v for k, v in sorted(hist.items())}, '9': N_DEEP}
    if rec['depth_histogram'] != want:
        raise ValueError(f'{key}: depth histogram {rec["depth_histogram"]} != {want}')
    for b in BRANCHES:
        br = rec['branches'][b]
        if len(br['deep']) != N_DEEP or len(br['shallow']) != ms.N_SHALLOW:
            raise ValueError(f'{key}: {b} not scored on every task')
        vals = list(br['deep'].values()) + list(br['shallow'].values())
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f'{key}: {b} non-finite score')
        if b != 'WAKE' and br['library_sha256'] == br['library_sha256_before_sleep']:
            raise ValueError(f'{key}: {b} sleep left the library unchanged')
    for b in ('INC', 'CYC'):
        if len(rec['branches'][b]['deep_routes']) != N_DEEP:
            raise ValueError(f'{key}: {b} did not route every deep task')


# ---------------------------------------------------------------- scoring (the independent scorer is score_m2)
def solved_fraction(rec, branch):
    v = list(rec['branches'][branch]['deep'].values())
    return sum(x < SOLVED for x in v) / len(v)


def summarize(records):
    streams = [records[k] for k in sorted(records)]
    f = {b: [solved_fraction(r, b) for r in streams] for b in BRANCHES}
    shallow = {b: [float(np.median(list(r['branches'][b]['shallow'].values()))) for r in streams] for b in BRANCHES}
    n = len(streams)
    formed = sum(x < FORMED for x in shallow['CYC'])
    inc_solves = sum(max(a, b) >= 0.8 for a, b in zip(f['INC'], f['KEEP']))
    cyc_solves = sum(x >= 0.8 for x in f['CYC'])
    paired = sum(c - max(i, k) >= 0.5 for c, i, k in zip(f['CYC'], f['INC'], f['KEEP']))
    if formed < 7:
        label = 'FORMATION_BROKEN'
    elif inc_solves >= 7:
        label = 'NO_NECESSITY'
    elif cyc_solves >= 7 and paired >= 7:
        label = 'CYCLE_SOLVES'
    else:
        label = 'PARTIAL'
    return {'label': label, 'denominator_streams': n, 'formed_cyc': formed, 'incumbent_solves': inc_solves,
            'cyc_solves': cyc_solves, 'paired_gain': paired, 'solved_fraction': f, 'shallow_median': shallow,
            'harm_shallow': sum(c > 1.5 * min(i, k) for c, i, k in zip(shallow['CYC'], shallow['INC'], shallow['KEEP']))}


# ---------------------------------------------------------------- orchestration
def protocol(root, scale, worlds):
    return {'id': 'm2-motif-cycle-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'root': Path(root).as_posix(), 'worlds': list(worlds), 'streams': list(STREAMS), 'n_deep': N_DEEP,
            'branches': list(BRANCHES), 'min_count': MIN_COUNT, 'max_alphabet': MAX_ALPHABET, 'opt_steps': OPT_STEPS,
            'polish_cap': POLISH_CAP, 'solved': SOLVED, 'formed': FORMED, 'memory': o3.MEMORY,
            'extra_updates': o3.EXTRA_UPDATES, 'scale': scale,
            'input_sha256': {p: (digest(Path(p)) if Path(p).exists() else None)
                             for p in (PLAN.as_posix(), 'configs/v1.yaml')},
            'implementation_sha256': {m.__name__: digest(Path(m.__file__))
                                      for m in (ms, dn, dr, o2, o2c, o2d, o3, ll)} | {'m2': digest(Path(__file__))}}


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
        if (root / 'manifest.json').exists():
            import json
            if json.loads((root / 'manifest.json').read_text()) != manifest:
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
            import json
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
                                       f'free={free:.1f}GiB')
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
                        + f" alphabet {len(record['mining']['alphabet'])} images {record['mining']['images_in_alphabet']}")
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
                import shutil
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
    ap.add_argument('--sizing-world', type=int, default=None, help='EXPLORATORY single-stream sizing (not a verdict)')
    ap.add_argument('--stop-after', type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.dry_run:
        run([(DRY_WORLD, 0), (DRY_WORLD, 1)], root=DRY_ROOT, output=DRY_ROOT / 'report.json', jobs=2,
            scale=DRY_SCALE, stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print('M2 dry run complete')
        return
    if args.sizing_world is not None:
        w = args.sizing_world
        run([(w, 0)], root=Path(f'artifacts/m2_sizing_w{w}'), output=Path(f'reports/m2_sizing_w{w}.json'), jobs=1,
            scale=1, worlds=(w,))
        return
    require_clean_code(OUTPUT)
    run([(w, s) for w in WORLDS for s in STREAMS])
    print(f'M2 report: {OUTPUT}')


if __name__ == '__main__':
    main()
