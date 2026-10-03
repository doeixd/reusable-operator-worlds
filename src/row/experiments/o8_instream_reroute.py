"""O8 Tier 1: in-stream re-routing of EARLIER tasks against the current library, during the stream.

Plan: `O8_INSTREAM_REROUTE_PLAN.md` (frozen b693ea2). EXPLORATORY; development worlds 20-26 (band 3),
three streams, paired with O3's committed SHUFFLED and INTERLEAVED cells.

Arms (constructions):
- REROUTE_WAKE: o2.run_single's SHUFFLED construction plus the `Rerouter` prospective hook. After task
  t has trained, every earlier task i < t is re-routed by exhaustive search on its 64 reservoir examples
  (the Interleaver's draw, seed [1940, w, s, i]) against the current library, installed by O5's
  minimal logit swap. Task t itself is never touched at step t, so the last-task anchor holds exactly.
  No consolidation updates; the hook performs no random draws.
- REROUTE_INTERLEAVED: o3.run_interleaved's construction with the `Rerouter` called BEFORE the
  Interleaver's consolidation step for that task (the Interleaver itself is o3's, verbatim).

Gates: E1 REROUTE_WAKE with re-routing disabled reproduces O3's committed SHUFFLED_w20_s0 bitwise at
full scale; E2 REROUTE_INTERLEAVED disabled matches o3.run_interleaved bitwise at DRY_SCALE and enabled
differs; E3 REROUTE_WAKE enabled at DRY_SCALE changes routes and the library hash against the base.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch

from row.experiments import census_o8_staleness_position as census
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments.audit_j1c_curriculum import library_sha
from row.experiments.audit_j2a_staged_library import enum_route
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock

PLAN = Path('O8_INSTREAM_REROUTE_PLAN.md')
O3_REPORT = Path('reports/o3_online_sleep_v2.json')
ROOT = Path('artifacts/o8_instream_reroute')
DRY_ROOT = Path('artifacts/o8_instream_reroute_dry')
OUTPUT = Path('reports/o8_instream_reroute.json')
WORLDS = o3.WORLDS
STREAMS = o3.STREAMS
ARMS = ('REROUTE_WAKE', 'REROUTE_INTERLEAVED')   # registered order: rule A's arm first
REFERENCE = {'REROUTE_WAKE': 'SHUFFLED', 'REROUTE_INTERLEAVED': 'INTERLEAVED'}
MEMORY = o3.MEMORY
EXTRA_UPDATES = o3.EXTRA_UPDATES
THRESHOLD = 0.05
JOBS = 3
MIN_FREE_GIB = 4.5   # O7's registered precondition, under the PI's 2026-09-27 instruction
DRY_SCALE = o3.DRY_SCALE


def cells(worlds=WORLDS):
    return [(a, w, s) for a in ARMS for w in worlds for s in STREAMS]


def protocol(root=ROOT):
    return {'id': 'o8-instream-reroute-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'root': Path(root).as_posix(), 'worlds': list(WORLDS), 'streams': list(STREAMS),
            'arms': list(ARMS), 'memory': MEMORY, 'extra_updates': EXTRA_UPDATES, 'threshold': THRESHOLD,
            'kind': o2.KIND, 'lean': o2.LEAN,
            'input_sha256': {p: digest(Path(p)) for p in (PLAN.as_posix(), 'configs/v1.yaml', O3_REPORT.as_posix())},
            'implementation_sha256': digest(Path(__file__)), 'o2_sha256': digest(Path(o2.__file__)),
            'o3_sha256': digest(Path(o3.__file__)), 'o2d_sha256': digest(Path(o2d.__file__)),
            'census_sha256': digest(Path(census.__file__)), 'lifetime_sha256': digest(Path(ll.__file__))}


# ------------------------------------------------------------------ hooks

class Rerouter:
    """prospective_hook: after task t, re-derive the routes of all EARLIER tasks against the current library.

    `enabled=False` is the switch that must recover the baseline bitwise (gates E1, E2): it still draws the
    reservoir (numpy, its own seeds) but never touches the model."""

    def __init__(self, stream_tasks, world, stream, enabled=True):
        self.stream, self.world, self.s, self.enabled = stream_tasks, world, stream, enabled
        self.plan = {t.task_id: len(t.program.primitive_ids) for t in stream_tasks}
        self.by = {}
        self.changed_per_pass, self.seconds = [], 0.0

    def add_reservoir(self, index):
        task = self.stream[index]
        pick = np.random.default_rng(np.random.SeedSequence([1940, self.world, self.s, index]))
        idx = pick.choice(len(task.train_x), size=MEMORY, replace=False)   # the Interleaver's draw, same order
        self.by[task.task_id] = (torch.tensor(task.train_x[idx], dtype=torch.float32),
                                 torch.tensor(task.train_y[idx], dtype=torch.float32))

    def reroute(self, model, upto):
        """Exhaustive re-route of stream tasks [0, upto) with the library frozen for the search. Returns #changed."""
        library = FrozenLibrary(model)
        changed = 0
        with torch.no_grad():
            for i in range(upto):
                task = self.stream[i]
                d = self.plan[task.task_id]
                xs, ys = self.by[task.task_id]
                library.steps = d
                new = [int(r) for r in enum_route(library, xs, ys)]
                code = model.task_codes[task.task_id]
                cur = [int(r) for r in torch.argmax(code[:d], dim=-1).tolist()]
                if cur != new:
                    for step in range(d):   # O5's minimal logit swap, verbatim
                        old = cur[step]
                        if old != new[step]:
                            a, b = code[step, old].clone(), code[step, new[step]].clone()
                            code[step, old], code[step, new[step]] = b, a
                    changed += 1
        return changed

    def __call__(self, model, lifetime_index, world_task_index):
        self.add_reservoir(world_task_index)
        if self.enabled and world_task_index >= 1:
            t0 = time.perf_counter()
            self.changed_per_pass.append(self.reroute(model, world_task_index))
            self.seconds += time.perf_counter() - t0
        return None


class RerouteInterleaver:
    """The Rerouter, then o3's Interleaver verbatim (reservoir add + n_t consolidation updates)."""

    def __init__(self, cfg, stream_tasks, world, stream, total=EXTRA_UPDATES, enabled=True):
        self.rerouter = Rerouter(stream_tasks, world, stream, enabled)
        self.interleaver = o3.Interleaver(cfg, stream_tasks, world, stream, total)

    def __call__(self, model, lifetime_index, world_task_index):
        self.rerouter(model, lifetime_index, world_task_index)
        return self.interleaver(model, lifetime_index, world_task_index)


# ------------------------------------------------------------------ cells

def run_arm(arm, world, stream, work: Path, scale=1, enabled=True):
    cfg3, world3, stream_tasks, plan, canonical = o2.build_stream('SHUFFLED', world, stream)
    mixed = dataclasses.replace(world3, tasks=tuple(stream_tasks))
    output = work / 'lifetime'
    if arm == 'REROUTE_WAKE':
        hook = Rerouter(stream_tasks, world, stream, enabled)
        rerouter, interleaver = hook, None
    elif arm == 'REROUTE_INTERLEAVED':
        hook = RerouteInterleaver(cfg3, stream_tasks, world, stream, EXTRA_UPDATES, enabled)
        rerouter, interleaver = hook.rerouter, hook.interleaver
    else:
        raise ValueError(arm)
    ran = o2.run_cfg(cfg3, scale)
    summary = ll.run(dataclasses.replace(ran, output_directory=output), o2.KIND, world=mixed,
                     model=o2.planned_model(cfg3, plan), return_model=True,
                     replay_seed=o2.replay_seed_for(world, stream), prospective_hook=hook)
    model = summary.pop('terminal_model')
    canon_ids = {t.task_id for t in canonical}
    terminal = score(model, SimpleNamespace(tasks=[t for t in canonical if t.task_id in model.task_codes]))
    rows = o2.task_summaries(output)
    final = {r['task_id']: float(r['final_nmse']) for r in rows}
    last = rows[-1]
    last_task = next(t for t in stream_tasks if t.task_id == last['task_id'])
    last_terminal = score(model, SimpleNamespace(tasks=[last_task]))['per_task'][last_task.task_id]
    routes = model.hard_routes()
    pool = o2d.reservoir(stream_tasks, world, stream, MEMORY)
    stale = census.staleness(model, stream_tasks, plan, pool)
    stale_positions = [r['position'] for r in stale.pop('rows') if r['stale']]
    return {'arm': arm, 'world': world, 'stream': stream, 'scale': scale, 'reroute_enabled': enabled,
            'terminal_median': terminal['median'], 'terminal_below': terminal['below_0.05'],
            'terminal_per_task': terminal['per_task'], 'scored_tasks': len(terminal['per_task']),
            'end_of_task_median': o2._median([final[t] for t in final if t in canon_ids]),
            'end_of_task_median_by_depth': {str(d): o2._median([v for t, v in final.items() if plan[t] == d])
                                            for d in (1, 2, 3)},
            'first16_canonical_eot': o3.first16_canonical_eot(output, canon_ids),
            'anchor_task_id': last['task_id'], 'anchor_depth': plan[last['task_id']],
            'anchor_abs_error': abs(last_terminal - float(last['final_nmse'])),
            'stream_tasks': len(stream_tasks), 'trained_tasks': len(rows),
            'depth_histogram': {str(d): sum(1 for v in plan.values() if v == d) for d in (1, 2, 3)},
            'routes_checked': sum(1 for t in plan if t in routes),
            'route_lengths_match_plan': all(t in routes and len(routes[t]) == plan[t] for t in plan),
            'reroute_passes': len(rerouter.changed_per_pass),
            'routes_changed_per_pass': list(rerouter.changed_per_pass),
            'routes_changed_total': int(sum(rerouter.changed_per_pass)),
            'reroute_seconds': rerouter.seconds,
            'extra_updates': interleaver.updates if interleaver is not None else 0,
            'pool_size': len(interleaver.pool) if interleaver is not None else 0,
            'terminal_stale': stale['stale'], 'terminal_stale_by_tercile': stale['by_tercile'],
            'terminal_stale_by_depth': stale['by_depth'], 'terminal_stale_positions': stale_positions,
            'prequential': summary.get('cumulative_prequential_gaussian_log_loss'),
            'library_sha256': library_sha(model)}


def run_cell(arm, world, stream, root, scale=1):
    torch.set_num_threads(1)
    started = time.perf_counter()
    rec = run_arm(arm, world, stream, Path(root) / 'work' / f'{arm}_w{world}_s{stream}', scale)
    rec['seconds'] = time.perf_counter() - started
    return rec


def validate(rec):
    key = f"{rec['arm']}_w{rec['world']}_s{rec['stream']}"
    if rec['arm'] not in ARMS or not rec['reroute_enabled']:
        raise ValueError(f'{key}: cell identity')
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if rec['stream_tasks'] != 188 or rec['trained_tasks'] != 188 or not rec['route_lengths_match_plan']:
        raise ValueError(f'{key}: stream/route check')
    if rec['depth_histogram'] != o2.EXPECTED_DEPTHS['SHUFFLED'] or rec['routes_checked'] != 188:
        raise ValueError(f'{key}: depth histogram / routes checked')
    if rec['reroute_passes'] != 187 or len(rec['routes_changed_per_pass']) != 187:
        raise ValueError(f'{key}: {rec["reroute_passes"]} re-route passes, not 187')
    # Nothing trains after REROUTE_WAKE's last task and its route is never re-derived at its own step,
    # so terminal == end-of-task there. REROUTE_INTERLEAVED consolidates after the last task (O3's
    # INTERLEAVED precedent): its anchor error is recorded, not required to vanish.
    if rec['arm'] == 'REROUTE_WAKE':
        if rec['anchor_abs_error'] > o2.ANCHOR_TOLERANCE:
            raise ValueError(f'{key}: last-task anchor {rec["anchor_abs_error"]}')
        if rec['extra_updates'] != 0:
            raise ValueError(f'{key}: REROUTE_WAKE trained extra updates')
    if rec['arm'] == 'REROUTE_INTERLEAVED' and rec['extra_updates'] != EXTRA_UPDATES:
        raise ValueError(f'{key}: extra updates {rec["extra_updates"]} != {EXTRA_UPDATES}')
    if rec['terminal_stale'] is None or rec['terminal_stale'] < 0:
        raise ValueError(f'{key}: staleness not measured')


# ------------------------------------------------------------------ rules

def passing(m):
    return m is not None and np.isfinite(m) and m < THRESHOLD


def label_a(k, n_better, h):
    if h >= 3:
        return 'HARMS'
    if k >= 14 and n_better >= 16:
        return 'PREVENTS'
    if k >= 14 or n_better >= 16:
        return 'PARTIAL'
    return 'NO_EFFECT'


def label_b(f, n_better, h):
    if h >= 3:
        return 'HARMS_B'
    if f <= 1 and n_better >= 16:
        return 'REACHES_CEILING'
    if n_better >= 16:
        return 'IMPROVES'
    return 'NO_EFFECT_B'


def summarize(records, worlds=WORLDS, o3_cells=None):
    o3_cells = o3_cells or json.loads(O3_REPORT.read_text())['cells']
    out = {'denominator': len(worlds) * len(STREAMS)}
    for arm in ARMS:
        ref = REFERENCE[arm]
        new = {(w, s): records[f'{arm}_w{w}_s{s}']['terminal_median'] for w in worlds for s in STREAMS}
        old = {(w, s): o3_cells[f'{ref}_w{w}_s{s}']['terminal_median'] for w in worlds for s in STREAMS}
        k = sum(passing(v) for v in new.values())
        better = sum(passing(new[c]) and np.isfinite(new[c]) and new[c] < old[c] for c in new)
        harm = sum(passing(old[c]) and not passing(new[c]) for c in new)
        stale = [records[f'{arm}_w{w}_s{s}']['terminal_stale'] for w in worlds for s in STREAMS]
        out[arm] = {'reference': ref, 'reference_passes': sum(passing(v) for v in old.values()),
                    'k': k, 'f': len(new) - k, 'n_better': better, 'h': harm,
                    'terminal_stale_median': float(np.median(stale)),
                    'label': label_a(k, better, harm) if arm == 'REROUTE_WAKE' else label_b(len(new) - k, better, harm)}
    return out


# ------------------------------------------------------------------ gates

def gates(root=ROOT):
    g = Path(root) / 'gates'
    g.mkdir(parents=True, exist_ok=True)
    out = {}
    o3cells = json.loads(O3_REPORT.read_text())['cells']
    e1 = run_arm('REROUTE_WAKE', 20, 0, g / 'E1', 1, enabled=False)
    ref = o3cells['SHUFFLED_w20_s0']
    out['E1'] = {'passes': e1['library_sha256'] == ref['library_sha256']
                 and e1['terminal_per_task'] == ref['terminal_per_task'] and e1['routes_changed_total'] == 0}
    log_line(g / 'gates.log', f"E1 {out['E1']}")
    inter = o3.run_interleaved(20, 0, g / 'E2_inter', DRY_SCALE)
    off = run_arm('REROUTE_INTERLEAVED', 20, 0, g / 'E2_off', DRY_SCALE, enabled=False)
    on = run_arm('REROUTE_INTERLEAVED', 20, 0, g / 'E2_on', DRY_SCALE, enabled=True)
    out['E2'] = {'passes': off['library_sha256'] == inter['library_sha256']
                 and off['terminal_per_task'] == inter['terminal_per_task']
                 and on['routes_changed_total'] > 0 and on['library_sha256'] != inter['library_sha256']
                 and on['extra_updates'] == EXTRA_UPDATES,
                 'off_matches_interleaved': off['library_sha256'] == inter['library_sha256'],
                 'on_routes_changed': on['routes_changed_total']}
    log_line(g / 'gates.log', f"E2 {out['E2']}")
    base = o2.run_single('SHUFFLED', 20, 0, g / 'E3_base', DRY_SCALE)
    rw = run_arm('REROUTE_WAKE', 20, 0, g / 'E3_rw', DRY_SCALE, enabled=True)
    out['E3'] = {'passes': rw['routes_changed_total'] > 0 and rw['library_sha256'] != base['library_sha256']
                 and rw['anchor_abs_error'] <= o2.ANCHOR_TOLERANCE,
                 'routes_changed': rw['routes_changed_total'], 'anchor_abs_error': rw['anchor_abs_error']}
    log_line(g / 'gates.log', f"E3 {out['E3']}")
    out['all_pass'] = all(v['passes'] for v in out.values() if isinstance(v, dict))
    out['protocol_sha256'] = fingerprint(protocol())
    atomic_json(g / 'gates.json', out)
    return out


# ------------------------------------------------------------------ pool

def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    eta = (elapsed / done * (total - done)) if done else None
    atomic_json(Path(root) / 'status.json', {'state': state, 'cells_done': done, 'cells_total': total,
                                             'running': running, 'pid': os.getpid(), 'started_utc': started,
                                             'updated_utc': now(), 'eta_seconds': eta})


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, require_gates=True,
        worlds=WORLDS):
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale}
    sha = fingerprint(expected)
    manifest = {'protocol': expected, 'protocol_sha256': sha}
    started = time.time()
    with writer_lock(root / 'launcher.lock'):
        if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
            raise ValueError('protocol mismatch; retire the path rather than overwrite it')
        if require_gates:
            g = json.loads((ROOT / 'gates' / 'gates.json').read_text())
            if not g.get('all_pass') or g.get('protocol_sha256') != fingerprint(protocol()):
                raise RuntimeError('gates have not passed at this exact protocol')
        atomic_json(root / 'manifest.json', manifest)
        atomic_json(root / 'run.pid', {'pid': os.getpid(), 'started_utc': now()})
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
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, s, str(root), scale): (a, w, s) for a, w, s in todo}
                _status(root, 'running', len(records), total, [f'{a}_w{w}_s{s}' for a, w, s in todo[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, w, s = futures.pop(done_future)
                    key = f'{a}_w{w}_s{s}'
                    record = done_future.result()
                    validate(record)   # at every scale, so the dry run exercises it
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f"cell finished {key} {record['seconds']:.1f}s "
                                               f"terminal={record['terminal_median']:.6g} "
                                               f"changed={record['routes_changed_total']} stale={record['terminal_stale']} "
                                               f"reroute={record['reroute_seconds']:.1f}s")
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
    parser.add_argument('--gates', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.gates:
        print(json.dumps(gates(), indent=1))
        return
    if args.dry_run:
        dry = [('REROUTE_WAKE', 27, 0), ('REROUTE_INTERLEAVED', 27, 0)]   # held-back world 27, one cell per arm
        run(dry, root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE, stop_after=args.stop_after,
            require_gates=False, worlds=(27,))
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'O8 report: {OUTPUT}')


if __name__ == '__main__':
    main()
