"""G1T (Tier 1, EXPLORATORY): the confirmed online pipeline with teacher AND learner of the `tanh` form G0F selected.

Plan: `G1T_TANH_FORMATION_PLAN.md`. Development band 6, worlds 70-72 (G0's, G0F's and G1's worlds), stream 0.
Teacher: `row.form_world` f1, form `tanh`, gain 2 (`c_2 Q tanh(2 R z + b)`). Learner: `row.models.tanh_form`.
- REROUTE_WAKE: `o8.run_arm` verbatim; RW_SLEEP: `o3.run_sleep` verbatim on that terminal (the confirmed pipeline,
  O11), then rebuilt and measured exactly as G1 does (`g1.formed_diagnostics`), plus the formed vocabulary's S1;
- SHUFFLED: `o2.run_single('SHUFFLED')` verbatim (wake alone); SLEEP: `o3.run_sleep` verbatim on that terminal (sleep
  alone). With REROUTE_WAKE and RW_SLEEP these are O11's four mechanism arms (reviewer 86, priority 1);
- PLAIN: `o2.run_staged('PLAIN')` verbatim (no anchors); STAGED: `o2.run_staged('STAGED')` verbatim (J1c's length
  curriculum, library carried between stages).

`use_form()` rebinds names only, as G1 did; no committed module is edited:
  `o2.stage_setup` -> the form world's stages; `o2.planned_model` and `so2.build_fast` -> the tanh learner;
  `o2.lifetime` -> the same call with the tanh learner where it would have built the default kind (first stage);
  `FrozenLibrary` in `o8`, `census` and `g1` -> `FrozenTanhLibrary`.
`learned_lifetime.run` still receives the kind label `rotated_discrete_fast`; it selects the `discrete_model`
hyperparameters (learning rates, weight decay, replay) and labels the lifetime's own fingerprint and compute
accounting, which therefore describe the rotated kind. The learner actually trained is recorded in every cell.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil
import torch

from row import form_world
from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig
from row.experiments import audit_j1c_curriculum as j1c
from row.experiments import audit_j2a_staged_library as j2a
from row.experiments import audit_so2_online_gate as so2
from row.experiments import census_o8_staleness_position as census
from row.experiments import g0_generator_census as g0
from row.experiments import g1_formation_probe as g1
from row.experiments import learned_lifetime as ll
from row.experiments import o2_online_reliability as o2
from row.experiments import o2c_consolidation as o2c
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o8_instream_reroute as o8
from row.experiments.audit_rotated_g5r_diagnosis import require_clean_code
from row.experiments.audit_rotated_g5r_interference import score, world_config
from row.experiments.audit_so1r_route_only import FrozenLibrary
from row.experiments.so1_storage import atomic_json, digest, fingerprint, log_line, now, writer_lock
from row.models import tanh_form
from row.models.tanh_form import FrozenTanhLibrary, TanhFormLearner, build_tanh_learner
from types import SimpleNamespace

PLAN = Path('G1T_TANH_FORMATION_PLAN.md')
G0F_REPORT = Path('reports/g0f_form_census.json')
G1_REPORT = Path('reports/g1_formation_probe.json')
ROOT = Path('artifacts/g1t_tanh_formation')
DRY_ROOT = Path('artifacts/g1t_tanh_formation_dry')
OUTPUT = Path('reports/g1t_tanh_formation.json')
FORM, KNOB, CANDIDATE = 'tanh', 2.0, 'tanh2'
WORLDS = (70, 71, 72)
STREAM = 0
ARMS = ('REROUTE_WAKE', 'SHUFFLED', 'STAGED', 'PLAIN', 'RW_SLEEP', 'SLEEP')
DEPENDS = {'RW_SLEEP': 'REROUTE_WAKE', 'SLEEP': 'SHUFFLED'}   # submitted when the wake terminal exists
THRESHOLD = 0.05
IDENTITY_ARI = 0.8   # formed rotated vocabularies 0.97-1.00 (G1); untrained tanh2 libraries 0.42-0.47 (G0F S3)
JOBS = 3
MIN_FREE_GIB = 2.0
DRY_WORLD = 99
DRY_SCALE = o3.DRY_SCALE

_ORIGINAL = {}


# ------------------------------------------------------------------ form plumbing

def form_stage_setup(world_seed: int, stage: int, model_seed: int = 5000):
    """`audit_j1c_curriculum.stage_setup` with the f1 `tanh` generators at gain 2 (G1's `general_stage_setup` shape)."""
    base = load_config('configs/v1.yaml')
    length, tasks, updates, root, tail = j1c.STAGES[stage - 1]
    if stage == 3:
        cfg = world_config(base, world_seed)
        world = form_world.generate_form_world(cfg.world, FORM, KNOB)
    else:
        cfg = replace(base, world=CurriculumWorldConfig.from_world(
            replace(base.world, seed=world_seed), program_length=length, tasks=tasks, stage=f'length-{length}'))
        world = form_world.generate_form_curriculum_world(cfg.world, FORM, KNOB)
    cfg = replace(cfg, discrete_model=replace(cfg.discrete_model, task_steps=length, seed=model_seed))
    return cfg, world, updates, np.random.SeedSequence([root, world_seed, tail])


def tanh_planned_model(cfg, plan):
    return build_tanh_learner(cfg, plan=plan, task_steps=3)


def tanh_stage_model(cfg):
    return build_tanh_learner(cfg)


def tanh_lifetime(cfg, world, output, model, replay_seed, scale):
    if model is None:
        model = tanh_stage_model(cfg)
    return _ORIGINAL['lifetime'](cfg, world, output, model, replay_seed, scale)


def use_form() -> None:
    if not _ORIGINAL:
        _ORIGINAL.update(lifetime=o2.lifetime, stage_setup=o2.stage_setup, planned_model=o2.planned_model,
                         build_fast=so2.build_fast, o8=o8.FrozenLibrary, census=census.FrozenLibrary,
                         g1=g1.FrozenLibrary)
    o2.stage_setup = form_stage_setup
    o2.planned_model = tanh_planned_model
    o2.lifetime = tanh_lifetime
    so2.build_fast = tanh_stage_model
    o8.FrozenLibrary = census.FrozenLibrary = g1.FrozenLibrary = FrozenTanhLibrary


def restore() -> None:
    if _ORIGINAL:
        o2.lifetime, o2.stage_setup, o2.planned_model = (_ORIGINAL['lifetime'], _ORIGINAL['stage_setup'],
                                                         _ORIGINAL['planned_model'])
        so2.build_fast = _ORIGINAL['build_fast']
        o8.FrozenLibrary, census.FrozenLibrary, g1.FrozenLibrary = (_ORIGINAL['o8'], _ORIGINAL['census'],
                                                                    _ORIGINAL['g1'])


def key_of(arm, w):
    return f'{arm}_w{w}'


def cells(worlds=WORLDS):
    return [(a, w) for a in ARMS for w in worlds]


def work_of(root, arm, w):
    return Path(root) / 'work' / key_of(arm, w)


# ------------------------------------------------------------------ formed-vocabulary S1

def slot_callables(library, slots):
    def make(s):
        def apply(z):
            with torch.no_grad():
                out = library.candidates(torch.tensor(np.asarray(z), dtype=torch.float32))[:, s]
            return out.numpy().astype(np.float64)
        return apply
    return tuple(make(s) for s in slots)


def formed_s1(model, stream_tasks, plan, w, d=16, k_ops=6):
    """G0's `s1_linear`, verbatim draws, on the FORMED library: teacher primitive p is replaced by the slot most of
    p's length-1 tasks route to. Paired with the teacher S1 of the same world (same hyperplanes, programs, inputs)."""
    routes = model.hard_routes()
    votes = {p: Counter() for p in range(k_ops)}
    for t in stream_tasks:
        if plan[t.task_id] == 1:
            votes[t.program.primitive_ids[0]][int(routes[t.task_id][0])] += 1
    mapping = [votes[p].most_common(1)[0][0] for p in range(k_ops)]
    out = g0.s1_linear(slot_callables(FrozenTanhLibrary(model), mapping), k_ops, d, w)
    return {'mapping': mapping, 'distinct_slots': len(set(mapping)),
            'median': {k: float(np.median(v['accuracy'])) for k, v in out.items()}, 'records': out}


# ------------------------------------------------------------------ cells

def run_cell(arm, w, root, scale=1):
    torch.set_num_threads(1)
    use_form()
    started = time.perf_counter()
    work = work_of(root, arm, w)
    if arm == 'REROUTE_WAKE':
        rec = o8.run_arm(arm, w, STREAM, work, scale)
        rec['d1_ari_wake'] = g1.formed_diagnostics_d1(work / 'lifetime' / 'model.pt', w)
    elif arm == 'SHUFFLED':
        rec = o2.run_single('SHUFFLED', w, STREAM, work, scale)
        rec['d1_ari_wake'] = g1.formed_diagnostics_d1(work / 'lifetime' / 'model.pt', w)
    elif arm == 'SLEEP':
        rec = o3.run_sleep(w, STREAM, work_of(root, 'SHUFFLED', w) / 'lifetime' / 'model.pt')
    elif arm in ('PLAIN', 'STAGED'):
        rec = o2.run_staged(arm, w, STREAM, work, scale)
    elif arm == 'RW_SLEEP':
        terminal = work_of(root, 'REROUTE_WAKE', w) / 'lifetime' / 'model.pt'
        rec = o3.run_sleep(w, STREAM, terminal)   # the registered construction, verbatim
        cfg3, slept, st, canonical = o3.load_shuffled_terminal(terminal, w, STREAM)
        pool = o2d.reservoir(st, w, STREAM, o3.MEMORY)
        o2c.consolidate(cfg3, slept, pool, [t.task_id for t in st], [1941, w, STREAM, o3.MEMORY],
                        updates=o3.EXTRA_UPDATES)
        rec['rebuild_matches'] = score(slept, SimpleNamespace(tasks=canonical))['per_task'] == rec['terminal_per_task']
        cfg3b, world3, _, plan, _ = o2.build_stream('SHUFFLED', w, STREAM)
        rec['formed'] = g1.formed_diagnostics(slept, st, plan, j2a.held_out_tasks(cfg3b, world3))
        rec['formed_s1'] = formed_s1(slept, st, plan, w)
        rec['learner_class'] = type(slept).__name__
    else:
        raise ValueError(arm)
    rec.update({'arm': arm, 'world': w, 'stream': STREAM, 'scale': scale, 'form': FORM, 'knob': KNOB,
                'generator': form_world.VERSION, 'learner_implementation': TanhFormLearner.implementation,
                'seconds': time.perf_counter() - started})
    return rec


def validate(rec):
    key = key_of(rec['arm'], rec['world'])
    if rec['scored_tasks'] != 64 or len(rec['terminal_per_task']) != 64:
        raise ValueError(f'{key}: not scored on the canonical 64')
    if not all(math.isfinite(v) for v in rec['terminal_per_task'].values()):
        raise ValueError(f'{key}: non-finite terminal')
    if rec['arm'] == 'REROUTE_WAKE':
        o8.validate(rec)
    elif rec['arm'] == 'SHUFFLED':
        o2.validate_cell(rec)
    elif rec['arm'] == 'SLEEP':
        if rec['pool_size'] != 188 * o3.MEMORY or rec['extra_updates'] != o3.EXTRA_UPDATES:
            raise ValueError(f'{key}: sleep construction')
        if rec['library_sha256'] == rec['library_sha256_before']:
            raise ValueError(f'{key}: sleep left the library unchanged')
    elif rec['arm'] in ('PLAIN', 'STAGED'):
        want = {'1': 0, '2': 0, '3': 64} if rec['arm'] == 'PLAIN' else {'1': 60, '2': 64, '3': 64}
        if rec['depth_histogram'] != want or not rec['route_lengths_match_plan']:
            raise ValueError(f'{key}: {rec["arm"]} construction')
        if rec['anchor_abs_error'] > o2.ANCHOR_TOLERANCE:
            raise ValueError(f'{key}: last-task anchor {rec["anchor_abs_error"]}')
    else:
        if rec['pool_size'] != 188 * o3.MEMORY or rec['extra_updates'] != o3.EXTRA_UPDATES:
            raise ValueError(f'{key}: sleep construction')
        if rec['library_sha256'] == rec['library_sha256_before']:
            raise ValueError(f'{key}: sleep left the library unchanged')
        if not rec['rebuild_matches']:
            raise ValueError(f'{key}: diagnostic rebuild is not the slept model')
        if rec['learner_class'] != 'TanhFormLearner':
            raise ValueError(f'{key}: learner is {rec["learner_class"]}')
        if rec['formed']['d1_tasks'] != 60 or rec['formed']['identifiability_tasks'] != 64:
            raise ValueError(f'{key}: diagnostic denominators')
        if len(rec['formed_s1']['records']['1']['accuracy']) != g0.HYPERPLANES:
            raise ValueError(f'{key}: formed S1 denominator')


# ------------------------------------------------------------------ reading (exploratory; no verdict)

def passing(v):
    return v is not None and math.isfinite(v) and v < THRESHOLD


def forms(rec):
    """A slept cell FORMS iff its terminal passes AND its length-1 routes recover operation identity."""
    return passing(rec['terminal_median']) and rec['formed']['d1_ari'] >= IDENTITY_ARI


def reading(records, rotated_terminal, worlds=WORLDS):
    """`rotated_terminal[w]`: G1's committed m = 0 RW_SLEEP terminal (rotated teacher and learner, same world, stream,
    task ids, programs, inputs, order and seeds). A world is ANCHORED iff it passed there."""
    anchored = [w for w in worlds if passing(rotated_terminal[w])]
    sleep = {w: records[key_of('RW_SLEEP', w)] for w in worlds}
    k = sum(forms(sleep[w]) for w in anchored)
    impostors = [w for w in anchored if passing(sleep[w]['terminal_median']) and not forms(sleep[w])]
    if not anchored:
        label = 'ANCHOR_MISS'
    elif k == len(anchored):
        label = 'FORMS'
    elif k == 0:
        label = 'FAILS'
    else:
        label = 'MIXED'
    M = {key: v['terminal_median'] for key, v in records.items()}
    return {'label': label, 'anchored_worlds': anchored, 'forming_cells': k, 'denominator': len(anchored),
            'impostor_worlds': impostors,
            'wake_passes': sum(passing(M[key_of('REROUTE_WAKE', w)]) for w in worlds),
            'shuffled_passes': sum(passing(M[key_of('SHUFFLED', w)]) for w in worlds),
            'sleep_passes': sum(passing(M[key_of('SLEEP', w)]) for w in worlds),
            'staged_passes': sum(passing(M[key_of('STAGED', w)]) for w in worlds),
            'plain_passes': sum(passing(M[key_of('PLAIN', w)]) for w in worlds),
            'terminal': {key: M[key] for key in sorted(M)},
            'rotated_terminal': {str(w): rotated_terminal[w] for w in worlds},
            'formed': {str(w): sleep[w]['formed'] for w in worlds},
            'formed_s1_median': {str(w): sleep[w]['formed_s1']['median'] for w in worlds}}


def rotated_reference(worlds=WORLDS):
    cells_ = json.loads(G1_REPORT.read_text())['cells']
    return {w: cells_[f'RW_SLEEP_m0_w{w}']['terminal_median'] for w in worlds}


# ------------------------------------------------------------------ gates and protocol

def _stream_snapshot(w, with_targets):
    cfg3, world3, st, plan, canonical = o2.build_stream('SHUFFLED', w, STREAM)
    rows = [(t.task_id, t.program.primitive_ids, t.train_x.tobytes(), t.eval_x.tobytes())
            + ((t.train_y.tobytes(), t.eval_y.tobytes()) if with_targets else ()) for t in st]
    staged = [[(t.task_id, t.program.primitive_ids, t.train_x.tobytes()) for t in o2.stage_setup(w, s, 5000)[1].tasks]
              for s in (1, 2, 3)]
    return rows, plan, [t.task_id for t in canonical], staged


def gates(worlds=(70, DRY_WORLD)):
    """E1: the form stream equals the committed (rotated) stream in task ids, programs, inputs, plan and stage tasks.
    E2: every target differs (the form reaches the pipeline). F1: the teacher library reproduces G0F's `tanh2` S1
    record for world 70 exactly. L1: the search view reproduces the learner's hard prediction bitwise at depths 1-3.
    L2: every rebound name builds or searches the tanh learner."""
    out = {'E1': {}, 'E2': {}}
    for w in worlds:
        restore()
        ref, ref_y = _stream_snapshot(w, False), _stream_snapshot(w, True)[0]
        use_form()
        new, new_y = _stream_snapshot(w, False), _stream_snapshot(w, True)[0]
        out['E1'][str(w)] = new == ref
        out['E2'][str(w)] = all(a[4] != b[4] and a[5] != b[5] for a, b in zip(ref_y, new_y))
    use_form()
    lib = form_world.form_library(FORM, KNOB, 70, 6, 16)
    want = next(r for r in json.loads(G0F_REPORT.read_text())['records']
                if r['candidate'] == CANDIDATE and r['world'] == 70)['s1']
    out['F1'] = g0.s1_linear(lib, 6, 16, 70) == want
    cfg = load_config('configs/v1.yaml')
    model = tanh_planned_model(cfg, {'d1': 1, 'd2': 2, 'd3': 3})
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    l1 = []
    for t in ('d1', 'd2', 'd3'):
        model.begin_task(t)
        with torch.no_grad():
            model.task_codes[t].normal_()
    model.eval()
    for t in ('d1', 'd2', 'd3'):
        library = o8.FrozenLibrary(model)
        route = model.hard_routes()[t]
        library.steps = len(route)
        l1.append(torch.equal(library.hard(x, route), model(x, t)))
    out['L1'] = all(l1) and len(l1) == 3
    stage_cfg = o2.stage_setup(70, 1, 5000)[0]
    out['L2'] = (isinstance(o2.planned_model(cfg, {}), TanhFormLearner)
                 and isinstance(so2.build_fast(stage_cfg), TanhFormLearner)
                 and so2.build_fast(stage_cfg).task_steps == 1
                 and o8.FrozenLibrary is census.FrozenLibrary is g1.FrozenLibrary is FrozenTanhLibrary
                 and o2.lifetime is tanh_lifetime)
    restore()
    out['all_pass'] = (all(out['E1'].values()) and all(out['E2'].values()) and out['F1'] and out['L1']
                       and out['L2'])
    return out


def protocol(root=ROOT):
    modules = (o8, o3, o2, o2c, o2d, j1c, j2a, census, so2, g0, g1, ll, form_world, tanh_form,
               sys.modules[o2.PlannedDepthRotatedLearner.__module__], sys.modules[FrozenLibrary.__module__])
    return {'id': 'g1t-tanh-formation-v1', 'git_commit': o2.git_commit(), 'plan': PLAN.as_posix(), 'tier': 1,
            'exploratory': True, 'worlds': list(WORLDS), 'form': FORM, 'knob': KNOB, 'stream': STREAM,
            'arms': list(ARMS), 'threshold': THRESHOLD, 'identity_ari': IDENTITY_ARI, 'jobs': JOBS,
            'root': Path(root).as_posix(), 'generator': form_world.VERSION,
            'learner_implementation': TanhFormLearner.implementation, 'init_gain': tanh_form.INIT_GAIN,
            'input_sha256': {p.as_posix(): digest(p) for p in (PLAN, Path('configs/v1.yaml'), G0F_REPORT, G1_REPORT)},
            'implementation_sha256': digest(Path(__file__)),
            'module_sha256': {m.__name__: digest(Path(m.__file__)) for m in modules}}


def _status(root, state, done, total, running, started):
    elapsed = time.time() - started
    eta = elapsed / done * (total - done) if done else None
    atomic_json(Path(root) / 'status.json', {
        'state': state, 'cells_done': done, 'cells_total': total, 'running': running, 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)), 'updated_utc': now(),
        'eta_seconds': eta})


def _headline(rec):
    extra = ''
    if rec['arm'] == 'RW_SLEEP':
        f = rec['formed']
        extra = (f" d1_ari={f['d1_ari']:.3f} ident={f['identifiability_median']:.3g} ties={f['near_tie_tasks']}"
                 f" s1_formed={rec['formed_s1']['median']['1']:.3f}")
    elif rec['arm'] in ('REROUTE_WAKE', 'SHUFFLED'):
        extra = f" d1_ari_wake={rec['d1_ari_wake']:.3f} changed={rec.get('routes_changed_total')}"
    return f"{rec['seconds']:.1f}s terminal={rec['terminal_median']:.6g}{extra}"


def run(todo_cells, root=ROOT, output=OUTPUT, jobs=JOBS, scale=1, stop_after=None, require_gates=True,
        worlds=WORLDS):
    """The parent writes every log line (G0F learning: pool workers appending to one log can lose lines)."""
    root, output = Path(root), Path(output)
    (root / 'cells').mkdir(parents=True, exist_ok=True)
    expected = protocol(root) | {'scale': scale, 'cells': [list(c) for c in todo_cells]}
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
            gpath = root / 'gates.json'
            g = json.loads(gpath.read_text()) if gpath.exists() else None
            if not g or g.get('protocol_sha256') != sha:
                g = gates() | {'protocol_sha256': sha}
                atomic_json(gpath, g)
            log_line(root / 'run.log', f"GATES E1={g['E1']} E2={g['E2']} F1={g['F1']} L1={g['L1']} L2={g['L2']}")
            if require_gates and not g['all_pass']:
                raise RuntimeError(f'gates failed: {g}')
            for arm, w in todo_cells:
                key = key_of(arm, w)
                path = root / 'cells' / f'{key}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get('stamp') != {'protocol_sha256': sha} or not saved.get('complete')
                            or fingerprint(saved['record']) != saved['record_sha256']):
                        raise ValueError(f'cell integrity failure: {key}')
                    records[key] = saved['record']
                    log_line(root / 'run.log', f'reused validated cell {key}')
                else:
                    todo.append((arm, w))
            total = len(todo_cells)
            log_line(root / 'run.log', f'LAUNCH {sha} todo={len(todo)} reused={len(records)} jobs={jobs} '
                                       f'scale={scale} free={free:.1f}GiB')
            ready = [c for c in todo if c[0] not in DEPENDS or key_of(DEPENDS[c[0]], c[1]) in records]
            waiting = [c for c in todo if c not in ready]
            finished = 0
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(run_cell, a, w, str(root), scale): (a, w) for a, w in ready}
                for a, w in ready[:jobs]:
                    log_line(root / 'run.log', f'submitted {key_of(a, w)}')
                _status(root, 'running', len(records), total, [key_of(*c) for c in ready[:jobs]], started)
                while futures:
                    done_future = next(as_completed(futures))
                    a, w = futures.pop(done_future)
                    key = key_of(a, w)
                    record = done_future.result()
                    validate(record)   # at every scale, so the dry run exercises it
                    atomic_json(root / 'cells' / f'{key}.json',
                                {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record,
                                 'record_sha256': fingerprint(record), 'finished_utc': now()})
                    records[key] = record
                    finished += 1
                    log_line(root / 'run.log', f'cell finished {key} {_headline(record)}')
                    for dep in [c for c in waiting if DEPENDS[c[0]] == a and c[1] == w]:
                        waiting.remove(dep)
                        futures[pool.submit(run_cell, dep[0], w, str(root), scale)] = dep
                        log_line(root / 'run.log', f'submitted {key_of(*dep)}')
                    _status(root, 'running', len(records), total, [key_of(*c) for c in futures.values()], started)
                    if stop_after is not None and finished >= stop_after:
                        log_line(root / 'run.log', f'STOP after {finished} cells (restart test)')
                        pool.shutdown(wait=False, cancel_futures=True)
                        os._exit(3)
            if waiting:
                raise RuntimeError(f'dependent cells never submitted: {waiting}')
            report = {**manifest, 'complete': True, 'gates': g, 'cells': records, 'finished_utc': now()}
            if scale == 1:
                report['reading'] = reading(records, rotated_reference(worlds), worlds)
            atomic_json(output, report)
            log_line(root / 'run.log', f"EXIT 0 {report.get('reading', {}).get('label')}")
            _status(root, 'complete', len(records), total, [], started)
            atomic_json(root / 'exit.json', {'exit_code': 0, 'complete': True, 'finished_utc': now()})
        except BaseException:
            log_line(root / 'run.log', 'FAILED ' + traceback.format_exc().replace('\n', ' | '))
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
        run(cells((DRY_WORLD,)), root=DRY_ROOT, output=DRY_ROOT / 'report.json', scale=DRY_SCALE,
            stop_after=args.stop_after, worlds=(DRY_WORLD,))
        print(f'dry run complete: {DRY_ROOT}')
        return
    require_clean_code(OUTPUT)
    run(cells())
    print(f'G1T report: {OUTPUT}')


if __name__ == '__main__':
    main()
