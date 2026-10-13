"""G0: generator census for the generality program (`G0_GENERATOR_CENSUS_PLAN.md`, Tier 0, no lifetimes).

For each candidate generator (alpha, gain, K, shared) of `row.general_world` g1 and each development world, measures:
- S1 LINEAR PREDICTABILITY (B1-mid's statistic, held out): a random unit hyperplane on the state after k teacher
  steps (random program); accuracy of the best least-squares linear separator on the INPUT, fit on 2,048 inputs and
  scored on 2,048 others; 24 hyperplanes per world, k = 1 (primary, B1-mid's case) and k = 3.
- S2 ROUTE IDENTIFIABILITY ON THE TEACHER LIBRARY: for 32 canonical depth-3 tasks (support = the task's 128 training
  examples), the support NMSE of the best WRONG route among all K^3 teacher routes (`runner-up NMSE`); the true route
  has NMSE 0. Small = some wrong program nearly fits (inference hard); large = trivially identifiable.
- S3 DEPTH-1 CLUSTERING (N1's statistic): ARI between teacher labels and support-argmin slot labels on an UNTRAINED
  learner library (slots 2K, model seed 5000), 60 length-1 tasks; and the mean per-position ARI on the 64 canonical
  depth-3 tasks.
- S4 SCALE: per-task output variance of the 64 canonical tasks (median, max/min) and the state-norm ratio
  ||P5...P1 x|| / ||x|| over 64 random five-step programs (median, max); everything finite.
- S5 COST: seconds for 200 learner updates (batch of two tasks) at slots 2K, d = 16.

Teacher programs enter only these measurements; nothing is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig
from row.general_world import VERSION, general_library, generate_general_curriculum_world, generate_general_world
from row.world import Program

PLAN = Path('G0_GENERATOR_CENSUS_PLAN.md')
ROOT = Path('artifacts/g0_generator_census')
OUTPUT = Path('reports/g0_generator_census.json')
BASELINE_OUTPUT = Path('reports/g0_baseline_premeasure.json')

WORLDS = (70, 71, 72)
NONLINEARITY = ((0.35, 1.0), (0.7, 1.0), (1.0, 1.0), (1.5, 1.0), (0.35, 2.0), (0.35, 4.0), (0.7, 2.0), (1.0, 2.0),
                (0.7, 4.0))
PRIMITIVES = (6, 8, 12)
SHARED = (2.0, 4.0, 8.0, 16.0)
SHARED_NONLINEARITY = ((0.35, 1.0), (1.0, 1.0), (0.7, 2.0))
BASE = (0.35, 1.0, 6, 0.0)
HYPERPLANES, FIT_N, SCORE_N, S1_STEPS = 24, 2048, 2048, (1, 3)
S2_TASKS = 32
S3_ANCHORS, MODEL_SEED = 60, 5000
S4_PROGRAMS, S4_STEPS = 64, 5
S5_UPDATES = 200
SEED = 7101
JOBS = 3


def candidates():
    grid = [(a, g, k, 0.0) for k in PRIMITIVES for a, g in NONLINEARITY]
    return grid + [(a, g, 6, m) for m in SHARED for a, g in SHARED_NONLINEARITY]


def key_of(c):
    a, g, k, m = c
    return f'a{a}_g{g}_K{k}_m{m}'


def world_cfg(c, w):
    a, g, k, m = c
    return replace(load_config('configs/v1.yaml').world, seed=w, alpha=a, teacher_primitives=k)


def nmse_np(pred, y):
    return float(np.mean((pred - y) ** 2) / np.mean((y - y.mean(axis=0)) ** 2))


def s1_linear(lib, k_ops, d, w):
    out = {}
    for steps in S1_STEPS:
        accs, balance = [], []
        for i in range(HYPERPLANES):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, w, steps, i]))
            prog = Program(tuple(int(o) for o in rng.integers(0, k_ops, size=steps)))
            wv = rng.normal(size=d)
            wv /= np.linalg.norm(wv)
            xf, xs = rng.normal(size=(FIT_N, d)), rng.normal(size=(SCORE_N, d))
            lf, ls = prog.execute(lib, xf) @ wv > 0, prog.execute(lib, xs) @ wv > 0
            coef = np.linalg.lstsq(np.c_[xf, np.ones(FIT_N)], 2.0 * lf - 1.0, rcond=None)[0]
            accs.append(float(np.mean((np.c_[xs, np.ones(SCORE_N)] @ coef > 0) == ls)))
            balance.append(float(min(ls.mean(), 1 - ls.mean())))
        out[str(steps)] = {'accuracy': accs, 'minority_fraction': balance}
    return out


def s2_runner_up(world, lib):
    k_ops = len(lib)
    rows = []
    for t in world.tasks[:S2_TASKS]:
        x, y = t.train_x, t.train_y
        states = {(): x}
        for _ in range(3):
            states = {r + (o,): lib[o](z) for r, z in states.items() for o in range(k_ops)}
        true = t.program.primitive_ids
        errs = {r: nmse_np(z, y) for r, z in states.items()}
        if errs[true] > 1e-20:
            raise ValueError('true route does not reproduce its task')
        wrong = sorted(v for r, v in errs.items() if r != true)
        rows.append({'runner_up_nmse': wrong[0], 'second_wrong_nmse': wrong[1],
                     'wrong_below_0.05': int(sum(v < 0.05 for v in wrong))})
    return rows


def learner(k_ops, d, steps=3):
    from row.models.learned_models import FastRotatedDiscreteLibraryLearner
    sel = load_config('configs/v1.yaml').discrete_model
    return FastRotatedDiscreteLibraryLearner(
        d=d, operator_slots=2 * k_ops, operator_rank=sel.operator_rank, task_steps=steps,
        alpha=sel.operator_alpha_init, initial_temperature=sel.initial_temperature,
        final_temperature=sel.final_temperature, seed=MODEL_SEED, learnable_alpha=sel.learnable_alpha,
        activation=sel.operator_activation)


def s3_clustering(c, w, world3):
    from row.experiments.audit_h47_baselines import ari_nmi
    from row.experiments.l0d_depth4_execution_gate import DepthLibrary
    a, g, k_ops, m = c
    cfg = world_cfg(c, w)
    lib = DepthLibrary(learner(k_ops, cfg.state_dim))
    world1 = generate_general_curriculum_world(
        CurriculumWorldConfig.from_world(cfg, program_length=1, tasks=S3_ANCHORS), gain=g, shared=m)

    def routes(tasks, depth):
        out = []
        for t in tasks:
            x = torch.tensor(t.train_x, dtype=torch.float32)
            y = torch.tensor(t.train_y, dtype=torch.float32)
            with torch.no_grad():
                mse = lib.all_route_support_mse_depth(x, y, depth)
            if not bool(torch.isfinite(mse).all()):
                raise ValueError('nonfinite support losses')
            flat, route = int(torch.argmin(mse)), []
            for _ in range(depth):
                route.append(flat % lib.slots)
                flat //= lib.slots
            out.append(route[::-1])
        return out

    r1 = routes(world1.tasks, 1)
    ari1, _ = ari_nmi([t.program.primitive_ids[0] for t in world1.tasks], [r[0] for r in r1])
    r3 = routes(world3.tasks, 3)
    ari3 = [ari_nmi([t.program.primitive_ids[p] for t in world3.tasks], [r[p] for r in r3])[0] for p in range(3)]
    return {'ari_depth1': float(ari1), 'ari_depth3_mean': float(np.mean(ari3)),
            'slots_used_depth1': len({r[0] for r in r1})}


def s4_scale(world, lib, d, w):
    var = [float(np.var(t.train_y)) for t in world.tasks]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 4, w]))
    ratios = []
    for _ in range(S4_PROGRAMS):
        prog = Program(tuple(int(o) for o in rng.integers(0, len(lib), size=S4_STEPS)))
        x = rng.normal(size=(256, d))
        ratios.append(float(np.median(np.linalg.norm(prog.execute(lib, x), axis=1) / np.linalg.norm(x, axis=1))))
    finite = all(np.isfinite(t.train_y).all() and np.isfinite(t.eval_y).all() for t in world.tasks)
    return {'output_variance_median': float(np.median(var)), 'output_variance_max_over_min': max(var) / min(var),
            'norm_ratio_median': float(np.median(ratios)), 'norm_ratio_max': float(max(ratios)), 'finite': finite}


def s5_cost(k_ops, d=16):
    torch.manual_seed(0)
    model = learner(k_ops, d)
    ids = [f't{i}' for i in range(2)]
    codes = [model.begin_task(i) for i in ids]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # begin_task registers the codes as parameters
    x, y = torch.randn(2, d), torch.randn(2, d)
    start = time.perf_counter()
    for _ in range(S5_UPDATES):
        opt.zero_grad()
        loss = torch.mean((model.forward_tasks(x, ids) - y) ** 2)
        loss.backward()
        opt.step()
    return time.perf_counter() - start


def _log(root, line):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'run.log', 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")


def measure_cached(job):
    """One (candidate, world) record, written durably and reused on relaunch only under the same protocol."""
    c, w, root, stamp = job
    path = root / 'cells' / f'{key_of(c)}_w{w}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('stamp') == stamp and saved.get('complete'):
            return saved['record']
    rec = measure((c, w))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'stamp': stamp, 'complete': True, 'record': rec}), encoding='utf-8')
    tmp.replace(path)
    _log(root, f"cell {key_of(c)} w{w} {rec['seconds']:.1f}s s1={np.median(rec['s1']['1']['accuracy']):.3f} "
               f"s2={np.median([t['runner_up_nmse'] for t in rec['s2']]):.3f} ari1={rec['s3']['ari_depth1']:.3f}")
    return rec


def measure(job):
    c, w = job
    torch.set_num_threads(1)
    a, g, k_ops, m = c
    cfg = world_cfg(c, w)
    lib = general_library(cfg, gain=g, shared=m)
    world3 = generate_general_world(cfg, gain=g, shared=m)
    started = time.perf_counter()
    rec = {'candidate': key_of(c), 'alpha': a, 'gain': g, 'K': k_ops, 'shared': m, 'world': w,
           's1': s1_linear(lib, k_ops, cfg.state_dim, w), 's2': s2_runner_up(world3, lib),
           's3': s3_clustering(c, w, world3), 's4': s4_scale(world3, lib, cfg.state_dim, w)}
    rec['seconds'] = time.perf_counter() - started
    return rec


def summarize(records):
    out = {}
    for c in sorted({r['candidate'] for r in records}):
        rs = [r for r in records if r['candidate'] == c]
        out[c] = {
            'alpha': rs[0]['alpha'], 'gain': rs[0]['gain'], 'K': rs[0]['K'], 'shared': rs[0]['shared'], 'worlds': [r['world'] for r in rs],
            's1_k1_median': float(np.median([v for r in rs for v in r['s1']['1']['accuracy']])),
            's1_k3_median': float(np.median([v for r in rs for v in r['s1']['3']['accuracy']])),
            's1_minority_median': float(np.median([v for r in rs for v in r['s1']['1']['minority_fraction']])),
            's2_runner_up_median': float(np.median([t['runner_up_nmse'] for r in rs for t in r['s2']])),
            's2_frac_runner_up_below_0.05': float(np.mean([t['runner_up_nmse'] < 0.05 for r in rs for t in r['s2']])),
            's3_ari_depth1': [r['s3']['ari_depth1'] for r in rs],
            's3_ari_depth3': [r['s3']['ari_depth3_mean'] for r in rs],
            's4_output_variance_median': float(np.median([r['s4']['output_variance_median'] for r in rs])),
            's4_variance_max_over_min': float(max(r['s4']['output_variance_max_over_min'] for r in rs)),
            's4_norm_ratio_median': float(np.median([r['s4']['norm_ratio_median'] for r in rs])),
            's4_norm_ratio_max': float(max(r['s4']['norm_ratio_max'] for r in rs)),
            's4_finite': all(r['s4']['finite'] for r in rs)}
    return out


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol(baseline_only):
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    inputs = [Path('configs/v1.yaml'), Path('src/row/general_world.py'), Path(__file__)]
    if not baseline_only:
        inputs.insert(0, PLAN)
    return {'id': 'g0-generator-census-v1', 'generator_version': VERSION, 'git_commit': commit,
            'worlds': list(WORLDS), 'nonlinearity': [list(x) for x in NONLINEARITY], 'primitives': list(PRIMITIVES),
            'shared': list(SHARED), 'shared_nonlinearity': [list(x) for x in SHARED_NONLINEARITY],
            'hyperplanes': HYPERPLANES, 'fit_n': FIT_N, 'score_n': SCORE_N, 's2_tasks': S2_TASKS,
            's3_anchors': S3_ANCHORS, 'model_seed': MODEL_SEED, 'seed': SEED,
            'input_sha256': {p.as_posix(): _sha(p) for p in inputs}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-only', action='store_true',
                    help='measure the current generator only (threshold baselines, before the plan is frozen)')
    ap.add_argument('--jobs', type=int, default=JOBS)
    args = ap.parse_args()
    cands = [BASE] if args.baseline_only else candidates()
    proto = protocol(args.baseline_only)
    stamp = hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()
    root = ROOT / ('baseline' if args.baseline_only else 'full')
    _log(root, f'LAUNCH {stamp[:12]} commit={proto["git_commit"]} jobs={len(cands) * len(WORLDS)} pool={args.jobs}')
    jobs = [(c, w, root, stamp) for c in cands for w in WORLDS]
    # Largest K first so the tail is short.
    jobs.sort(key=lambda j: -j[0][2])
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        records = list(ex.map(measure_cached, jobs))
    cost = {str(k): s5_cost(k) for k in sorted({c[2] for c in cands})}
    report = {'protocol': proto, 'protocol_sha256': stamp, 'records': records, 'summary': summarize(records),
              's5_seconds_per_200_updates': cost,
              'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    out = BASELINE_OUTPUT if args.baseline_only else OUTPUT
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=1), encoding='utf-8')
    tmp.replace(out)
    _log(root, f'EXIT 0 records={len(records)}')
    print(json.dumps({'summary': report['summary'], 's5': cost}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
