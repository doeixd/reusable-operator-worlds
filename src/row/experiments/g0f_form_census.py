"""G0F: operator-FORM census for decision 20 (a) (`G0F_FORM_CENSUS_PLAN.md`, Tier 0, nothing trained except S6).

For each candidate operator form (`row.form_world` f1: `tanh`, `gate`, `fold` at several strengths; plus the current
rotated form as the anchor) and each development world 70-72, measures with G0's own functions where they apply:
- S1 linear predictability of a computed state from the input (`g0.s1_linear`, verbatim);
- S2 teacher runner-up NMSE among all K^3 routes (`g0.s2_runner_up`, verbatim; reported, not registered here);
- S3 depth-1 clustering on an UNTRAINED SAME-FORM library: 2K slots drawn from the candidate's own form at the same
  strength with seed `MODEL_SEED`, support-argmin over slots, ARI against teacher labels for 60 length-1 tasks, and the
  mean per-position ARI over all (2K)^3 routes for the 64 canonical depth-3 tasks;
- S4 scale (`g0.s4_scale`, verbatim);
- S6 (reported) single-operator learnability: a same-form operator with free parameters, initialized from untrained
  slot 0, fitted by Adam to each teacher primitive (128 inputs), held-out NMSE on 1,024 inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from row.experiments import g0_generator_census as g0
from row.form_world import VERSION, form_library, generate_form_curriculum_world, generate_form_world
from row.general_world import general_library, generate_general_curriculum_world, generate_general_world

PLAN = Path('G0F_FORM_CENSUS_PLAN.md')
ROOT = Path('artifacts/g0f_form_census')
OUTPUT = Path('reports/g0f_form_census.json')
BASELINE_OUTPUT = Path('reports/g0f_baseline_premeasure.json')

WORLDS = (70, 71, 72)
K = 6
BASE = ('rot', 0.35)
CANDIDATES = (BASE, ('tanh', 0.5), ('tanh', 1.0), ('tanh', 2.0), ('tanh', 4.0), ('gate', 1.0), ('gate', 2.0),
              ('gate', 4.0), ('fold', 1.0), ('fold', 2.0), ('fold', 4.0))
MODEL_SEED = g0.MODEL_SEED
S3_ANCHORS = g0.S3_ANCHORS
S6_FIT_N, S6_EVAL_N, S6_STEPS, S6_LR = 128, 1024, 2000, 0.01
SEED = 7201
JOBS = 3


def key_of(c):
    return f'{c[0]}{c[1]:g}'


def world_cfg(w):
    return replace(load_config('configs/v1.yaml').world, seed=w, teacher_primitives=K)


def teacher(c, cfg):
    """(library, canonical depth-3 world, 60-task length-1 world)."""
    form, knob = c
    c1 = CurriculumWorldConfig.from_world(cfg, program_length=1, tasks=S3_ANCHORS)
    if form == 'rot':
        return general_library(cfg), generate_general_world(cfg), generate_general_curriculum_world(c1)
    return (form_library(form, knob, cfg.seed, K, cfg.state_dim), generate_form_world(cfg, form, knob),
            generate_form_curriculum_world(c1, form, knob))


def untrained(c, d):
    form, knob = c
    if form == 'rot':
        cfg = replace(load_config('configs/v1.yaml').world, seed=MODEL_SEED, teacher_primitives=2 * K)
        return general_library(cfg)
    return form_library(form, knob, MODEL_SEED, 2 * K, d)


def _argmin_routes(lib, x, y, depth):
    states = {(): x}
    for _ in range(depth):
        states = {r + (o,): lib[o](z) for r, z in states.items() for o in range(len(lib))}
    errs = {r: float(np.mean((z - y) ** 2)) for r, z in states.items()}
    if not all(np.isfinite(v) for v in errs.values()):
        raise ValueError('nonfinite support loss')
    return min(sorted(errs), key=lambda r: errs[r])


def s3_same_form(slots, world1, world3):
    from row.experiments.audit_h47_baselines import ari_nmi
    r1 = [_argmin_routes(slots, t.train_x, t.train_y, 1) for t in world1.tasks]
    ari1, _ = ari_nmi([t.program.primitive_ids[0] for t in world1.tasks], [r[0] for r in r1])
    r3 = [_argmin_routes(slots, t.train_x, t.train_y, 3) for t in world3.tasks]
    ari3 = [ari_nmi([t.program.primitive_ids[p] for t in world3.tasks], [r[p] for r in r3])[0] for p in range(3)]
    return {'ari_depth1': float(ari1), 'ari_depth3_mean': float(np.mean(ari3)),
            'slots_used_depth1': len({r[0] for r in r1})}


class FormOperator(torch.nn.Module):
    """A same-form operator with free parameters (orthogonal Q = Q0 expm(S - S^T)), initialized from a numpy draw."""

    def __init__(self, p):
        super().__init__()
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
        self.form = p.form if hasattr(p, 'form') else 'rot'
        self.Q0 = t(p.Q)
        self.S = torch.nn.Parameter(torch.zeros_like(self.Q0))
        if self.form == 'rot':
            self.U, self.V, self.b = (torch.nn.Parameter(t(a)) for a in (p.U, p.V, p.b))
            self.a = torch.nn.Parameter(t(p.alpha))
        elif self.form == 'tanh':
            self.A, self.b = torch.nn.Parameter(t(p.knob * p.A)), torch.nn.Parameter(t(p.b))
            self.s = torch.nn.Parameter(t(p.scale))
        elif self.form == 'gate':
            self.A, self.b = torch.nn.Parameter(t(p.knob * p.A)), torch.nn.Parameter(t(p.knob * p.b))
        else:
            self.A = torch.nn.Parameter(t(p.A))

    def forward(self, z):
        Q = self.Q0 @ torch.linalg.matrix_exp(self.S - self.S.T)
        if self.form == 'rot':
            return (z + self.a * torch.tanh(z @ self.V.T + self.b) @ self.U.T) @ Q.T
        if self.form == 'tanh':
            return self.s * torch.tanh(z @ self.A.T + self.b) @ Q.T
        if self.form == 'gate':
            return (z * 2.0 * torch.sigmoid(z @ self.A.T + self.b)) @ Q.T
        for w in self.A:
            w = w / torch.linalg.norm(w)
            z = z - 2.0 * torch.clamp(z @ w, max=0.0)[:, None] * w[None, :]
        return z @ Q.T


def s6_learnability(lib, slot0, w, d):
    rows = []
    for k, prim in enumerate(lib):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 6, w, k]))
        xf, xe = rng.normal(size=(S6_FIT_N, d)), rng.normal(size=(S6_EVAL_N, d))
        yf, ye = prim(xf), prim(xe)
        op = FormOperator(slot0)
        opt = torch.optim.Adam(op.parameters(), lr=S6_LR)
        X, Y = torch.tensor(xf), torch.tensor(yf)
        for _ in range(S6_STEPS):
            opt.zero_grad()
            loss = torch.mean((op(X) - Y) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = op(torch.tensor(xe)).numpy()
        rows.append({'primitive': k, 'heldout_nmse': g0.nmse_np(pred, ye), 'fit_nmse': g0.nmse_np(
            op(X).detach().numpy(), yf)})
    return rows


def measure(job):
    c, w = job
    torch.set_num_threads(1)
    cfg = world_cfg(w)
    started = time.perf_counter()
    lib, world3, world1 = teacher(c, cfg)
    slots = untrained(c, cfg.state_dim)
    rec = {'candidate': key_of(c), 'form': c[0], 'knob': c[1], 'K': K, 'world': w,
           's1': g0.s1_linear(lib, K, cfg.state_dim, w), 's2': g0.s2_runner_up(world3, lib),
           's3': s3_same_form(slots, world1, world3), 's4': g0.s4_scale(world3, lib, cfg.state_dim, w),
           's6': s6_learnability(lib, slots[0], w, cfg.state_dim)}
    rec['seconds'] = time.perf_counter() - started
    return rec


def _log(root, line):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'run.log', 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")
        f.flush()


def _atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    tmp.replace(path)


def measure_cached(job):
    c, w, root, stamp = job
    path = root / 'cells' / f'{key_of(c)}_w{w}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('stamp') == stamp and saved.get('complete'):
            return saved['record']
    _log(root, f'start {key_of(c)} w{w}')
    rec = measure((c, w))
    _atomic(path, {'stamp': stamp, 'complete': True, 'record': rec})
    _log(root, f"cell {key_of(c)} w{w} {rec['seconds']:.1f}s s1={np.median(rec['s1']['1']['accuracy']):.3f} "
               f"s2={np.median([t['runner_up_nmse'] for t in rec['s2']]):.3f} ari1={rec['s3']['ari_depth1']:.3f} "
               f"s6={np.median([r['heldout_nmse'] for r in rec['s6']]):.4f}")
    return rec


def summarize(records):
    out = {}
    for key in sorted({r['candidate'] for r in records}):
        rs = [r for r in records if r['candidate'] == key]
        out[key] = {
            'form': rs[0]['form'], 'knob': rs[0]['knob'], 'worlds': [r['world'] for r in rs],
            's1_k1_median': float(np.median([v for r in rs for v in r['s1']['1']['accuracy']])),
            's1_k3_median': float(np.median([v for r in rs for v in r['s1']['3']['accuracy']])),
            's1_minority_median': float(np.median([v for r in rs for v in r['s1']['1']['minority_fraction']])),
            's2_runner_up_median': float(np.median([t['runner_up_nmse'] for r in rs for t in r['s2']])),
            's3_ari_depth1': [r['s3']['ari_depth1'] for r in rs],
            's3_ari_depth3': [r['s3']['ari_depth3_mean'] for r in rs],
            's4_output_variance_median': float(np.median([r['s4']['output_variance_median'] for r in rs])),
            's4_variance_max_over_min': float(max(r['s4']['output_variance_max_over_min'] for r in rs)),
            's4_norm_ratio_max': float(max(r['s4']['norm_ratio_max'] for r in rs)),
            's4_finite': all(r['s4']['finite'] for r in rs),
            's6_heldout_median': float(np.median([x['heldout_nmse'] for r in rs for x in r['s6']])),
            's6_heldout_max': float(max(x['heldout_nmse'] for r in rs for x in r['s6']))}
    return out


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol(baseline_only):
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    inputs = [Path('configs/v1.yaml'), Path('src/row/form_world.py'), Path('src/row/general_world.py'),
              Path('src/row/experiments/g0_generator_census.py'), Path(__file__), g0.BASELINE_OUTPUT]
    if not baseline_only:
        inputs[:0] = [PLAN, BASELINE_OUTPUT]
    return {'id': 'g0f-form-census-v1', 'form_version': VERSION, 'git_commit': commit, 'worlds': list(WORLDS),
            'K': K, 'candidates': [list(c) for c in CANDIDATES], 'model_seed': MODEL_SEED, 'seed': SEED,
            's6': [S6_FIT_N, S6_EVAL_N, S6_STEPS, S6_LR], 'input_sha256': {p.as_posix(): _sha(p) for p in inputs}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-only', action='store_true', help='the rotated anchor only (before freezing)')
    ap.add_argument('--jobs', type=int, default=JOBS)
    args = ap.parse_args()
    cands = [BASE] if args.baseline_only else list(CANDIDATES)
    proto = protocol(args.baseline_only)
    stamp = hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()
    root = ROOT / ('baseline' if args.baseline_only else 'full')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'run.pid').write_text(str(os.getpid()))
    jobs = [(c, w, root, stamp) for c in cands for w in WORLDS]
    _log(root, f'LAUNCH {stamp[:12]} commit={proto["git_commit"]} jobs={len(jobs)} pool={args.jobs}')
    started = time.time()
    records = []
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for rec in ex.map(measure_cached, jobs):
                records.append(rec)
                el = time.time() - started
                _atomic(root / 'status.json', {
                    'commit': proto['git_commit'], 'pid': os.getpid(), 'started_unix': started, 'done': len(records),
                    'total': len(jobs), 'last_update_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'eta_seconds': el / len(records) * (len(jobs) - len(records))})
        report = {'protocol': proto, 'protocol_sha256': stamp, 'records': records, 'summary': summarize(records),
                  'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        _atomic(BASELINE_OUTPUT if args.baseline_only else OUTPUT, report)
    except Exception as exc:
        _log(root, f'ERROR {exc!r}')
        _atomic(root / 'exit.json', {'exit_code': 1, 'error': repr(exc), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        raise
    _log(root, f'EXIT 0 records={len(records)}')
    _atomic(root / 'exit.json', {'exit_code': 0, 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(json.dumps(report['summary'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
