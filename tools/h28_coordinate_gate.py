"""Oracle-only development construction checks for H28-C; no learner training."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
from row.world import Primitive

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'h28-c-conjugacy-fixture-v0'
PROGRAMS = ((0, 1, 2), (5, 3, 1))
TOL = 1e-10


def orthogonal(rng, d):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.where(np.diag(r) < 0, -1, 1)


def checked_inverse(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('require finite square coordinate matrix')
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 2 + 1e-12:
        raise ValueError('coordinate condition number exceeds fixture bound 2')
    return np.linalg.inv(matrix)


def fixture():
    library = tuple(Primitive.random(0, k, 16, 8, .35) for k in range(6))
    streams = [np.random.default_rng(np.random.SeedSequence([0, 2820, i])) for i in (1, 2, 3)]
    support, query = streams[0].normal(size=(16, 16)), streams[1].normal(size=(32, 16))
    rng = streams[2]
    matrices = {'identity': np.eye(16), 'permutation': np.roll(np.eye(16), 1, axis=0),
                'orthogonal': orthogonal(rng, 16),
                'dense': orthogonal(rng, 16) @ np.diag(np.geomspace(1, 2, 16)) @ orthogonal(rng, 16).T}
    return library, {'support': support, 'query': query}, matrices


def realized(primitive, x, matrix, inverse):
    return primitive(x @ inverse.T) @ matrix.T


def weight_only(primitive, x, matrix, inverse):
    naive = Primitive(matrix @ primitive.U, primitive.V @ inverse, primitive.b, primitive.alpha)
    return naive(x)


def discrepancy(prediction, target):
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('non-finite or mismatched arrays')
    energy = float(np.mean(np.sum(target ** 2, axis=-1)))
    if energy <= 1e-12:
        raise ValueError('vacuous target energy')
    return float(np.mean(np.sum((prediction-target)**2, axis=-1)) / energy)


def derivative_at_zero(primitive):
    hidden = np.tanh(primitive.b)
    output = np.tanh(primitive.alpha * (primitive.U @ hidden))
    inner = np.eye(primitive.U.shape[0]) + primitive.alpha * (
        primitive.U @ np.diag(1-hidden**2) @ primitive.V)
    return np.diag(1-output**2) @ inner


def finite_difference(primitive, step=1e-5):
    basis = np.eye(primitive.U.shape[0]) * step
    return ((primitive(basis)-primitive(-basis))/(2*step)).T


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((array.shape, array.dtype.str)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def run_checks():
    library, splits, matrices = fixture()
    rows, fingerprints = [], {}
    for split, initial in splits.items():
        fingerprints[split] = array_digest(initial)
        for name, matrix in matrices.items():
            inverse = checked_inverse(matrix)
            fingerprints[name] = array_digest(matrix)
            for program in PROGRAMS:
                canonical = initial.copy()
                rollout = initial @ matrix.T
                for step, op in enumerate(program):
                    # All per-operation predictions use THIS same canonical trajectory.
                    observed = canonical @ matrix.T
                    target = library[op](canonical)
                    correct = realized(library[op], observed, matrix, inverse) @ inverse.T
                    naive = weight_only(library[op], observed, matrix, inverse) @ inverse.T
                    missing = library[op](observed) @ inverse.T
                    # With tied inverses, conjugating identity still yields identity.
                    identity_core = ((observed @ inverse.T) @ matrix.T) @ inverse.T
                    rollout = realized(library[op], rollout, matrix, inverse)
                    rows.append(dict(split=split, context=name, program=list(program), step=step,
                                     oracle=discrepancy(correct, target),
                                     weight_only=discrepancy(naive, target),
                                     missing_adapter=discrepancy(missing, target),
                                     identity_core=discrepancy(identity_core, target),
                                     identity_invariance=discrepancy(identity_core, canonical),
                                     round_trip=discrepancy(observed @ inverse.T, canonical),
                                     oracle_rollout=discrepancy(rollout @ inverse.T, target)))
                    canonical = target
    summaries = {}
    for name, matrix in matrices.items():
        selected = [r for r in rows if r['context'] == name]
        summaries[name] = {key+'_max': max(r[key] for r in selected)
                           for key in ('oracle', 'weight_only', 'missing_adapter', 'identity_core',
                                       'identity_invariance', 'round_trip', 'oracle_rollout')}
        summaries[name]['condition_number'] = float(np.linalg.cond(matrix))
    for row in rows:
        if any(row[k] > TOL for k in ('oracle', 'identity_invariance', 'round_trip', 'oracle_rollout')):
            raise ValueError('oracle/inverse/identity control failed')
        if row['context'] in ('identity', 'permutation') and row['weight_only'] > TOL:
            raise ValueError('weight-only permutation anchor failed')
        if row['context'] == 'identity' and row['missing_adapter'] > TOL:
            raise ValueError('identity sharing anchor failed')
        if row['identity_core'] <= TOL:
            raise ValueError('non-identity target opportunity failed')
        if row['context'] in ('orthogonal', 'dense') and (
                row['missing_adapter'] <= TOL or row['weight_only'] <= TOL):
            raise ValueError('coordinate mismatch opportunity failed')
    jacobian = derivative_at_zero(library[0])
    numerical = finite_difference(library[0])
    derivative_error = float(np.max(np.abs(jacobian-numerical)))
    trace = float(np.trace(jacobian))
    negative_trace = 1.1*trace
    trace_errors = {name: abs(float(np.trace(m @ jacobian @ checked_inverse(m)))-trace)
                    for name, m in matrices.items()}
    if derivative_error > 1e-8 or max(trace_errors.values()) > 1e-10 or abs(negative_trace-trace) <= TOL:
        raise ValueError('linear-conjugacy negative witness failed')
    payload = sum(p.U.nbytes+p.V.nbytes+p.b.nbytes+np.asarray(p.alpha, dtype=np.float64).nbytes
                  for p in library)
    for k, p in enumerate(library):
        fingerprints[f'primitive_{k}'] = {name: array_digest(np.asarray(getattr(p, name)))
                                        for name in ('U', 'V', 'b', 'alpha')}
    return dict(version=VERSION, controls_ok=True, rows=rows, contexts=summaries,
                fixture_sha256=fingerprints,
                negative_witness=dict(trace=trace, scaled_trace=negative_trace,
                                      derivative_max_abs_error=derivative_error,
                                      conjugacy_trace_errors=trace_errors,
                                      scope='linear adapter impossibility for 1.1*A only'),
                payload=dict(core_float64_bytes=payload,
                             adapter_float64_bytes={n: m.nbytes for n, m in matrices.items()},
                             inverse_retained=False, includes_code_and_metadata=False,
                             total_description_length_measured=False),
                learned_adapters=False, learned_core=False, economic_value_measured=False)


def file_digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-check', required=True, action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'preserve prior output: {args.output}')
    start = time.perf_counter()
    result = run_checks()
    seconds = time.perf_counter()-start
    inputs = ('H28_C_COORDINATE_GATE_PLAN.md', 'tools/h28_coordinate_gate.py',
              'tests/test_h28_coordinate_gate.py', 'src/row/world.py')
    report = dict(status='PROVISIONAL_DEVELOPMENT_CHECK', accepted_scientific_result=False,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  dirty_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines(),
                  input_sha256={p: file_digest(ROOT/p) for p in inputs},
                  python=platform.python_version(), numpy=np.__version__, seconds=seconds, result=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], controls_ok=True, seconds=seconds,
                          boundaries=len(result['rows']), output=str(args.output))))


if __name__ == '__main__':
    main()
