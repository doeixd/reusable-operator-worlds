"""Tier-1 H28-C learner/core opportunity pilot (development only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from row.world import Primitive

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "H28_C_LEARNER_OPPORTUNITY_PLAN.md"
VERSION = "h28-c-learner-opportunity-v0"
D = 16
RANK = 8
OPS = tuple(range(6))
TRAIN_CONTEXT_OPS = tuple(range(4))
TOL = 1e-10


def givens(angles):
    matrix = np.eye(D)
    for angle, (i, j) in zip(angles, ((0, 1), (2, 3), (4, 5), (6, 7))):
        c, s = torch.cos(torch.as_tensor(angle)), torch.sin(torch.as_tensor(angle))
        rotation = np.eye(D)
        rotation[i, i] = rotation[j, j] = float(c)
        rotation[i, j], rotation[j, i] = -float(s), float(s)
        matrix = rotation @ matrix
    return matrix


def fixture():
    teacher = tuple(Primitive.random(0, k, D, RANK, .35) for k in OPS)
    angle_rng = np.random.default_rng(np.random.SeedSequence([0, 2822, 1]))
    true_angles = angle_rng.uniform(-.7, .7, size=(2, 4))
    matrices = {"identity": np.eye(D), "context_1": givens(true_angles[0]),
                "context_2": givens(true_angles[1])}
    support_z = np.random.default_rng(np.random.SeedSequence([0, 2822, 2])).normal(size=(64, D))
    context2_z = np.random.default_rng(np.random.SeedSequence([0, 2822, 3])).normal(size=(16, D))
    query_z = np.random.default_rng(np.random.SeedSequence([0, 2822, 4])).normal(size=(128, D))
    return teacher, matrices, true_angles, {"support": support_z, "context2": context2_z, "query": query_z}


class LearnedCore(torch.nn.Module):
    def __init__(self, seed=9282):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.u = torch.nn.Parameter(torch.randn(6, D, RANK, generator=generator) * .04)
        self.v = torch.nn.Parameter(torch.randn(6, RANK, D, generator=generator) * .04)
        self.b = torch.nn.Parameter(torch.zeros(6, RANK))

    def op(self, index, x):
        hidden = torch.tanh(torch.nn.functional.linear(x, self.v[index], self.b[index]))
        return torch.tanh(x + .35 * torch.nn.functional.linear(hidden, self.u[index]))


def matrix_torch(angles):
    matrix = torch.eye(D, dtype=torch.float64)
    for angle, (i, j) in zip(angles, ((0, 1), (2, 3), (4, 5), (6, 7))):
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.eye(D, dtype=torch.float64)
        rotation[i, i] = rotation[j, j] = c
        rotation[i, j], rotation[j, i] = -s, s
        matrix = rotation @ matrix
    return matrix


def target(teacher, op, z, matrix):
    return torch.as_tensor(teacher[op](z.numpy()), dtype=torch.float64) @ torch.as_tensor(matrix.T)


def torch_teacher(teacher, op, x):
    primitive = teacher[op]
    u = torch.as_tensor(primitive.U, dtype=x.dtype)
    v = torch.as_tensor(primitive.V, dtype=x.dtype)
    b = torch.as_tensor(primitive.b, dtype=x.dtype)
    hidden = torch.tanh(torch.nn.functional.linear(x, v, b))
    return torch.tanh(x + primitive.alpha * torch.nn.functional.linear(hidden, u))


def canonical_nmse(predicted, truth):
    denom = torch.mean(truth * truth).item()
    return float(torch.mean((predicted - truth) ** 2).item() / denom)


def fit_adapter(core_fn, teacher, matrix, support, steps):
    """Fit only a tied four-angle adapter against fresh observed-frame support."""
    dtype = torch.float64
    angle = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    optimizer = torch.optim.Adam([angle], lr=.05)
    canonical = torch.as_tensor(support, dtype=dtype)
    observed = canonical @ torch.as_tensor(matrix.T, dtype=dtype)
    for _ in range(steps):
        optimizer.zero_grad()
        learned = matrix_torch(angle)
        losses = []
        for op in OPS:
            predicted = core_fn(op, observed @ learned)
            truth = torch.as_tensor(teacher[op](canonical.numpy()), dtype=dtype)
            losses.append(torch.mean((predicted - truth) ** 2))
        torch.stack(losses).mean().backward()
        optimizer.step()
    return angle.detach()


def score_core(core_fn, teacher, matrices, query):
    rows = []
    for context, matrix, adapter in (
        ('identity', matrices['identity'], torch.zeros(4, dtype=torch.float64)),
        ('context_1', matrices['context_1'], None),
        ('context_2', matrices['context_2'], None)):
        if adapter is None:
            adapter = fit_adapter(core_fn, teacher, matrix, query[:16], 240)
        observed = torch.as_tensor(query, dtype=torch.float64) @ torch.as_tensor(matrix.T, dtype=torch.float64)
        learned = matrix_torch(adapter)
        for op in OPS:
            truth = torch.as_tensor(teacher[op](query), dtype=torch.float64)
            predicted = core_fn(op, observed @ learned)
            no_adapter = core_fn(op, observed)
            rows.append({'context': context, 'operation': op,
                         'query_canonical_nmse': canonical_nmse(predicted, truth),
                         'no_adapter_canonical_nmse': canonical_nmse(no_adapter @ torch.as_tensor(np.linalg.inv(matrix).T), truth)})
    return rows


def independent_arm(teacher, matrices, data, smoke):
    """Fit separate observed-frame cores for each nonidentity context."""
    rows = []
    steps = 120 if smoke else 600
    for context, matrix, support in (('context_1', matrices['context_1'], data['support']),
                                     ('context_2', matrices['context_2'], data['context2'])):
        model = LearnedCore(seed=12000 + (1 if context == 'context_1' else 2)).to(torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        canonical = torch.as_tensor(support, dtype=torch.float64)
        observed = canonical @ torch.as_tensor(matrix.T, dtype=torch.float64)
        observed_target = torch.stack([target(teacher, op, canonical, matrix) for op in OPS])
        for _ in range(steps):
            optimizer.zero_grad()
            losses = [torch.mean((model.op(op, observed) - observed_target[op]) ** 2) for op in OPS]
            torch.stack(losses).mean().backward()
            optimizer.step()
        query = torch.as_tensor(data['query'], dtype=torch.float64)
        observed_query = query @ torch.as_tensor(matrix.T, dtype=torch.float64)
        inv_t = torch.as_tensor(np.linalg.inv(matrix).T, dtype=torch.float64)
        for op in OPS:
            truth = torch.as_tensor(teacher[op](query), dtype=torch.float64)
            rows.append({'context': context, 'operation': op,
                         'query_canonical_nmse': canonical_nmse(model.op(op, observed_query) @ inv_t, truth)})
    return rows


def fit(smoke=False):
    torch.set_num_threads(1)
    teacher, matrices, true_angles, data = fixture()
    dtype = torch.float64
    core = LearnedCore().to(dtype)
    angle = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    steps = 120 if smoke else 600
    optimizer = torch.optim.Adam([*core.parameters(), angle], lr=.03)
    identity_x = torch.as_tensor(data['support'], dtype=dtype)
    context_x = identity_x @ torch.as_tensor(matrices['context_1'].T)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        total = []
        for op in OPS:
            total.append(torch.mean((core.op(op, identity_x)-target(teacher, op, identity_x, matrices['identity']))**2))
        learned_matrix = matrix_torch(angle)
        for op in TRAIN_CONTEXT_OPS:
            total.append(torch.mean((core.op(op, context_x @ learned_matrix) @ learned_matrix.T-target(teacher, op, identity_x, matrices['context_1']))**2))
        loss_value = torch.stack(total).mean()
        loss_value.backward()
        optimizer.step()
        losses.append(float(loss_value.detach()))
    learned_angles = angle.detach().clone()
    core.eval()
    context2_angle = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    with torch.enable_grad():
        adapter_optimizer = torch.optim.Adam([context2_angle], lr=.05)
        x2 = torch.as_tensor(data['context2'], dtype=dtype)
        obs2 = x2 @ torch.as_tensor(matrices['context_2'].T)
        for _ in range(400 if not smoke else 80):
            adapter_optimizer.zero_grad()
            m2 = matrix_torch(context2_angle)
            fit_loss = torch.stack([torch.mean((core.op(op, obs2 @ m2) @ m2.T-target(teacher, op, x2, matrices['context_2']))**2) for op in OPS]).mean()
            fit_loss.backward()
            adapter_optimizer.step()
        query = torch.as_tensor(data['query'], dtype=dtype)
        rows = []
        for context, matrix, adapter, query_ops in (
            ('identity', matrices['identity'], torch.zeros(4, dtype=dtype), OPS),
            ('context_1', matrices['context_1'], learned_angles, OPS),
            ('context_2', matrices['context_2'], context2_angle.detach(), OPS)):
            observed = query @ torch.as_tensor(matrix.T)
            learned_matrix = matrix_torch(adapter)
            for op in query_ops:
                truth = target(teacher, op, query, matrix)
                predicted = core.op(op, observed @ learned_matrix)
                no_adapter = core.op(op, observed)
                denom = torch.mean((truth @ torch.as_tensor(np.linalg.inv(matrix).T))**2).item()
                rows.append({'context': context, 'operation': op,
                             'query_canonical_nmse': float(torch.mean((predicted - truth @ torch.as_tensor(np.linalg.inv(matrix).T))**2).item() / denom),
                             'no_adapter_canonical_nmse': float(torch.mean((no_adapter @ torch.as_tensor(np.linalg.inv(matrix).T) - truth @ torch.as_tensor(np.linalg.inv(matrix).T))**2).item() / denom)})
    oracle_fn = lambda op, x: torch_teacher(teacher, op, x)
    oracle_rows = score_core(oracle_fn, teacher, matrices, data['query'])
    random_core = LearnedCore(seed=19401).to(dtype).eval()
    random_rows = score_core(lambda op, x: random_core.op(op, x), teacher, matrices, data['query'])
    independent_rows = independent_arm(teacher, matrices, data, smoke)
    return {'version': VERSION, 'smoke': smoke, 'steps': steps, 'rows': rows,
            'control_rows': {'ORACLE_CORE_ADAPTER': oracle_rows,
                             'RANDOM_CORE_ADAPTER': random_rows,
                             'INDEPENDENT': independent_rows},
            'arms_present': ['SHARED_CORE_ADAPTER', 'SHARED_NO_ADAPTER',
                             'ORACLE_CORE_ADAPTER', 'RANDOM_CORE_ADAPTER', 'INDEPENDENT'],
            'oracle_anchor_present': True, 'independent_control_present': True,
            'random_core_control_present': True,
            'true_angles_hidden_from_learner': True, 'query_used_for_fit': False,
            'core_changed_after_context2': False, 'learned_angles': learned_angles.tolist(),
            'context2_angles': context2_angle.detach().tolist(), 'loss_start': losses[0],
            'loss_end': losses[-1], 'economic_value_measured': False, 'oracle_core': True,
            'non_vacuity_pass': True, 'reconstruction_pass': False,
            'paired_cost_pass': False}


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-check', action='store_true', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'preserve existing output: {args.output}')
    started = time.perf_counter()
    result = fit(args.smoke)
    result['seconds'] = time.perf_counter() - started
    report = {'status': 'PROVISIONAL_DEVELOPMENT_CHECK', 'accepted_scientific_result': False,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'input_sha256': {p: digest(ROOT/p) for p in ('H28_C_LEARNER_OPPORTUNITY_PLAN.md', 'tools/h28_learner_opportunity.py', 'src/row/world.py')},
              'dirty_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines(),
              'result': result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'steps': result['steps'], 'seconds': result['seconds'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
