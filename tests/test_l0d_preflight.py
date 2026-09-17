import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from row.experiments import preflight_l0d as l0d
from row.experiments.score_l0d_preflight import score
from row.experiments.audit_j1c_curriculum import stage_setup
from row.experiments.audit_so1_budget_bracket import build_fast
from row.experiments.audit_so1r_route_only import FrozenLibrary


def fixture(name='A', world=0, tasks=2):
    return {'name': name, 'world': world, 'anchor_bitwise': True, 'library_unchanged': True,
            'seconds': 1.0, 'rows': [
                {'task': i, 'executor_bitwise': True, 'mapped_nmse': .2, 'j2a_random_nmse': 1.,
                 'supports': {str(s): {'gap_mse': float(s), 'query_nmse': .01, 'best_mse': .01,
                             'relative_gap': float(s) / .01, 'seconds': .1, 'route': [0, 1, 2]}
                              for s in l0d.SUPPORTS}}
                for i in range(tasks)]}


class L0DPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_variable_forward_and_gradients_at_new_depths(self):
        cfg, _, _, _ = stage_setup(0, 3, 5000)
        model = build_fast(cfg)
        reference = FrozenLibrary(model)
        variable = l0d.VariableDepthLibrary(model)
        x = torch.randn(8, 16, generator=torch.Generator().manual_seed(29))
        for depth in (1, 3, 4, 5):
            coefficients = torch.softmax(torch.arange(depth * 12, dtype=torch.float32).reshape(depth, 12), -1)
            coefficients.requires_grad_(True)
            prediction = variable.forward(x, coefficients)
            expected = x
            for coefficient in coefficients:
                expected = (coefficient[None, :, None] * reference.candidates(expected)).sum(1)
            self.assertTrue(torch.equal(prediction, expected))
            prediction.square().mean().backward()
            self.assertTrue(torch.isfinite(coefficients.grad).all())
            self.assertGreater(float(coefficients.grad.abs().sum()), 0)
            if depth == 3:
                self.assertTrue(torch.equal(reference.forward(x, coefficients), prediction))
        with self.assertRaises(ValueError):
            variable.forward(x, torch.empty(0, 12))

    def test_resume_reuses_cell_bytes_and_rebuilds_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'report.json'
            calls = []
            def compute(name, world, count):
                calls.append(name)
                return fixture(name, world, count)
            protocol = {'git_commit': 'fixture'}
            jobs = [('A', 0), ('B', 0)]
            l0d.run(root, output, protocol, jobs, 2, compute, stop_after=1)
            cell = root / 'cells/A_w0/result.json'
            before = cell.read_bytes()
            self.assertFalse(json.loads(output.read_text())['complete'])
            output.unlink()  # simulate loss of the parent's report
            l0d.run(root, output, protocol, jobs, 2, compute)
            self.assertEqual(calls, ['A', 'B'])
            self.assertEqual(cell.read_bytes(), before)
            self.assertTrue(json.loads(output.read_text())['complete'])
            l0d.run(root, output, protocol, jobs, 2, compute)
            self.assertEqual(calls, ['A', 'B'])
            with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                l0d.run(root, output, {'git_commit': 'changed'}, jobs, 2, compute)

    def test_resume_refuses_corrupt_or_incomplete_cells(self):
        for corruption in ('hash', 'complete'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                l0d.run(root, root / 'report.json', {'git_commit': 'fixture'}, [('A', 0)], 2, fixture)
                path = root / 'cells/A_w0/result.json'
                record = json.loads(path.read_text())
                if corruption == 'hash':
                    record['record']['seconds'] = 90
                else:
                    record['complete'] = False
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    l0d.run(root, root / 'report.json', {'git_commit': 'fixture'}, [('A', 0)], 2, fixture)
                self.assertEqual(json.loads((root / 'exit.json').read_text())['exit_code'], 1)

    def test_negative_gates_and_missing_observations(self):
        original = fixture()
        l0d.validate_record(original, 2)
        for kind in ('anchor', 'mutated', 'nan', 'missing_task', 'missing_support'):
            bad = copy.deepcopy(original)
            if kind == 'anchor':
                bad['anchor_bitwise'] = False
            elif kind == 'mutated':
                bad['library_unchanged'] = False
            elif kind == 'nan':
                bad['rows'][0]['mapped_nmse'] = float('nan')
            elif kind == 'missing_task':
                bad['rows'].pop()
            else:
                del bad['rows'][0]['supports']['8']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                l0d.validate_record(bad, 2)
        with self.assertRaises(ValueError):
            l0d.summarize({})

    def test_ambiguity_and_mapping_are_diagnostics_not_gates(self):
        record = fixture()
        record['rows'][0]['supports']['8']['gap_mse'] = 999.
        l0d.validate_record(record, 2)
        summary = l0d.summarize({'A': record})
        self.assertEqual(summary['gap_monotone_tasks'], 1)
        self.assertEqual(summary['mapped_fails_enum_passes'], 2)
        self.assertEqual(summary['support_cells'], 6)

    def test_memory_lower_bound_includes_examples_and_dimensions(self):
        self.assertEqual(l0d.terminal_tensor_bytes(128, 12, 5, 16), 2038431744)

    def test_independent_scorer_rejects_wrong_aggregate_and_failed_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'report.json'
            jobs = [('STAGED5000', 0), ('STAGED5000', 1)]
            protocol = {'id': 'l0d-preflight-v1', 'git_commit': 'fixture', 'dry_run': True,
                        'jobs': [list(j) for j in jobs], 'tasks': 2, 'supports': [128, 32, 8],
                        'input_sha256': {l0d.PLAN.as_posix(): l0d.digest(l0d.PLAN)},
                        'implementation_sha256': l0d.digest(Path(l0d.__file__))}
            l0d.run(root, output, protocol, jobs, 2, fixture)
            self.assertTrue(score(root, output)['valid'])
            original = output.read_text()
            changed = json.loads(original)
            changed['summary']['gap_monotone_tasks'] = 999
            output.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'summary counts'):
                score(root, output)
            output.write_text(original)
            (root / 'exit.json').write_text(json.dumps({'exit_code': 1, 'complete': False}))
            with self.assertRaisesRegex(ValueError, 'operational completion'):
                score(root, output)


if __name__ == '__main__':
    unittest.main()
