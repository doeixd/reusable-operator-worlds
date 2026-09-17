import copy
import json
import tempfile
import unittest
from pathlib import Path

from row.experiments import l0d_ambiguity_gate as gate
from row.experiments.score_l0d_ambiguity_gate import score


def record():
    rows = []
    for index in range(16):
        supports = {}
        for support in gate.SUPPORTS:
            supports[str(support)] = {'route': [0, 1, 2], 'query_nmse': .01,
                'best_mse': .1, 'gap_mse': .1, 'relative_gap': 1., 'seconds': .01}
        for support in (4, 2, 1):
            supports[str(support)].update(route_changed_from_128=False, query_delta_from_128=0.)
        rows.append({'task': index, 'program': [0, 1, 2], 'supports': supports})
    return {'name': gate.NAME, 'world': gate.WORLD, 'model_seed': gate.MODEL_SEED,
            'library_sha256': 'fixture', 'rows': rows, 'seconds': 0.1}


class AmbiguityGateTests(unittest.TestCase):
    def test_validate_rejects_missing_and_inconsistent_observations(self):
        base = record()
        gate.validate(base)
        for mode in ('missing', 'route', 'delta'):
            broken = copy.deepcopy(base)
            if mode == 'missing':
                broken['rows'].pop()
            elif mode == 'route':
                broken['rows'][0]['supports']['1']['route_changed_from_128'] = True
            else:
                broken['rows'][0]['supports']['1']['query_delta_from_128'] = 1.
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                gate.validate(broken)

    def test_summary_preserves_favorable_and_unfavorable_differences(self):
        base = record()
        base['rows'][0]['supports']['1'].update(route=[3, 3, 3], route_changed_from_128=True,
                                                query_nmse=.2, query_delta_from_128=.19)
        result = gate.summary(base)
        self.assertEqual(result['route_changes']['1'], 1)
        self.assertEqual(result['query_worsens']['1'], 1)
        self.assertEqual(result['query_improves']['1'], 0)

    def test_independent_scorer_rejects_changed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {'id': 'fixture', 'input_sha256': {}, 'implementation_sha256': 'x', 'supports': [128, 4, 2, 1]}
            sha = gate.fingerprint(protocol)
            manifest = {'protocol': protocol, 'protocol_sha256': sha}
            cell = {'stamp': {'protocol_sha256': sha}, 'complete': True, 'record': record()}
            cell['record_sha256'] = gate.fingerprint(cell['record'])
            (root / 'cell.json').write_text(json.dumps(cell))
            (root / 'manifest.json').write_text(json.dumps(manifest))
            report = {**manifest, 'complete': True, 'cell': cell['record'], 'summary': {'interpretation': 'opportunity gate only; no PX7 verdict'}}
            output = root / 'report.json'; output.write_text(json.dumps(report))
            (root / 'status.json').write_text(json.dumps({'state': 'complete'}))
            (root / 'exit.json').write_text(json.dumps({'exit_code': 0, 'complete': True}))
            with self.assertRaises(ValueError):
                score(root, output)


if __name__ == '__main__':
    unittest.main()
