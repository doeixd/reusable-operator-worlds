import json
import tempfile
import unittest
from pathlib import Path

from row.experiments import g3a_inference_census as g3
from row.experiments import score_g3a_inference_census as sc
from row.experiments.so1_storage import fingerprint


def _record(m, w, regret_by_support):
    rows = []
    for i in range(g3.PROGRAMS):
        by = {}
        for s in map(str, g3.SUPPORTS):
            regret, floor = regret_by_support[s](i)
            by[s] = {'regret': regret, 'null_floor': floor, 'binds': regret > floor,
                     'nmse_hat_qb': 0.01 + regret, 'nmse_star_qb': 0.01,
                     'inference_failure': 0.01 + regret >= 0.05, 'routes_differ': regret != 0,
                     'identifiability': 1.0, 'near_tie_size': 1, 'near_tie_disagreement': 0.0}
        rows.append({'task': i, 'program': [i // 36, (i // 6) % 6, i % 6], 'by_support': by})
    by_support = {}
    for s in map(str, g3.SUPPORTS):
        rs = [r['by_support'][s] for r in rows]
        import numpy as np
        by_support[s] = {'median_regret': float(np.median([r['regret'] for r in rs])),
                         'median_floor': float(np.median([r['null_floor'] for r in rs])),
                         'cell_counts': bool(np.median([r['regret'] for r in rs]) > np.median([r['null_floor'] for r in rs])),
                         'binds': sum(r['binds'] for r in rs), 'inference_failures': sum(r['inference_failure'] for r in rs),
                         'median_identifiability': 1.0}
    return {'shared': m, 'world': w, 'programs': g3.PROGRAMS, 'anchor_terminal': True, 'anchor_formed': True,
            'rows': rows, 'by_support': by_support}


ZERO = {s: (lambda i: (0.0, 0.0)) for s in map(str, g3.SUPPORTS)}


class G3aTests(unittest.TestCase):
    def _records(self, m4):
        recs = {}
        for w in g3.WORLDS:
            recs[g3.key_of(0.0, w)] = _record(0.0, w, ZERO)
            recs[g3.key_of(4.0, w)] = _record(4.0, w, m4)
        return recs

    def test_labels(self):
        self.assertEqual(g3.reading(self._records(ZERO))['label'], 'NO_HEADROOM')
        minority = dict(ZERO, **{'2': lambda i: (0.1, 0.0) if i < 13 else (0.0, 0.0)})   # 13/64 = 0.20 bind
        r = g3.reading(self._records(minority))
        self.assertEqual((r['label'], r['rule_b_supports']), ('PARTIAL', ['2']))
        tiny = dict(ZERO, **{'2': lambda i: (0.1, 0.0) if i < 5 else (0.0, 0.0)})        # 5/64 = 0.08 < 0.10
        self.assertEqual(g3.reading(self._records(tiny))['label'], 'NO_HEADROOM')
        most = dict(ZERO, **{'8': lambda i: (0.1, 0.0)})
        self.assertEqual(g3.reading(self._records(most))['label'], 'HEADROOM')
        recs = self._records(ZERO)
        for w in g3.WORLDS:
            recs[g3.key_of(0.0, w)] = _record(0.0, w, most)
        self.assertEqual(g3.reading(recs)['label'], 'CONTROL_FIRES')

    def test_scorer_recomputes_and_refuses(self):
        minority = dict(ZERO, **{'2': lambda i: (0.1, 0.0) if i < 13 else (0.0, 0.0)})
        recs = self._records(minority)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'cells').mkdir()
            for k, r in recs.items():
                (root / 'cells' / f'{k}.json').write_text(json.dumps(
                    {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': r, 'record_sha256': fingerprint(r)}))
            rep = {'protocol': {'input_sha256': {}}, 'protocol_sha256': 'x', 'cells': recs, 'reading': g3.reading(recs)}
            (root / 'report.json').write_text(json.dumps(rep))
            out = sc.score(root / 'report.json', root / 'cells', check_inputs=False)
            self.assertEqual((out['label'], out['runner_agrees']), ('PARTIAL', True))
            bad = json.loads((root / 'cells' / 'm4_w71.json').read_text())
            bad['record']['rows'][0]['by_support']['2']['regret'] = 9.0
            (root / 'cells' / 'm4_w71.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                sc.score(root / 'report.json', root / 'cells', check_inputs=False)


if __name__ == '__main__':
    unittest.main()
