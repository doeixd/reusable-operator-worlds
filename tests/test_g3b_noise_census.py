import json
import tempfile
import unittest
from pathlib import Path

from row.experiments import g3a_inference_census as g3a
from row.experiments import g3b_noise_census as g3b
from row.experiments import score_g3b_noise_census as sc
from row.experiments.so1_storage import fingerprint


def _record(m, w, binder):
    """binder(v, s, i) -> True if program i binds at noise v, support s."""
    rows = []
    for i in range(g3b.PROGRAMS):
        by_noise = {}
        for v in g3b.NOISE:
            by_noise[f'{v:g}'] = {str(s): {'regret': 0.1 if binder(v, s, i) else 0.0, 'null_floor': 0.0,
                                           'binds': binder(v, s, i), 'nmse_hat_qb': 0.01, 'nmse_star_qb': 0.01,
                                           'routes_differ': binder(v, s, i), 'identifiability': 1.0}
                                  for s in g3b.SUPPORTS}
        rows.append({'task': i, 'program': [i], 'by_noise': by_noise})
    summary = {f'{v:g}': {str(s): {'binds': sum(r['by_noise'][f'{v:g}'][str(s)]['binds'] for r in rows)}
                          for s in g3b.SUPPORTS} for v in g3b.NOISE}
    return {'shared': m, 'world': w, 'programs': g3b.PROGRAMS, 'anchor_terminal': True, 'anchor_g3a': True,
            'rows': rows, 'summary': summary}


NONE = lambda v, s, i: False


class G3bTests(unittest.TestCase):
    def _records(self, b4, b0=NONE):
        recs = {}
        for w in g3b.WORLDS:
            recs[g3a.key_of(0.0, w)] = _record(0.0, w, b0)
            recs[g3a.key_of(4.0, w)] = _record(4.0, w, b4)
        return recs

    def test_labels(self):
        self.assertEqual(g3b.reading(self._records(NONE))['label'], 'NO_BIND')
        win = lambda v, s, i: v == 0.1 and s == 2 and i < 13          # 0.20 at (0.1, 2)
        r = g3b.reading(self._records(win))
        self.assertEqual((r['label'], r['window']), ('NOISE_WINDOW', [[0.1, 2]]))
        self.assertEqual(r['onset_noise']['m4']['2'], 0.1)
        self.assertEqual(g3b.reading(self._records(win, win))['label'], 'NOISE_ONLY')
        high = lambda v, s, i: v >= 0.3 and s == 2 and i < 13
        self.assertEqual(g3b.reading(self._records(high))['label'], 'HIGH_NOISE_ONLY')
        small_margin = lambda v, s, i: v == 0.1 and s == 2 and i < 13
        control = lambda v, s, i: v == 0.1 and s == 2 and i < 7        # 0.20 vs 0.11: margin 0.09 < 0.10
        self.assertEqual(g3b.reading(self._records(small_margin, control))['label'], 'NOISE_ONLY')

    def test_scorer_recomputes_and_refuses(self):
        win = lambda v, s, i: v == 0.03 and s == 8 and i < 20
        recs = self._records(win)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'cells').mkdir()
            for k, r in recs.items():
                (root / 'cells' / f'{k}.json').write_text(json.dumps(
                    {'stamp': {'protocol_sha256': 'x'}, 'complete': True, 'record': r, 'record_sha256': fingerprint(r)}))
            rep = {'protocol': {'input_sha256': {}}, 'protocol_sha256': 'x', 'cells': recs, 'reading': g3b.reading(recs)}
            (root / 'report.json').write_text(json.dumps(rep))
            out = sc.score(root / 'report.json', root / 'cells', check_inputs=False)
            self.assertEqual((out['label'], out['window'], out['runner_agrees']), ('NOISE_WINDOW', [[0.03, 8]], True))
            bad = json.loads((root / 'cells' / 'm4_w72.json').read_text())
            bad['record']['anchor_g3a'] = False
            (root / 'cells' / 'm4_w72.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                sc.score(root / 'report.json', root / 'cells', check_inputs=False)


if __name__ == '__main__':
    unittest.main()
