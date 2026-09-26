import math
import unittest

from row.experiments import o3_online_sleep as o3
from row.experiments import score_o3_online_sleep as scorer


class O3Tests(unittest.TestCase):
    def test_schedule_is_exact(self):
        n = o3.schedule(8192, 188)
        self.assertEqual(sum(n), 8192)
        self.assertEqual(len(n), 188)
        self.assertLessEqual(max(n) - min(n), 1)
        self.assertEqual(sum(o3.schedule(0, 188)), 0)

    def test_cells(self):
        c = o3.cells()
        self.assertEqual(len(c), 21 * 3 + 7)
        self.assertEqual(c[0][0], 'INTERLEAVED')
        self.assertFalse({w for _, w, _ in c} & set(range(0, 20)))

    def test_labels_agree(self):
        for k in range(22):
            self.assertEqual(o3.label(k), scorer.label(k))

    def test_contrast_rule_and_agreement(self):
        cases = [
            [[-0.5, -0.5, -0.5]] * 7,
            [[0.5, 0.5, 0.5]] * 7,
            [[-0.01, 0.01, 0.02]] * 7,
            [[-0.3, 0.3, 0.3]] * 7,
        ]
        want = ['PHASE_MATTERS', 'INTERLEAVED_BETTER', 'EQUIVALENT', 'INDETERMINATE']
        for rw, w in zip(cases, want):
            self.assertEqual(o3.contrast_label(rw), w)
            self.assertEqual(scorer.contrast(rw), w)

    def test_nonfinite_r(self):
        for ms, mi in ((float('nan'), 0.1), (0.1, float('nan')), (float('nan'), float('nan')), (0.01, 0.1)):
            self.assertEqual(o3.contrast_r(ms, mi), scorer.r_value(ms, mi))
        self.assertEqual(o3.contrast_r(float('nan'), 0.1), math.inf)

    def test_summary_floor(self):
        recs = {f'{a}_w{w}_s{s}': {'terminal_median': 0.01} for a, w, s in o3.cells()}
        self.assertEqual(o3.summarize(recs)['primary'], 'FLOOR_FAILED')
        for w in o3.WORLDS:
            recs[f'PLAIN_w{w}_s0'] = {'terminal_median': 1.9}
        s = o3.summarize(recs)
        self.assertEqual((s['primary'], s['contrast']), ('WAKE_SLEEP_RELIABLE', 'INDETERMINATE'))


class ValidateTests(unittest.TestCase):
    def _rec(self, arm, **kw):
        r = {'arm': arm, 'world': 20, 'stream': 0, 'scored_tasks': 64,
             'terminal_per_task': {str(i): 0.01 for i in range(64)}, 'stream_tasks': 188, 'trained_tasks': 188,
             'route_lengths_match_plan': True, 'anchor_abs_error': 0.0, 'extra_updates': 8192,
             'library_sha256': 'a', 'library_sha256_before': 'b'}
        r.update(kw)
        return r

    def test_v1_failure_is_now_valid_for_interleaved(self):
        # the exact v1 failure: INTERLEAVED consolidates after its last task, so its anchor is nonzero
        o3.validate(self._rec('INTERLEAVED', anchor_abs_error=0.003132016919603301))

    def test_shuffled_anchor_still_enforced(self):
        with self.assertRaises(ValueError):
            o3.validate(self._rec('SHUFFLED', anchor_abs_error=0.003))

    def test_extra_updates_enforced(self):
        with self.assertRaises(ValueError):
            o3.validate(self._rec('INTERLEAVED', extra_updates=8191))


if __name__ == '__main__':
    unittest.main()
