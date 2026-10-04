import json
import unittest
from pathlib import Path

from row.experiments import o10_rw_sleep as o10
from row.experiments import score_o10_rw_sleep as scorer


class O10Tests(unittest.TestCase):
    def test_labels_partition_and_agree(self):
        seen = set()
        for r in range(9):
            for b in range(38):
                self.assertEqual(o10.label(r, b), scorer.verdict(r, b))
                seen.add(o10.label(r, b))
        self.assertEqual(seen, {'HARMS', 'REPAIRS', 'PARTIAL', 'NO_REPAIR'})
        self.assertEqual((o10.label(7, 2), o10.label(6, 0), o10.label(3, 0), o10.label(8, 3)),
                         ('REPAIRS', 'PARTIAL', 'NO_REPAIR', 'HARMS'))

    def test_target_is_o9_failures(self):
        cells = json.loads(Path('reports/o9_sealed_online.json').read_text())['cells']
        fails = [(w, s) for w, s in o10.cells() if cells[f'REROUTE_WAKE_w{w}_s{s}']['terminal_median'] >= 0.05]
        self.assertEqual(fails, o10.TARGET)
        self.assertEqual((len(o10.cells()), len(o10.harm())), (45, 37))

    def test_summary_nan_counts_as_broken(self):
        recs = {f'w{w}_s{s}': {'terminal_median': 0.01, 'stale_after_sleep': 0} for w, s in o10.cells()}
        self.assertEqual(o10.summarize(recs)['label'], 'REPAIRS')
        for w, s in o10.harm()[:3]:
            recs[f'w{w}_s{s}']['terminal_median'] = float('nan')
        self.assertEqual(o10.summarize(recs)['label'], 'HARMS')


if __name__ == '__main__':
    unittest.main()
