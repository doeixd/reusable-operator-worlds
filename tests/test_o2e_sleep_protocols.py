import json
import unittest
from types import SimpleNamespace

from row.experiments import o2e_sleep_protocols as o2e
from row.experiments import score_o2e_sleep_protocols as scorer
from row.experiments.audit_rotated_g5r_interference import score


class O2ETests(unittest.TestCase):
    def test_labels_agree(self):
        for r in range(7):
            for b in range(13):
                self.assertEqual(o2e.label(r, b), scorer.label(r, b))

    def test_cells(self):
        self.assertEqual(len(o2e.cells()), 42)

    def test_staged_terminal_reloads_exactly(self):
        # the G0 anchor on one cell per arm (full G0 runs at launch)
        report = json.load(open('reports/o2_online_reliability.json'))['cells']
        for arm, base in o2e.ARMS.items():
            _, model, tasks, canonical = o2e.load_terminal(arm, 13, 1)
            self.assertEqual(len(tasks), scorer.EXPECTED_TASKS[arm])
            per = score(model, SimpleNamespace(tasks=canonical))['per_task']
            self.assertEqual(per, report[f'{base}_w13_s1']['terminal_per_task'])


if __name__ == '__main__':
    unittest.main()
