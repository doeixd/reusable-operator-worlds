import unittest

import torch

from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments import o7_gradient_reroute as o7
from row.experiments import score_o7_gradient_reroute as scorer


class O7Tests(unittest.TestCase):
    def test_labels_partition_and_agree(self):
        for r in range(11):
            for b in range(13):
                self.assertEqual(o7.label(r, b), scorer.label(r, b))
        self.assertEqual((o7.label(8, 1), o7.label(7, 1), o7.label(4, 0), o7.label(3, 0), o7.label(10, 2)),
                         ('MATCHES_EXHAUSTIVE', 'PARTIAL', 'PARTIAL', 'NO_RESCUE', 'HARMS'))

    def test_cells_match_scorer_and_plan(self):
        self.assertEqual([list(c) for c in o7.TARGET], scorer.TARGET)
        self.assertEqual([list(c) for c in o7.HARM], scorer.HARM)
        self.assertEqual((len(o7.TARGET), len(o7.HARM)), (10, 12))
        self.assertFalse(set(map(tuple, o7.TARGET)) & set(map(tuple, o7.HARM)))

    def test_summary_counts_nan_as_failing(self):
        recs = {f'{b}_w{w}_s{s}': {'terminal_median': 0.01} for b, w, s in o7.cells()}
        self.assertEqual(o7.summarize(recs)['label'], 'MATCHES_EXHAUSTIVE')
        recs['O2_w13_s1']['terminal_median'] = float('nan')
        recs['O2_w15_s1']['terminal_median'] = 0.2
        self.assertEqual(o7.summarize(recs)['label'], 'HARMS')

    def test_safe_chooser_never_worse_on_support(self):
        _, model, st, canon = o3.load_shuffled_terminal(o5.terminal_path('O2', 14, 0), 14, 0)
        _, _, _, plan, _ = o2.build_stream('SHUFFLED', 14, 0)
        pool = o2d.reservoir(st, 14, 0, 64)
        lib = o7.FrozenLibrary(model)
        cur_routes = model.hard_routes()
        task = next(t for t in st if plan[t.task_id] == 3)
        pts = [(x, y) for x, y, t in pool if t == task.task_id]
        xs = torch.tensor([p[0] for p in pts], dtype=torch.float32)
        ys = torch.tensor([p[1] for p in pts], dtype=torch.float32)
        cur = [int(r) for r in cur_routes[task.task_id][:3]]
        o7.OPT_STEPS = 50
        new, _ = o7.choose(lib, xs, ys, 3, cur, 'gradient')
        self.assertLessEqual(o7.support_mse(lib, xs, ys, new), o7.support_mse(lib, xs, ys, cur))


if __name__ == '__main__':
    unittest.main()
