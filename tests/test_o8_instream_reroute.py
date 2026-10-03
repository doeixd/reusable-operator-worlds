import unittest

import numpy as np
import torch

from row.experiments import o2_online_reliability as o2
from row.experiments import o2d_sleep_memory as o2d
from row.experiments import o3_online_sleep as o3
from row.experiments import o5_reroute_sleep as o5
from row.experiments import o8_instream_reroute as o8
from row.experiments import score_o8_instream_reroute as scorer
from row.experiments.audit_j1c_curriculum import library_sha


class O8Tests(unittest.TestCase):
    def test_labels_partition_and_agree_with_scorer(self):
        seen_a, seen_b = set(), set()
        for k in range(22):
            for nb in range(22):
                for h in range(9):
                    self.assertEqual(o8.label_a(k, nb, h), scorer.label_a(k, nb, h))
                    seen_a.add(o8.label_a(k, nb, h))
                for h in range(18):
                    self.assertEqual(o8.label_b(21 - k, nb, h), scorer.label_b(21 - k, nb, h))
                    seen_b.add(o8.label_b(21 - k, nb, h))
        self.assertEqual(seen_a, {'HARMS', 'PREVENTS', 'PARTIAL', 'NO_EFFECT'})
        self.assertEqual(seen_b, {'HARMS_B', 'REACHES_CEILING', 'IMPROVES', 'NO_EFFECT_B'})
        self.assertEqual((o8.label_a(14, 16, 2), o8.label_a(14, 15, 0), o8.label_a(13, 16, 0), o8.label_a(13, 15, 0),
                          o8.label_a(21, 21, 3)), ('PREVENTS', 'PARTIAL', 'PARTIAL', 'NO_EFFECT', 'HARMS'))
        self.assertEqual((o8.label_b(1, 16, 0), o8.label_b(2, 16, 0), o8.label_b(0, 15, 0), o8.label_b(0, 21, 3)),
                         ('REACHES_CEILING', 'IMPROVES', 'NO_EFFECT_B', 'HARMS_B'))

    def test_cells_and_constants_match_plan(self):
        self.assertEqual(len(o8.cells()), 42)
        self.assertEqual(o8.WORLDS, (20, 21, 22, 23, 24, 25, 26))
        self.assertEqual(tuple(scorer.WORLDS), o8.WORLDS)
        self.assertEqual(o8.cells()[:21], [('REROUTE_WAKE', w, s) for w in o8.WORLDS for s in o8.STREAMS])
        self.assertEqual((o8.MEMORY, o8.EXTRA_UPDATES, o8.THRESHOLD, o8.MIN_FREE_GIB), (64, 8192, 0.05, 4.5))
        self.assertEqual(o8.REFERENCE, scorer.RULES)

    def test_summary_counts_nan_as_failing_and_never_better(self):
        o3cells = {f'{ref}_w{w}_s{s}': {'terminal_median': 0.03 if (w, s) == (20, 0) else 0.2}
                   for ref in ('SHUFFLED', 'INTERLEAVED') for w in o8.WORLDS for s in o8.STREAMS}
        recs = {f'{a}_w{w}_s{s}': {'terminal_median': 0.01, 'terminal_stale': 3} for a, w, s in o8.cells()}
        summ = o8.summarize(recs, o3_cells=o3cells)
        self.assertEqual((summ['REROUTE_WAKE']['k'], summ['REROUTE_WAKE']['n_better'], summ['REROUTE_WAKE']['h']), (21, 21, 0))
        self.assertEqual(summ['REROUTE_WAKE']['label'], 'PREVENTS')
        self.assertEqual(summ['REROUTE_INTERLEAVED']['label'], 'REACHES_CEILING')
        recs['REROUTE_WAKE_w20_s0']['terminal_median'] = float('nan')
        summ = o8.summarize(recs, o3_cells=o3cells)
        self.assertEqual((summ['REROUTE_WAKE']['k'], summ['REROUTE_WAKE']['n_better'], summ['REROUTE_WAKE']['h']), (20, 20, 1))

    def test_rerouter_reservoir_matches_o2d_and_swap_takes(self):
        """On a saved terminal: the hook's reservoir is o2d's draw, re-routing earlier tasks installs the
        exhaustive route (hard_routes agrees), is idempotent, and does nothing when disabled."""
        w, s = 20, 0
        _, model, st, _ = o3.load_shuffled_terminal(o5.terminal_path('O3', w, s), w, s)
        pool = o2d.reservoir(st, w, s, o8.MEMORY)
        hook = o8.Rerouter(st, w, s, enabled=True)
        for i in range(12):
            hook.add_reservoir(i)
        by = {}
        for x, y, t in pool:
            by.setdefault(t, []).append(x)
        for i in range(12):
            np.testing.assert_array_equal(hook.by[st[i].task_id][0].numpy(),
                                          np.stack(by[st[i].task_id]).astype(np.float32))
        before = library_sha(model)
        changed = hook.reroute(model, 12)
        self.assertEqual(library_sha(model), before)   # the library is never modified by re-routing
        self.assertGreaterEqual(changed, 0)
        routes = model.hard_routes()
        from row.experiments.audit_so1r_route_only import FrozenLibrary
        from row.experiments.audit_j2a_staged_library import enum_route
        lib = FrozenLibrary(model)
        for i in range(12):
            d = hook.plan[st[i].task_id]
            lib.steps = d
            xs, ys = hook.by[st[i].task_id]
            self.assertEqual([int(r) for r in routes[st[i].task_id][:d]], [int(r) for r in enum_route(lib, xs, ys)])
        self.assertEqual(hook.reroute(model, 12), 0)   # idempotent
        off = o8.Rerouter(st, w, s, enabled=False)
        codes = {t.task_id: model.task_codes[t.task_id].detach().clone() for t in st[:12]}
        for i in range(12):
            off(model, i, i)
        self.assertEqual(off.changed_per_pass, [])
        for t in st[:12]:
            self.assertTrue(torch.equal(codes[t.task_id], model.task_codes[t.task_id].detach()))

    def test_validate_rejects_broken_constructions(self):
        base = {'arm': 'REROUTE_WAKE', 'world': 20, 'stream': 0, 'reroute_enabled': True, 'scored_tasks': 64,
                'terminal_per_task': {str(i): 0.01 for i in range(64)}, 'stream_tasks': 188, 'trained_tasks': 188,
                'route_lengths_match_plan': True, 'depth_histogram': o2.EXPECTED_DEPTHS['SHUFFLED'],
                'routes_checked': 188, 'reroute_passes': 187, 'routes_changed_per_pass': [0] * 187,
                'anchor_abs_error': 0.0, 'extra_updates': 0, 'terminal_stale': 5}
        o8.validate(dict(base))
        for bad in ({'reroute_enabled': False}, {'reroute_passes': 186}, {'anchor_abs_error': 1e-3},
                    {'extra_updates': 10}, {'trained_tasks': 187}):
            with self.assertRaises(ValueError):
                o8.validate(base | bad)
        ri = base | {'arm': 'REROUTE_INTERLEAVED', 'extra_updates': 8192, 'anchor_abs_error': 0.3}
        o8.validate(ri)
        with self.assertRaises(ValueError):
            o8.validate(ri | {'extra_updates': 8191})


if __name__ == '__main__':
    unittest.main()
