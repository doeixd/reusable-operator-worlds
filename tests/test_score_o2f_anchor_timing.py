import unittest

from row.experiments import score_o2f_anchor_timing as o2f


class LabelTests(unittest.TestCase):
    def test_labels_partition_as_registered(self):
        self.assertEqual(o2f.label(4, 4), 'UNMEASURABLE_WORLD_DRIVEN')
        self.assertEqual(o2f.label(7, 6), 'EARLY_DOSE_PREDICTS')    # 0.857 >= 0.85
        self.assertEqual(o2f.label(7, 5), 'INCONCLUSIVE')           # 0.714
        self.assertEqual(o2f.label(10, 6), 'NO_TIMING_EFFECT')      # 0.60 <= 0.60
        for n in range(15):
            for a in range(n + 1):
                self.assertIn(o2f.label(n, a), ('UNMEASURABLE_WORLD_DRIVEN', 'EARLY_DOSE_PREDICTS',
                                                'NO_TIMING_EFFECT', 'INCONCLUSIVE'))

    def test_registered_plan_hash_verifies(self):
        self.assertEqual(o2f.sha(o2f.PLAN), o2f.PLAN_SHA256)

    def test_predictors_rederive_from_order_seeds(self):
        import json
        registered = json.loads(o2f.PREDICTORS.read_text())['cells']
        E = {(c['arm'], c['world'], c['stream']): c['E_len1_in_first32'] for c in registered}
        self.assertEqual(o2f.recompute_predictors(), E)


if __name__ == '__main__':
    unittest.main()
