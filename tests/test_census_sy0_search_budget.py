import unittest

from row.experiments import census_sy0_search_budget as sy0


def rec(o, opt):
    r = {'O': {'nmse': o}}
    r.update({f'OPT{k}': {'nmse': v} for k, v in zip(sy0.BUDGETS, opt)})
    return r


class SY0Tests(unittest.TestCase):
    def test_solved_uses_gap_per_task(self):
        self.assertEqual(sy0.solved_budgets(rec(0.01, [0.1, 0.0116, 0.0117, 0.009])), [500, 2000])
        self.assertEqual(sy0.solved_budgets(rec(0.01, [1, 1, 1, 1])), [])

    def test_label_flat_is_no_opportunity(self):
        self.assertEqual(sy0.label({d: 0.3 for d in sy0.DEPTHS})[0], 'NO_OPPORTUNITY')

    def test_label_rising_fires(self):
        u = {d: 0.0 if d < 7 else 0.6 for d in sy0.DEPTHS}
        self.assertEqual(sy0.label(u)[0], 'OPPORTUNITY')

    def test_label_rising_but_low_level_does_not_fire(self):
        u = {d: 0.0 if d < 9 else 0.25 for d in sy0.DEPTHS}
        self.assertEqual(sy0.label(u)[0], 'NO_OPPORTUNITY')

    def test_missing_record_fails_closed(self):
        from pathlib import Path
        with self.assertRaises(SystemExit):
            sy0.load_records(Path('reports/does_not_exist'))


if __name__ == '__main__':
    unittest.main()
