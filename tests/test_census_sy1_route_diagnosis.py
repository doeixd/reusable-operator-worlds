import unittest

from row.experiments import census_sy1_route_diagnosis as sy1


def task(depth, hard, sf, solved):
    return {'depth': depth, 'hard': hard, 'search_failure': sf,
            'polish': {'solved': solved, 'evaluations': 10}}


class SY1Tests(unittest.TestCase):
    def label(self, tasks):
        return sy1.summarize(tasks)[1]['label']

    def test_all_search_repairable(self):
        self.assertEqual(self.label([task(8, True, True, True)] * 10),
                         'SEARCH_FAILURE_LOCALLY_REPAIRABLE')

    def test_search_not_repairable(self):
        self.assertEqual(self.label([task(8, True, True, False)] * 10),
                         'SEARCH_FAILURE_NOT_LOCALLY_REPAIRABLE')

    def test_selection(self):
        self.assertEqual(self.label([task(9, True, False, False)] * 10), 'SELECTION_FAILURE')

    def test_mixed(self):
        ts = [task(9, True, True, False)] * 5 + [task(9, True, False, False)] * 5
        self.assertEqual(self.label(ts), 'MIXED')

    def test_shallow_and_easy_tasks_ignored(self):
        ts = [task(5, True, False, False)] * 10 + [task(8, False, False, False)] * 10 \
            + [task(8, True, True, True)] * 2
        self.assertEqual(self.label(ts), 'SEARCH_FAILURE_LOCALLY_REPAIRABLE')

    def test_polish_descends(self):
        import torch
        lib = [lambda z, a=a: z + a for a in (0.0, 1.0, 2.0)]
        x = torch.zeros(4, 1)
        y = torch.full((4, 1), 3.0)
        route, best, _, _ = sy1.polish(lib, x, y, [0, 0], 3)
        self.assertEqual(best, 0.0)


if __name__ == '__main__':
    unittest.main()
