import unittest

from row.experiments import census_cf0_loop_structure as cf0


def rows(rate_by_size, seconds=1.0):
    out = []
    for (a, b) in cf0.SIZES:
        k = round(rate_by_size[(a, b)] * 24)
        for n_support in cf0.SUPPORTS:
            for j in range(24):
                out.append({'a': a, 'b': b, 'support': n_support, 'success': j < k, 'exact': j < k,
                            'seconds': seconds, 'evaluations': 10, 'query_nmse': 0.1})
    return out


class CF0Tests(unittest.TestCase):
    def test_all_pass(self):
        self.assertEqual(cf0.label(rows({s: 1.0 for s in cf0.SIZES}))[0], 'NO_SEARCH_NECESSITY')

    def test_largest_fails(self):
        r = {s: 1.0 for s in cf0.SIZES}
        r[cf0.SIZES[-1]] = 0.5
        self.assertEqual(cf0.label(rows(r))[0], 'SEARCH_BINDS')

    def test_slow_binds(self):
        self.assertEqual(cf0.label(rows({s: 1.0 for s in cf0.SIZES}, seconds=700))[0], 'SEARCH_BINDS')

    def test_graded(self):
        r = {s: 1.0 for s in cf0.SIZES}
        r[cf0.SIZES[-1]] = 0.75
        self.assertEqual(cf0.label(rows(r))[0], 'GRADED')


if __name__ == '__main__':
    unittest.main()
