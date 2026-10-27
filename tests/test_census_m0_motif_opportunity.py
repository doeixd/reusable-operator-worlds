import unittest

import numpy as np

from row.experiments import census_m0_motif_opportunity as m0


def cells(rates):
    out = []
    for g in m0.GENERATORS:
        for s in m0.WORLDS:
            deep = []
            for j in range(m0.N_DEEP):
                row = {}
                for arm in ('MINED', 'RANDOM', 'ORACLE', 'INCUMBENT'):
                    if g == 'UNIFORM' and arm in ('ORACLE', 'INCUMBENT'):
                        continue
                    k = round(rates[f'{g}_{arm}'] * m0.N_DEEP)
                    row[arm] = {'success': j < k}
                deep.append(row)
            out.append({'generator': g, 'world': s, 'deep': deep})
    return out


BASE = {'MOTIF_MINED': 1.0, 'MOTIF_RANDOM': 0.0, 'MOTIF_ORACLE': 1.0, 'MOTIF_INCUMBENT': 0.0,
        'UNIFORM_MINED': 0.0, 'UNIFORM_RANDOM': 0.0}


class M0Tests(unittest.TestCase):
    def test_opportunity(self):
        self.assertEqual(m0.label(cells(BASE))[0], 'MOTIF_OPPORTUNITY')

    def test_oracle_invalid(self):
        self.assertEqual(m0.label(cells({**BASE, 'MOTIF_ORACLE': 11 / 12}))[0], 'INVALID_ORACLE')

    def test_construct_fails(self):
        for k, v in (('MOTIF_MINED', 10 / 12), ('MOTIF_RANDOM', 3 / 12), ('UNIFORM_MINED', 3 / 12)):
            self.assertEqual(m0.label(cells({**BASE, k: v}))[0], 'CONSTRUCT_FAILS', k)

    def test_incumbent_bands(self):
        self.assertEqual(m0.label(cells({**BASE, 'MOTIF_INCUMBENT': 11 / 12}))[0], 'NO_SEARCH_NECESSITY')
        self.assertEqual(m0.label(cells({**BASE, 'MOTIF_INCUMBENT': 8 / 12}))[0], 'GRADED')
        self.assertEqual(m0.label(cells({**BASE, 'MOTIF_INCUMBENT': 6 / 12}))[0], 'MOTIF_OPPORTUNITY')

    def test_incomplete_refused(self):
        c = cells(BASE)[:-1]
        with self.assertRaises(SystemExit):
            m0.label(c)

    def test_generator_shapes_and_null(self):
        motifs, shallow, deep = m0.make_programs(80, 'MOTIF', 6)
        self.assertEqual(len(set(motifs)), m0.K_MOTIFS)
        self.assertTrue(all(len(set(m)) > 1 for m in motifs))
        self.assertTrue(all(any(tuple(p[j:j + 3]) in motifs for j in range(len(p) - 2)) for p in shallow))
        self.assertTrue(all(len(p) == 9 and all(tuple(p[j:j + 3]) in motifs for j in (0, 3, 6)) for p in deep))
        _, u_shallow, u_deep = m0.make_programs(80, 'UNIFORM', 6)
        self.assertEqual([len(p) for p in u_shallow], [len(p) for p in shallow])
        self.assertNotEqual(u_deep, deep)

    def test_mining_counts_threshold(self):
        alphabet, _ = m0.mine([(0, 1, 2)] * 3 + [(3, 4, 5)] * 2)
        self.assertEqual(alphabet, [(0, 1, 2)])

    def test_macro_search_recovers_planted(self):
        w = m0.cf0.world(80)
        motifs, _, deep = m0.make_programs(80, 'MOTIF', 6)
        xs, ys, _, _ = m0.task_data(w, 80, 'MOTIF', 1, 0, deep[0])
        r = m0.macro_search(w.library, motifs, xs, ys)
        self.assertTrue(r['success'])
        self.assertEqual(tuple(r['found']), deep[0])
        self.assertEqual(r['enum_evaluations'], m0.K_MOTIFS ** 3)

    def test_discrimination_monotone(self):
        d = m0.discrimination(draws=20000)
        self.assertGreater(d['mined_pass_ge_0.9']['0.99'], d['mined_pass_ge_0.9']['0.8'])
        self.assertLess(d['incumbent_binds_le_0.5']['0.7'], 0.05)


if __name__ == '__main__':
    unittest.main()
