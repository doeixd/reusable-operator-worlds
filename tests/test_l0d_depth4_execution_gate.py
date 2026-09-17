import unittest
from row.experiments.audit_so1r_route_only import unflatten

class Depth4Tests(unittest.TestCase):
    def test_depth_four_terminal_tensor_shape_and_route_index(self):
        self.assertEqual(128 * 12**4 * 16 * 4, 169869312)
        index = (((1 * 12) + 2) * 12 + 3) * 12 + 4
        self.assertEqual(unflatten(index, 12, 4), [1, 2, 3, 4])

if __name__ == '__main__': unittest.main()
