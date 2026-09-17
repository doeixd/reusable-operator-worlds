import unittest
from row.experiments.l0d_depth5_memory_gate import BLOCK

class Depth5MemoryTests(unittest.TestCase):
    def test_expected_block_count_and_lower_bound(self):
        self.assertEqual((12**5 + BLOCK - 1)//BLOCK, 243)
        self.assertEqual(128*12**5*16*4, 2038431744)

if __name__=='__main__':unittest.main()
