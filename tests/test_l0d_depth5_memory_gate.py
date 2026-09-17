import unittest
import torch
from row.experiments.l0d_depth4_execution_gate import DepthLibrary
from row.experiments.l0d_depth5_memory_gate import BLOCK, chunked_mse
from row.config import load_config
from row.experiments.audit_rotated_g5r_interference import world_config
from row.experiments.audit_so1_budget_bracket import build_fast

class Depth5MemoryTests(unittest.TestCase):
    def test_expected_block_count_and_lower_bound(self):
        self.assertEqual((12**5 + BLOCK - 1)//BLOCK, 243)
        self.assertEqual(128*12**5*16*4, 2038431744)

    def test_chunked_route_losses_match_complete_depth_three_vector(self):
        cfg = world_config(load_config('configs/v1.yaml'), 0)
        model = build_fast(cfg)
        library = DepthLibrary(model)
        x = torch.randn(4, 16, generator=torch.Generator().manual_seed(4))
        y = torch.randn(4, 16, generator=torch.Generator().manual_seed(5))
        expected = library.all_route_support_mse_depth(x, y, 3)
        actual, blocks, largest = chunked_mse(library, x, y, 3, 37)
        self.assertEqual(blocks, (12 ** 3 + 36) // 37)
        self.assertTrue(torch.allclose(expected, actual, rtol=1e-6, atol=1e-6))
        self.assertEqual(int(torch.argmin(expected)), int(torch.argmin(actual)))
        self.assertGreater(largest, 0)

if __name__=='__main__':unittest.main()
