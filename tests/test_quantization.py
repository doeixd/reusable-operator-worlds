import unittest

import torch

from row.experiments.quantize_artifact import symmetric_int8_dequantize


class QuantizationTests(unittest.TestCase):
    def test_symmetric_quantization_is_bounded_and_preserves_zero(self) -> None:
        values = torch.tensor([-2.0, -0.25, 0.0, 0.75, 2.0])
        restored, scale = symmetric_int8_dequantize(values)
        self.assertGreater(scale, 0.0)
        self.assertEqual(float(restored[2]), 0.0)
        self.assertLessEqual(float(torch.max(torch.abs(restored - values))), scale / 2 + 1e-7)


if __name__ == "__main__":
    unittest.main()
