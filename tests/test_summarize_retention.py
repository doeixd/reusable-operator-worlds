from __future__ import annotations

import unittest

from row.experiments.summarize_retention import _aggregate


class SummarizeRetentionTests(unittest.TestCase):
    def test_aggregate_requires_constant_accounting(self) -> None:
        rows = [
            {
                "world_seed": 0,
                "total_retained_bits": 100,
                "shared_weight_bits": 80,
                "task_state_bits": 20,
                "inference_multiply_adds": 10,
                "quantized_minus_float_nmse_mean": 1e-5,
                "maximum_task_nmse_increase": 2e-5,
                "float_final_nmse_mean": 0.01,
                "quantized_final_nmse_mean": 0.01001,
            },
            {
                "world_seed": 1,
                "total_retained_bits": 101,
                "shared_weight_bits": 80,
                "task_state_bits": 21,
                "inference_multiply_adds": 10,
                "quantized_minus_float_nmse_mean": 1e-5,
                "maximum_task_nmse_increase": 2e-5,
                "float_final_nmse_mean": 0.01,
                "quantized_final_nmse_mean": 0.01001,
            },
        ]
        with self.assertRaisesRegex(ValueError, "changed across worlds"):
            _aggregate("model", rows)


if __name__ == "__main__":
    unittest.main()
