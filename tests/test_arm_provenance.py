"""The E5 scratch-arm confusion, as a regression test."""
import copy
import unittest

import torch
from torch import nn

from row.arm_provenance import arms_differ, assert_arm, describe_arm, tensor_digest


def _model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))


class TestArmProvenance(unittest.TestCase):
    def test_fresh_and_copied_arms_are_distinguishable(self):
        """The exact E5 bug: a deep copy of a trained model labelled 'scratch'."""
        trained = _model(1)
        with torch.no_grad():
            for p in trained.parameters():
                p.add_(1.0)                      # stand in for "this model was trained"
        fresh = _model(2)
        a = describe_arm("S", fresh, init_source="fresh", steps=10, fresh_seed=2)
        b = describe_arm("S", copy.deepcopy(trained), init_source="copy:trained", steps=10)
        self.assertNotEqual(a["checkpoint_hash"], b["checkpoint_hash"])
        self.assertIn("init_source", arms_differ(a, b))

    def test_assert_arm_fails_closed_on_wrong_construction(self):
        record = describe_arm("S", _model(3), init_source="copy:trained", steps=10)
        with self.assertRaises(SystemExit):
            assert_arm(record, init_source="fresh")

    def test_assert_arm_passes_on_the_registered_construction(self):
        record = describe_arm("S", _model(3), init_source="fresh", steps=10)
        assert_arm(record, init_source="fresh", steps=10)

    def test_rejects_unknown_init_source(self):
        with self.assertRaises(ValueError):
            describe_arm("S", _model(3), init_source="somehow", steps=1)

    def test_digest_is_stable_and_order_independent(self):
        self.assertEqual(tensor_digest(_model(5)), tensor_digest(_model(5)))
        self.assertNotEqual(tensor_digest(_model(5)), tensor_digest(_model(6)))

    def test_trainable_and_frozen_counts_partition_the_model(self):
        model = _model(7)
        for p in model[0].parameters():
            p.requires_grad_(False)
        record = describe_arm("F", model, init_source="trained", steps=1)
        total = sum(p.numel() for p in model.parameters())
        self.assertEqual(record["trainable_parameters"] + record["frozen_parameters"], total)
        self.assertGreater(record["frozen_parameters"], 0)

    def test_identical_arms_report_no_differences(self):
        a = describe_arm("O", _model(9), init_source="trained", steps=5)
        b = describe_arm("O", _model(9), init_source="trained", steps=5)
        self.assertEqual(arms_differ(a, b), [])


if __name__ == "__main__":
    unittest.main()
