import unittest

from row.experiments import audit_so3_consolidation as so3

GATES_OK = {"G0_replay_seed_neutral": True, "G1_prefix_sharing": True, "G2_non_vacuity": True,
            "G3_anchor": True, "G4_transfer": True}


def medians(base, lr, store):
    """Each argument: {world: [stream0, stream1, stream2]}."""
    return {arm: {w: dict(enumerate(values[w])) for w in so3.WORLDS}
            for arm, values in (("BASE", base), ("LR_HALF", lr), ("STORE_8", store))}


def flat(value):
    return {w: [value] * 3 for w in so3.WORLDS}


class LadderTests(unittest.TestCase):
    def test_store_passes_and_is_stream_robust(self):
        base = {w: [0.10, 0.12, 0.09] for w in so3.WORLDS}
        labels = so3.classify(medians(base, flat(0.2), flat(0.03)), GATES_OK)
        self.assertEqual(labels["STORE_8"], "STORE_8_PASSES")
        self.assertTrue(labels["STORE_8_STREAM_ROBUST"])
        self.assertEqual(labels["LR_HALF"], "LR_HALF_FAILS")
        self.assertEqual(labels["program"], "SO3_PASSES")

    def test_store_pass_within_base_stream_spread_is_confounded(self):
        # BASE has one lucky stream at 0.02 in every world: STORE_8 at 0.04 passes but is not below BASE's minimum.
        base = {w: [0.10, 0.02, 0.12] for w in so3.WORLDS}
        labels = so3.classify(medians(base, flat(0.2), flat(0.04)), GATES_OK)
        self.assertEqual(labels["STORE_8"], "STORE_8_STREAM_CONFOUNDED")
        self.assertEqual(labels["program"], "SO3_PARTIAL")

    def test_lr_pass_needs_beating_base_in_the_same_worlds(self):
        base = {w: [0.04, 0.04, 0.04] for w in so3.WORLDS}  # BASE itself passes everywhere
        labels = so3.classify(medians(base, flat(0.045), flat(0.2)), GATES_OK)
        self.assertEqual(labels["LR_HALF"], "LR_HALF_FAILS")
        self.assertEqual(labels["base_world_passes"], 3)

    def test_partial_and_fails(self):
        base = flat(0.2)
        labels = so3.classify(medians(base, flat(0.09), flat(0.15)), GATES_OK)
        self.assertEqual(labels["LR_HALF"], "LR_HALF_PARTIAL")
        self.assertEqual(labels["STORE_8"], "STORE_8_FAILS")
        self.assertEqual(labels["program"], "SO3_PARTIAL")
        self.assertEqual(so3.classify(medians(base, flat(0.19), flat(0.18)), GATES_OK)["program"], "SO3_FAILS")

    def test_two_of_three_worlds_suffice_and_median_is_over_streams(self):
        base = flat(0.2)
        store = {3: [0.01, 0.02, 0.9], 4: [0.03, 0.9, 0.04], 5: [0.9, 0.9, 0.01]}  # medians 0.02, 0.04, 0.9
        self.assertEqual(so3.classify(medians(base, flat(0.3), store), GATES_OK)["STORE_8"], "STORE_8_PASSES")

    def test_harness_failed_on_gate_or_missing_cell(self):
        m = medians(flat(0.2), flat(0.01), flat(0.01))
        self.assertEqual(so3.classify(m, GATES_OK | {"G1_prefix_sharing": False})["program"], "HARNESS_FAILED")
        del m["STORE_8"][5][2]
        self.assertEqual(so3.classify(m, GATES_OK)["program"], "HARNESS_FAILED")


class ConstructionTests(unittest.TestCase):
    def test_stream_seeds(self):
        self.assertIsNone(so3.replay_seed_for(3, 0))
        seeds = {so3.replay_seed_for(w, s) for w in so3.WORLDS for s in (1, 2)}
        self.assertEqual(len(seeds), 6)
        self.assertEqual(so3.replay_seed_for(4, 2), so3.replay_seed_for(4, 2))

    def test_arm_configs_change_one_field(self):
        base, _, _, _ = so3.stage_setup(3, 3, so3.MODEL_SEED)
        self.assertEqual(base.discrete_model.seed, 6000)
        self.assertEqual(so3.changed_fields(base, so3.arm_config(base, "BASE")), [])
        self.assertEqual(so3.changed_fields(base, so3.arm_config(base, "LR_HALF")), ["discrete_model.global_learning_rate"])
        self.assertEqual(so3.changed_fields(base, so3.arm_config(base, "STORE_8")),
                         ["discrete_model.replay_examples_per_task"])
        self.assertEqual(so3.arm_config(base, "STORE_8").discrete_model.replay_examples_per_task, 8)


if __name__ == "__main__":
    unittest.main()
