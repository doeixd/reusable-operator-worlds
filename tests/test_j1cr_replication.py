import unittest

from row.experiments import audit_j1cr_replication as j1cr
from row.experiments.audit_j1c_curriculum import stage_setup


def staged(median, seed=j1cr.MODEL_SEED, carry=("sha1", "sha2")):
    def stage(k, sha, start):
        return {"terminal_median": median if k == 3 else 0.01, "shared_relative_change": 1.0,
                "code_relative_change": 1.0, "library_sha256": sha, "library_sha256_at_start": start,
                "final_per_task": {f"t{k}_{i}": 0.01 for i in range(4)}}
    return {"arm": "STAGED-R", "model_seed": seed, "terminal_median": median,
            "stages": {"1": stage(1, "sha1", None), "2": stage(2, "sha2", carry[0]),
                       "3": stage(3, "sha3", carry[1])}}


def control(median, seed=j1cr.MODEL_SEED):
    return {"arm": "NON-STAGED-R", "model_seed": seed, "terminal_median": median,
            "stages": {"3": {"terminal_median": median, "shared_relative_change": 1.0,
                             "code_relative_change": 1.0, "library_sha256": "x",
                             "final_per_task": {"a": median}}}}


class J1cRTests(unittest.TestCase):
    def test_second_initialization_reaches_the_model(self):
        cfg, _, _, _ = stage_setup(1, 3, j1cr.MODEL_SEED)
        self.assertEqual(cfg.discrete_model.seed, j1cr.MODEL_SEED)
        default, _, _, _ = stage_setup(1, 3)
        self.assertEqual(default.discrete_model.seed, 5000)

    def test_non_staged_budget_equals_the_staged_total(self):
        self.assertEqual(j1cr.TOTAL_UPDATES, 16384 + 16384 + 32768)

    def test_classification_ladder_and_guards(self):
        cells = {f"STAGED-R_w{w}": staged(m) for w, m in zip(j1cr.WORLDS, (0.006, 0.005, 0.007))}
        cells |= {f"NON-STAGED-R_w{w}": control(0.95) for w in j1cr.WORLDS}
        self.assertEqual(j1cr.classify(cells), "REPLICATES")
        self.assertEqual(j1cr.classify(dict(cells, **{"STAGED-R_w1": staged(0.005, seed=5000)})), "HARNESS_FAILED")
        self.assertEqual(j1cr.classify(dict(cells, **{"STAGED-R_w1": staged(0.005, carry=("sha1", "wrong"))})),
                         "HARNESS_FAILED")
        one = {f"STAGED-R_w{w}": staged(m) for w, m in zip(j1cr.WORLDS, (0.9, 0.005, 0.8))}
        one |= {f"NON-STAGED-R_w{w}": control(0.95) for w in j1cr.WORLDS}
        self.assertEqual(j1cr.classify(one), "PARTIAL")
        none = {f"STAGED-R_w{w}": staged(0.9) for w in j1cr.WORLDS}
        none |= {f"NON-STAGED-R_w{w}": control(0.95) for w in j1cr.WORLDS}
        self.assertEqual(j1cr.classify(none), "FAILS_TO_REPLICATE")


if __name__ == "__main__":
    unittest.main()
