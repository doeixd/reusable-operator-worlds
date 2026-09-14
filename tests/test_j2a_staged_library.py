import unittest

import numpy as np

from row.experiments import audit_j2a_staged_library as j2a
from row.experiments.audit_j1c_curriculum import stage_setup


def cell(held=0.01, trained=0.01, random_trained=2.0, bitwise=True, finite=True):
    return {"trained": {"as_trained": trained, "enum": trained, "opt": trained, "random": random_trained},
            "held_out": {"enum": held, "random": 2.0}, "held_out_below_threshold": 64,
            "export_ratio": held / trained, "g": 0.0, "as_trained_bitwise": bitwise,
            "enum_equals_as_trained_route": 0.9, "finite": finite}


def cells(staged_held):
    out = {f"{n}_w{w}": cell(held=h) for n, hs in staged_held.items() for w, h in zip(j2a.WORLDS, hs)}
    out |= {f"{n}_w{w}": cell(held=1.5) for n in ("NONSTAGED3001", "RESET5000") for w in j2a.WORLDS}
    return out


class J2ATests(unittest.TestCase):
    def test_held_out_programs_are_unseen_distinct_and_reproducible(self):
        cfg, world, _, _ = stage_setup(1, 3, 5000)
        tasks = j2a.held_out_tasks(cfg, world)
        trained = {tuple(int(p) for p in t.program.primitive_ids) for t in world.tasks}
        programs = [t["program"] for t in tasks]
        self.assertEqual(len(programs), 64)
        self.assertEqual(len(set(programs)), 64)
        self.assertFalse(set(programs) & trained)
        again = j2a.held_out_tasks(cfg, world)
        self.assertEqual([t["program"] for t in again], programs)
        self.assertTrue(np.array_equal(again[0]["train_x"], tasks[0]["train_x"]))
        other, world2, _, _ = stage_setup(2, 3, 5000)
        self.assertNotEqual([t["program"] for t in j2a.held_out_tasks(other, world2)], programs)

    def test_held_out_targets_follow_the_teacher_program(self):
        cfg, world, _, _ = stage_setup(1, 3, 5000)
        task = j2a.held_out_tasks(cfg, world)[0]
        library = world.tasks[0].teacher_library
        expected = task["eval_x"]
        for primitive in task["program"]:
            expected = library[primitive](expected)
        self.assertTrue(np.allclose(task["eval_y"], expected))

    def test_classification_ladder(self):
        good = {"STAGED5000": (0.01, 0.01, 0.01), "STAGED3001": (0.01, 0.01, 0.01)}
        self.assertEqual(j2a.classify(cells(good)), "EXPORTS")
        one_bad = {"STAGED5000": (0.01, 0.01, 0.9), "STAGED3001": (0.01, 0.01, 0.01)}
        self.assertEqual(j2a.classify(cells(one_bad)), "EXPORTS")  # 5 of 6 still export
        three = {"STAGED5000": (0.01, 0.9, 0.9), "STAGED3001": (0.01, 0.01, 0.9)}
        self.assertEqual(j2a.classify(cells(three)), "EXPORTS_WEAKLY")
        none = {"STAGED5000": (0.9, 0.9, 0.9), "STAGED3001": (0.9, 0.9, 0.9)}
        self.assertEqual(j2a.classify(cells(none)), "DOES_NOT_EXPORT")

    def test_harness_guards(self):
        good = {"STAGED5000": (0.01, 0.01, 0.01), "STAGED3001": (0.01, 0.01, 0.01)}
        broken = cells(good)
        broken["STAGED5000_w1"] = cell(bitwise=False)
        self.assertEqual(j2a.classify(broken), "HARNESS_FAILED")
        floor = cells(good)
        floor["STAGED3001_w2"] = cell(random_trained=0.001)  # random beats search
        self.assertEqual(j2a.classify(floor), "HARNESS_FAILED")
        self.assertEqual(j2a.classify({k: v for k, v in cells(good).items() if k != "RESET5000_w0"}),
                         "HARNESS_FAILED")
        # An export ratio above the registered bound fails even below threshold.
        ratio = cells(good)
        ratio["STAGED5000_w0"] = cell(held=0.045, trained=0.002)
        ratio["STAGED5000_w1"] = cell(held=0.045, trained=0.002)
        self.assertEqual(j2a.classify(ratio), "EXPORTS_WEAKLY")


if __name__ == "__main__":
    unittest.main()
