import math
import unittest
from dataclasses import replace

from row.config import load_config
from row.curriculum_world import CurriculumWorldConfig, generate_curriculum_world
from row.experiments import audit_so2_online_gate as so2
from row.experiments.learned_lifetime import _adapt_novel_composition, _novel_composition_available
from row.rotated_world import generate_rotated_world


def cell(arm, median, margin, carry=("sha1", "sha2")):
    stages = {"3": {"library_sha256": "sha3", "library_sha256_at_start": carry[1] if arm == "STAGED" else None}}
    if arm == "STAGED":
        stages |= {"1": {"library_sha256": "sha1", "library_sha256_at_start": None},
                   "2": {"library_sha256": "sha2", "library_sha256_at_start": carry[0]}}
    return {"arm": arm, "model_seed": so2.MODEL_SEED, "terminal_median": median,
            "stages": stages, "margin": {"margin": margin}}


def cells(staged_medians, staged_margins):
    out = {f"STAGED_w{w}": cell("STAGED", m, g) for w, m, g in zip(so2.WORLDS, staged_medians, staged_margins)}
    out |= {f"PLAIN_w{w}": cell("PLAIN", 0.95, 0.1) for w in so2.WORLDS}
    return out


class SO2Tests(unittest.TestCase):
    def test_probe_reports_unavailable_when_every_program_is_used(self):
        base = load_config("configs/v1.yaml")
        cfg = replace(base, world=CurriculumWorldConfig.from_world(
            replace(base.world, seed=1), program_length=1, tasks=60, stage="length-1"))
        world = generate_curriculum_world(cfg.world)
        self.assertFalse(_novel_composition_available(world, cfg))
        result = _adapt_novel_composition(None, world, cfg, 0.05)  # must not touch the model
        self.assertFalse(result["available"])
        # The canonical world keeps its unseen programs, so the probe stays available there.
        canonical = replace(base, world=replace(base.world, seed=1))
        self.assertTrue(_novel_composition_available(generate_rotated_world(canonical.world), canonical))

    def test_margin_is_natural_log_of_geometric_means(self):
        rows = [{"trained": 0.01, "scratch": 0.1}, {"trained": 0.04, "scratch": 0.4}]
        margin = math.log(so2.geo([r["scratch"] for r in rows])) - math.log(so2.geo([r["trained"] for r in rows]))
        self.assertAlmostEqual(margin, math.log(10), places=12)
        self.assertGreater(margin, so2.MARGIN)

    def test_classification_ladder_and_transfer_guards(self):
        self.assertEqual(so2.classify(cells((0.7, 0.01, 0.02), (0.2, 1.0, 1.1))), "SO2_PASSES")
        self.assertEqual(so2.classify(cells((0.7, 0.01, 0.02), (0.2, 0.5, 0.6))), "SO2_ACQUIRES_ONLY")
        self.assertEqual(so2.classify(cells((0.7, 0.8, 0.9), (1.0, 1.0, 1.0))), "SO2_FAILS")
        broken = cells((0.01, 0.01, 0.01), (1.0, 1.0, 1.0))
        broken["STAGED_w1"] = cell("STAGED", 0.01, 1.0, carry=("sha1", "wrong"))
        self.assertEqual(so2.classify(broken), "HARNESS_FAILED")
        leaky = cells((0.01, 0.01, 0.01), (1.0, 1.0, 1.0))
        leaky["PLAIN_w1"]["stages"]["3"]["library_sha256_at_start"] = "sha2"
        self.assertEqual(so2.classify(leaky), "HARNESS_FAILED")

    def test_stage_worlds_are_the_registered_ones(self):
        lengths, counts = [], []
        for stage in (1, 2, 3):
            cfg, world, _, _ = so2.stage_setup(1, stage, so2.MODEL_SEED)
            lengths.append(cfg.discrete_model.task_steps)
            counts.append(len(world.tasks))
        self.assertEqual(lengths, [1, 2, 3])
        self.assertEqual(counts, [60, 64, 64])


if __name__ == "__main__":
    unittest.main()
