import unittest

import torch

from row.config import load_config
from row.experiments import audit_so1_anchor_diagnostic as diag
from row.experiments import score_so1_anchor_diagnostic as scorer
from row.experiments.audit_rotated_g5r_interference import world_config


def cells_from(medians_by_world):
    """medians_by_world[w] = dict of cell key -> terminal median."""
    return {k: {str(w): {"terminal_median": m[k]} for w, m in medians_by_world.items()}
            for k in diag.CELLS}


def stage_d(values):
    return {"cells": {"C_lo": {str(w): {"terminal_median": v} for w, v in values.items()}}}


def world_row(sd, r=None, p=None, f0=None, s100=0.8, f100=0.8, extra=(0.8, 0.8, 0.8)):
    r = sd if r is None else r
    return {"R": r, "P": r if p is None else p, "F0": sd if f0 is None else f0,
            "S100": s100, "F100": f100, "F120": extra[0], "F121": extra[1], "F122": extra[2]}


class ClassificationTests(unittest.TestCase):
    SD = {0: 0.90, 1: 0.79, 2: 0.90}

    def run_both(self, worlds):
        cells = cells_from(worlds)
        q = diag.quantities(cells, stage_d(self.SD))
        rows = [q["per_world"][str(w)] for w in (0, 1, 2)]
        self.assertEqual(q["classification"], scorer.classify(rows))
        return q

    def test_equivalent(self):
        q = self.run_both({w: world_row(sd) for w, sd in self.SD.items()})
        self.assertEqual(q["classification"], "IMPLEMENTATION_EQUIVALENT")

    def test_harness_failed_on_imperfect_reproduction(self):
        worlds = {w: world_row(sd) for w, sd in self.SD.items()}
        worlds[1]["R"] = self.SD[1] + 2e-6
        self.assertEqual(self.run_both(worlds)["classification"], "HARNESS_FAILED")

    def test_trajectory_sensitive(self):
        worlds = {w: world_row(sd) for w, sd in self.SD.items()}
        worlds[2]["F0"] = self.SD[2] - 0.15
        worlds[2]["P"] = self.SD[2] + 0.05
        self.assertEqual(self.run_both(worlds)["classification"], "TRAJECTORY_SENSITIVE")

    def test_divergent_when_perturbation_is_stable(self):
        worlds = {w: world_row(sd) for w, sd in self.SD.items()}
        worlds[2]["F0"] = self.SD[2] - 0.15
        worlds[1]["F100"], worlds[1]["P"] = 0.9, self.SD[1] + 0.05
        # world 2 misses with a stable perturbation -> divergence
        self.assertEqual(self.run_both(worlds)["classification"], "IMPLEMENTATION_DIVERGES")

    def test_matched_stream_100_miss_counts(self):
        worlds = {w: world_row(sd) for w, sd in self.SD.items()}
        worlds[0]["F100"] = 0.85  # vs S100 0.8
        self.assertEqual(self.run_both(worlds)["classification"], "IMPLEMENTATION_DIVERGES")


class PerturbationTests(unittest.TestCase):
    def test_eps_zero_is_bitwise_and_eps_positive_moves_only_shared(self):
        cfg = world_config(load_config("configs/v1.yaml"), 0)
        for kind in ("sequential", "fast"):
            base = diag.BUILDERS[kind](cfg)
            same = diag.builder_for(kind, 0.0, 0)(cfg)
            moved = diag.builder_for(kind, diag.EPS, 0)(cfg)
            self.assertTrue(all(torch.equal(a, b) for a, b in
                                zip(base.state_dict().values(), same.state_dict().values())))
            diffs = [float((a - b).abs().max()) for a, b in
                     zip(base.shared_parameters(), moved.shared_parameters())]
            self.assertTrue(any(d > 0 for d in diffs))
            self.assertTrue(all(d < 1e-5 for d in diffs))


if __name__ == "__main__":
    unittest.main()
