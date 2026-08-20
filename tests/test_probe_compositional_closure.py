from __future__ import annotations

import unittest

import numpy as np
import torch

from row.experiments.probe_compositional_closure import (
    compose_teacher,
    saturation_fraction,
)
from row.experiments.score_v5_causal import crossing_of
from row.world import Primitive, WorldConfig


class CompositionalClosureHelpers(unittest.TestCase):
    def test_saturation_fraction(self) -> None:
        zeros = np.zeros((4, 3))
        ones = np.ones((4, 3))
        self.assertEqual(saturation_fraction(zeros), 0.0)
        self.assertEqual(saturation_fraction(ones), 1.0)

    def test_compose_teacher_is_sequential_application(self) -> None:
        primitive = Primitive.random(seed=0, primitive_index=0, d=4, rank=2, alpha=0.35)
        x = np.random.default_rng(0).normal(size=(8, 4))

        class _World:
            library = (primitive,)

        once = compose_teacher(_World, [0], x)
        twice = compose_teacher(_World, [0, 0], x)
        again = primitive(once)
        np.testing.assert_allclose(twice, again)

    def test_world_config_still_caps_tasks_at_program_count(self) -> None:
        with self.assertRaises(ValueError):
            WorldConfig(teacher_primitives=2, program_length=2, tasks=5)


class CrossingHelper(unittest.TestCase):
    def test_crossing_interpolates(self) -> None:
        rows = [{"H": 8, "c": 100.0}, {"H": 16, "c": 300.0}]
        # V_retain = c - 200 crosses between 8 and 16 at H=12
        self.assertAlmostEqual(crossing_of(rows, 200.0), 12.0)

    def test_no_crossing_when_always_negative(self) -> None:
        rows = [{"H": 8, "c": 10.0}, {"H": 16, "c": 20.0}]
        self.assertIsNone(crossing_of(rows, 100.0))
