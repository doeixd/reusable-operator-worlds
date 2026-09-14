"""Exact calibration checks, with no ROW imports, model loads or training."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('h28_t0', Path(__file__).resolve().parents[1] / 'tools/h28_t0.py')
h28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h28)


class ExactControlTests(unittest.TestCase):
    def test_fair_bit_entropy(self):
        rows = [({'y': 0}, .5), ({'y': 1}, .5)]
        self.assertAlmostEqual(h28.entropy_given(rows, ()), math.log(2))
        self.assertEqual(h28.entropy_given(rows, ('y',)), 0)

    def test_invalid_joint_mass(self):
        for probabilities in ([], [0.4, 0.4], [-.1, 1.1], [float('nan'), .5]):
            with self.assertRaises(ValueError):
                h28.entropy_given([({'y': i}, p) for i, p in enumerate(probabilities)], ())

    def test_grouping_duplicate_rows_preserves_entropy(self):
        rows = h28.bit_rows('closed')
        split = [(row, p/2) for row, p in rows for _ in range(2)]
        self.assertAlmostEqual(h28.entropy_given(rows, ('z',)), h28.entropy_given(split, ('z',)))

    def test_closed_has_predictive_signal(self):
        result = h28.gaps(h28.bit_rows('closed'))
        noise_entropy = -.1*math.log(.1)-.9*math.log(.9)
        self.assertAlmostEqual(result['macro_entropy'], noise_entropy)
        self.assertGreater(result['predictive_gain'], .3)
        self.assertTrue(h28.nontrivial_gate(result, True))

    def test_joint_only_leakage_is_not_missed(self):
        result = h28.gaps(h28.bit_rows('joint'))
        for got, wanted in zip(result['gaps'], [0, 0, math.log(2)]):
            self.assertAlmostEqual(got, wanted)
        self.assertFalse(h28.nontrivial_gate(result, True))

    def test_micro_and_world_controls_are_distinct(self):
        self.assertAlmostEqual(h28.gaps(h28.bit_rows('micro'))['gaps'][0], math.log(2))
        self.assertAlmostEqual(h28.gaps(h28.bit_rows('micro'))['gaps'][1], 0)
        self.assertAlmostEqual(h28.gaps(h28.bit_rows('world'))['gaps'][0], 0)
        self.assertAlmostEqual(h28.gaps(h28.bit_rows('world'))['gaps'][1], math.log(2))

    def test_weak_probe_false_zero_exposed(self):
        rows = h28.bit_rows('joint')
        self.assertEqual(h28.gaps(rows, weak=True)['gaps'], [0, 0, 0])
        self.assertGreater(h28.gaps(rows)['gaps'][2], .6)

    def test_trivial_encoders_rejected(self):
        self.assertFalse(h28.nontrivial_gate(h28.gaps(h28.bit_rows('constant')), True))
        self.assertFalse(h28.nontrivial_gate(h28.gaps(h28.bit_rows('closed')), False))

    def test_rotation_and_permutation(self):
        for r in (.1, .5, .8, .99, -.8):
            self.assertAlmostEqual(h28.gaussian_psi(r, 0), 0)
            self.assertAlmostEqual(h28.gaussian_psi(r, math.pi/2), 0)
            expected = -.5*math.log(1-r*r)+math.log(1-r*r/2)
            self.assertAlmostEqual(h28.gaussian_psi(r, math.pi/4), expected)
            self.assertGreater(expected, 0)

    def test_singular_gaussian_rejected(self):
        for r in (0, 1, -1, float('nan')):
            with self.assertRaises(ValueError):
                h28.gaussian_psi(r, .5)

    def test_seed_parser_fails_closed(self):
        self.assertEqual(h28.saved_seed('world:\n  seed: 0\nmodel:\n  seed: 100\n'), 0)
        self.assertEqual(h28.saved_seed('world:\n  seed: 600\n'), 600)
        for config in ('world: {seed: 0}', 'world:\n  seed: 0\n  seed: 1',
                       'world:\n  seed: 0\nworld:\n  seed: 0', 'model:\n  seed: 0'):
            with self.assertRaises(ValueError):
                h28.saved_seed(config)

    def test_inventory_checks_prior_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = 'artifacts/example/cells/STAGED_w0/stage3'
            stage = root / relative
            stage.mkdir(parents=True)
            (stage/'config.yaml').write_text('world:\n  seed: 0\n')
            (stage/'model.pt').write_bytes(b'opaque binary; never deserialized')
            record = {'stamp': {'world': 0},
                      'artifact_sha256': {'stage3/model.pt': h28.sha256(stage/'model.pt')}}
            (stage.parent/'result.json').write_text(json.dumps(record))
            with patch.object(h28, 'SOURCES', (relative,)):
                self.assertTrue(h28.inventory(root)[0]['prior_manifest']['model_hash_matches'])
                (stage/'model.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    h28.inventory(root)

    def test_inventory_rejects_wrong_seed_before_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = 'artifacts/example/world_0/lifecycle'
            stage = root/relative
            stage.mkdir(parents=True)
            (stage/'config.yaml').write_text('world:\n  seed: 600\n')
            with patch.object(h28, 'SOURCES', (relative,)):
                with self.assertRaisesRegex(ValueError, 'non-development-zero'):
                    h28.inventory(root)

    def test_missing_inventory_never_substitutes_world(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(h28, 'SOURCES', ('artifacts/missing/world_0',)):
                result = h28.inventory(Path(directory))
            self.assertEqual(result, [{'path': 'artifacts/missing/world_0', 'status': 'MISSING'}])


if __name__ == '__main__':
    unittest.main()
