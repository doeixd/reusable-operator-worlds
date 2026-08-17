from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from row.provenance import resolved_config_sha256, validate_artifact


def _resolved(width: int = 32) -> dict[str, object]:
    return {
        "world": {"seed": 3, "reuse_rho": 1.0, "program_length": 3},
        "dense_model": {
            "seed": 3000,
            "hidden_width": width,
            "global_learning_rate": 0.001,
            "task_learning_rate": 0.05,
            "replay_ratio": 1.0,
        },
        "evaluation": {"support_points": (0, 1, 2)},
        "order": "forward",
        "output": {"directory": "artifact"},
    }


class ProvenanceTests(unittest.TestCase):
    def test_hash_treats_yaml_list_and_python_tuple_equally(self) -> None:
        with_tuple = _resolved()
        with_list = json.loads(json.dumps(with_tuple))
        self.assertEqual(
            resolved_config_sha256(with_tuple), resolved_config_sha256(with_list)
        )

    def test_validation_backfills_legacy_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            resolved = _resolved()
            (output / "config.yaml").write_text(
                yaml.safe_dump(resolved), encoding="utf-8"
            )
            (output / "git_commit.txt").write_text("abc123\n", encoding="utf-8")
            fingerprint = validate_artifact(output, resolved, "dense")
            self.assertTrue(fingerprint["backfilled_from_resolved_config"])
            self.assertTrue((output / "fingerprint.json").exists())

    def test_validation_accepts_numerically_equal_int_and_float(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            actual = _resolved()
            actual["world"]["reuse_rho"] = 1
            (output / "config.yaml").write_text(
                yaml.safe_dump(actual), encoding="utf-8"
            )
            (output / "git_commit.txt").write_text("abc123\n", encoding="utf-8")
            validate_artifact(output, _resolved(), "dense")

    def test_validation_rejects_architecture_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "config.yaml").write_text(
                yaml.safe_dump(_resolved(width=128)), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_artifact(output, _resolved(width=32), "dense")


if __name__ == "__main__":
    unittest.main()
