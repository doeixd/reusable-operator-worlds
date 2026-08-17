from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from row.experiments.check_scrambled_ids import _compare_artifacts


class CheckScrambledIdsTests(unittest.TestCase):
    def test_comparison_normalizes_only_opaque_task_names(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            scrambled = root / "scrambled"
            canonical.mkdir()
            scrambled.mkdir()
            for directory, task_id in (
                (canonical, "task_a"),
                (scrambled, "task_b"),
            ):
                (directory / "world_programs.json").write_text(
                    json.dumps(
                        [{"task_index": 0, "task_id": task_id, "primitive_ids": [1, 2]}]
                    ),
                    encoding="utf-8",
                )
                (directory / "metrics.jsonl").write_text(
                    json.dumps({"task_id": task_id, "nmse": 0.25}) + "\n",
                    encoding="utf-8",
                )
                summary = {"loss": 1.5}
                if directory == scrambled:
                    summary["task_id_scramble_seed"] = 7
                (directory / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                torch.save(
                    {
                        "model_state_dict": {
                            f"task_codes.{task_id}": torch.tensor([0.5]),
                            "shared": torch.tensor([1.0]),
                        }
                    },
                    directory / "model.pt",
                )
            result = _compare_artifacts(canonical, scrambled)
            self.assertTrue(result["exact_invariance"])


if __name__ == "__main__":
    unittest.main()
