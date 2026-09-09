import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from row.config import load_config
from row.experiments.audit_so1_budget_bracket import protocol, run_job, run_grid
from row.experiments.audit_so1_budget_bracket import anchor_check
from row.experiments.score_so1_budget_bracket import (
    LEVELS, persistence_from_scores, summarize_cells, validate_cell,
    validate_report,
)
from row.experiments.so1_storage import atomic_json, cell_stamp, digest, fingerprint, load_cell, resolved, writer_lock
from row.rotated_world import generate_rotated_world


class SO1RestartTests(unittest.TestCase):
    def test_full_config_changes_are_fingerprinted(self):
        base = load_config("configs/v1.yaml")
        reference = fingerprint(protocol(base))
        for changed in (
            replace(base, world=replace(base.world, alpha=0.36)),
            replace(base, discrete_model=replace(base.discrete_model, operator_rank=7)),
            replace(base, evaluation=replace(base.evaluation, target_precision=0.01)),
        ):
            self.assertNotEqual(reference, fingerprint(protocol(changed)))

    def test_writer_lock_excludes_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "writer.lock"
            with writer_lock(path):
                with self.assertRaises((RuntimeError, OSError)):
                    with writer_lock(path):
                        self.fail("two writers acquired the same cell")
            with writer_lock(path):
                pass

    def test_durable_cell_survives_parent_interruption_and_rejects_corruption(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            job = {"config": "configs/v1.yaml", "key": "test_oracle", "world": 0,
                   "oracle": True, "batch": 2, "updates": 16, "cell_index": 100,
                   "sampling_index": 100, "git_commit": "unit-test",
                   "protocol_sha256": "test-protocol", "artifact": str(path)}
            cell = run_job(job)
            # Simulate a lost parent aggregate: the worker's artifact is enough.
            with patch("row.experiments.audit_so1_budget_bracket._run_job", side_effect=AssertionError("must not retrain")):
                self.assertEqual(run_job(job), cell)
            cfg = load_config(job["config"])
            world = generate_rotated_world(cfg.world)
            validate_cell(cell, path, cell_stamp(job), cfg, world)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_cell(path, cell_stamp(dict(job, git_commit="different")))
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_cell(path, cell_stamp(dict(job, protocol_sha256="different")))
            with (path / "model.pt").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_job(job)

    def test_independent_persistence_negative_and_nonmonotone_controls(self):
        def points(values):
            return {str(i): {"median": v} for i, v in enumerate(values)}
        self.assertEqual(persistence_from_scores(points([1, .04, .03, .02])), (1, "persistent"))
        self.assertEqual(persistence_from_scores(points([1, 1, .04])), (2, "crossed, persistence unobservable"))
        self.assertEqual(persistence_from_scores(points([1, .04, 1, .02])), (1, "crossed, not persistent"))
        self.assertEqual(persistence_from_scores(points([1, 1, 1])), (None, "not crossed"))

    def test_independent_ladder_and_dose_monotonicity(self):
        def make(value):
            return {"terminal_median": value, "checkpoints": {"0": {"median": 2}, "1": {"median": value}}}
        cells = {f"O_b{b}_g{g}": {str(w): make(1.) for w in (0, 1, 2)} for b in (2, 64) for g in LEVELS}
        self.assertEqual(summarize_cells(cells)["classification"], "NO_ORACLE_CELL_PASSES")
        cells["O_b64_g262144"] = {str(w): make(.01) for w in (0, 1, 2)}
        cells["L_b64_g262144"] = {str(w): make(1.) for w in (0, 1, 2)}
        summary = summarize_cells(cells)
        self.assertEqual(summary["classification"], "ORACLE_PASSES_LEARNED_FAILS")
        self.assertEqual(summary["envelope"]["64"]["lowest_passing"], 262144)
        self.assertIsNone(summary["envelope"]["64"]["lowest_passing_excluding_unobservable"])
        cells["L_b64_g262144"]["1"] = make(.01)
        cells["L_b64_g262144"]["2"] = make(.01)
        self.assertEqual(summarize_cells(cells)["classification"], "ORACLE_AND_LEARNED_PASS")
        cells["O_b2_g32768"]["0"] = make(2.)
        self.assertFalse(summarize_cells(cells)["dose_monotonicity"]["2"]["0"])

    def test_anchor_failure_is_verified_as_instrument_failure_never_science(self):
        # The real model/artifact check is exercised above. This fixture isolates
        # report completeness, freshness, anchor arithmetic and stop semantics.
        base = load_config("configs/v1.yaml")
        p = protocol(base)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate_path = root / "gate.json"
            gate = {"git_commit": "test", "protocol_sha256": fingerprint(p), "gate": "PASS",
                    "memory_passes": True, "bitwise_identical_cells": 9, "expected_cells": 9,
                    "serial": [{}] * 9, "pooled": [{}] * 9}
            atomic_json(gate_path, gate)
            cells = {key: {w: {"terminal_median": 3., "passes": False} for w in ("0", "1", "2")}
                     for key in ("O_b2_g16384", "O_b64_g262144")}
            stage_d = json.loads(Path("reports/rotated_g5r_interference.json").read_text())
            report = {"git_commit": "test", "protocol": p, "protocol_sha256": fingerprint(p),
                      "launch": {"gate_path": str(gate_path), "gate_sha256": digest(gate_path), "artifact_root": str(root)},
                      "started_utc": "2026-09-09T00:00:00+00:00", "finished_utc": "2026-09-09T02:00:00+00:00",
                      "cells": cells, "anchor": anchor_check(cells, stage_d), "complete": False,
                      "classification": "ANCHOR_FAILED_NOTHING_READ"}
            report_path, exit_path = root / "report.json", root / "exit.json"
            manifest_path = root / "run_manifest.json"
            atomic_json(manifest_path, {"git_commit": "test", "protocol_sha256": fingerprint(p),
                                        "started_utc": report["started_utc"]})
            report["launch"].update(manifest_path=str(manifest_path), manifest_sha256=digest(manifest_path))
            atomic_json(report_path, report)
            atomic_json(exit_path, {"git_commit": "test", "exit_code": 1, "finished_utc": "2026-09-09T03:00:00+00:00"})
            with patch("row.experiments.score_so1_budget_bracket.generate_rotated_world", return_value=None), \
                 patch("row.experiments.score_so1_budget_bracket.validate_cell", return_value="2026-09-09T01:00:00+00:00"):
                checked = validate_report(report_path, Path("configs/v1.yaml"), exit_path)
                self.assertFalse(checked["scientific_result_accepted"])
                self.assertEqual(checked["verified_cells"], 6)
                report["envelope"] = {}
                atomic_json(report_path, report)
                with self.assertRaisesRegex(ValueError, "analysis crossed failed anchor"):
                    validate_report(report_path, Path("configs/v1.yaml"), exit_path)
                del report["envelope"]
                del report["cells"]["O_b2_g16384"]["0"]
                atomic_json(report_path, report)
                with self.assertRaisesRegex(ValueError, "incomplete cell"):
                    validate_report(report_path, Path("configs/v1.yaml"), exit_path)

    def test_grid_stops_at_anchors_and_pairs_conditional_streams(self):
        base = load_config("configs/v1.yaml")
        stage_d = json.loads(Path("reports/rotated_g5r_interference.json").read_text())
        for anchor_failure in (True, False):
            with self.subTest(anchor_failure=anchor_failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = SimpleNamespace(config=Path("configs/v1.yaml"), output=root / "report.json",
                                       gate=root / "gate.json", hard_cap=2, measured_rss_mib=768,
                                       artifact_root=root / "cells")
                atomic_json(args.gate, {"git_commit": "test", "gate": "PASS", "implementation": "batched_rotation_v1",
                                       "protocol_sha256": fingerprint(protocol(base)), "budget_mib": 768})
                dispatched, saved = [], {}
                def pool(function, jobs, budget, **kwargs):
                    for job in jobs:
                        dispatched.append(job)
                        median = .04
                        name = {"O_b2_g16384": "C_lo", "O_b64_g262144": "C_hi"}.get(job["key"])
                        if name:
                            median = 3. if anchor_failure else stage_d["cells"][name][str(job["world"])]["terminal_median"]
                        result = {"terminal_median": median, "passes": median <= .05,
                                  "persistence": "not crossed", "seconds": 0.}
                        path = Path(job["artifact"])
                        atomic_json(path / "result.json", {})
                        saved[str(path)] = result
                        kwargs["on_result"](job, result)
                    return []
                with patch("row.experiments.audit_so1_budget_bracket.require_clean_code"), \
                     patch("row.experiments.audit_so1_budget_bracket.subprocess.run", return_value=SimpleNamespace(returncode=0)), \
                     patch("row.experiments.audit_so1_budget_bracket.git_commit", return_value="test"), \
                     patch("row.experiments.audit_so1_budget_bracket.run_pool", side_effect=pool), \
                     patch("row.experiments.audit_so1_budget_bracket.load_cell", side_effect=lambda path, stamp: saved[str(path)]):
                    if anchor_failure:
                        with self.assertRaisesRegex(SystemExit, "ANCHOR FAILED"):
                            run_grid(args, base)
                        self.assertEqual(len(dispatched), 6)
                        self.assertFalse(json.loads(args.output.read_text())["complete"])
                    else:
                        run_grid(args, base)
                        self.assertEqual(len(dispatched), 36)
                        learned = [j for j in dispatched if not j["oracle"]]
                        self.assertEqual(len(learned), 6)
                        for job in learned:
                            paired = next(j for j in dispatched if j["oracle"] and j["world"] == job["world"]
                                          and j["batch"] == job["batch"] and j["updates"] == job["updates"])
                            self.assertEqual(job["sampling_index"], paired["sampling_index"])
                            self.assertEqual(job["resolved_sha256"], paired["resolved_sha256"])


if __name__ == "__main__":
    unittest.main()
