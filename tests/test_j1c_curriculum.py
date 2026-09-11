import unittest

import torch

from row.experiments import audit_j1c_curriculum as j1c
from row.experiments.audit_rotated_g5r_interference import _assignment, offline_cell
from row.experiments.audit_so1_budget_bracket import build_fast, checkpoints_for


def stage_cell(arm="STAGED", median=0.01, carry=("sha1", "sha2")):
    def stage(k, sha, start):
        return {"terminal_median": median if k == 3 else 0.01, "shared_relative_change": 1.0,
                "code_relative_change": 1.0, "library_sha256": sha, "library_sha256_at_start": start,
                "final_per_task": {f"t{k}_{i}": 0.01 for i in range(4)}}
    return {"arm": arm, "terminal_median": median,
            "stages": {"1": stage(1, "sha1", None), "2": stage(2, "sha2", carry[0]),
                       "3": stage(3, "sha3", carry[1])}}


class J1cTests(unittest.TestCase):
    def test_stage3_loop_matches_offline_cell_learned_routes(self):
        torch.set_num_threads(1)
        cfg, world, _, stream = j1c.stage_setup(1, 3)
        reference = offline_cell(cfg, world, _assignment(cfg, world), oracle=False, updates=16, batch=2,
                                 cell_index=103, checkpoints_requested=checkpoints_for(16),
                                 build=build_fast, retain_per_task=True)
        _, mine = j1c.train_stage(cfg, world, 16, stream)
        self.assertEqual(mine["final_per_task"], reference["final_per_task"])

    def test_stage_setup_lengths_tasks_and_disjoint_ids(self):
        seen = []
        for stage, (length, tasks) in enumerate(((1, 60), (2, 64), (3, 64)), start=1):
            cfg, world, updates, _ = j1c.stage_setup(1, stage)
            self.assertEqual(cfg.discrete_model.task_steps, length)
            self.assertEqual(len(world.tasks), tasks)
            self.assertTrue(all(len(t.program.primitive_ids) == length for t in world.tasks))
            seen.append({t.task_id for t in world.tasks})
        self.assertFalse(seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
        self.assertEqual([u for _, _, u, _ in (j1c.stage_setup(1, s) for s in (1, 2, 3))], [16384, 16384, 32768])

    def test_classification_and_transfer_guards(self):
        baseline = {w: 0.95 for w in j1c.WORLDS}
        cells = {f"STAGED_w{w}": stage_cell(median=m) for w, m in zip(j1c.WORLDS, (0.7, 0.01, 0.02))}
        cells |= {f"RESET_w{w}": stage_cell("RESET", 0.9, ("sha1", None)) for w in j1c.WORLDS}
        self.assertEqual(j1c.classify(cells, baseline), "J1C_ACQUIRES")
        broken = dict(cells, STAGED_w1=stage_cell(median=0.01, carry=("sha1", "wrong")))
        self.assertEqual(j1c.classify(broken, baseline), "HARNESS_FAILED")
        leaky = dict(cells, RESET_w1=stage_cell("RESET", 0.9, ("sha1", "sha2")))
        self.assertEqual(j1c.classify(leaky, baseline), "HARNESS_FAILED")
        mid = {f"STAGED_w{w}": stage_cell(median=m) for w, m in zip(j1c.WORLDS, (0.7, 0.3, 0.4))}
        mid |= {f"RESET_w{w}": stage_cell("RESET", 0.9, ("sha1", None)) for w in j1c.WORLDS}
        self.assertEqual(j1c.classify(mid, baseline), "J1C_IMPROVES")
        bad = {f"STAGED_w{w}": stage_cell(median=0.8) for w in j1c.WORLDS}
        bad |= {f"RESET_w{w}": stage_cell("RESET", 0.9, ("sha1", None)) for w in j1c.WORLDS}
        self.assertEqual(j1c.classify(bad, baseline), "J1C_FAILS")


if __name__ == "__main__":
    unittest.main()
