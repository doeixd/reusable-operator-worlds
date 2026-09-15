import copy
import unittest

from row.experiments import audit_so2p_plasticity as p


def arm(term=0.126, below=6, eot=0.077, lost=21, drift=0.52, sha="s", anchor=0.0, name="BASE"):
    change = p.ARMS[name]
    return {"terminal_median": term, "terminal_below": below, "end_of_task_median": eot, "lost_threshold": lost,
            "drift_from_stage2": {"median": drift}, "library_sha256": sha, "anchor_abs_error": anchor,
            "change": list(change) if change else None,
            "changed_fields": [f"discrete_model.{change[0]}"] if change else []}


def arms(**overrides):
    out = {n: arm(sha=f"sha_{n}", name=n) for n in p.ARMS}
    # Default LR arms: drift falls monotonically, nothing improves -> DISFAVOURED.
    for n, d in zip(p.LR_ORDER, (0.52, 0.40, 0.30, 0.20)):
        out[n]["drift_from_stage2"]["median"] = d
    for n, values in overrides.items():
        out[n].update(values)
    return out


GATES_OK = {"G0_reproduction": True, "G1_non_vacuity": True, "G2_anchor": True}


class TriageTests(unittest.TestCase):
    def test_disfavoured_when_drift_falls_but_nothing_improves(self):
        self.assertEqual(p.triage(arms(), GATES_OK), "DISFAVOURED")

    def test_live(self):
        self.assertEqual(p.triage(arms(**{"LR_1/4": {"terminal_median": 0.03, "terminal_below": 40}}), GATES_OK),
                         "LIVE")

    def test_live_refused_when_acquisition_slows(self):
        a = arms(**{"LR_1/4": {"terminal_median": 0.03, "terminal_below": 40, "end_of_task_median": 0.2}})
        self.assertEqual(p.triage(a, GATES_OK), "PARTIAL")  # halves terminal, lower drift, but eot > 2x base

    def test_partial_by_lost_count(self):
        self.assertEqual(p.triage(arms(**{"REPLAY_2x": {"lost_threshold": 9, "drift_from_stage2": {"median": 0.4}}}),
                                  GATES_OK), "PARTIAL")

    def test_improvement_without_lower_drift_is_not_partial(self):
        a = arms(**{"REPLAY_4x": {"lost_threshold": 3, "drift_from_stage2": {"median": 0.6}}})
        self.assertEqual(p.triage(a, GATES_OK), "DISFAVOURED")

    def test_uninformative_when_drift_not_monotone_or_gate_fails(self):
        a = arms()
        a["LR_1/10"]["drift_from_stage2"]["median"] = 0.45
        self.assertEqual(p.triage(a, GATES_OK), "UNINFORMATIVE")
        self.assertEqual(p.triage(arms(), GATES_OK | {"G2_anchor": False}), "UNINFORMATIVE")
        missing = arms()
        del missing["REPLAY_4x"]
        self.assertEqual(p.triage(missing, GATES_OK), "UNINFORMATIVE")


class GateTests(unittest.TestCase):
    def so2(self):
        return {"stages": {"3": {"library_sha256": "sha_BASE", "library_sha256_at_start": "c",
                                 "terminal_per_task": {"t0": 0.1, "t1": 0.2}}}}

    def full(self):
        a = arms()
        for r in a.values():
            r["library_sha256_at_start"] = "c"
            r["terminal_per_task"] = {"t0": 0.1, "t1": 0.2}
        return a

    def test_all_gates_pass(self):
        self.assertEqual(p.gates(self.full(), self.so2()), GATES_OK)

    def test_g0_fails_on_per_task_drift(self):
        a = self.full()
        a["BASE"]["terminal_per_task"]["t1"] = 0.2 + 1e-3
        self.assertFalse(p.gates(a, self.so2())["G0_reproduction"])

    def test_g1_fails_when_arm_did_not_change_library_or_changed_extra_field(self):
        a = self.full()
        a["LR_1/2"]["library_sha256"] = "sha_BASE"
        self.assertFalse(p.gates(a, self.so2())["G1_non_vacuity"])
        b = self.full()
        b["REPLAY_2x"]["changed_fields"] = ["discrete_model.replay_examples_per_task", "discrete_model.seed"]
        self.assertFalse(p.gates(b, self.so2())["G1_non_vacuity"])


class ConfigTests(unittest.TestCase):
    def test_each_arm_changes_exactly_its_field(self):
        base, _, _, _ = p.stage_setup(p.WORLD, 3, p.MODEL_SEED)
        self.assertEqual(p.changed_fields(base, p.arm_config(base, "BASE")), [])
        for name, (field, value) in ((n, c) for n, c in p.ARMS.items() if c):
            cfg = p.arm_config(base, name)
            self.assertEqual(p.changed_fields(base, cfg), [f"discrete_model.{field}"])
            self.assertEqual(getattr(cfg.discrete_model, field), value)
        self.assertEqual(base.discrete_model.global_learning_rate, 0.001)
        self.assertEqual(base.discrete_model.replay_examples_per_task, 4)


if __name__ == "__main__":
    unittest.main()
