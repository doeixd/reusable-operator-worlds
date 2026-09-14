# H28 T0 development checkpoint: inventory and exact controls

Status: PROVISIONAL DEVELOPMENT CHECK, not an accepted scientific result.
This is implementation validation and a bounded metadata census, not a learned
closure audit, an H28 run, or a verdict on CL3-CL8. Source HEAD at execution:
`a31a397`; the new utility/plan/tests were uncommitted. Full provenance and hashes
are in [the current r2 record](h28_t0_development_20260914_r2.json).
The [first record](h28_t0_development_20260914.json) is preserved: r2 adds three
inventory integrity tests; no controls, source artifacts, or numerical protocol
changed. The two records agree on controls and inventory (timing/provenance differ).

# What was done

Implemented [tools/h28_t0.py](../tools/h28_t0.py) under
[H28_T0_INSTRUMENT_PLAN.md](../H28_T0_INSTRUMENT_PLAN.md), with only Python's
standard library. Fourteen tests passed, including joint-only leakage, a weak
detector's false zero, trivial encoders, wrong-world rejection, tampered model
hashes, and missing artifacts. No torch import, tensor deserialization, model
forward, optimizer update, world generation, or sealed-artifact read occurred.

Exact finite controls produced their prescribed entropy differences: the joint
XOR case has zero individual micro/world improvement but ln(2) joint improvement.
The Gaussian transported-encoder control preserves zero Psi at identity and
permutation and gives positive Psi at 45 degrees for all four specified r values.
These validate mathematical examples and scoring code, not trained probes or
empirical ROW emergence. Full-state rejection checks the explicit
discarded-distinction requirement; it does not measure an encoder's capacity.

# Bounded inventory

All six selected configs declare world seed 0. Only the exact plan paths were
inspected; r100 in the V5 path names the condition, not world seed 100.

| Candidate | Existing model hash checked against prior durable record | history.pt present | promotion_snapshots.npz present |
|---|---|---|---|
| J1c STAGED, seed 5000, stage 3 | matches | no | no |
| J1c RESET, seed 5000, stage 3 | matches | no | no |
| J1c-R STAGED-R, seed 3001, stage 3 | matches | no | no |
| J1c-R NON-STAGED-R, seed 3001, stage 3 | matches | no | no |
| V4 structured lifecycle | no prior binary manifest checked; new hash only | no | no |
| V5 H20b r100 lifecycle | no prior binary manifest checked; new hash only | no | no |

The J1c durable records hash model.pt, not the config/model_state.json sidecars;
those sidecars received fresh identity hashes only. Nothing here certifies full
reconstruction or reproduces the accepted terminal score. The file-name census
does not prove that no historical checkpoint exists anywhere in the repository.
It establishes that these six specific terminal directories do not contain the
listed history/snapshot files. The earlier J1c stage checkpoints have different
task/formation stages, not paired before/after PROMOTE interventions.

# Source audit and implication

[PromotingSharedResidualLearner.forward](../src/row/models/promoting_models.py)
combines routed full-state operators with shared/private residual innovations.
Its internal low-rank hidden activation is a possible encoder candidate, but
the existing artifact contract does not designate it as a common coarse state
with a closed next-state law. Slot IDs and static promoted weight tensors do
not supply that interface either. This is an interface-definition gap, not a
proof that no useful quotient exists.

[LifecycleLibraryLearner.sleep](../src/row/models/lifecycle_models.py) can record
member residuals and a born abstraction. The
[lifetime serializer](../src/row/experiments/learned_lifetime.py) writes those
separately when present. Such snapshots still omit the complete historical
basis/routes and are insufficient alone to replay pre-promotion dynamics.
[The effective-operator loader](../src/row/experiments/audit_effective_operator.py)
also shows why terminal evaluation must restore references and retired tasks
outside the tensor state. The next audit must reproduce a saved score before
using new on-trajectory transitions.

Decision for this bounded checkpoint: **direct causal PROMOTE closure audit not
ready**. Do not fit probes against arbitrary next activations and label the gap
a promotion effect. Choose an explicit candidate interface and establish its
non-vacuity, or proceed with H28-C's separately registered coordinate-reuse
opportunity test. No branch-wide NO_OPPORTUNITY or learned-method null is claimed.

# Performance and reproducibility

Pre-run performance pass: 100 exact-control evaluations took 0.03395 seconds.
The final check took 0.00098 seconds for controls and 0.22603 seconds for census
plus provenance (interpreter startup excluded). No model update applies.
No speedup was applied; the decisive economy choice was avoiding model loads,
critic training, recursive artifact searches, and a lifetime entirely. No
before/after acceleration claim is made.

Validation commands:

```powershell
python -m unittest discover -s tests -p test_h28_t0.py -v
python tools/h28_t0.py --development-check --output reports/h28_t0_development_20260914_r2.json
```

The development-output command refuses to overwrite a prior record; a deliberate
new check needs a fresh output path. It is subsecond and not a restartable
lifetime launcher. Longer work must use the full repository run discipline.

SO2 was stopped after an error when this work began. Its partial report remains
and another session's resume responsibilities are not yet resolved. No SO2
source, report, launch record, or commit was changed by this task. Commit the
prepared H28 work when doing so preserves that run's provenance; do not promote
these provisional outputs into accepted results by relabeling them.
