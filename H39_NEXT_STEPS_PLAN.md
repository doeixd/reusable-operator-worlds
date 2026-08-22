# After the H39 confirmation: audits, then membership discovery

Status: Stage A (audits) is frozen with this file's hash; it runs on
existing artifacts only. Stages B and C are DESIGNS, to be frozen as their
own plans after Stage A passes; nothing in them is licensed by this file.
Written 2026-08-22 from review 62 and the sealed result.

# Stage A — two audits of the sealed result (frozen here; no new lifetimes)

Both read frozen artifacts from `artifacts/h39_confirmation` and
`artifacts/v6_clean`; neither touches a lifetime or a sealed seed beyond
reading completed cells. One scorer, `audit_h39_confirmation_followups.py`,
one report, `reports/h39_confirmation_followups.json`, written atomically.

## A1 — "made, not mined", apples-to-apples (review 62 point 1)

Question: does the census construction — fit a fresh task-local coordinate
in the affine span of what the finished learner STORED — fail on M's own
argument matrices the way it failed on O's residuals? If post-hoc
extraction from M also fails while M's online alpha succeeds, the
difference is WHEN the coordinates were formed, not which learner.

Construction, per sealed world 700-729, on M's final artifact:
- Post-hoc span fit (the census): PCA over the 64 trained family tasks'
  effective residual vectors in M (`W alpha`-free: M has no residual
  schema; use the private residuals plus promoted abstractions exactly as
  census C0 did), rank 8 and max rank; fit route code + coordinate, eps
  zero, k = 128, B1. Ratio to O's full-fit anchor: `R_census(M)`.
- Online alpha fit: M's registered alpha-only endpoint from the sealed
  report: `R_M` (geomean 1.636).
- Also on O: `R_census(O)` at max rank (the development census repeated on
  the 30 sealed ordinary artifacts).

Decision (fixed): A1 PASSES if the geometric mean of `R_census(M)` at max
rank exceeds the geometric mean of `R_M` by at least 0.5, AND
`R_census(O)` at max rank exceeds `R_M` by at least 0.5. Then "formed
online versus extracted afterward" is the operative difference on the same
learner and the same worlds. If `R_census(M) <= R_M + 0.2`, the census
failure was a property of O's population, not of post-hoc extraction, and
the "made, not mined" sentence is withdrawn from the closure. Between:
recorded as PARTIAL with both numbers.

Predicted: `R_census(M)` max-rank geomean in [2.2, 3.5]; `R_census(O)` in
[2.2, 3.5]; A1 passes.

## A2 — compensation in the alpha-zero ablation (review 62 point 4)

Question: E5 zeroed alpha on trained family tasks WITHOUT letting anything
else re-adapt. Can the route code and private residual compensate for a
removed argument if allowed to re-fit?

Construction, per sealed world, on M's final artifact, for every trained
family task: zero both slots' alphas; freeze the representation and the
alphas; re-fit ONLY the task's route code and private residual on the
task's own 128 training examples (B1, 2,000 updates, support-only);
measure evaluation NMSE. Report the ratio to the intact model
(`compensated_ratio`) beside E5's uncompensated ratio (mean 3.64).

Decision (fixed): A2 PASSES if the mean over worlds of the median
compensated ratio is at least 1.25 — the registered usage bar — i.e. the
argument's contribution cannot be recovered by re-optimizing the other
task-local pathways. If the compensated ratio falls below 1.10, E5 is
reinterpreted: alpha is used but substitutable, and the "causally used"
sentence is weakened to "used, and not cheaply replaced" only if A2 holds.
Between: PARTIAL.

Predicted: compensated ratio in [1.4, 2.5]; A2 passes. (Route + residual
cannot express a 32-direction operator change with a 198-scalar rank-2
residual; but the private residual is what did the work in the H39 pilot,
so this is a real test.)

## Stage A gate

Stage B is frozen only if A1 and A2 both PASS or are PARTIAL with the
closure updated. A FAIL on either rewrites the closure's interpretation
before any discovery plan is written. Cost: ~3 h of fitting.

# Stage B — H47, membership discovery (design; to be frozen separately)

## The comparison that is not already answered

The confirmed arm M already discovers membership by soft routing with no
labels. The open question is therefore how much the ECONOMICS lose to
label-free discovery relative to an arm that is TOLD the membership.

Arms, development worlds 0-2 (later 0-9 if the design survives), two
parameterized slots at K = 32 throughout:
- **M** (exists): soft routing, no labels — the discovered-membership arm.
- **L** (labelled oracle): identical, but each family task's route logits
  at the family step are initialized with a fixed +4 bias toward its
  family's designated slot (families 0-1 -> slot 11, 2-3 -> slot 10) and
  those logits are FROZEN; unseen-family futures get no bias (they must be
  discovered at adaptation time, as in M). This is an upper reference for
  what perfect membership buys, not a learner.
- **H** (hard commitment): M with the route softmax annealed to a
  near-one-hot selection over the two parameterized slots by task 24, the
  V2-style premature-commitment arm review 62 predicts will lose.
- **G** (exists): the frozen-direction control, reused.

Endpoints: J_present (lifetime loss), R_alpha, R_full, E5 usage, all as in
the confirmation scorer; plus the diagnostic teacher-membership agreement
(adjusted Rand index between the learner's dominant parameterized slot per
family task and the teacher family, and route entropy over the two slots).

Decision rules (to be frozen with baseline-checked tolerances):
- **membership cost**: paired (M - L) on J_present, R_alpha, R_full; H47
  is SUPPORTED if M is within registered tolerances of L on all three
  (proposal: J_present within 300 nats; R_alpha and R_full within 0.15)
  in at least 2 of 3 worlds; and
- **soft beats hard**: M better than H on R_alpha in at least 2 of 3
  worlds (review 62's prediction).
- Teacher ARI is reported and decides nothing.
Non-vacuity: L's frozen logits bitwise at their init; H's final routes
have entropy below a registered bound; M's and H's argument matrices
moved. Thresholds are set after measuring M's existing route entropy and
slot agreement on worlds 0-2, before freezing.

Predicted (ours, registered when frozen): M within tolerance of L on
J_present and R_full, probably not on R_alpha in all worlds; H worse than
M on R_alpha 3/3.

Cost: L and H on 3 worlds = 6 lifetimes (~1 h on a pool of 3) plus
scoring.

# Stage C — H48, cardinality discovery (design sketch; frozen only after B)

Candidate pool of K_max = 6 parameterized slots (indices 11..6), each with
K = 32 directions and a per-slot presence gate charged at
`lambda * D*(U_k)` bits in the prequential objective, decided at sleeps
(not online; the MDL-gate lesson: compress after evidence accumulates).
Controls: fixed 1, fixed 2 (M), fixed 4, fixed 6, discovered. Worlds
generated with F_schema in {1, 2, 4, 8} families at constant 16 tasks per
family. Deciding statistic: active count versus F, and J_present / R_alpha
versus the fixed-correct arm. Discovery wins only if it is within
tolerance of fixed-correct on every F without activating all K_max.
Predicted (review 62): the hard one; too many or too few.

# Order and what each stage may change

A -> B -> C. A can rewrite the closure's interpretation. B can say that
label-free membership is or is not economically free, and whether
premature commitment is the failure mode. C can say whether the
abstraction mechanism scales with recurrence rather than with supplied
capacity. None of them opens a sealed seed; a second confirmation block
(seeds 800-829) would be written only if C passes on development worlds.
