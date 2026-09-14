# H28 T0: bounded artifact inventory and exact instrument controls

Status: DEVELOPMENT INSTRUMENT PLAN, 2026-09-14; not a frozen experiment or
accepted scientific result. This is the first bounded step of
H28_CLOSURE_RESEARCH_PLAN.md. Its outputs describe artifact availability and
verify mathematical controls; they cannot score CL3-CL8 or establish learned
closure. No model loading, world generation, learned probes, training, or
sealed-data access is permitted in this step.

# Inventory scope

Inspect only these six development-world-0 directories, selected by artifact
family and control role before inspecting their model contents:

- `artifacts/j1c_curriculum/cells/STAGED_w0/stage3`
- `artifacts/j1c_curriculum/cells/RESET_w0/stage3`
- `artifacts/j1cr_replication/cells/STAGED-R_w0/stage3`
- `artifacts/j1cr_replication/cells/NON-STAGED-R_w0/stage3`
- `artifacts/v4_dev/structured/world_0/lifecycle`
- `artifacts/v5_h20b/r100/world_0/lifecycle`

Here r100 is a condition name, not world seed 100. Independently check the
saved config's world seed before reading further metadata. Record missing paths
rather than substituting another world. No recursive search into seed bands.
The two lifecycle artifacts cover the promotion mechanism inherited from V3;
the selected V3 task-group directories found during reconnaissance contain
shared-residual learners, not promoted models. Do not silently treat those as
PROMOTE artifacts.

Record file names, byte sizes, SHA256, saved seed, presence of complete terminal
state sidecars, and presence of histories or promotion snapshots. For J1c sources
compare stage-file hashes with the already-existing parent durable record;
fresh hashes without a prior trusted manifest establish identity, not validity.
Do not read task programs, predictions, metric arrays, or tensor contents.

Interpretation requires a separate source audit: a residual operator may contain
an internal low-rank bottleneck, but this is not yet a specified coarse-state
interface with a transition law. It is a possible future candidate, not a reason
to declare all coarse-grainings impossible. Promotion residual snapshots alone
do not recover the complete pre-promotion model, routes, and context. Missing
historical state limits the proposed causal audit; it does not refute PROMOTE.

# Exact finite controls

Use three independent uniform bits z, n, w. The microscopic observation is (z,n),
and w is the realization/context input kept separate for the world probe. The
complete controlled state includes both. No operation is hidden: the action is
fixed. Enumerate the following transition laws exactly, without sampling:

| Control | Target y | Expected micro/world/joint gaps, nats |
|---|---|---|
| closed | z XOR Bernoulli(0.1), noise independent | 0 / 0 / 0 |
| micro-leaky | z XOR n | ln(2) / 0 / ln(2) |
| world-dependent | z XOR w | 0 / ln(2) / ln(2) |
| jointly leaky | z XOR n XOR w | 0 / 0 / ln(2) |
| collapsed | constant 0 | 0 / 0 / 0, but zero predictive gain |

Compute conditional entropy by grouping the enumerated joint mass. Compare
H(y|z) against H(y|z,n), H(y|z,w), and H(y|z,n,w), using natural logarithms.
These are ideal predictors, not trained probes. Weak probes that ignore added
inputs return zero gaps even on the leaky cases; this failure must be exposed.
Require target entropy and predictive gain above numerical tolerance for a
non-vacuous positive. For the full-state encoding control, explicitly mark that
no micro distinctions were discarded and refuse the combined abstraction claim.
These checks implement logical non-vacuity only; learned representations will
need measured capacity, fidelity, and overlap gates in a later plan.

# Exact coordinate control

Check the Gaussian construction in the parent plan at r = 0.1, 0.5, 0.8, 0.99.
Compute the MI terms from the correlation coefficients of each rotated
coordinate at angles 0, pi/4, pi/2. Identity and permutation give Psi = 0;
45-degree mixing gives positive Psi and agrees with the derived formula.
The macro law is held fixed by transporting the encoder. No neural critic is
trained and no empirical ROW sensitivity is asserted.

# Validation, performance, and provenance

Use only the Python standard library. Unit checks include wrong/unnormalized
joint probabilities, dependence requiring joint augmentation, weak-detector
failure, collapsed/full-state rejection, permutation invariance, and the
independent analytic formula. Tolerances are numerical (1e-12), not calibrated
scientific effect thresholds. Time repeated control evaluations and one bounded
inventory; no model update timing applies because no model update occurs.

Record source/plan hashes, HEAD, dirty status, and whether the output came from
a development check. During SO2 repair/resume coordination keep its launch HEAD
unchanged. Preparatory code and checks never modify files SO2 imports. Output
from uncommitted development code must be marked provisional and cannot become
an accepted experiment result by renaming it. Commit the plan, utility and tests
when compatible with SO2; a later accepted audit requires the normal frozen
plan/clean-code/reconstruction/scorer gates.

# Stop and next action

Stop on failed exact controls or mismatched saved seed/hash. If no candidate has
a declared nontrivial coarse-state interface, record that the direct existing-
artifact closure comparison is not yet specified. The next action is to define
and calibrate one such interface, or start the separately registered H28-C
coordinate-reuse opportunity gate. Do not train a closure regularizer, open a
lifetime, or infer a PROMOTE effect from this inventory.
