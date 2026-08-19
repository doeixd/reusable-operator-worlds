# Plan for writing the V4 specification

A blueprint, not the spec. It fixes the decisions the spec must not
re-litigate, orders the work that has to happen before the spec can be
written honestly, and maps every planned section to the document it comes
from. Written 2026-08-19, immediately after the V3 sealed block closed
5/5.

The V3 spec is the model to imitate: hypotheses with explicit falsifiers,
worlds with validity gates that can fail, endpoints defined operationally
before any run, a sealed protocol with parameter intervals rather than
signs, and limits written into the document rather than left to a reader
to infer.

## Source documents

| Source | What V4 takes from it |
| --- | --- |
| [`notes/v4-sketch.txt`](v4-sketch.txt) Rev 1 | First-pass hypotheses H14-H17, nonstationary world families, the branch table, two new design principles (lifecycle provenance, no silent mutation) |
| [`reviews/reviewer-feedback-26.txt`](../reviews/reviewer-feedback-26.txt) | The V4 charter and ladder ordering, real-options framing of DELETE, the hysteresis lag control, migration costs, stable handles, per-operation refusal controls, deterministic edit ordering, cascading distortion, lineage metadata, the GC correspondence, the V6 decision dataset |
| [`reviews/reviewer-feedback-27.txt`](../reviews/reviewer-feedback-27.txt) | Confirmation of the ladder order, MERGE inherits gauge freedom, the outstanding matched-compute audit, the library-size-wander risk |
| [`row_v3_experimental_spec.md`](../row_v3_experimental_spec.md) §12 | What V3 established and did not; the substitutability ladder; the four-way leave-one-out gate; constitutional law V4 inherits |
| [`V3_CONFIRMATION_PLAN.md`](../V3_CONFIRMATION_PLAN.md) | The template for V4's own confirmation plan: registered outcomes, intervals, surrendered control, analysis rules |
| [`PREDICTIONS.md`](../PREDICTIONS.md) | Two standing V4 design inputs (P-2026-08-19-G horizon, P-2026-08-19-H lifetime sweep) and the V3 outcome record |
| [`notes/learnings.txt`](learnings.txt) | The sealed-V3 section: four failed testbeds, the ladder, graded refusal |
| [`AGENTS.md`](../AGENTS.md) | Machine constraints (4-6 concurrent lifetimes), provenance and commit conventions |
| Reviews [14](../reviews/reviewer-feedback-14.txt), [15](../reviews/reviewer-feedback-15.txt), [17](../reviews/reviewer-feedback-17.txt) | Earlier convergent material: hysteresis, memory hierarchy, search debt, prospective value |

## Decisions this plan fixes (so the spec inherits them settled)

**D1. The ladder is RETAIN/DELETE first, and the sketch's H14 is split.**
`v4-sketch.txt` bundles merge, fork, and delete into one hypothesis.
Feedback-26 and -27 both argue against that, and they are right: bundled
operations make interactions undiagnosable, and DELETE alone attacks V3's
measured weakness (5.3 abstractions for 2 families; 6.2 on sealed worlds)
with the simplest possible mechanism. The spec therefore has four rungs,
each gated on its predecessor's sealed verdict:

    V4.1 SURVIVAL       PROMOTE + RETAIN/DELETE
    V4.2 DEDUPLICATION  MERGE
    V4.3 NONSTATIONARITY copy-on-write + FORK
    V4.4 INVESTMENT     hysteresis, dormancy, horizon

**D2. The primary causal comparison is PROMOTE+DELETE against frozen V3
PROMOTE-only.** Same worlds, same seeds, same data, the V3 learner
untouched as the comparator. This mirrors V3's own primary comparison
(promoted against unpromoted) and keeps the claim causal.

**D3. Library size is a diagnostic, never an outcome.** Feedback-26 §7 is
explicit and the spec must say so: score `J` and substitutability. Five
abstractions for two teacher families may be a finer decomposition than
the teacher ontology rather than fragmentation, and "library size → 2"
would reward matching an ontology the learner has no access to.

**D4. Every structural edit is justified by substitution, never
identity.** V3's constitutional rule, carried verbatim. MERGE in
particular must ask whether one refitted replacement can serve all
dependents of both abstractions, with the same four-way controls
(existing pair / merged function / global fit / delete).

**D5. Migration costs are logged from V4.1 and charged from V4.2.**
Reference rewrites, validation, and refitting are real and make lifecycle
operations look free if ignored. V4.1 logs them so the accounting exists;
MERGE is the first operation whose economics they materially change.

**D6. Stable handles are introduced when an operation first needs them,
not preemptively.** Feedback-26 §5 predicts an indirection table will be
forced by MERGE. V4.1 does not need one. The spec should note the
prediction so that if it happens it is recorded as forced rather than
designed.

**D7. V4.1 stays in the validated V3 world.** Nonstationarity waits for
V4.3. The V3 testbed is the only world in the program with a passing
four-way substitutability gate, and changing the world and the operator
in the same step is what cost V3 four redesigns.

## Prerequisites, in order, before the spec is frozen

**P1. Close the V3 loose end: the matched-compute audit.** Feedback-27
flags this as load-bearing and it was not run. Give the unpromoted
comparator the same optimization-step budget and adaptation checkpoints
as the promoted learner, so the +1,174-nat lifetime gain cannot be extra
SGD in disguise. The future-block probe is already budget-matched; the
lifetime comparison is not. Report as a V3 robustness addendum, not a new
V3 claim.

**P2. Run the two standing post-H11 predictions**, both of which are V4
design inputs rather than optional extras:
[P-2026-08-19-G](../PREDICTIONS.md) (horizon: promotion rate should track
expected remaining lifetime) and P-2026-08-19-H (lifetime sweep for the
amortization threshold N*). Horizon sensitivity and N* between them
locate where DELETE's thresholds should sit.

**P3. Build the survival instruments.** Per-abstraction lineage from day
one, as feedback-26 §15 argues: id, birth time, supporting tasks, current
dependents, reuse count, cumulative realized savings, functional
descriptor, parents and children, edit history. This is for the
experimenter, not the learner, and without it a library that has been
merged and forked is unreconstructable.

**P4. Build the dormancy world and its controls.** The real-options test
needs A→B→A (temporary dormancy; must retain) against A→B→B… (permanent
disappearance; must eventually delete). This is the DELETE analogue of
V3's structureless control and the spec cannot register a refusal
criterion without it.

**P5. Define and run the V4 validity gate.** V3's lesson is that a world
must be shown capable of exercising the operator before the operator is
tuned on it. For DELETE the analogue is an offline clairvoyant oracle
(feedback-26 §10): compute the best lifecycle decisions with full
knowledge of the future stream, and define lifecycle regret
`R = J_online − J_oracle`. If the oracle's advantage over PROMOTE-only is
negligible, the world cannot test DELETE and must be redesigned before
anything is tuned.

**P6. Resolve the branch table.** `v4-sketch.txt`'s table resolves to its
first case — V3 passed on all three predictions — so V4 proceeds as
sketched rather than into the blocked branches. The spec's §0 records
this resolution explicitly.

## Planned structure of the V4 spec

Mirroring `row_v3_experimental_spec.md` section for section, so the two
read as a series.

**§0 Charter, and what V3 established.** The sealed 5/5 result with its
scope limits, and the one-line V4 question: *can a learner maintain a
compact computational vocabulary by letting abstractions be born
speculatively and making persistence expensive?* Includes the branch-table
resolution (P6) and the honest statement of V3's residue: birth is noisy,
survival is untested.

**§1 Hypotheses.** Renumbered from the sketch to match the ladder:

- **H14 Survival (primary, V4.1).** Evidence from subsequent reuse
  distinguishes abstractions that deserved to exist from abstractions that
  merely looked promising at birth. PROMOTE+DELETE beats frozen V3
  PROMOTE-only on `J` at preserved prediction and future-task adaptation.
  Falsifier: DELETE fires indiscriminately, or `J` fails to improve.
- **H15 Deduplication (V4.2).** Functionally redundant abstractions
  consolidate under mutual substitutability, reducing library plus
  reference description length at preserved behavior. Falsifier: merged
  replacements cannot serve both dependent sets within tolerance.
- **H16 Safe shared update (V4.3).** Copy-on-write beats both
  update-in-place (old-task interference) and never-update (retained
  private bits) on lifetime cost at matched bits. FORK is PROMOTE
  conditioned on an existing parent, per feedback-26 §8, not a separate
  mechanism.
- **H17 Hysteresis (V4.4, the flagship).** `r_create > r_delete`, with the
  **rate-sweep control preregistered**: run several sweep speeds and
  require the gap to survive as `v → 0`. If it vanishes the loop was
  adaptation lag and the investment framing dies. This control is the
  single most important addition feedback-26 makes to the sketch.
- **H18 Priced retrieval (exploratory).** With `gamma > 0`, optimal
  library size becomes interior. Kept exploratory because ROW-scale
  libraries may be too small for the term to bite; report honestly if so.

**§2 Worlds.** V4.1 in the frozen V3 testbed unchanged; the dormancy
world and its permanent-disappearance control; redundancy-onset worlds for
MERGE; regime-split worlds for FORK; nonstationary `r(t)` schedules with
the rate sweep for H17. Every world carries a validity gate that can fail,
and the frequency-versus-value world from feedback-26 §11 (frequent
low-value reuse against rare high-value reuse) belongs here, because it
tests that the lifecycle manages economic value rather than popularity.

**§3 The learner.** The V3 `PromotingSharedResidualLearner` with a
lifecycle layer: per-abstraction accounts, a deterministic greedy edit
policy (generate candidates, estimate `ΔJ`, apply the best, recompute,
repeat until no edit improves — feedback-26 §13, which prevents edit order
from becoming a hidden hyperparameter), and the logged candidate queue.
Retention value stated as a real option:
`V_retain(A) = P(return) × value-if-returned − D_retain(A)`.

**§4 Endpoints and instruments.** The four-way substitutability audit
carried forward as standard. New: the survival table (births, survivors,
final useful, by condition), lifecycle regret against the offline oracle,
the migration-cost ledger, and a semantic regression suite — every
dependent task's behavior checked against a stable checkpoint after every
edit, so individually acceptable edits cannot accumulate distortion
(feedback-26 §14).

**§5 Statistical plan.** Development worlds 0-9 with the 0-2 / 3-9 split
reported separately from the start, since worlds 0-2 will again absorb the
debugging. Paired per-world deltas, bootstrap intervals, sign tests.

**§6 Sealed protocol.** Seeds 400-429, untouched until
`V4_CONFIRMATION_PLAN.md` is frozen and hashed into
`tools/check_prereg.py`. Parameter intervals, not signs. One surrendered
control per rung. One registered re-derivation per gate.

**§7 Staged operations and what is deferred.** V4.1 through V4.4 gated in
order; MACRO/LOOP/BRANCH explicitly deferred to V5 with the reasoning
recorded (feedback-26 §"I would not do recursive/macros in V4": five
redundant primitives can produce dozens of redundant macros, and search
debt explodes before the vocabulary is sane).

**§8 Registered diagnostics and failure branches.** V3 §4.3's pattern,
adapted: per-operation refusal controls (dormant-but-returning must
retain; superficially similar but behaviorally incompatible must not
merge; noisy one-off deviation must not fork; genuinely obsolete must
eventually delete), and the named interpretation of each failure mode.

**§9 Execution notes.** Machine constraints from `AGENTS.md`, the
resumable detached-driver pattern, provenance requirements, and the
decision dataset feedback-26 §17 asks for — `(library state, candidate
edit, ΔJ, outcome)` logged from the first run, because it is the training
set a learned restructuring policy would need in V6, and it costs nothing
to collect now.

## Open questions the spec must answer, not inherit

1. **What is DELETE's evidence window?** A fixed number of tasks, an
   exponential decay, or a change-point detector. The real-options framing
   demands some `P(return)` estimate and the spec must freeze one before
   any tuning.
2. **Does deleting an abstraction restore its dependents' private
   residuals, or must they re-adapt?** The honest choice is re-adaptation,
   since the residuals were retired; the cost of that must be in the
   ledger.
3. **Is the V4.1 comparator frozen-V3-PROMOTE or a re-run V3 with the
   same seeds?** Same seeds, re-run, to keep pairing exact.
4. **How is `gamma` (retrieval price) calibrated** if H18 is attempted at
   all — and is ROW's library ever large enough for it to matter?

## Sequencing

1. P1-P2 (V3 loose ends) — small, and they inform the thresholds.
2. P3-P4 (instruments and dormancy world).
3. P5 (validity gate). **If it fails, redesign the world before writing
   the spec.** This is the V3 lesson that cost four redesigns.
4. Write the V4 spec against this blueprint.
5. Develop V4.1 on worlds 0-9.
6. Freeze `V4_CONFIRMATION_PLAN.md`, hash it, unseal 400-429.
7. Retire `notes/v4-sketch.txt` the day the spec exists, as the V3 sketch
   was retired.
