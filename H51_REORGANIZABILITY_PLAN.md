# H51: the reorganizability testbed — is cheap restructuring a property of the wake representation?

Status: DRAFT (freeze commit to be recorded in `tools/check_prereg.py` before
any code). Development worlds 0-2, the `schema_groups = 2` world at K = 4. No
new worlds, no sealed seeds.

**Numbering note.** Review 72 proposes three successors it calls H51
(provenance), H52 (decomposable innovation basis), and H53 (overcomplete
provisional schemas), and then advises starting cheaper with a synthetic
reorganizability testbed. This plan IS that testbed; the reviewer's H51 and H52
enter as its arms `R_1` and `R_2`. The reviewer's H53 is out of scope here and
keeps its label for a possible successor. Ledger entries use THIS file's
numbering (H51 = the testbed) with the mapping recorded in `PREDICTIONS.md`.

# Question

H50 established that ordinary wake leaves a representation that bounded,
budget-matched migration cannot move into the fertile organization: at 4,096
steps under the TRUE partition, recovery of L_4's retrospective separation is
~0% and future fertility does not recover. The registered reading is that
organization is a formation-time property.

H51 asks whether that is a property of THIS wake or of finished representations
in general:

> Does a wake process that preserves separately-addressable task innovations
> make fertile counterfactual organizations locally reachable, at matched
> present performance?

The endpoint is a property of the representation, not of the learner's score:

    C_restructure(R) = min m in the scored grid such that the H50 SEPARATION
                       rule holds when the H50 migration operator is applied
                       to R.

# Arms (wake representations)

Every arm is a full 64-task lifetime on the same world seeds (0, 1, 2), same
`schema_groups = 2` generator, same task order, same examples, same replay
policy, same evaluation sets, `pslot_count = 2`, `slot_args K = 4`.

- **R_0 — ordinary.** The existing frozen `M_4` artifacts (H49/H50's `M`).
  Re-used, not re-run.
- **R_1 — provenance-preserving wake** (reviewer's H51). Identical learner and
  identical predictions; retirement additionally STORES each retired task's
  pre-retirement local state (route code, alphas, private residual) as a frozen
  innovation trace that never enters prediction. Bitwise control: with tracing
  off the artifact must equal `R_0` tensor-for-tensor (the H39 frozen-argument
  rule). Migration may initialize from and optimize these traces.
- **R_2 — decomposable-innovation wake** (reviewer's H52). The wake learner
  represents each task's private innovation in a shared, separately addressable
  component basis: `FactorizedLifecycleLearner`'s residual factorization
  (residual = `W . a + eps`, implemented and exercised as branch C) composed
  with `ParameterizedSlotLearner`'s two parameterized slots. These are today
  two SEPARATE `kind`s in `learned_lifetime.py` and do not compose; H51
  requires implementing the composition (a `pslot_factorized` kind sharing both
  parents' begin/end-task hooks). Components stay addressable for the whole
  lifetime; no schema is imposed and no group label is ever visible. Required
  equivalence check: at `schema_dim = 0` (or components frozen at zero) the
  composed learner must reproduce `R_0` bitwise, and at `slot_args = 0` it must
  reproduce the branch-C artifact bitwise.
- **R_3 — oracle organized wake.** The existing frozen `L_4` artifacts (trained
  under the TRUE partition). Re-used, not re-run. This is the reference the
  recovery fraction is taken against and the instrument's positive control:
  H49 already measured SEPARATION on `L_4` at m = 0 (margins 0.60 / 0.30 /
  0.65), so the endpoint is known to be expressible.

Retrain-and-select is NOT a branch here; `R_3` is its oracle ceiling.

# Balance gates (the critical control; checked BEFORE any migration is scored)

Reorganizability must not be bought with capacity. Per world, each of `R_1`,
`R_2` against `R_0`:

- **G1 present performance.** Lifetime cumulative prequential Gaussian log loss
  within 10%, and mean final family-task NMSE within 20%.
- **G2 description.** Retained-`D*` proxy (the H49 accounting) within 20%.
- **G3 parameter budget.** Trainable-scalar count within 20%; the count is
  reported exactly for every arm.
- **G4 migration budget.** The migration operator's trainable-variable count is
  matched across arms within 20% and the STEP count matched exactly. Where an
  arm exposes strictly more migration variables (R_1's traces, R_2's
  components), a matched-budget companion is required: a variant of that arm
  whose extra variables are frozen, so a difference in `C_restructure` cannot be
  a difference in migration degrees of freedom alone.

An arm failing G1-G3 is **unscoreable for the causal claim** and reported
descriptively only, with the failing quantity as a covariate (the V5.1 confound
rule). An arm failing G4 is rerun with the companion before its
`C_restructure` is read.

# The migration operator and scoring instrument

Verbatim H50, no changes: `audit_h50_reorganization.py`'s `migrate()` and the
H49 re-fit scorer, same candidate list (TRUE, WRONG-A, WRONG-B, RANDOM-1,
RANDOM-2 with H49's seeds `[49, world, r]`, and **SHAM**), same optimizer groups
and learning rates, same seeded batches shared across arms, same scored grid
`m in {0, 16, 64}` with ALL 64 LOO tasks, substitutability only for TRUE /
best-wrong / SHAM, sibling diagnostics computed last and selecting nothing.
Migration-cost ingredients (steps, wall-clock, post-migration old-task loss,
functional drift) recorded for every cell.

Reusing the instrument unchanged is deliberate: H51's independent variable is
the wake representation, and nothing else may move.

# Decision rules

Per arm, using H50's rule exactly (`Delta(h) = log C_LOO(h) - log C_LOO(TRUE)`):

- **SEPARATION at m** iff every wrong/random candidate's `Delta >= +0.15` in
  >= 2 of 3 worlds AND `log C_LOO(SHAM) - log C_LOO(TRUE) >= +0.15` AND TRUE's
  substitutability exceeds every scored wrong candidate's by >= 0.30.
- `C_restructure(arm)` = the smallest scored `m` with SEPARATION, or `None`.
- **Recovery fraction** at each m, against `R_3`, as in H50 Amendment 1.

Outcomes, registered:

1. **REORGANIZABILITY IS REAL** — `C_restructure(R_1) < C_restructure(R_0)` or
   `C_restructure(R_2) < C_restructure(R_0)` (i.e. finite where R_0's is
   `None`, or smaller), with the arm passing G1-G4. The headline is which
   preserved structure bought it and at what present cost.
2. **PARTIAL** — no arm reaches SEPARATION, but recovery fraction at m = 64
   increases monotonically in the preservation ordering
   `R_0 <= R_1 <= R_2` by >= 0.25 in >= 2 of 3 worlds. Reported as a graded
   effect, licensing a stronger preservation mechanism (the reviewer's H53),
   not a discovery claim.
3. **FORMATION-TIME, CONFIRMED** — neither arm separates and recovery fractions
   are flat (all pairwise differences < 0.25). Then path-dependence is not an
   artifact of retiring task-local evidence or of dense innovation storage, and
   the open question moves to what `R_3` does differently that no
   preservation-only change reproduces.

Non-vacuity, all required before any verdict:
- `R_1` with tracing off is bitwise `R_0`.
- `R_2` at zero component rank reproduces `R_0`'s objective family (documented
  equivalence, checked on the smoke world).
- Every arm's migration changes its parameters (mean absolute change > 0) and
  improves post-migration family NMSE over its own m = 0.
- The `R_3` positive control reproduces H49's m = 0 separation margins to
  within numerical tolerance when re-scored by this run's instrument.
- Traces and components are never used in prediction during wake (verified by a
  frozen-channel ablation: zeroing them mid-lifetime must not change any wake
  prediction for `R_1`).

# Registered predictions

Reviewer (review 72): `ordinary ~ provenance < decomposable < oracle` in
reorganizability; a tweak that only preserves residual vectors will not be
enough; highest prior on some version of `R_2`.

Ours: `R_1` moves nothing (`C_restructure = None`, recovery within 0.1 of
`R_0`) because H50 already showed the retrospective objective, not the missing
evidence, is what fails to discriminate; `R_2` improves the recovery fraction
but does not reach SEPARATION at m <= 64. We register outcome 2 as our modal
expectation and outcome 3 as second. If `R_2` DOES separate, we expect it to
carry a measurable present-performance cost near the G1 boundary, and that cost
is the quantity to report, not the separation alone.

# Cost

Two new wake arms x 3 worlds = 6 lifetimes (pool of 3, `slots = 12`), plus the
G4 companions where required. Migration + scoring reuses H50's cost model:
~5 h per representation at `m in {16, 64}` with all 64 LOO tasks, so ~10-15 h
total background. `R_0` and `R_3` rows are re-used from H49/H50, not re-run.

Economy, if compute forces cuts (review 71 rule): drop the `m = 4` grid point
entirely; never thin the LOO sample; run substitutability only on TRUE /
best-wrong / SHAM.

# Explicitly out of scope

A learned proposer; the reviewer's H53 overcomplete provisional schemas; any
change to the migration operator or the scoring instrument; the
`gamma * C_restructure` objective as a trained term (it is a conceptual
criterion here, measured, not optimized); sealed seeds.

# Amendment 1 (2026-08-24, before any code): how R_1 is realized, and why it needs no lifetime

Discovered while reading the existing artifacts, before implementation.

1. **Provenance never enters prediction, so a provenance-preserving wake IS
   R_0.** Review 72's H51 stores which computation came from which experience;
   a trace that is never predicted from and never optimized during wake cannot
   change the wake trajectory. The arm is therefore not a different lifetime —
   it is R_0 plus retained evidence that the MIGRATION operator may use. R_1
   requires no new run, and passes G1-G3 exactly (bitwise R_0).
2. **The trace is not already available, and this was measured, not assumed.**
   The frozen `M_4` artifacts already carry `history.pt`: each task's route
   code, private `eps`, and effective residual snapshotted AT TASK COMPLETION,
   before any sleep. Replay keeps training completed tasks afterwards, so those
   snapshots differ substantially from the final stored state — mean relative
   difference over the 72 recorded tasks, worlds 0/1/2: `eps` 0.91 / 1.03 /
   1.00, route code 0.49 / 0.61 / 0.49. H50's migration saw only the final
   state. (Had this come out ~0, R_1 would have been vacuous and is recorded
   here as the check that decided it.)
3. **Two R_1 variants, both scored.**
   - `R_1a` **trace initialization**: H50's migration verbatim, except each
     family task's route code and `eps` are initialized from its completion
     trace before masking. Trainable-variable count identical to `R_0` — G4
     holds exactly, with no companion needed.
   - `R_1b` **trace recombination** (the reviewer's "sleep can recombine these
     traces"): during migration a task's residual is
     `sum_j c_j trace_j + eps`, over the traces of the tasks CO-ASSIGNED to its
     slot under the candidate partition, with `c` learned at the task learning
     rate. The wrong/random partitions supply the G4 control by construction:
     they give each task the same number of coefficients over a differently
     chosen trace set, so a `TRUE` advantage cannot be a coefficient-count
     advantage. `SHAM` recombines over all family tasks' traces (same
     coefficient count as the largest partition cell, no structure).
4. `R_2` is unchanged and still needs the composed learner and three
   lifetimes.
5. Registered prediction for the new split: `R_1a` moves nothing (recovery
   within 0.1 of `R_0`); `R_1b` improves the recovery fraction but does not
   reach SEPARATION, because the traces are exactly the EXTENSIONAL population
   that H39's Stage-A audits already found infertile (post-hoc extraction from
   the ordinary learner's realized population cost 2.59x online acquisition).
   If `R_1b` DOES separate, the H39 conclusion needs revisiting, and that —
   not the separation — is the headline.
