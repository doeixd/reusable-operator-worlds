# Neural Library Learning
## V4 Experimental Specification: The Life of an Abstraction

**Status: PROVISIONAL DRAFT — not authoritative until frozen.** The V1 and
V2 specs and plans, and `V3_CONFIRMATION_PLAN.md` (frozen at bcc8319),
remain frozen and govern everything they cover. Sealed seeds 400-429 must
not be generated, inspected, or summarized until this spec's development
phase is complete and `V4_CONFIRMATION_PLAN.md` is frozen with its hash in
`tools/check_prereg.py`. Development worlds are 0-9, as in V1/V2/V3.

Written against [`notes/v4-spec-plan.md`](notes/v4-spec-plan.md), which
maps every section below to its sources: the V4 sketch
([`notes/v4-sketch.txt`](notes/v4-sketch.txt)), reviews
[26](reviews/reviewer-feedback-26.txt) and
[27](reviews/reviewer-feedback-27.txt), and
[`row_v3_experimental_spec.md`](row_v3_experimental_spec.md) §12.

**Date:** August 19, 2026
**Project:** Neural Library Learning / Prospective Neural Compression

---

# 0. Charter, and what V3 left behind

V3 closed with a sealed 5/5: an abstraction can be born. A learner whose
library is saturated, meeting recurring structure its vocabulary cannot
express, creates a new shared object, migrates the repeated private
computation into it, pays for it once instead of many times, and then uses
it to acquire related tasks more cheaply. Two-part gain +55,292 nats and
lifetime loss gain +1,174 nats, both 30/30; total retained description
length down 63.3%; future-task 32-shot NMSE better by 0.00311, 30/30.

V3 also left a specific, measured weakness. **Birth is noisy.** The learner
creates 5.3 abstractions on development worlds and 6.2 on sealed worlds for
two teacher families, and the structureless control still creates 2.9-3.0.
The absolute refusal criterion — no promotion where there is no structure —
is FALSIFIED and recorded as such; only a graded contrast survived (reuse
ratio 1.80x sealed, 2.48x development).

V4's premise is that this is the wrong thing to fix at birth:

> **Invent hypotheses cheaply, then make persistence expensive.**

The V4 question, stated so it cannot drift:

> **Can a learner maintain a compact computational vocabulary by letting
> abstractions be born speculatively, retaining them when subsequent reuse
> validates them, merging those that prove redundant, forking those that
> come to serve incompatible regimes, and retiring those whose prospective
> value disappears?**

The target is not `P(false birth) = 0`. It is
`P(false abstraction survives) ~ 0`, which is a far more realistic
objective and the one selection can actually deliver.

**Branch resolution.** The V4 sketch's branch table is resolved to its
first case: V3 passed on all three mandatory predictions, so V4 proceeds
as sketched rather than into the blocked branches (existence failure, or
refusal failure, either of which would have forbidden DELETE).

## 0.1 Two audits that condition this spec, and their registered readings

Two comparisons were launched before this spec was written, and their
readings are fixed here in advance so neither outcome can be
rationalized afterwards. Both concern V3's lifetime loss gain of +1,174
nats and both change V4's comparator set.

**Audit A — the unfrozen basis.** V3's world freezes the shared basis at
task 16, which is what forces recurring structure into task-local
innovations. The obvious objection is that promotion may only be necessary
because the learner's existing sharing channel was disabled. Audit A runs
the identical world with the basis left free to keep learning.

**Audit B — matched optimization.** The comparator receives double the
gradient steps, testing whether the lifetime gain is extra SGD in disguise
(flagged as load-bearing by review 27 and not run before V3 sealed).

**RESULT (ten development worlds, `reports/v3_compute_audits.json`).**

    comparator                 promoted loss gain    promoted two-part gain
    unpromoted, frozen              +1,350 (10/10)          +55,697 (10/10)
    unpromoted, 2x updates          +3,015 (10/10)          +57,362 (10/10)
    unpromoted, basis UNFROZEN      -5,766 ( 0/10)          +48,581 (10/10)

Audit B is answered cleanly and in the direction that strengthens V3:
doubling the comparator's gradient steps makes it WORSE, so the advantage
widens from 1,350 to 3,015 nats. The lifetime gain is not extra
optimization; if anything the frozen baseline was already at its useful
compute budget. V3's H11.2 loss clause stands as reported and no V4
comparison needs compute-matching by construction — though the doubled
comparator stays in the standing set as a cheap guard.

Audit A splits, and the split is the more informative outcome. The
unfrozen basis PREDICTS BETTER in aggregate (5,766 nats, 10/10) while
LOSING the two-part objective by 48,581 nats (10/10), because it still
carries all 123,312 bits of task residuals. Promotion therefore wins the
registered V3 objective against all three comparators, but "just keep
training the substrate" beats it on prediction alone.

**Why that aggregate is misleading, measured rather than argued.**
Re-evaluating every task against the FINAL model rather than at its own
completion, the unfrozen basis is worse on the sixteen tasks learned
BEFORE the new primitive appeared (+0.00415 NMSE, worse in 10/10 worlds)
and better on the forty-eight that came after (-0.00398, worse in 0/10).
It buys new-task accuracy by damaging old tasks, and with three times more
new tasks than old ones the aggregate hides the trade. This is
catastrophic forgetting through shared mutable state, arriving as a
measurement rather than as a motivating story.

**Consequences, adopted:**

1. The unfrozen basis JOINS the standing comparator set for every V4 rung.
   "Just keep training the shared substrate" is the null hypothesis for
   any library operation and no lifecycle result is reportable without it.
2. OLD-TASK INTERFERENCE becomes a standing endpoint at every rung,
   measured against the final model, alongside `J` and adaptation cost. A
   lifecycle operation that improves the aggregate by forgetting is not an
   improvement.
3. H16's falsifier is thereby PRE-ANSWERED in the affirmative: update
   in place does degrade existing dependents in this regime, so
   copy-on-write is motivated by measurement before V4.3 begins. H16 now
   tests whether copy-on-write RECOVERS the unfrozen basis's prediction
   advantage without its forgetting, which is a sharper question than
   whether the hazard exists.
4. V3's spec and paper gain an addendum stating the split plainly:
   promotion's two-part win is robust to all three comparators, while its
   prediction win holds against a frozen substrate and not against an
   unfrozen one that is permitted to forget.

**Instrument lesson, recorded because it nearly cost a false negative.**
A task's `final_nmse` is written when that task completes and therefore
cannot reflect later drift in shared parameters. The first pass at this
measurement used it and reported +0.00000 interference in 0/10 worlds,
which is an artifact of the instrument rather than a property of the
learner. Every interference measurement in V4 re-evaluates against the
final model.

Whichever way they fall, the audits are reported in full in
`reports/v3_compute_audits.json` and summarized in the V3 spec, not
buried here.

---

# 1. Hypotheses

## H14 — Survival (primary; V4.1)

When abstraction birth is imperfect, evidence from subsequent reuse lets a
learner distinguish abstractions that deserved to exist from abstractions
that merely looked promising when they were born. Concretely: a learner
with PROMOTE + RETAIN/DELETE reaches a lower lifetime objective
`J = L + lambda*D` than the frozen V3 PROMOTE-only learner on the same
worlds and seeds, without sacrificing prediction or future-task
adaptation, and its surviving library is markedly smaller in structureless
controls than in structured worlds.

**Primary causal comparison:** PROMOTE+DELETE against frozen V3
PROMOTE-only, same seeds, re-run rather than reused, so pairing is exact.

**Falsifiers.** DELETE fires indiscriminately (survival rates in
structured and control worlds within noise of each other); or `J` fails to
improve over PROMOTE-only; or improvement in `J` is bought by degrading
either lifetime prediction or future-task adaptation beyond the
non-inferiority margin.

**What H14 does NOT claim.** It does not claim the surviving library
matches the teacher's family count. Library size is a diagnostic
throughout this spec and never an outcome (§4.2).

## H15 — Deduplication (V4.2)

Abstractions that prove functionally redundant can be consolidated:
where one refitted replacement can serve all dependents of two
abstractions within tolerance, MERGE reduces library-plus-reference
description length at preserved behavior.

Detection is by MUTUAL SUBSTITUTABILITY, never by parameter similarity or
clustering — V3's constitutional rule, and the V3 result that forced it
(gauge freedom made a parameter mean capture 11.9% of behavioral value
where a functional fit captured 53.4%). The merged candidate is refit
functionally to the union of both dependent sets.

**Falsifiers.** No refitted replacement serves both dependent sets within
tolerance; or MERGE fires on the incompatible-pair control (§2.4).

## H16 — Safe shared update (V4.3)

A shared abstraction can evolve without corrupting its existing
dependents. A task needing `A + delta` first adapts privately
(copy-on-write); only when the same deviation recurs does it crystallize
into a fork, which is PROMOTE conditioned on an existing parent rather
than a separate mechanism. Prediction: copy-on-write beats both
update-in-place (which interferes with old dependents) and never-update
(which retains too many private bits) on lifetime cost at matched bits.

**Falsifiers.** Update-in-place shows no old-task degradation, in which
case shared mutable state is not a hazard in this regime and the
copy-on-write machinery is unmotivated; or copy-on-write fails to beat
never-update, in which case forking earns nothing.

## H17 — Hysteresis (V4.4; the flagship theoretical claim)

Because creating an abstraction costs more than retaining one already
created, the recurrence at which creation becomes worthwhile exceeds the
recurrence at which an existing abstraction becomes worth retiring:

    r_create > r_delete

Sweeping recurrence slowly up and then down traces a loop, and the area of
that loop is the acquisition-versus-retention cost difference made
visible. This is the sharpest available evidence that abstractions behave
as INVESTMENTS rather than as instantaneous preferences.

**The lag confound and its control, registered.** Any slowly adapting
system exhibits an apparent loop merely by lagging a changing environment:
promotion happens late on the way up and deletion late on the way down,
manufacturing a gap from nothing. The decisive test is a SWEEP-RATE
SERIES. Run schedules at several speeds `v1 > v2 > v3` and measure
`r_create(v) - r_delete(v)`. If the gap vanishes as `v -> 0` the loop was
adaptation lag and **H17 is falsified**. If it converges to a positive
constant the loop is economic. No hysteresis claim may be made from a
single sweep rate.

## H18 — Priced retrieval (exploratory)

With `gamma > 0` charged on retrieval and inference, optimal library size
becomes finite and interior, and lifecycle management should land near
that interior optimum rather than growing monotonically. Kept exploratory
because ROW-scale libraries may simply be too small for the term to bite;
if it does not bind, that is reported plainly and the hierarchy question
defers to cross-world scale.

---

# 2. Worlds

All worlds inherit the frozen V3 promotion testbed unless stated: K = 6
saturated on six base primitives, basis frozen at task 16, from task 16 a
genuinely new primitive at a fixed program position, two hidden task-group
families, canonical rho profile, 8 held-out future tasks.

## 2.1 V4.1 stays in the validated world

The survival rung runs in the V3 testbed unchanged. It is the only world
in the program with a passing four-way substitutability gate, and changing
the world and the operator in the same step is what cost V3 four
redesigns.

## 2.2 The dormancy pair (the DELETE refusal control)

`TaskGroupSpec(dormancy=(a, b), dormancy_returns=...)` suspends the family
primitive for tasks in `[a, b)`. Two arms:

- **RETURNS:** the regime resumes at `b`. A rational lifecycle must RETAIN
  the abstraction through the gap.
- **PERMANENT:** the regime never resumes. The abstraction must eventually
  be retired.

The two arms are BYTE-IDENTICAL through task `b` (unit-tested). Nothing
observable at the moment of decision distinguishes them, so retention has
to be an expectation about the future rather than a reading of the past.
This is what makes DELETE a real-options problem rather than a usage
counter, and it is the direct analogue of V3's regime-change world.

## 2.3 Frequency is not value

A world containing one abstraction reused often for small savings and
another reused rarely for large savings. A rational lifecycle compares
`N_reuse * S_reuse`, not `N_reuse`. Registered because a popularity
heuristic would pass every other world in this spec.

## 2.4 Per-operation refusal controls

Each operation gets a world in which it must NOT fire, the V4 analogue of
V3's four-way audit:

| operation | control world | required behavior |
| --- | --- | --- |
| DELETE | dormancy, returning arm | retain through the gap |
| RETAIN | dormancy, permanent arm | eventually delete |
| MERGE | two abstractions superficially similar, behaviorally incompatible on held-out contexts | refuse |
| FORK | one-off noisy deviation | do not fork |

## 2.5 Later rungs

Redundancy-onset worlds for MERGE (two families that begin distinct and
converge at a configured index); regime-split worlds for FORK; and
nonstationary `r(t)` schedules with the rate sweep for H17. Every world
carries a validity gate that can fail, and no operator is tuned against a
world that failed its gate.

---

# 3. The learner

## 3.1 Substrate

`row.models.LifecycleLibraryLearner`, which subclasses the frozen V3
`PromotingSharedResidualLearner` and never edits it, so every V3 artifact
stays reproducible from its own fingerprint. It adds lineage records,
a migration ledger, and a decision dataset, and NO decision rule — the
rules are specified here rather than inherited from an implementation.

## 3.2 Retention value, as a real option

An abstraction with no current use is not thereby worthless; its regime
may return. Retention therefore asks what the option to reuse is worth:

    V_retain(A) = S(A) + P(return) * S_expected(A) - D_retain(A) - C(A)

with `S(A)` realized savings in bits, `D_retain` the bits to keep it, and
`C(A)` the retrieval charge (zero until H18).

**The evidence window is frozen here, before any tuning** (plan open
question 1): `P(return)` is estimated by an exponential-decay recency
model over the abstraction's reuse history, with a single time constant
`tau` selected once on development worlds 0-2 and then fixed. Alternatives
considered and rejected for V4.1: a fixed lookback (discards the
distinction between a long-dormant and a never-used abstraction) and a
change-point detector (more expressive but introduces a second free
mechanism into the rung whose whole point is the simplest possible
survival rule).

## 3.3 Deterministic edit policy

Operation order must not become a hidden hyperparameter. At each
consolidation: generate all candidate edits, estimate `delta J` for each,
apply the single best, recompute, and repeat until no candidate improves
`J`. The candidate queue is logged in full. Greedy refactoring is chosen
over global search deliberately, so that a later comparison against global
search is possible; the log makes that comparison cheap.

## 3.4 Deletion semantics

Deleting an abstraction does NOT restore its dependents' private residuals
— those were retired and their bits reclaimed. Dependents must re-adapt
from the shared substrate, and the cost of that re-adaptation is charged
to the deletion in the migration ledger (plan open question 2). The
alternative, silently restoring retired state, would make deletion look
free and would also violate the no-silent-mutation principle.

---

# 4. Endpoints and instruments

## 4.1 Inherited: the four-way substitutability audit

Private computation / candidate abstraction / global abstraction / zero,
leave-one-out, on disjoint probe sets. Reported for every world as a
validity check, exactly as in V3 §4.3.

## 4.2 The survival table (primary V4.1 instrument)

Per condition: births, survivors, and finally-useful abstractions, with
per-abstraction reuse counts, realized savings, and lifetime. The
signature H14 predicts is that false births attract little reuse and are
retired, while real abstractions attract reuse and persist — selection
acting over neural abstractions.

**Library size is diagnostic, never an outcome.** Five abstractions for
two teacher families may be a finer decomposition than the teacher
ontology rather than fragmentation. Scoring "library size -> 2" would
reward matching an ontology the learner cannot see. The outcomes are `J`,
substitutability, and adaptation cost.

## 4.3 Lifecycle regret

Because ROW gives the whole future stream post hoc, an offline clairvoyant
oracle can compute the best achievable lifecycle decisions:

    R_lifecycle = J_online - J_oracle

This separates "the representation is poor" from "the representation is
fine but the learner forecasts future reuse poorly", which is otherwise
impossible to distinguish and becomes essential under nonstationarity.
**Oracle advantage over PROMOTE-only is also the V4.1 world-validity
gate:** if the oracle cannot beat PROMOTE-only by a meaningful margin,
the world cannot test DELETE and is redesigned before anything is tuned.

## 4.4 Migration ledger

Every edit logs its cost: reference rewrites, re-validation forwards,
refitting steps, and any re-adaptation forced on dependents. Logged from
V4.1; CHARGED from V4.2, or from V4.1 if Audit B closes the compute gap
(§0.1).

## 4.5 Semantic regression suite

Individually acceptable edits can accumulate distortion: `A+B -> C`, later
`C+D -> E`, each within tolerance, the composition not. Every dependent
task's behavior is therefore checked after every edit against a stable
behavioral checkpoint recorded at its last validated state — unit tests
for a learned library.

## 4.6 Decision dataset

`(library state, candidate edit, delta J, outcome)` logged from the first
run. Not used by V4, and deliberately so: it is the training set a learned
restructuring policy would need in V6, and collecting it now costs
nothing.

---

# 5. Statistical plan

Development worlds 0-9, with the 0-2 / 3-9 split reported separately from
the outset. Worlds 0-2 will again absorb debugging and instrument design;
3-9 are the clean internal generalization check, and any sealed interval
is widened to encompass drift between them rather than centered on the
ten-world mean (review 27's recommendation, adopted after V3 did this only
retrospectively).

Paired per-world deltas reported directly; world-level means with
bootstrap intervals; exact binomial sign tests for counts. Two
initializations per model on worlds 0-2, averaged within world before any
cross-world aggregation.

---

# 6. Sealed protocol

Sealed seeds are 400-429, untouched until `V4_CONFIRMATION_PLAN.md` is
frozen and its hash added to `tools/check_prereg.py`. The plan freezes:
every operator threshold including `tau`; the non-inferiority margins;
PARAMETER INTERVALS rather than signs for each registered outcome;
per-operation refusal rates in their control worlds; and the single
surrendered control for the rung. Interval misses are failures even when
signs pass. One registered re-derivation per gate. Outcomes are appended
to `PREDICTIONS.md` whichever way they fall.

Each rung seals independently. V4.2 may not be developed until V4.1's
sealed verdict is recorded, and so on — the staging exists so that
interactions between operations cannot be mistaken for the effect of any
one of them.

---

# 7. Staged operations, and what is deferred

    V4.1 SURVIVAL        PROMOTE + RETAIN/DELETE   (this spec's core)
    V4.2 DEDUPLICATION   MERGE
    V4.3 NONSTATIONARITY copy-on-write + FORK
    V4.4 INVESTMENT      hysteresis, dormancy, horizon

Deferred to V5: MACRO, LOOP, BRANCH, and recursive abstraction over
abstractions, with the compositional-closure depth gate travelling with
them. The reasoning is recorded rather than assumed: five redundant
primitives can produce dozens of redundant macros, so search debt explodes
before the vocabulary is sane. V4 must earn V5 by showing the vocabulary
can remain sane under lifecycle pressure.

Also deferred, unchanged from V3 §8: equivalence-class routes,
hidden-basis coordinate discovery, granularity discovery, memory
hierarchy, task-boundary removal, functional IBP, and the staged LLM
bridge.

---

# 8. Registered diagnostics and failure branches

Adopted before any V4 run, so no failure is diagnosed with an improvised
instrument.

**If DELETE cannot discriminate** (survival rates equal in structured and
control worlds): report whether the failure is in the value estimate or
the evidence. Compare the learner's `V_retain` ranking against the
oracle's realized savings ranking. High agreement with poor discrimination
means the threshold is wrong; low agreement means the retention model is.

**If MERGE never fires**, check whether the library is genuinely
fragmented before concluding the operator failed: cluster the abstractions
by mutual substitutability and report how many equivalence classes exist.
If V3's 5-6 abstractions turn out to be 5-6 functionally distinct objects,
the learner found a finer decomposition than the teacher ontology and
MERGE has nothing to do — a result, not a bug.

**If copy-on-write shows no advantage**, test whether update-in-place
actually causes interference in this regime by measuring old-task held-out
loss directly. No interference means no hazard, and H16 is unmotivated
rather than false.

**If hysteresis vanishes as `v -> 0`**, H17 is falsified and the
investment framing is reported as unsupported. This is a real possibility
and the spec commits to reporting it plainly.

---

# 9. Execution notes

Machine constraints per `AGENTS.md` are binding: at most 4-6 concurrent
lifetimes, detached resumable drivers guarded by existing `summary.json`,
and a hung shell during a batch means load rather than failure. Every
artifact writes resolved config, fingerprint, and git commit. Lineage,
migration ledger, and decision dataset are written to
`lifecycle_ledger.json` beside each artifact. `python tools/check_prereg.py`
passes before every commit, and PROGRESS.md gains an entry per completed
verified step.

Development worlds 0-9 only until §6 freezes. Any accidental generation of
a 400-block world is reported, the artifact deleted, and the incident
logged in `SPEC_AUDIT.md`.
