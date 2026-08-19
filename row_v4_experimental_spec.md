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

**Citation convention.** `[26 §N]` and `[27 §N]` refer to sections of
[`reviews/reviewer-feedback-26.txt`](reviews/reviewer-feedback-26.txt) and
[`reviews/reviewer-feedback-27.txt`](reviews/reviewer-feedback-27.txt),
with line numbers given as `L###` so the original reasoning can be read
rather than taken on trust. Design choices that came from a review are
cited; choices that came from measurement cite the report instead.

---

# A. Inventory: what exists, what must be built

An agent picking this up should start here. Everything in the first table
is working and tested; everything in the second is V4's job.

## A.1 Already built and usable

| Component | Path | What it gives V4 |
| --- | --- | --- |
| `PromotingSharedResidualLearner` | `src/row/models/promoting_models.py` | PROMOTE, functional abstraction fitting, disjoint proposal/validation probes, refusal ledger. **Frozen — subclass, never edit** (V3 sealed config, `bcc8319`) |
| `LifecycleLibraryLearner` | `src/row/models/lifecycle_models.py` | Lineage records, migration ledger, decision dataset, realized-savings accounts. Adds no decision rule yet |
| `TaskGroupSpec` / world generator | `src/row/task_group_world.py` | The frozen V3 testbed, plus `dormancy=(a,b)` and `dormancy_returns` for the real-options arms |
| Four-way substitutability gate | `src/row/experiments/audit_substitutability.py` | Private / family / global / zero, leave-one-out. The V4 world-validity instrument |
| Future-block probe | `src/row/experiments/audit_future_block.py` | Matched-budget adaptation on held-out tasks; the H14a "no future-adaptation regression" check |
| Sealed scorer pattern | `src/row/experiments/score_v3_sealed.py` | Thresholds transcribed from a frozen plan, written before results are seen. Copy this shape |
| Lifetime runner | `src/row/experiments/mixed_lifetime.py` | `--task-group-eta`, `--family-onset`, `--freeze-basis-at`, `--dormancy`, `--sleeps`, `--operator-slots` |
| Detached driver pattern | `tools/run_v3_taskgroup.py`, `tools/run_v3_sealed.py` | Resumable, `summary.json`-guarded, logs to `tools/*.log` |

## A.2 To be built for V4.1

| # | Component | Prior art to imitate | Done when |
| --- | --- | --- | --- |
| B1 | Conservative lifecycle oracle | `audit_promotion_oracle.py` | Reports `J_oracle` and regret per world; validity gate runs off it (§4.3) |
| B2 | RETAIN/DELETE rule on `LifecycleLibraryLearner` | `PromotingSharedResidualLearner.sleep` | Age-neutral, grace period `G`, `V_retain` per §3.2, every decision in the ledger |
| B3 | Varying-gap dormancy world family | `task_group_world.py` dormancy hooks | Gap lengths sweepable; arms byte-identical to `b` (already unit-tested) |
| B4 | Frequency-versus-value world | §2.3 recipe | LOW and HIGH families measurably differ in per-use saving |
| B5 | Old-task interference endpoint | the §0.1 measurement | Re-evaluates against the FINAL model, never `final_nmse` |
| B6 | Survival-table scorer | `score_group_clustering.py` | Births / survivors / finally-useful per condition |
| B7 | Semantic regression suite | new | Per-dependent behavioral checkpoints, checked after every edit |

## A.3 Commands

    # a V4.1 development lifetime
    python -m row.experiments.mixed_lifetime --config configs/v1.yaml         --model lifecycle --world-seed 0 --task-group-eta 0.9         --task-groups 2 --operator-slots 6 --new-primitive-families         --family-onset 16 --freeze-basis-at 16 --sleeps 24 32 48 64         --output artifacts/v4_dev/world_0/lifecycle

    # the world-validity gate, which must pass before any tuning
    python -m row.experiments.audit_lifecycle_oracle --worlds 0 1 2

    # batches: detached, resumable, 4 jobs (AGENTS.md ceiling)
    python -m tools.run_v4_lifecycle --worlds 0 1 2 3 4 5 6 7 8 9 --jobs 4

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
`P(false abstraction survives) ~ 0` [26 §"The key conceptual shift", L27;
26 §1, L621], which is a far more realistic objective and the one
selection can actually deliver. The corresponding one-line charter is
[26 L612]: *invent hypotheses cheaply, then make persistence expensive.*

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

## 0.2 The resource model, which V4 has to extend

**The problem, stated plainly.** Through V1-V3 the objective was

    J = L_preq + lambda * D_T

with `D_T` the description RETAINED AT THE END. Under that objective a
dormant abstraction costs nothing to keep. Deleting it at task 30 and
deleting it at task 63 give the same `D_T`, and the early deletion can
only hurt `L` if the regime returns. Retention is weakly dominant, the
optimal policy is *keep everything until a final cleanup*, and the
question V4.1 was written to ask — WHEN should an abstraction be retired —
does not follow from the objective at all. Worse, §3.2's retention rule as
first drafted charged `D_retain(A)` as an ongoing rent that `J` never
levies, so the learner would have been optimizing a quantity the scorer
does not measure.

This was caught in review before execution and it is load-bearing, so the
resource model is extended rather than patched:

    J = L_preq
        + lambda * D_T                (final retained description, V1-V3)
        + kappa  * SUM_t D_live(t)    (memory occupancy: bit-time)
        + gamma  * C_retrieval        (cost of a large live library)
        + mu     * C_edit             (search, fitting, migration)

`D_T` answers *how many bits must I ultimately retain?* The occupancy term
answers a question V1-V3 never had to ask: *what does it cost to keep
something AVAILABLE while the world changes?* Those are different
questions, and V4 is the first version in which they come apart — just as
V3 forced the distinction between parameter identity and functional
substitutability. This is not scope creep; it is the resource model
catching up with the experiment.

**How the terms are handled in V4.1.**

- `lambda = ln 2`, unchanged, so every prior result stays comparable.
- `kappa` is SWEPT AND REPORTED AS A CURVE, not fitted. Following the
  two-currency discipline V2 adopted after the shared-residual reversal,
  the headline is reported at more than one price of memory.
- **`kappa = 0` is a registered outcome in its own right**, and an
  informative null: at zero occupancy price, deletion can only pay as
  final-state compression, so any improvement there is compression rather
  than lifecycle. Every timing claim requires `kappa > 0`, and the spec
  says which claims live at which price.
- `gamma = 0` until H18; `mu` per §0.3.

A hard budget `D_live(t) <= B` was considered and rejected for V4.1: it
would also make stale abstractions costly, but by imposing an arbitrary
cliff rather than a price, and it makes the interesting quantity
(willingness to pay for availability) unobservable.

## 0.3 What creation actually costs, and whether it is charged

H17 predicts `r_create > r_delete` because creating an abstraction costs
more than retaining one. That is only true if something nonrecoverable is
actually charged. Decomposed:

    C_create = C_search      candidate generation and clustering
             + C_fit         functional refitting (300 Adam steps each)
             + D_selection   log2(M) for choosing among M candidates
             + L_acquisition prequential loss paid while the abstraction
                             is being discovered rather than used

`D_selection` is already charged in V3's promotion criterion and carries
over. `L_acquisition` is charged automatically, because prequential
scoring bills every example before the update that follows it. `C_search`
and `C_fit` are LOGGED at `mu = 0` in V4.1-V4.3.

**Consequence, registered:** with `mu = 0`, the charged asymmetry between
creating and retaining rests on `D_selection` and `L_acquisition` alone.
If the measured hysteresis gap is of the same order as those two terms,
the investment interpretation is supported; if the gap is much larger,
something else is producing it and H17's mechanism claim is NOT supported
even when its sign is. This comparison is registered as part of H17 rather
than left to interpretation.

---

# 1. Hypotheses

## H14 — Survival (primary; V4.1)

When abstraction birth is imperfect, evidence from subsequent reuse lets a
learner distinguish abstractions that deserved to exist from abstractions
that merely looked promising when they were born. Concretely: a learner
with PROMOTE + RETAIN/DELETE improves on the frozen V3 PROMOTE-only
learner. Registered as TWO outcomes, because they can dissociate and a
single composite would hide which one failed:

- **H14a (economic).** `J` as defined in §0.2 is lower than PROMOTE-only
  on the same worlds and seeds, without degrading lifetime prediction,
  future-task adaptation, or old-task retention beyond their
  non-inferiority margins. Reported at several prices of memory, with
  **`kappa = 0` separated out**: an improvement that survives only at
  `kappa = 0` is final-state compression, not lifecycle management, and
  must be reported as such. The timing claims live at `kappa > 0`.
- **H14b (differential survival).** Survival is selective: the fraction
  of births that survive is materially higher in structured worlds than in
  matched structureless controls. **Scored age-standardized**: only
  abstractions that reached at least `G` tasks of age plus two
  consolidation points are eligible, and a survival-versus-age curve is
  reported alongside the scalar. Without this, "structured survives more"
  could mean only "structured abstractions were born later and had fewer
  opportunities to be deleted", and the two conditions are not guaranteed
  to share a birth-time distribution. This is
  the "births / survives / final useful" table [26 §"One result I would
  especially love from V4", L1522] and the reviewer's own framing of it as
  selection acting over neural abstractions [26 §"I would explicitly
  measure abstraction survival", L405].

H14a can pass while H14b fails (deleting everything shrinks `D` and would
improve `J` in both conditions equally — which is compression, not
selection). H14b can pass while H14a fails (selection works but the bits
saved do not repay the migration cost). Only the conjunction supports the
charter.

**Primary causal comparison:** PROMOTE+DELETE against frozen V3
PROMOTE-only, same seeds, re-run rather than reused, so pairing is exact.
The unfrozen basis and the 2x-updates comparator (§0.1) are reported
alongside as standing comparators.

**Falsifiers.** `J` fails to improve (H14a); or survival rates in
structured and control worlds are within noise of each other (H14b); or
improvement is bought by degrading prediction, future adaptation, or
old-task retention past their margins. A learner that deletes its entire
library in both conditions falsifies H14b even if `J` improves.

**What H14 does NOT claim.** It does not claim the surviving library
matches the teacher's family count. Library size is a diagnostic
throughout this spec and never an outcome (§4.2).

## H15 — Deduplication (V4.2)

Abstractions that prove functionally redundant can be consolidated:
where one refitted replacement can serve all dependents of two
abstractions within tolerance, MERGE reduces library-plus-reference
description length at preserved behavior.

Detection is by MUTUAL SUBSTITUTABILITY, never by parameter similarity or
clustering [26 §6, L913: "Can one replacement serve all dependents of both
abstractions?"] — V3's constitutional rule, and the V3 result that forced
it
(gauge freedom made a parameter mean capture 11.9% of behavioral value
where a functional fit captured 53.4%). The merged candidate is refit
functionally to the union of both dependent sets.

**Falsifiers.** No refitted replacement serves both dependent sets within
tolerance; or MERGE fires on the incompatible-pair control (§2.4).

**Before concluding MERGE failed**, check whether the library is
fragmented at all [26 §7, L997]: cluster the abstractions by mutual
substitutability and count equivalence classes. If V3's five or six
abstractions are five or six functionally DISTINCT objects, the learner
found a finer decomposition than the teacher ontology and MERGE has
nothing to do — a result, not a bug.

## H16 — Safe shared update (V4.3)

A shared abstraction can evolve without corrupting its existing
dependents. A task needing `A + delta` first adapts privately
(copy-on-write); only when the same deviation recurs does it crystallize
into a fork, which is PROMOTE conditioned on an existing parent rather
than a separate mechanism [26 §8, L1053] — so the library becomes
genealogical and delta encoding between parent and child becomes a later
compression opportunity. Distinguishing a one-off noisy deviation from
genuine fork-worthy divergence requires repeated divergence plus
cross-task substitutability plus prospective value [26 §9, L1124], which
is V3's own acceptance ladder applied one level up. Prediction: copy-on-write beats both
update-in-place (which interferes with old dependents) and never-update
(which retains too many private bits) on lifetime cost at matched bits.

**The hazard is already measured, so the question is sharper than
"does interference exist".** §0.1 found the unfrozen substrate degrades
pre-onset tasks by +0.00415 NMSE in 10/10 worlds while improving
post-onset ones. H16's primary question is therefore:

> Can copy-on-write recover the plastic substrate's new-task advantage
> WITHOUT its old-task interference?

with the three-corner comparison — freeze (stable, less plastic), update
in place (plastic, forgetful), copy-on-write (hoped: plastic, stable, and
compact) — as the reporting frame.

**Falsifiers.** Copy-on-write fails to recover a material share of the
unfrozen substrate's new-task advantage; or it recovers the advantage but
inherits the interference; or it fails to beat never-update on `J`, in
which case forking earns nothing.

## H17 — Hysteresis (V4.4; the flagship theoretical claim)

Because creating an abstraction costs more than retaining one already
created, the recurrence at which creation becomes worthwhile exceeds the
recurrence at which an existing abstraction becomes worth retiring:

    r_create > r_delete

Sweeping recurrence slowly up and then down traces a loop, and the area of
that loop is the acquisition-versus-retention cost difference made
visible. This is the sharpest available evidence that abstractions behave
as INVESTMENTS rather than as instantaneous preferences.

**The lag confound and its control, registered** [26 §3, L751; the
decisive test at L784; endorsed in 27 §"The V4 roadmap is well-conceived",
L257]. Any slowly adapting
system exhibits an apparent loop merely by lagging a changing environment:
promotion happens late on the way up and deletion late on the way down,
manufacturing a gap from nothing. The decisive test is a SWEEP-RATE
SERIES. Run schedules at several speeds `v1 > v2 > v3` and measure
`r_create(v) - r_delete(v)`. If the gap vanishes as `v -> 0` the loop was
adaptation lag and **H17 is falsified**. If it converges to a positive
constant the loop is economic. No hysteresis claim may be made from a
single sweep rate.

**H17 runs on the V4.1 mechanism, not the accumulated system.** MERGE and
FORK can themselves shift apparent create and delete thresholds, so the
flagship economic test uses PROMOTE + RETAIN/DELETE only, with the V4.1
configuration frozen. It is executed after V4.2 and V4.3 chronologically
but against the V4.1 learner, and that ordering is registered here so it
cannot later look like a choice made to protect the result.

**A second confound, and a binding constraint on V4.1 because of it.**
Review 26 observes that generational garbage collection — treat young
allocations as speculative and monitor them aggressively, require stronger
evidence to reclaim old ones — *naturally generates hysteresis*. That is
offered as an attraction, and it is one architecturally, but for H17 it is
fatal: if V4.1's retention rule hard-codes any asymmetry between young and
old abstractions, then `r_create > r_delete` is BUILT INTO THE RULE and
its observation is a tautology rather than a finding.

Therefore, registered as a constraint on the earlier rung: **V4.1's
retention rule must be age-neutral.** The only concession is a fixed
GRACE PERIOD of `G` tasks after birth during which an abstraction is
exempt from deletion, which is required because `s_bar(A)` is undefined
before an abstraction has been reused at all, and which applies equally to
every abstraction regardless of when it was born. `G` is fixed in advance
and reported. Any age-dependent retention threshold beyond that grace
period is forbidden in V4.1 and V4.4.

If a generational rule is later wanted for its engineering merits, it is
introduced only AFTER H17 has been tested with the age-neutral rule, and
any hysteresis it produces is reported as a property of the rule rather
than of the economics.

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

A world containing two hidden families with deliberately mismatched
frequency and value: family LOW is used by many tasks but its primitive is
close to something the frozen basis can already express (small per-use
saving), while family HIGH is used by few tasks but its primitive is far
from anything expressible (large per-use saving). Concretely, LOW's new
primitive is drawn as a small perturbation of an existing base primitive
and assigned to 32 tasks; HIGH's is drawn independently and assigned to 8.
A rational lifecycle compares `N_reuse * s_bar`, not `N_reuse` [26 §11,
L1201], and must retain HIGH.

**Validity check, run before any tuning** — the counts alone do not
guarantee the world tests what it claims. Measure `s_bar` for both
families with the deep-copy no-reuse branch and require

    N_HIGH * s_bar_HIGH  >  N_LOW * s_bar_LOW

by a margin recorded in advance. If 8 rare-but-valuable tasks do not
outweigh 32 frequent-but-cheap ones, the world does not distinguish value
from popularity and its per-use gap is widened until it does. Registered because a popularity heuristic would pass every
other world in this spec, and because `V_retain` as specified in §3.2
multiplies rate by per-use saving precisely so that it can.

## 2.4 Per-operation refusal controls

Each operation gets a world in which it must NOT fire [26 §12, L1241],
the V4 analogue of V3's four-way audit:

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
stays reproducible from its own fingerprint. It adds lineage records [26 §15, L1357: track lineage from day one,
"not because the learner needs all of this, because you will"], a
migration ledger, and a decision dataset, and NO decision rule — the
rules are specified here rather than inherited from an implementation.

## 3.2 Retention value, as a real option

An abstraction with no current use is not thereby worthless; its regime
may return [26 §2, L688, with the A->B->A world at L719]. Retention therefore asks what the option to reuse is worth:

    V_retain(A) = p_return * (C_recreate + L_reacquire + lambda * D_temporary)
                  - kappa * D(A) * T_retained

**Why this and not "expected future savings".** The spec permits DELETE
followed by a later re-PROMOTE, so the counterfactual to retaining is NOT
"never have `A` again". It is "delete it and recreate it if evidence later
warrants". If the regime returns, BOTH policies eventually enjoy the reuse
savings, so those savings cancel and cannot be the value of retention. The
first drafting of this rule used `p_return * H * s_bar(A)`, which prices
the wrong thing and would have overvalued retention by the entire future
benefit of an abstraction that deletion would have recovered anyway.

What retention actually buys is avoided REDISCOVERY: the search and
refitting to build it again (`C_recreate`), the prequential loss paid
while relearning before it is available (`L_reacquire`), and the private
bits dependents carry in the interim (`D_temporary`). What it costs is
occupancy — `D(A)` bits held for `T_retained` further tasks, priced at
`kappa` (§0.2). That is a real option in the strict sense: a price paid to
preserve immediate access to something that may become useful again.

**Estimating the terms online, without leakage.** `C_recreate` and
`L_reacquire` are estimated from the abstraction's OWN birth record —
what its promotion cost and how long its members took to become
predictable — which is in lineage from V3 onward. `p_return` is the
recency-weighted reuse rate with time constant `tau`. No term may use
teacher structure, held-out data, or the future stream; any that cannot be
computed from observed history is not admissible in the rule and belongs
only in the oracle.

**`s_bar(A)` and the zero-reuse case.** Mean realized saving per reuse is
counterfactual — reuse cost against what private adaptation would have
cost — so it is measured with a deep-copied no-reuse branch at the point
of reuse. That branch is experimental machinery: it is declared here, its
compute is logged, and it touches no lifetime state. **After the grace
period an abstraction with zero realized reuse has `s_bar(A) = 0`, not
undefined.** The first drafting left it undefined and therefore
effectively infinite, which would have made the single worst abstraction
in the library immortal — the exact opposite of the intended rule.

**Delete-and-re-promote is a legitimate policy, not a failure.** In the
returning arm the learner may retire an abstraction during the gap and
promote an equivalent one when the regime resumes. Whether that beats
retaining depends on `D_retain` against the cost of re-promotion, which
is precisely the economic question. Scoring therefore uses `J` and the
resurrection is recorded in lineage as a re-birth with its parent noted;
"must retain" is a prediction about which policy wins, not a rule imposed
on the learner.

## 3.3 Deterministic edit policy

Operation order must not become a hidden hyperparameter [26 §13, L1273]. At each
consolidation: generate all candidate edits, estimate `delta J` for each,
apply the single best, recompute, and repeat until no candidate improves
`J`. The candidate queue is logged in full. Greedy refactoring is chosen
over global search deliberately, so that a later comparison against global
search is possible; the log makes that comparison cheap.

## 3.4 Addressing, and a recorded prediction about it

V4.1 addresses abstractions by physical index, which is adequate while
PROMOTE and DELETE are the only operations: deletion tombstones an entry
rather than compacting the list, so live references never shift.

Review 26 predicts this will not survive MERGE, because consolidating two
abstractions forces every dependent of both to have its reference
rewritten, and that an indirection layer — stable abstraction IDs mapped
to current implementations, i.e. a symbol table — will be forced by the
economics rather than chosen. **The prediction is recorded here so that if
it happens it is recorded as FORCED rather than designed**, which is the
same discipline that made PROMOTE's own necessity credible. V4.1 does not
build it; V4.2 reports whether reference-rewrite cost in the migration
ledger justified it.

## 3.5 Deletion semantics

Deleting an abstraction does NOT restore its dependents' private
residuals — those were retired and their bits reclaimed. But "the
dependents re-adapt" is not a specification until the data they re-adapt
FROM is named, and the admissible answers differ enormously in what they
imply.

**DELETE is admissible only if, for every live dependent, one of these
holds**, checked before the edit and logged either way:

1. the dependent can be substituted onto another existing abstraction
   within `epsilon` on its validation probe; or
2. a private fallback is reconstructed FROM THE SAME REPLAY MEMORY every
   learner already has (the `TaskReplayBuffer` contents for that task, and
   nothing else), with the reconstructed residual's bits charged to
   `D_task` and the reconstruction steps charged to `C_edit`.

If neither holds the deletion is REFUSED and the refusal is recorded.
Using a task's evaluation examples or any teacher structure to rebuild a
dependent is leakage and is prohibited; using stored training examples is
legitimate precisely because that memory already exists and is already
paid for in every condition, including the comparators.

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

**Library size is diagnostic, never an outcome** [26 §7, L1035: "Do not
score V4 by library size -> 2"]. Note the tension with [27 §"The one thing
I would watch most carefully", L293], which warns that a wandering library
size is the biggest risk to the narrative. Both are right: size is what a
reader will look at first, and scoring it would still be wrong. It is
reported prominently and scored never. Five abstractions for
two teacher families may be a finer decomposition than the teacher
ontology rather than fragmentation. Scoring "library size -> 2" would
reward matching an ontology the learner cannot see. The outcomes are `J`,
substitutability, and adaptation cost.

## 4.3 Lifecycle regret

Because ROW gives the whole future stream post hoc, an offline clairvoyant
oracle can compute the best achievable lifecycle decisions [26 §10,
L1156], separating "the representation is bad" from "the representation is
fine but the learner forecasts future reuse poorly":

    R_lifecycle = J_online - J_oracle

**Defined tractably, because the unrestricted version is exponential —
but it MUST scan deletion times.** The oracle takes the online learner's
births as given and optimizes each abstraction's deletion time
independently, holding all other decisions at the learner's values:

    for each abstraction A:
        for t_d in {sleep checkpoints} union {infinity}:
            evaluate J with A deleted at t_d
        keep the best t_d

which is `O(|L| * T)` in sleeps, entirely affordable.

**An earlier drafting of this oracle compared only "retain for the whole
lifetime" against "delete at birth", and that version is rejected**: it is
blind to *useful early, obsolete later*, which is the central case V4
exists to study, and it was to be used as the world-validity gate. It
could therefore have reported that a world has no room for DELETE in
exactly the world where timed deletion is the whole opportunity. The
timed version remains a lower bound (it cannot exploit interactions
between abstractions), so regret is still conservative, and that
restriction is stated wherever regret is quoted.

This separates "the representation is poor" from "the representation is
fine but the learner forecasts future reuse poorly", which is otherwise
impossible to distinguish and becomes essential under nonstationarity.
**Oracle advantage over PROMOTE-only is also the V4.1 world-validity
gate:** if the oracle cannot beat PROMOTE-only by a meaningful margin,
the world cannot test DELETE and is redesigned before anything is tuned.

## 4.4 Migration ledger

Every edit logs its cost [26 §4, L825]: reference rewrites, re-validation
forwards, refitting steps, and any re-adaptation forced on dependents. If
these are treated as free, lifecycle operations look artificially
attractive. Logged from V4.1 and CHARGED from V4.2 (`mu > 0`). Audit B is resolved —
extra optimization made the comparator worse, not better (§0.1) — so the
conditional clause in an earlier drafting no longer applies and `mu = 0`
stands for V4.1.

## 4.5 Semantic regression suite

Individually acceptable edits can accumulate distortion [26 §14, L1321]:
`A+B -> C`, later `C+D -> E`, each within tolerance, the composition
not. Every dependent
task's behavior is therefore checked after every edit against a stable
behavioral checkpoint recorded at its last validated state — unit tests
for a learned library.

## 4.6 Decision dataset

`(library state, candidate edit, delta J, outcome)` logged from the first
run [26 §17, L1427]. Not used by V4, and deliberately so: it is the training set a learned
restructuring policy would need in V6, and collecting it now costs
nothing.

---

# 4A. Fixed operational constants

Frozen here so they cannot drift during development:

| constant | value | rationale |
| --- | --- | --- |
| consolidation points | after tasks 24, 32, 48, 64 | V3's schedule, unchanged, so promotion behaves identically |
| grace period `G` | 8 tasks after birth | `s_bar` is undefined before first reuse; age-neutral otherwise (§H17) |
| deletion deadline (permanent arm) | 30 tasks after regime end | makes "eventually delete" falsifiable |
| substitutability tolerance | `epsilon = 0.02` NMSE | V3's frozen value, on disjoint probes |
| `lambda` | `ln 2` | the two-part exchange rate, unchanged |
| `mu` | 0 in V4.1, > 0 from V4.2 | edit cost logged first, charged once MERGE makes it material |
| `gamma` | 0 until H18 | retrieval logged, not charged |
| `kappa` (occupancy) | SWEPT, reported as a curve, with `kappa = 0` a registered outcome | the new currency (§0.2); a single value would hide whether a result is compression or lifecycle |
| `gamma` calibration (H18 only) | swept, not fitted | reported as a curve `J(gamma)`; if no `gamma` in a plausible range produces an interior optimum, H18 is reported as not binding at ROW scale (plan open question 4) |

---

# 5. Statistical plan

Development worlds 0-9, with the 0-2 / 3-9 split reported separately from
the outset. Worlds 0-2 will again absorb debugging and instrument design;
3-9 are the clean internal generalization check, and any sealed interval
is widened to encompass drift between them rather than centered on the
ten-world mean [27 L239, adopted after V3 did this only retrospectively].

Paired per-world deltas reported directly; world-level means with
bootstrap intervals; exact binomial sign tests for counts. Two
initializations per model on worlds 0-2, averaged within world before any
cross-world aggregation.

---

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

**The surrendered control for V4.1** is the frequency-versus-value world
(§2.3): it is run on development worlds and reported, but not on sealed
seeds. It tests a property of the retention RULE rather than of the
learner's behavior in the target regime, and the rung's sealed budget is
better spent on the dormancy pair.

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
them. The reasoning is recorded rather than assumed [26 §"I would not do
recursive/macros in V4", L493; endorsed 27 L285]: five redundant
primitives can produce dozens of redundant macros, so search debt explodes
before the vocabulary is sane. V4 must earn V5 by showing the vocabulary
can remain sane under lifecycle pressure.

Also deferred, unchanged from V3 §8: equivalence-class routes,
hidden-basis coordinate discovery, granularity discovery, memory
hierarchy, task-boundary removal, functional IBP, and the staged LLM
bridge.

## 7.1 Definition of done, per rung

A rung is complete when all of the following hold. An agent should not
advance to the next rung on a partial verdict.

**V4.1 SURVIVAL**
1. World-validity gate passes: conservative oracle beats PROMOTE-only by a
   margin recorded before any tuning (§4.3).
2. `tau` and `G` selected on the no-dormancy structured world and frozen;
   every value tried is logged.
3. Development on worlds 0-9 with 0-2 and 3-9 reported separately.
4. H14a and H14b each scored against the standing comparator set (frozen,
   2x-updates, unfrozen basis) with old-task interference reported.
5. All four refusal controls (§2.4) run; failures reported, not tuned away.
6. `V4_CONFIRMATION_PLAN.md` frozen with intervals, hashed into
   `tools/check_prereg.py`, sealed block 400-429 run and scored, outcomes
   appended to `PREDICTIONS.md` whichever way they fall.

**V4.2 DEDUPLICATION** — as above, plus the fragmentation check [26 §7]
run BEFORE concluding anything about MERGE, and a report on whether
reference-rewrite cost forced the indirection layer predicted in §3.4.

**V4.3 NONSTATIONARITY** — as above, plus the three-policy comparison
(update-in-place, never-update, copy-on-write) with old-task interference
as a first-class endpoint, since §0.1 has already measured the hazard to
be real.

**V4.4 INVESTMENT** — as above, plus the sweep-rate series at no fewer
than three speeds. A hysteresis claim from a single rate is not a result.

---

## 7.2 The trilemma V4 is really about

The unfrozen-basis audit (§0.1) turned a hypothetical into a measurement,
and it frames the whole program more sharply than any single operation
does. Three strategies for a shared substrate meeting new recurring
structure:

| strategy | new-task prediction | old-task retention | description length |
| --- | --- | --- | --- |
| freeze the substrate | worse | preserved | good (V3: -63%) |
| update it in place | **better** (+5,766 nats) | **damaged** (+0.00415, 10/10) | poor |
| copy-on-write / library evolution | ? | ? | ? |

The first two corners are measured. V4.3 asks whether the third exists. If
copy-on-write captures the plastic substrate's plasticity without its
forgetting, that may matter more than MERGE — it would be a constructive
answer to the stability-plasticity dilemma in a setting where all three
axes are measured simultaneously rather than traded two at a time.

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

## 9.1 Working agreements for whoever runs this

These are the conventions the previous three versions learned the hard
way. Violating them does not usually produce an error; it produces a
result that cannot be trusted later.

1. **Never edit a frozen class.** `PromotingSharedResidualLearner` is part
   of V3's sealed configuration. Subclass it. The same will apply to V4's
   own classes once `V4_CONFIRMATION_PLAN.md` is hashed.
2. **A gate that cannot fail is not a gate.** Before tuning any operator
   on a world, run the validity instrument and be willing to redesign the
   world. Four V3 testbeds failed this way, and each failure was cheaper
   than the false positive it prevented.
3. **Measure interference against the FINAL model.** A task's
   `final_nmse` is written at that task's completion and cannot see later
   drift in shared parameters. This produced a clean, wrong +0.00000 in
   0/10 worlds once already (§0.1).
4. **Structural edits are justified by substitution, never identity.**
   Parameters are gauge-equivalent; functions are not.
5. **Log refusals as carefully as actions.** The refusal ledger is
   evidence, not debug output.
6. **A hung shell during a batch means load, not failure.** Check artifact
   counts before killing anything; lifetimes write `summary.json` only at
   completion.
7. **Append outcomes to `PREDICTIONS.md` whichever way they fall**, and
   never edit a prediction after its experiment has run. Corrections are
   appended and dated.
8. **Report the number that would embarrass you.** V3's absolute-refusal
   criterion is recorded as falsified; V3's prediction win is recorded as
   holding only against a frozen substrate. Both are in the specs because
   a reader would otherwise have to discover them independently.


---

# 10. Pre-run checklist

1. **P5, the validity gate, before anything is tuned.** Run the
   conservative oracle (§4.3) against PROMOTE-only on the V4.1 world. If
   the oracle's advantage is not materially positive, the world cannot
   test DELETE and is redesigned first. This is the V3 lesson that four
   testbed failures paid for, and it is the single most important step in
   this list.
2. Run P2's two standing predictions — P-2026-08-19-G (horizon) and
   P-2026-08-19-H (lifetime sweep for `N*`) — both of which locate where
   the deletion horizon should sit.
3. Select `tau` once on the structured world WITHOUT dormancy, and log
   every value tried. The dormancy worlds stay held out.
4. Build the varying-gap dormancy family (§3.2) and the
   frequency-versus-value world (§2.3), each with its own validity check.
5. Implement RETAIN/DELETE on `LifecycleLibraryLearner`, with the
   age-neutral rule and the grace period.
6. Develop on worlds 0-9, reporting 0-2 and 3-9 separately from the first
   run.
7. Freeze `V4_CONFIRMATION_PLAN.md` with parameter intervals, hash it into
   `tools/check_prereg.py`, and only then unseal 400-429.
8. Retire `notes/v4-sketch.txt` on the day this spec is frozen, as the V3
   sketch was retired.

# 11. Open questions this spec does NOT settle

Recorded so they are not mistaken for oversights, and so a later choice
cannot be presented as having been planned:

1. **Does deletion interact with promotion's own threshold?** A learner
   that knows it can delete later may rationally promote more freely. V4.1
   holds PROMOTE's configuration frozen at V3's values precisely so this
   interaction is not silently exercised, but the interaction is real and
   V4.2 should measure it.
2. **Is the conservative oracle tight enough to be useful?** If measured
   regret is dominated by the oracle's own restriction (births taken as
   given, deletion only at birth), the instrument is too blunt and needs
   the timed-deletion version.
3. **What is the right unit of forgetting?** Old-task interference is
   measured as mean NMSE degradation over pre-onset tasks. Whether a
   worst-case or a per-task-threshold measure is more appropriate is
   unresolved; both are logged.
4. **Is `kappa` an honest currency or a knob?** Memory occupancy is
   introduced because the objective demanded it (§0.2), but nothing in ROW
   fixes its price. The sweep makes the dependence visible rather than
   hidden, and `kappa = 0` is reported as a registered null — but if every
   interesting V4 result requires a `kappa` outside a defensible range,
   that is a finding about the resource model and must be reported as one
   rather than absorbed into a choice of constant.
5. **Whether MERGE and DELETE can be separated at all in practice.**
   Deleting one of two redundant abstractions is observationally close to
   merging them. If V4.1's DELETE turns out to be doing deduplication
   implicitly, V4.2's contribution shrinks and the spec should say so
   rather than staging an operation with nothing left to do.
