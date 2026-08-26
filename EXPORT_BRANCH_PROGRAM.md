# Export → Composition → Synthesis: the branch program

Status: FROZEN as the branch's decision tree and terminology contract (commit
hash in `tools/check_prereg.py`). Each RUNG gets its own frozen plan before its
own code; this document fixes what may be claimed, in what order, and what each
failure licenses, so that no threshold can be made conditional on data already
seen. Source: review 75 (`reviews/reviewer-feedback-75.txt`).

**This branch does not touch the running formation line.** H53 continues
unmodified, and its eventual result may not be used to change any threshold
registered here.

# The question

Everything ROW has confirmed so far is about ECONOMICS: reusable structure
lowers lifetime cost (V1-V3), the amortization law is quantitative (V4R, V5),
and useful coordinates are made during learning rather than mined afterwards
(H39). None of it establishes that the learner acquired a COMPUTATIONAL
LANGUAGE.

> Does learning produce frozen computational objects that remain useful when
> exported into programs the lifetime never optimized?

**Constitutional criterion.** Synthesis starts when a frozen learned object
composes in a program that was never jointly optimized with it.

**Terminology contract, binding on every document in this repository from now
on.** Until an object passes an export/composition test it is a SHARED OBJECT,
an OPERATOR BASIS element, a SLOT, or an ABSTRACTION. The word PRIMITIVE is
reserved for an object that has passed. Likewise: unseen-task reuse is not
COMPOSITION until unseen PROGRAM COMBINATIONS are demonstrated, and route
optimization is not SYNTHESIS until a compact program variable exists or the
search space is explicitly program-structured.

# Five layers, kept apart so failure localizes

    representation reuse != exportable computation != composition
        != program writing != program synthesis != primitive invention

| Layer | Question |
|---|---|
| Vocabulary | Does a frozen learned operation do useful work elsewhere? |
| Composition | Do frozen operations work in unseen combinations? |
| Write | Can the solution be represented by a compact program variable? |
| Search | Can the correct program be inferred cheaply from examples? |
| Invention | Can new primitives be created when economically justified? |

# The rungs

## E0 — existing-artifact export audit (no new lifetimes)

**E0.1 contextual functional substitutability.** Parameter similarity is not
the identity test (the slot-index-versus-primitive-index lesson). For learned
object `A_i` and teacher primitive `P_j`,
`d(A_i, P_j) = E_{c,x}[ || c[A_i](x) - c[P_j](x) ||^2 ]`, where `c[.]` places
the operation in several positions and surrounding compositions from a frozen
probe set; Hungarian assignment. Report the best assignment, its margin over
second best, the distance, stability across position and across context, and
random/global assignment controls.

E0.1 governs TERMINOLOGY, not E1/E2's verdict: weak teacher alignment does not
falsify composition, because the learner may have found a different but
composable basis.

**E0.2 residual load-bearing.** Four conditions on trained high-reuse tasks:
intact; correct library with the private residual disabled; library disabled
with residual retained; residual re-fit under the frozen library. Score
`R_residual = (L_no_residual - L_full) / (L_no_library - L_full)` when the
denominator is well behaved, with absolute losses always reported. High
reliance means the library is a useful prior and the task program still lives
in private state — a warning, not a stop.

## E1 — frozen-library export (the first decisive new test)

Can a library learned in one lifetime be frozen and used on UNSEEN TEACHER
PROGRAMS by adapting only task-local state? No operator updates. Held-out
programs must be built from primitives that appeared in training, in the
positions they appeared, with disjoint support/query examples and no
performance-based selection of the test set.

Arms: **O** oracle program (tests `C_express`), **R** route/program-only
adaptation (tests `C_express + C_find`), **S** scratch, **F** full finetune
(measures repair: `C_repair = L_R - L_F`), **D** dense frozen control, **W**
wrong/permuted library.

Primary: `G_export = (L_S - L_R) / (L_S - L_O)`, reported only when the
denominator is positive and nontrivial, always beside raw
`L_O, L_R, L_S, L_F, L_D, L_W`.

Registered branches:
- **O beats S/D/W** → the frozen library contains transferable computation.
- **R approaches O and beats S** → the task can cheaply identify and use it.
- **O passes, R fails** → vocabulary exists, the WRITER/SEARCH is missing. Go
  to E3/E5; do not change the library.
- **O fails** → STOP the synthesis interpretation for this substrate. Licensed
  sentence: "ROW learned economically useful shared computation, but these
  objects are not exportable program primitives." Decisive and publishable.

## E2 — support-split compositional generalization (the milestone)

The first rung permitted to make a composition claim. A lifetime is built so
WHOLE PROGRAMS are withheld: `P_train` a strict subset of `{1..K}^D`, with
coverage constraints (every primitive occurs, in every position, in multiple
surrounding contexts, with balanced frequencies). Two strata: **H1** unseen
triple with seen pair-contexts; **H2** at least one adjacent pair never seen —
the stronger systematic test. Split frozen before training.

First condition `r = 1` (or the cleanest high-reuse cell), with a low-reuse
endpoint as a preregistered control. No `r`-sweep yet.

Same arm hierarchy as E1; no operator updates in the compositional arm.
Registered readings: pass; trained-works-but-held-out-fails (pipeline
regularities, not operations); oracle-works-but-inference-fails (composition
without search); oracle-fails (the object INTERFACE is not compositionally
stable — the next architectural target is the operator ABI, not the router).

## E3 — can the program be written?

**E3a post-hoc compilability** first: compile successful soft routes to
`z in {1..K}^D` by frozen argmax/beam without operator updates, charging
`D log2 K` plus registered argument and residual bits; compare predictive
degradation, `D*`, the two-part code, and wrong-library and wrong-depth
controls. **E3b explicit discrete program variable** only if E3a is
inadequate. The claim is never "discrete works" but "at high recurrence an
explicit symbolic program wins predictive AND description economics; at low
recurrence it does not."

## E4 — functional primitive identity (cross-cutting audit)

Does export/composition quality CORRELATE with contextual functional
substitutability, across positions, neighbours, unseen compositions, and later
depth changes? A positive correlation connects primitive-like identity to
compositional behaviour. Cheap prediction without teacher alignment does not
license "statistical reuse only" — look for the alternative basis first.

## E5 — program recognizer / amortized synthesizer

`q_phi(z, alpha | D_support)` over the FROZEN library, trained only on training
programs and tested on E2's held-out programs. Opaque task IDs are bookkeeping
and may carry no predictive information. Compared against oracle,
optimization-based route inference, enumerative/beam search, scratch, and (as a
compute reference) H53-style multi-hypothesis maintenance, on query loss,
examples required, search evaluations, device compute, and program accuracy.

If the recognizer succeeds where structural-head discovery did not, the
registered reading is: **we were searching the wrong latent space — programs,
not partitions of the network.**

## E6 — primitive invention under an explicit birth charge

Not before E1, preferably not before E2. A new object pays `lambda D*(A)` at
creation in the same currency as its savings; the law to test is
`H_remaining * s_bar_program > lambda D*(A)`. Independently vary remaining
horizon, functional recurrence, candidate size, and accidental similarity
without true substitutability. REFUSAL controls are required: no invention in
structureless worlds, on accidental activation similarity, on computations that
do not export, or when future amortization cannot repay birth.

## E7 — residual load-bearing (cross-cutting)

Tracked across trained programs, E1 unseen programs, E2 held-out compositions,
and depth generalization. Prediction: `R_residual` falls as export quality
rises. If residuals stay load-bearing at `r = 1` even when routing appears to
work, the library is a reusable PRIOR, not the task's executable program.

## E8 — length generalization

Only after same-depth E2 passes, and only if the executor genuinely supports
variable length (a hardcoded `D = 3` executor makes failure uninterpretable).
Freeze a depth-3 library; test `D = 2` and `D = 4`, oracle programs first, then
inferred. The four-way interpretation matrix from review 75 is adopted verbatim.

# Run order and the branch decision tree

    Phase 0  E0 + E4 + E7 on existing high-r artifacts (no new lifetimes)
    Phase 1  E1 frozen export
    Phase 2  E2 support-split composition          <- the milestone
    Phase 3  E8 length generalization
    Phase 4  E3 -> E5 write, then search
    Phase 5  E6 primitive invention

    Frozen oracle export?
      NO  -> shared representation, NOT program primitives            [STOP]
      YES v
    Held-out composition?
      NO  -> exportable task features/pipelines, NOT a compositional language
      YES v
    Compact writable program?
      NO  -> composition exists, syntax missing
      YES v
    Program inferable cheaply?
      NO  -> language exists, synthesizer/search missing
      YES v
    New primitives invented economically?
      -> a learned compositional computational language

# Methodological rules (binding for the branch)

1. Freeze before every rung; thresholds may depend on earlier results only
   through this document's decision tree.
2. Existence before discovery: never optimize a recognizer if oracle frozen
   programs fail.
3. Teacher ontology is diagnostic, not ground truth about the best
   representation.
4. Functional substitution beats parameter identity.
5. TRUE must beat plausible WRONG controls for any structural claim.
6. Support/query split mandatory for program inference.
7. Whole PROGRAMS, not examples, define E2's separation.
8. No future-query labels in route/program selection.
9. Absolute losses always reported beside normalized gap-closure statistics.
10. Program success may not depend on operator finetuning.
11. Residual/task-local repair measured separately.
12. No object is called a primitive before export is demonstrated.
13. Unseen-task reuse is not composition without unseen program combinations.
14. Route optimization is not synthesis without a compact program variable or an
    explicitly program-structured search space.
15. Report the embarrassing number.

# Registered predictions

Review 75: partial functional identity rather than clean one-to-one teacher
recovery (E0/E4); residuals more load-bearing than we would like (E7);
reasonable chance E1's oracle passes at high reuse; lower confidence in E1's
route inference, so the modal interesting result is **oracle export works,
learned route inference lags**, localizing the next problem to writing/search
rather than vocabulary; E2 genuinely uncertain and not implied by E1; E3
probably needs a cleaner operator interface than the early hard-library
substrate; E5 optimistic conditional on E2; E6 is where the amortization law
should reappear cleanly, but only if E1/E2 show real export value.

Ours, recorded now so the branch can score us too: we expect E0.2/E7 to show
HIGH residual reliance on the current substrate — H39's Stage-A audits already
found the residual channel carries ~2% of family computation while the routed
basis carries the rest, which predicts LOW reliance for family structure but
says nothing about ordinary task identity, so we register this as the rung we
are least able to predict. We agree with review 75's modal E1 outcome. On E2 we
are pessimistic: the crossover result (V1) shows the basis is preferred only
above `r ~ 0.83`, and H47 showed the learner absorbs distinct latent structure
into one channel, which is the opposite of the stable per-object interface E2
requires. If E2's oracle arm fails, our registered reading is that the operator
ABI — not the router and not the library's contents — is the thing to redesign.


# Amendment 1 (2026-08-25, before any rung is run): five changes that make this a better plan

No data from this branch exists. These changes are made now, while nothing can
be fitted to a result.

## 1. E1 must include a route-expressible substrate, or its STOP is unearned

Review 75 selects "the strongest existing high-reuse artifacts, where the
reusable substrate clearly bought lifetime loss." On this project that selects
the continuous/parameterized-slot family — and our OWN record already says that
family is the least likely to pass an oracle-route test, for a reason that has
nothing to do with exportability:

    Continuous, 64 tasks: own learned mixture route                    0.00191 NMSE
                          teacher route through Hungarian-matched slots 0.00937   (~4.9x)
    Discrete,   64 tasks: own learned hard route                       0.00456
                          teacher route through matched slots           0.00484   (~1.06x)

E1's oracle arm IS "teacher route through the frozen library". The continuous
learner's solution is distributed and compensatory — it is not expressible as a
route through individually matched objects, and we measured that in V1. Running
E1 only on the strongest-loss artifacts would therefore produce a negative about
MIXTURE ROUTING and license a STOP about EXPORT.

Registered: E1 runs on at least one **route-expressible substrate** (the
discrete-library family) alongside the strongest-loss substrate, despite its
weaker lifetime economics. **The branch STOP condition requires the oracle arm to
fail on the route-expressible substrate too.** A failure only on the mixture
substrate is recorded as "mixture routing is not route-expressible" — a
substrate-selection result, not an export result. The dissociation between the
substrate that wins the economics and the substrate that has a vocabulary is
itself a registered possible finding.

## 2. E1.0 — a non-vacuity gate on SEEN programs, before any unseen-program claim

Oracle routing on UNSEEN programs is uninterpretable if oracle routing fails on
programs the lifetime actually optimized. Registered gate, run first:

    freeze the library, supply the teacher route, evaluate on TRAINED programs.
    E1 proceeds only if oracle-route NMSE <= 2.0x the intact model NMSE.

The threshold is set against the known baselines above rather than guessed: the
two measured substrates sit at 1.06x and 4.9x, and 2.0x is near their geometric
midpoint (2.28x), so the gate cleanly separates the substrate whose solution is
route-expressible from the one whose is not. It is recomputed on the actual
artifacts used; a substrate that fails E1.0 has its E1 rows reported as
UNINTERPRETABLE rather than as export failures.

## 3. E1 needs an adaptable-state table as strict as H53's partition

"Adapt only task-local state" is ambiguous in this codebase: task-local means a
36-scalar route code AND a 198-scalar rank-2 residual AND the slot arguments,
against a 16-dimensional three-step task. If the residual is adaptable,
`G_export` may measure residual CAPACITY rather than library value, and every arm
converges for uninteresting reasons.

Registered: every tensor is classified adaptable/frozen per arm, in a table, in
E1's own plan. Arm **R** is run in two registered variants, both reported:

    R-route  route code + registered slot arguments only; residual FROZEN at its
             task-free initial state                         <- PRIMARY
    R-full   residual additionally adaptable                 <- capacity reference

`G_export` is reported for both. The primary claim uses R-route; R-full exists so
that "the library exports" cannot be manufactured by a 198-scalar patch.

## 4. E2 held-out strata must be shown to EXIST before the world is designed

With `K = 6, D = 3` there are 36 adjacent primitive pairs per position boundary
and a lifetime uses 64 programs. The H2 (pair-novel) stratum may be
unconstructible at usable size while also satisfying E2 coverage constraints
(every primitive in every position, in multiple surrounding contexts). This is the
horizon-grid bracketing rule one level up: verify the held-out stratum exists
before building a world around it.

Registered as a Phase 0 gate, **E2-feas**: enumerate the constraint system and
report the maximum achievable |H1| and |H2| under the coverage rules. If |H2| is
below a usable size, E2 must change its generator (more primitives, or depth 4)
BEFORE it is designed — and the plan must state that changing `K` breaks
comparability with every existing artifact.

## 5. E9 — export-constrained formation, registered now as the constructive successor

The branch as written audits existing artifacts for exportability. But the most
robust finding this project has — H29 (the finished span does not contain the
wanted structure), H50 (~0% migration recovery), H51 (three preservation
strategies, all `C_restructure = None`) — is that **a property not built into
formation is not recoverable afterward.** Exportability is such a property. So an
E1 negative on learners that were never trained under any export pressure is the
expected outcome, and treating it purely as a terminus would repeat the mistake
the H39 line already corrected.

Registered in advance, so it cannot be reverse-engineered from E1 results:

**Trigger.** E9 is licensed if E1 oracle fails on the route-expressible substrate
(the true STOP of section 1), OR if E1 passes but E2 oracle fails.

**Design.** A lifetime that carries an export constraint DURING formation:
periodic episodes in which the task must be solved with the library frozen and an
externally supplied route, and/or a held-out-composition term evaluated on
programs withheld from that lifetime. The objects must stay usable under an
interface they do not control while they form.

**Controls, matched and mandatory.** Same architecture, same parameter budget,
same total training compute, same task stream, no export pressure — plus a
WRONG-CONSTRAINT arm (export episodes with permuted routes), because a constraint
that helps regardless of its content is a regularizer, not exportability.
Lifetime economics must be reported beside the export gain: if export is bought
with present cost, that price IS the result.

**Endpoint.** The same E1 test, unchanged, applied afterward. The question is
whether `G_export` under R-route is materially higher for the export-constrained
learner than for its matched control.

**Registered prediction.** We expect E9 to work where the audit fails, and to
cost present-task loss — the same trade H39 found when the argument channel had
to be maintained during learning rather than extracted after it.

## 6. What a clean sweep would, and would not, license

Stated now so that success is not over-read. The teacher programs are fixed-depth
sequential compositions: no branching, no data-dependent control flow, no reuse
of intermediate values, one combinator. A clean pass through E0-E8 licenses
exactly:

> Neural-learned objects can serve as the primitives of a fixed-depth
> sequential-composition DSL, and a program over them can be inferred from
> examples.

It does not license "program synthesis" in the usual sense, and it is a
FOUNDATION claim rather than a capability. Control flow, loops, intermediate
values and macro creation are outside this branch and require a different
teacher.


# Amendment 2 (2026-08-25, review 76, before the E1 rung plan is written): the residual is a repair channel, never evidence

Phase 0 measured that the private residual is INERT AS TRAINED (`R_residual` =
0.0000 in all three worlds) but recovers 41-46% of the trained loss when re-fitted
under a frozen library. Usage and capacity are different quantities, and an
export test that lets a fresh residual adapt could therefore look excellent while
the supposedly exported program contributes almost nothing. That is a false
positive this branch would not otherwise have caught.

Binding from here, on every E1 and E2 test:

## 1. Three interfaces, reported together, never collapsed

    E1-P   PURE PROGRAM      frozen library; ONLY route/program state inferred;
                             private residual absent or frozen at its task-free
                             state.                                    <- PRIMARY
    E1-PR  PROGRAM+RESIDUAL  frozen library; route AND residual inferred.
                             A full-system endpoint; NOT evidence of export.
    E1-R   RESIDUAL ONLY     library contribution neutralised (routed parent
                             replaced by the identity); the SAME residual
                             adaptation budget as E1-PR.

with `Delta_library = L_R - L_PR` reported as the performance uniquely
attributable to the frozen library. Every export or composition claim rests on
**E1-P**. `E1-PR` may never carry one on its own, and the R-route/R-full split
of Amendment 1 is subsumed by this three-way version.

## 2. The oracle-with-no-residual arm is the cleanest vocabulary test, and its
failure branch is frozen now

For an unseen program, `L_oracle-program, eps = 0` asks directly whether the
frozen library contains the computation the program needs. Registered readings:

- **oracle-no-residual works** -> the library genuinely executes that program.
- **oracle+residual works while oracle-no-residual fails** -> the residual is
  REPAIRING the library, and we do NOT have an exported program. Recorded as
  such; no export claim is licensed, and the successor question is the operator
  interface, not the router.

## 3. E7 stops being a one-time audit

Program-only / program+residual / residual-only are reported for every E1 and E2
cell, not once in Phase 0. The trained-lifetime answer ("almost all of it lives
in the library") is established; whether it SURVIVES ON UNSEEN PROGRAMS is the
export question itself.

## 4. A distinction that must appear in every write-up

    what the learned representation EXPORTS
        != what a powerful downstream adaptor can RECONSTRUCT AROUND IT

Unleashing 2,000 steps of private-channel adaptation gives the test-time learner
a capability the lifetime policy deliberately declined to use — the residual sat
idle while the shared route absorbed the task. Reporting the second number as the
first is the same methodological error H37 produced from the other direction.

## 5. What E1 is now testing, restated

Not "is there anything worth exporting" — Phase 0 answered that: task identity is
carried by the routed library, and on the two substrates E1 actually concerns it
is carried there ARCHITECTURALLY, since neither has a residual channel at all.
The sharper question is:

> does a library that fully specifies its TRAINED tasks retain stable semantics
> on a program it never trained on?

with the two live alternatives being STABLE REUSABLE OPERATIONS versus A
DISTRIBUTED ROUTED FUNCTION BASIS THAT ONLY WORKS IN FAMILIAR PROGRAM CONTEXTS.
E1 separates export; E2 separates systematic composition.

## 6. Where the confound can and cannot bite (our note, recorded with the amendment)

The eligible substrate makes the primary path clean by construction: **DISC has
no private residual channel**, so `E1-P` is the only interface it HAS, and
`E1-PR` / `E1-R` are undefined there rather than merely discouraged. The
three-way split becomes load-bearing exactly when a residual-bearing substrate
enters the branch — which is E9's export-constrained formation, since that will
be built on the `pslot` family. Registered consequence: E9's plan inherits this
amendment in full, and its export endpoint must be E1-P.
