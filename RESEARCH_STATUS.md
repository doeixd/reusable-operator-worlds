# ROW research status: what is live, blocked, pending, closed

This is an INDEX, rewritten in place whenever a line of research changes state,
and committed with the change that caused it. The scientific record lives
elsewhere and is append-only: verdicts, hypotheses and corrections in
`PREDICTIONS.md`; completed steps in `PROGRESS.md`; results in `reports/`. If
this file disagrees with those, they win and this file is stale.

Last rewritten: 2026-09-22, after a program-level review of what the plans
leave reachable: SG0 Revision 3 (a registered NO-HEADROOM sub-triage that
pre-decides its own successor) and PROGRAM_LADDER_PLAN Revision 2 (the
learner-rung gate is unreachable; ladder decision 7). Nothing is running.

# Recommended next action: SG0, the synthesis headroom gate

- **State:** DRAFT `SYNTHESIS_OPPORTUNITY_GATE_PLAN.md` (2026-09-22), working
  hypotheses SG1-SG4 in `PREDICTIONS.md`, generator reasoning in
  `notes/identifiability-sketch.txt`. Tier 0, artifacts already held, minutes of
  compute, no world, no learner, no lifetime. Needs PI approval
  (ladder decision 5) before implementation.
- **Why it exists.** Counting by cause, eight rungs have failed for absence of
  opportunity rather than learner failure: the 2026-08-31 loop census, E6.2, E7,
  H47, H48b, H49, E9 and L0d's four gates. E5 (writer costlier than search and
  short of oracle quality) and E5.1 (space 3.58e7x, search seconds 3.30x, at
  oracle parity) plausibly belong to the same list.
- **The hypothesis it tests.** Support data here OVERDETERMINES the route, so
  the incumbent search is already at the ceiling and no proposer can pay. If
  true, "search is easy" (E5/E5.1) and "there is no ambiguity" (L0d) are one
  fact about the GENERATOR, and every proposer, discovery and uncertainty rung
  on this substrate is unmeasurable.
- **What it measures.** COMMITMENT REGRET, the query cost of choosing the
  support-optimal route rather than the query-optimal one, against a registered
  sampling-noise floor, with a selection-split guard so the hindsight-best route
  is not fit and scored on the same query examples. None of the four L0d gates
  measured this: they asked whether the selected route CHANGES, not whether it
  is RIGHT, which is why all four are consistent with regret 0 AND with regret
  large.
- **Registered triage and its consequences, fixed in advance:**
  - `NO-HEADROOM` - the amortized-proposer branch on this substrate is CLOSED,
    PX7 is retired as UNMEASURABLE HERE (not refuted), and the L0d census is
    WITHDRAWN rather than left unfrozen. **A registered SUB-TRIAGE (Revision 3)
    then splits it, from quantities the same run already records:
    `NO-HEADROOM-EVIDENTIAL` (near-ties are non-trivial and disagree on query,
    so restricting the support distribution is a live intervention and SG3 is
    licensed) versus `NO-HEADROOM-SATURATED` (the route space is functionally
    separated, SG3 prerequisite (a) fails here, and the recommendation is to
    bank). Secondary and diagnostic: it decides a program question, not a
    scientific one. Working hypothesis SG5 in `PREDICTIONS.md`.**
  - `HEADROOM-LOCALIZED` - the localizing statistic is the measured ambiguity
    source the census has lacked; freeze the census around identifiability, not
    support size.
  - `HEADROOM-UNEXPLAINED` - freeze nothing; this is the route-margin hunt
    repeating, and that cost three withdrawn candidates.
- **Contingent successor** (ladder decision 6, now PRE-DECIDED BY MEASUREMENT
  rather than left as an either/or): under `NO-HEADROOM`, the sub-triage above
  says which. `EVIDENTIAL` -> pursue the identifiability generator (restrict the
  SUPPORT input distribution while leaving query untouched, so
  evidence-equivalent routes diverge on query). `SATURATED` -> bank the
  economics and negative-results papers and stop opening rungs on this
  substrate. The PI still decides; what changed is that the decision now arrives
  with its measurement attached instead of after the cheap answer has landed.

# Awaiting a PI decision (nothing running)

## Track B: online learnability of the strong (rotated) substrate

- **State:** SO4 CLOSED as `SO4_FAILS`. The online-failure stop rule (the THIRD
  Track-B stop rule in `POST_E6_RESEARCH_PROGRAM.md`) stands with three blocks of
  evidence. No B2 statement. C2 and the program ladder's learner rungs stay
  closed.
- **What the three online runs jointly show** (working hypothesis, not a
  verdict): the protocol acquires the substrate in SOME worlds at every seed
  tried - 6 of 10 development worlds across SO2 (1/3, seed 5000), SO3 (3/3, seed
  6000) and SO4 (2/4, seed 7000) - and within one world the replay stream alone
  moves terminal error by up to an order of magnitude. World identity dominates;
  the SO2-P consolidation settings did not improve on the unchanged protocol
  anywhere.
- **Development worlds 0-9 are now ALL spent** for this protocol.

| step | result | commit |
|---|---|---|
| rotated substrate spec | rotation makes iteration AND branching necessary | (spec + Amendments 1-2) |
| G5 / G5R / diagnosis / Stage D | not learnable online; localized downstream of operator fitting; `BUDGET_LIMITED` | 8b9b802 |
| SO0 / SO1 / SO1R | no budget axis; oracle passes, learned fails; routes recoverable, so the wall is co-formation | 1042878 / dd57e24 / 9c8d058 |
| J0 / J1 | routing tracks library quality; search in the loop locks in | f9bcc30 / 01305d4 |
| J1c / J1c-R / J2A | length curriculum acquires OFFLINE at 2 seeds; exports 64/64 unseen programs | d4e6808 / 3bf6a59 / 686e4d3 |
| curriculum cost | same compute, +124 tasks of data | aa79aa3 |
| SO2 | `SO2_FAILS`: terminal 1/3, margin 3/3 (one stream per world) | 6a4f707 |
| census / SO2-P / correction | loss is post-acquisition and recency-weighted; one-world rescue; replay arms stream-confounded | 3d87a0f / afb1ac6 / b250973 |
| SO3 | `SO3_FAILS`; unchanged protocol passed 3/3 fresh worlds; stream spread 0.009-0.066 | 0e0bd67 |
| SO4 | `SO4_FAILS`: terminal 2/4, margin 3/4; world 6 negative margin | acf8af9 |

- **World-quality census: DONE** (plan, module and tests `8f0369a`; report
  `reports/world_quality_census.json`; PROGRESS 2026-09-16). Tier 0, committed
  reports only, no new world. Registered triage: **MIXED**.
  - Pooled over 24 cells: stage-2 terminal error predicts stage-3 outcome
    (Spearman +0.669, permutation 0.0010; stage-2 end-of-task +0.698) but the
    fail/pass median separation is 1.66x, short of the registered 2x.
  - **Within the four worlds whose streams disagree** (SO3 w4, w5; SO4 w7, w8),
    with world identity constant, separation CLEARS the bar: 3.07x (stage-1),
    2.73x (stage-2), 2.41x (stage-2 end-of-task). In 3 of the 4, the failing
    stream is the worst-prefix stream, and in all three it is stream 0.
  - **SO4 w8 inverts it:** best prefix -> worst stage 3 (0.173, 33 tasks lost);
    worst prefix -> pass (0.016). Any prefix-based account must explain this.
  - **World 6's collapse is visible in its prefix:** the worst stage-2 values in
    the census (end-of-task 0.175-0.309) and all three streams then failed.
  - The stage-1-to-stage-2 improvement ratio carries nothing (rho -0.043).
- **Library-geometry census: DONE** (plan, module and tests `79b1b74`; report
  `reports/library_geometry_census.json`; PROGRESS 2026-09-16). Tier 0, 24 frozen
  stage-2 libraries, all guards including the bitwise anchor. Registered triage:
  **GEOMETRY-EXPLAINS**, but fired by ONE measure and resting on ONE cell.
  - `route_margin` (how sharply stage-2 support data identifies a route): pooled
    rho -0.514, permutation 0.0045; failing/passing medians 1.35 vs 4.20 (0.32x),
    and 1.20 vs 14.76 within mixed-outcome worlds.
  - The w8 clause passes by 2.6% (s0 1.605 passed, s2 1.201 failed, s1 1.170
    failed worst), so one cell carries the label.
  - Per run: -0.50 (SO2) / **+0.12 WRONG SIGN (SO3)** / -0.64 (SO4); the measure
    spans 0.23-67.48 and the plan states it is not comparable across task sets.
  - High margin neither guarantees success (w6 s1: 2.55 -> 2.17) nor is necessary
    (w9 s0: 0.87 -> passed).
  - The registered PRIMARY measure `min_pair` carries NOTHING (+0.095, separation
    1.01): the "near-duplicate slots" hypothesis is not what fired.
  - Among the 12 lowest-error cells the ordering inverts (`route_margin` +0.014;
    `effective_rank` +0.483; `mean_pair` +0.448).
- **Route-margin normalization census: DONE** (plan, module and tests `0d01702`;
  report `reports/route_margin_normalization.json`; PROGRESS 2026-09-16). Tier 0,
  all guards including a clean recovery gate (16/16 in every library). Registered
  triage: **SURVIVES**, on ONE marginal test.
  - Test A (within-world concordance, `route_margin`): k = 5/7, exact
    Binomial(7,1/3) tail 0.0453.
  - Test B (`self_margin`, library-intrinsic and comparable across worlds): rho
    **+0.143**, permutation 0.264, separation 1.11 - null, and SIGN-FLIPPED
    against `route_margin`'s pooled -0.514.
  - Within-world z-scoring drops `route_margin` from -0.514 to -0.221: most of the
    pooled effect was cross-world scale.
  - `self_margin` concordance 3/7 (chance); stage-2 ERROR reference 1/7 (worse
    than chance at picking the failing stream).
  - Test A's survival plausibly re-detects that stream 0 is usually the failing
    stream, which the world-quality census already established.
- **MECHANISM HUNT CLOSED.** All three candidates are withdrawn: slot duplication
  (flat), route identifiability (null once comparable), prefix error (1/7 per
  world). Two cells rule out any monotone account (w6 s1 highest `self_margin`
  fails at 2.17; w9 s2 nearly lowest passes at 0.016). The defensible position is
  that online staged formation is world- AND stream-dependent with NO IDENTIFIED
  MECHANISM, which is what `paper/draft.md` states.
- **Successor options (PI DECISION 7, now informed by ALL THREE censuses):**
  1. **Prefix-screen intervention** (registered, needs a new world band): act on
     stages 1-2 - budget, stopping criterion, or stream selection - with the
     stage-2 statistic as a pre-stage-3 eligibility screen. Its weakness is SO4
     w8.
  2. **Explain SO4 w8 first, Tier 0:** the one cell where a good prefix failed.
     Its saved prefix and stage-3 model exist, so the question "what did stage 3
     do to a good library" is answerable with no new lifetime.
  3. **Stage-3 reliability intervention** (the SO2-P line, now weakly supported:
     the census says the entering library matters, not only stage 3).
  4. **Stop the online line** and report it as world-dependent, which is what the
     paper now says.
  5. **NEW, and now Claude's recommendation: a Tier 0 normalization step.**
     `route_margin` is the only geometric quantity that tracked the outcome, but
     it is scale-incomparable across task sets, flips sign in one run, and its
     registered clause turned on a 2.6% gap in a single cell. Recompute it
     normalized within world, or on a common task set, over the 24 cells already
     in hand, and see whether the relation survives. Minutes, no new world.
  - **Option 5 is now DONE** (the normalization census above) and it did not
    rescue the screen: the comparable measure is null.
  - **Claude's recommendation: option 4 for Track B - stop the mechanism hunt and
    report the online result as world-dependent**, which the paper already does.
    No world band is justified for options 1 or 3 on the present evidence: every
    observable tried is either null when made comparable, or worse than chance
    per world.
  - **Next research step instead: L0d**, the program-inference census
    (`L0D_INFERENCE_CENSUS_PLAN.md`), which is runnable now, needs no world, and
    belongs to the program ladder rather than to Track B.
- **Any further online lifetime needs a new development band** (decision 5):
  worlds 0-9 are spent.

# Drafted, awaiting PI decisions

## Program ladder: compositional operators -> control flow -> higher-order programs

- **State:** DRAFT `PROGRAM_LADDER_PLAN.md`, REVISED 2026-09-22 (Revision 1:
  SG0 now precedes the inference rungs; recommended order and PI decisions
  updated). Hypotheses PX1-PX9 in `PREDICTIONS.md` (committed `fbb547a`);
  SG1-SG4 added 2026-09-22.
- **Learner rungs L1-L8 are gated on an UNREACHABLE condition** (Revision 2,
  2026-09-22). The gate was "SO3 or a successor clears online learnability";
  SO2, SO3 and SO4 all failed, worlds 0-9 are spent, and Track B's recommended
  disposition is to stop. Nothing can clear it, so the rungs are not pending -
  they are undecided. **Ladder decision 7** puts the fork to the PI: either
  amend stop rule 2 for OFFLINE scope on frozen J2A libraries (every claim
  scoped "offline, supplied curriculum", with a matched non-staged baseline and
  the curriculum compute charged to the claim), or CLOSE L1-L8 and let the
  ladder's live content be its Tier 0 gates plus SG0. L0a-L0d and SG0 train no
  learner and are unaffected either way.
- **L0d preflight COMPLETE, no PX7 verdict.** Full census still drafted in
  `L0D_INFERENCE_CENSUS_PLAN.md`.
  - PI-run preflight at 4a6aa47 exited 0 in about 21 seconds: twelve libraries,
    192 library/task pairs, 576 support measurements. Independent scorer,
    provenance/finite/anchor checks and preregistration/invalid checks pass.
    Report: `reports/l0d_preflight.json`; logs and reproducible descriptive
    counts: `reports/l0d_preflight_2026-09-17/`.
  - All 96 staged-library/task pairs select identical routes with support
    128/32/8 and retain identical query NMSE (pooled median 0.00572); all pass
    0.05. The 96 control pairs all fail at every support, despite route changes.
    The present evidence reduction has no observed hard-route commitment cost
    on the usable staged libraries.
  - **Bounded ambiguity gate COMPLETE** (`L0D_AMBIGUITY_GATE_PLAN.md`: one
    existing library, support 1/2/4 against its
    saved 128-example anchor): all 16 tasks retain the exact 128 route
    and query error at supports 4/2/1; independent scorer passes. This closes
    the evidence-size opportunity on this fixed depth-three library. No larger
    inference comparison is justified by these negative gates alone.
  - The full census draft needs defined posterior/commit-late semantics,
    assignment-independent eligibility, measured ambiguity, exhaustive triage,
    and memory/cost sizing before freezing. Its old teacher-route and monotone-
    gap gates are not accepted; see the preflight plan's design audit.
  - The earlier proposed depths 3-5 and ENUM/OPT/beam/posterior/commit-late grid
    stays deferred. The preflight's depth-three executor equivalence passes;
    deeper execution, uncertainty semantics and eligibility need new gates.
  - **Depth-four execution gate COMPLETE:** commit `1faf77d`, report
    `reports/l0d_depth4_execution_gate.json`, archived logs under
    `reports/l0d_depth4_execution_gate_2026-09-17/`. On STAGED5000/world 0,
    16 deterministic length-four programs all pass support-only exhaustive
    ENUM at query NMSE 0.05 (median 0.00872); diagnostic mapped routes agree.
    This is scoped frozen-vocabulary usability evidence, not a PX7 verdict.
    Beam, posterior and commit-late remain untested and require their own
    frozen protocol.
  - **Depth-five memory gate COMPLETE:** run 2026-09-17 at `2db0254`, scored,
    archived and recorded 2026-09-22 (report
    `reports/l0d_depth5_memory_gate_v2.json`; operational records, validation
    and query addendum in `reports/l0d_depth5_memory_gate_2026-09-17/`;
    PROGRESS 2026-09-22). Classification `MEMORY_SAFE_SEARCH`.
    All 12^5 = 248,832 routes searched in 243 blocks of 1,024 for one
    deterministic length-five program on the frozen STAGED5000/world-0 library,
    in 245.77 s. Largest block terminal 8,388,608 bytes against a
    2,038,431,744-byte (1.90 GiB) unchunked single-terminal lower bound;
    selected support MSE bitwise equal to direct hard execution of the same
    route. Independent scorer, `check_prereg.py` and `check_invalid.py` all
    pass; the plan's depth-three equivalence gate (chunked vs all-route loss
    vector within 1e-6, identical argmin) passes in the test suite.
    Two disclosed deviations: the plan-required selected-route query NMSE was
    never recorded by the runner and was recovered post hoc (0.01339981,
    passes 0.05, one program only - NOT comparable to the depth-four 16-program
    median as a depth trend); peak RSS was never recorded and is unrecoverable.
    This is a memory-feasibility result for one library, world, program and
    depth. It is not a depth-five execution or search horizon and not a PX7
    verdict.
  - **Four consecutive L0d opportunity gates have now failed to produce route
    ambiguity** on a usable frozen vocabulary: support 128/32/8 (preflight),
    support 4/2/1 (sparse-evidence gate), depth four (execution gate) and depth
    five (memory gate). PX7(a) is about the cost of premature commitment under
    ambiguity, and this substrate has not yet yielded ambiguity whose effect it
    could measure. Per the opportunity-gate rule, the census cannot be frozen
    until a pre-specified, MEASURED ambiguity source exists; repeating
    evidence-size sweeps on this library is closed.
  - **NEXT for L0d: gated on SG0** (see the top section and
    `PROGRAM_LADDER_PLAN.md` Revision 1). The full census stays unfrozen and is
    WITHDRAWN outright if SG0 returns NO-HEADROOM; it is frozen around SG0's
    statistic only under HEADROOM-LOCALIZED. Do not launch the original
    depth/mechanism grid on the strength of these negative gates, and do not
    repeat evidence-size sweeps on this library.
- **Prior evidence the ladder must respect:** the 2026-08-31 loop census and its
  correction (a straight-line generator cannot price loops); the E6 macro line
  (macros pay but cannot be timed or compiled); the E7 census (no real
  parameterized-macro family against the learner's own null).

# Paused

## H28 closure / coordinate reuse (H28-C, then H28-Q)

- **State:** provisional development checks only; no verdict. Oracle coordinate
  fixture validated; oracle-core adapter pilot transfers; learner harness has
  oracle, independent and random-core controls, reconstruction and
  parameter-byte accounting (`11c9471`).
- **Missing:** the amortized economic endpoint; a frozen
  `H28_C_LEARNER_OPPORTUNITY_PLAN.md`; independent scorer; restart test.
- **Next:** add the economic endpoint and freeze the plan. Free to resume: no run
  holds HEAD.

# Available in parallel

## Track A: role-filler semantics (review 84)

- RF0a done; RF0b UNRESOLVED (`71a4411`); RF1 not planned in detail; RF2 not
  started. The convergence gate (Track A RF1 plus Track B) is required before
  typed higher-order combinators, and the ladder's PX6 depends on RF1.

## Budgeted execution economy

- `BUDGETED_EXECUTION_ECONOMY_PLAN.md`, a PI-merged DRAFT marked do-not-freeze.
  B0/B1 are Tier 0/1 and queued; not started.

## Infrastructure backlog

- `CONCURRENCY_PLAN.md` still marked DRAFT (with Amendment 1); the pool practice
  it describes is in use.
- Non-bitwise speedups (batched `forward_tasks`, batched slots) remain versioned
  candidates in `notes/performance_audit.txt`, admissible only at a plan boundary
  with an equivalence gate.
- **Host memory is a real constraint:** SO4's first attempt died on paging
  exhaustion at about 5 GB free. Ask the PI to close large applications before
  any multi-hour launch, and run one heavy job at a time.

# Design only

## Track D: self-hosted program synthesis (SH0-SH4)

- Design in `POST_E6_RESEARCH_PROGRAM.md`; SH0 waits on C0's typed IR.

# Closed (do not reopen without a new plan)

- **V1-V5 sealed programs**; V5 closure includes review-55 withdrawals.
- **V6 / V6R** adaptation geometry (development).
- **H39** sealed; the formation line H47-H53 closed negative.
- **Export branch:** sealed export block confirmed (seeds 800-829); E5's writer
  failed on quality; E5.1 cost scaling; the E6 macro line; the E7 census; E8D and
  E9 withdrawn before freezing.
- **Iteration world spec** withdrawn; superseded by the rotated substrate.
- **SO2, SO3, SO4** as programs, and the SO2-P/SO3 consolidation candidates.

# Hypothesis families (all in `PREDICTIONS.md`)

| family | line | status |
|---|---|---|
| CF1-CF7 | staged formation (Track B) | working; J1c supported staging, CF4 refuted by J1 |
| CL1-CL8 | H28 closure / coordinate reuse | working, untested |
| PX1-PX9 | program ladder | working, recorded 2026-09-15 |
| B0-B3 | H47 membership / opportunity | closed |
| SO0-SO4, J-rungs | Track B plans | registered inside each frozen plan |

# Decisions pending from the PI

1. Ladder: may teacher-side and oracle-executor gates (L0a-L0c) run before online
   learnability is established? The plan reads "yes".
2. Ladder ordering resolved by continuing after the completed world censuses:
   L0d preflight is complete. The sparse-evidence gate is next; the full
   depth/mechanism protocol stays unfrozen.
3. (Folded into 7.)
4. Should ladder rungs L1-L3 target remote workers when they open?
5. Development-world budget: worlds 0-9 are spent for the rotated staged
   protocol. Allocate a new development band, or stop opening worlds for it?
6. (Answered by SO4: the re-test ran and failed.)
7. ANSWERED: the cross-run world census ran, along with the library-geometry and
   route-margin normalization censuses. The Track B mechanism hunt is closed
   with all three candidates withdrawn.
8. **NEW, recommended first: approve SG0**, the Tier 0 headroom gate
   (`SYNTHESIS_OPPORTUNITY_GATE_PLAN.md`). Minutes, no world, and each of its
   three registered outcomes determines a different successor, including
   closing a branch.
9. **NEW, contingent on SG0 returning NO-HEADROOM:** pursue the identifiability
   generator (`notes/identifiability-sketch.txt`), or bank the economics and
   negative-results papers and stop opening rungs on this substrate?

# Housekeeping owed

- **Statistic fix, verified latent (`8f0369a`).** The old `spearman` helper used
  `argsort(argsort(x))`, which gives tied values a strict order: it returned
  +1.0 for a CONSTANT predictor. `census_world_quality.spearman` uses average
  ranks and returns nan on zero variance. No committed number moved (the
  per-task log-ratio series have no ties; three recomputed cells reproduce their
  stored values exactly). Recorded in `AGENTS.md` and `notes/learnings.txt`.
- **`SPEC_AUDIT.md` full re-audit still owed.** A PARTIAL record-consistency
  re-audit (rotated substrate through SO3) was appended 2026-09-15; SO4 is not
  yet covered. Still owed: code-level audit of runner and scorer against plan,
  with at least one recomputed cell per milestone.
- **Stop-rule numbering:** documents written 2026-09-15 call the online-failure
  rule "stop rule 2"; in `POST_E6_RESEARCH_PROGRAM.md` it is the third Track-B
  rule. `SO4_B2_RETEST_PLAN.md` and this file name it correctly.
- **`PROGRESS.md` is not strictly chronological:** the 2026-09-14/15 SO2-line
  entries sit before that day's H28 entries; see the placement note.
- **Paper draft** is current through SO4. The copy in
  `~/Downloads/ROW_paper_draft_2026-09-15.md` predates the SO4 addendum.
