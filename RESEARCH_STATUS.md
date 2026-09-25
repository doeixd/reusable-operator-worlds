# ROW research status: what is live, blocked, pending, closed

This is an INDEX, rewritten in place whenever a line of research changes state,
and committed with the change that caused it. The scientific record lives
elsewhere and is append-only: verdicts, hypotheses and corrections in
`PREDICTIONS.md`; completed steps in `PROGRESS.md`; results in `reports/`. If
this file disagrees with those, they win and this file is stale.

Last rewritten: 2026-09-24, after O1 Tier 1 returned LIVE. Nothing is running.
Order-free anchors form the library ONLINE as often as the staged curriculum
(2 of 3 worlds each), but in different worlds, so reliability is the open
question. `O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md` asks it and is DRAFTED,
awaiting PI decision 11. Follow-up audits A1, A2 and P1 are done.


# N1 CLOSED: ANCHORS_SUFFICE, 3 of 3 worlds (2026-09-22)

- **Verdict:** `ANCHORS_SUFFICE`. Run `3d2f5c6`, 12 cells, exit 0, independent
  scorer valid, prereg/invalid/adequacy pass. Report
  `reports/n1_anchor_supply.json`; archive `reports/n1_anchor_supply_2026-09-22/`.
  DEVELOPMENT evidence on worlds 0-2, never confirmatory.

| world | STAGED | INTERLEAVED | SHAM | NONE |
|---|---|---|---|---|
| 0 | 0.00623 | **0.00925** | 1.03669 | 0.95749 |
| 1 | 0.00514 | **0.01011** | 1.06240 | 0.90375 |
| 2 | 0.00725 | **0.00658** | 1.08676 | 0.90367 |

- **Anchors, not order, not volume.** `INTERLEAVED` (anchors at random
  positions, length never revealed) passes 0.05 in all 3 worlds. `SHAM` has the
  same pool, the same examples and verifiably identical minibatch draws, with
  length-3 fillers instead of anchors, and is the WORST arm - worse than the
  64-task floor. All three `STAGED` cells reproduce J1c bitwise.
- **Why it matters:** ordering needs program length, which an online learner
  lacks; an anchor supply needs only that easy tasks be present. SO2-SO4 carried
  the ORDER online. N1 says the order was never the active ingredient, and
  offers review 85's C1 an alternative that needs no confidence signal: inject,
  do not defer.
- **Not established:** anything online; anchor dose, proportion or length; other
  worlds; anything confirmatory.
- **Natural successors:** a Tier 0/1 dose census (how few anchors, of which
  length) on the same harness, then an ONLINE anchor-supply rung - which needs a
  new world band, so decision 5 is live again with a concrete purpose.
- **N1b CLOSED (2026-09-23): `L1_SUFFICES_ONLY`, `k* = 32`.** Length-1 anchors
  alone reach 0.0067-0.0175 (as good as all 124); 64 length-2 anchors do nothing
  (1.12-1.16, no better than none) despite above-chance clustering; 8 mixed
  anchors fail, 32 pass, monotone, no near-threshold cell. Run `fc8a742`, report
  `reports/n1b_anchor_dose.json`. NB1 and NB3 supported, NB2 not.
  **Post hoc (NB4, confounded):** every passing cell's length-1 anchors covered
  all 6 operations, every failing cell's covered <=5 - including one at 5 of 6
  that failed completely. Successor N1c holds anchor count fixed and varies
  coverage.
- **N1c CLOSED (2026-09-23): `K6_INSUFFICIENT`, `COUNT_COMPENSATES` - NB4
  REFUTED.** Six length-1 anchors fail with or without full coverage (1.04-1.09);
  eighteen covering only 5 of 6 operations suffice in 2 of 3 worlds (0.0129,
  0.0259; world 0 at 0.199). The registered split shows that in worlds 1 and 2
  the tasks USING the uncovered operation pass too - the library formed it from
  compositions - and world 0's failure is LOCAL (not-using 0.025, using 1.13).
  N1b's coverage pattern was a count effect. Run `6251a17`, report
  `reports/n1c_anchor_coverage.json`.
- **What the offline anchor line now says an online design needs:** a QUOTA of
  single-operation tasks - bracketed between 6 and roughly 16-18 over a 188-task
  stream at this budget - not their order (N1), not length-2 tasks (N1b), and not
  one for every operation (N1c). The successor is an ONLINE anchor-supply rung,
  which needs a new world band. **Worlds 10-19 were allocated by the PI on
  2026-09-23**. Tier 0 feasibility PASSED (bitwise equivalence at uniform depth
  through the real online lifetime; a mixed-length stream runs end to end).
- **O1 Tier 1 COMPLETE (2026-09-24): LIVE, exploratory.** Run `2e99765`, which
  launched 2026-09-24 after a first attempt failed closed on memory. Scorer valid.
  Report `reports/o1_online_anchor.json`, archive
  `reports/o1_online_anchor_20260924/`. Terminal medians, worlds 10/11/12:
  - `STAGED`: 0.272 / **0.0269** / **0.0188**
  - `SHUFFLED`: **0.0366** / 0.473 / **0.0184**
  - `MIXED_L1`: 0.191 / 0.149 / **0.0358**
  - `PLAIN`: 2.00 / 1.92 / 1.91
  Order is not needed online, and not sufficient for reliability either: the
  two 2-of-3 arms fail in different worlds. Without length-2 tasks, `MIXED_L1`
  passes only 1 of 3 (offline N1b: 3/3). One stream per world, so these are
  indications only.
- **O2 DRAFTED (2026-09-24), not frozen:** `O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md`.
  Worlds 13-19, three replay streams, arms `SHUFFLED` / `STAGED` (lifetimes only,
  behind a bitwise gate against O1's cells) / `MIXED_L1` / `PLAIN`. Registered
  rule k-of-21: `>= 19` RELIABLE, `<= 16` UNRELIABLE. At the measured
  null (STAGED cell rate 14/24 = 0.583), false-fire is <= 0.021 and detection
  >= 0.838 at four heterogeneity levels. About 3.6 h.
  Awaiting PI decision 11.

# SG0 CLOSED: NO-HEADROOM (2026-09-22)

- **Verdict:** `NO-HEADROOM`, `k = 0` of 36 staged cells against a registered
  `k <= 2` fixed before any data existed. Run `229fb97`, 72 cells, exit 0,
  independent scorer `valid: true`, prereg and invalid checks pass. Report
  `reports/sg0_full_v2.json`; archive `reports/sg0_full_2026-09-22/`; PROGRESS
  and PREDICTIONS 2026-09-22.
- **574 of 576 staged tasks have EXACTLY zero commitment regret**; the two
  exceptions are depth-4 tasks at 0.00994 and 0.00469 and neither cell clears its
  floor. No staged task has any route within 1% of the winner on support, at
  either depth or any support size. The bootstrap floor is exactly zero in every
  staged cell: 200 resamples never changed the selected route.
- **Non-vacuity passes decisively:** the same code path gives 500/576 control
  routes differing, regret to 0.663, 142/576 non-trivial near-ties, and 11 of 36
  control cells clearing their own floor.
- **Anchors pass:** J2A `enum_route`/`enum` at depth 3, and the committed
  depth-four execution gate at depth 4 world 0. Staged median query NMSE 0.00580
  and 0.00772 sit on the committed gates' 0.00571961 and 0.00871712.
- **In force, as registered in advance:** the amortized-proposer branch on this
  substrate is CLOSED; **PX7 is retired as UNMEASURABLE HERE**, not refuted; the
  full **L0d census is WITHDRAWN** rather than left indefinitely unfrozen.
- **Reading:** on the usable staged vocabulary the route space is not merely
  well-evidenced but FUNCTIONALLY SEPARATED, and every inference mechanism
  competes for a quantity measured here at zero. Consistent with the working
  reading that E5, E5.1 and L0d's four gates share one cause in the generator.
  A statement about THIS substrate at depths 3-4, not about program synthesis.
- **Sub-triage AMENDED AND SCORED: `NO-HEADROOM-SATURATED`**
  (`SG0_SUBTRIAGE_AMENDMENT.md`, module `row.experiments.sg0_subtriage`,
  report `reports/sg0_subtriage.json`, 9 tests). Amendment 1 fixes all three
  Revision 3 defects: aggregation is per program with both denominators (36
  cells, 576 programs); `eps` becomes a sensitivity curve calibrated on the
  control arm instead of a bare constant; the borrowed spread floor is deferred
  with a refusal clause that fires if staged near-ties are ever non-empty.
  Staged `j = 0` of 36 cells and **0 of 576 programs have ANY route within 1%
  of their winner**; staged median identifiability 53.9 against the controls'
  0.025, a ~2,000x separation. The reading holds across a tenfold widening of
  the tolerance (bound 0/2/2 of 36 at eps 0.01/0.05/0.10) and is not claimed
  beyond eps 0.10. Non-vacuity passes: controls read EVIDENTIAL with `j = 10`,
  142 of 576 tie-carrying programs and disagreement to 0.389.
  **It lives in a separate file because the plan is hashed into the run's
  protocol** - editing `SYNTHESIS_OPPORTUNITY_GATE_PLAN.md` would make
  `score_sg0_headroom_gate` refuse the report as `input changed since the run`.
  Still a DIAGNOSTIC: Revision 3 was written while the grid was running, so it
  is not preregistered for this run and decides only ladder decision 6. It is
  admissible post hoc because the staged answer is invariant to every part of
  it - 576/576 singleton near-tie sets leave nothing for any aggregation rule
  or floor to move.
- **One run was retired before scoring**
  (`INVALID_SG0_DEPTH4_WRONG_TARGET_LENGTH`): it searched depth-4 routes against
  depth-3 targets. Depth-3 evidence unaffected; plan Revision 4 fixes the program
  source per depth and requires a cross-check against a committed result at every
  depth.

# SG6 CLOSED BEFORE OPENING: no graded vocabulary-quality axis (2026-09-22)

- **The proposal.** SG0's reading is that a well-formed vocabulary dissolves the
  inference problem: staged median identifiability 53.9 against the controls'
  0.025, and 576 of 576 staged programs with no route within 1% of their winner.
  The natural successor was a Tier 0 census asking whether inference difficulty
  TRACKS vocabulary quality across the twelve held libraries - which would have
  turned ten negatives into one positive claim about when program inference is a
  real problem at all.
- **The gate was run FIRST, before any plan was written**
  (`row.experiments.sg6_quality_gate`, `reports/sg6_quality_gate.json`,
  `tests/test_sg6_quality_gate.py`, 9 tests), and it refused the rung. Verdict
  `NOT-MEASURABLE-HERE`.
  - Quality is BIMODAL with nothing between the clusters: staged
    0.00465-0.00725, control 1.26012-1.30157, a gap 30.2x the wider cluster's
    own spread.
  - Pooled Spearman against identifiability is **-0.692** and looks like a
    finding; within staged it is **+0.086** and within control **+0.429** -
    null, and the WRONG SIGN in both.
  - The pooled number re-detects cluster membership. Effective n is 2, not 12.
    Identical in form to the route-margin candidate that survived pooled and
    died once made comparable, except caught before the plan rather than after
    three withdrawals.
- **Not refuted, unmeasurable here.** Testing the hypothesis needs a generator
  with a graded vocabulary-maturity knob, its own balance gates and a
  non-vacuity check - a new substrate, not a census over these artifacts.
- **Tenth rung to fail an adequacy gate, and the FIRST to fail before a plan
  existed. Its gate was DISCRIMINATION (the sample has no graded axis), not
  opportunity or necessity** (relabelled 2026-09-24, A2 claims audit). The other
  nine cost a branch each; this cost one command. Recorded in AGENTS.md: the
  adequacy gates decide whether a plan gets written, not a step inside one.

# Two new plans drafted (2026-09-22): N1 and N2

Both are DRAFTS needing PI approval, both are offline, and both need the same
world-budget ruling (see below). They are the first plans written under
`DESIGN_ADEQUACY.md` and carry `# Necessity` and `# Discriminating power`
sections with MEASURED numbers.

- **N1, `N1_ANCHOR_SUPPLY_PLAN.md` — is it the ORDER, or the supply of anchors?**
  J1c confounds a monotone short-to-long ORDERING with the mere PRESENCE of
  length-1 tasks on which routing is clustering rather than search. Four arms at
  matched task count and budget: `STAGED` (the curriculum, and a bitwise anchor
  against the committed J1c cell), `INTERLEAVED` (same anchors, random positions,
  no ordering, length never revealed), `SHAM` (matched anchor count and
  positions, filled with length-3 tasks), `NONE` (the matched non-staged floor).
  **If anchors suffice without ordering the online problem changes shape:** a
  curriculum needs program length, which an online learner does not have, while
  an anchor supply needs only that easy tasks be present. It is also a different
  fix to the deadlock blocking C1 - inject, do not defer.
  Gates, measured: refusal cost is 150-200x (0.0047-0.0072 against 0.92-0.97,
  published); the registered triage has false-fire **0.0000** and detection
  **1.0000** against the null and full-effect samplers built from those
  published values, and the three outcomes partition 0..3 worlds.
  **`check_adequacy` now covers this plan.** STATUS: **NOT freeze-ready**, and
  the blocker is no longer the world ruling, which is now GIVEN (below).
  **Audit 1 (2026-09-22) found the arms arithmetically inconsistent.** The plan
  says anchors "REPLACE length-3 tasks" with task count "matched exactly", but
  the committed J1c protocol is 60 + 64 anchors and 64 final-stage tasks:
  **124 anchors cannot replace tasks inside a 64-task stream.** Matching all
  arms at 188 tasks instead keeps `STAGED` verbatim but then (a) `NONE` at 188
  length-3 tasks is a NEW construction whose floor is not the published
  0.92-0.97, so the non-vacuity referent and the null sampler are borrowed from
  a different arm, and (b) `SHAM` collapses toward `NONE`, since both become 188
  length-3 tasks differing only in which task sits where - losing the control the
  design turns on. Four things must be registered before freezing: the single
  matching invariant, `NONE`'s construction and its non-vacuity referent, how
  `SHAM` stays distinct (the pool has up to 216 distinct length-3 programs), and
  re-derived samplers. The NECESSITY section is unaffected: its 150-200x refusal
  cost is measured from published values and stands. A second, non-blocking audit
  notes the anchor-difficulty check compares a margin over 12 routes against one
  over 1,728 - the route-margin comparability failure in a new place.
  **AMENDMENT 1 (2026-09-22) answers all four and the plan is now freeze-ready
  pending PI approval.** Invariant: 65,536 updates for every arm, 188 task slots
  for the three contrast arms. `NONE` stays the published 64-slot floor, so its
  non-vacuity referent is its own published 0.92-0.97 rather than a borrowed
  value. `SHAM` becomes 188 slots - the canonical 64 plus 124 DISTINCT additional
  length-3 programs (216 exist, 64 used, 152 available) at `INTERLEAVED`'s anchor
  positions - so it is no longer a near-duplicate of the floor, and the three
  arms differ in exactly one respect each: `STAGED` vs `INTERLEAVED` isolates
  ORDER, `INTERLEAVED` vs `SHAM` isolates ANCHOR EASINESS.
  Samplers re-derived by SWEEPING `SHAM`'s unmeasured location
  (`reports/n1_design/`): false-fire 0.0000 and full detection 1.0000 at every
  location, but **half-effect detection collapses below `SHAM` ~ 0.45** (0.0000
  at 0.20-0.25, 0.3125 at 0.30-0.35). Registered guard: if measured `SHAM` falls
  under 0.45 in a world, `PARTIAL` is UNINTERPRETABLE there, never
  `INSUFFICIENT` - a ratio test whose denominator collapses gives a false
  negative, not a negative. **AUDIT 2 IS NOW ANSWERED AND MEASURED (Amendment 2):**
  the route-margin form is replaced by the adjusted Rand index between teacher
  primitive labels and support-argmin slot labels on an UNTRAINED library - a
  chance-corrected, scale-free statistic, so depth 1 (12 routes) and depth 3
  (1,728) are the same quantity. Verdict `PREMISE_HOLDS` 3/3: depth-1 ARI
  +0.7677 / +0.5850 / +0.4595 against depth-3 mean ARI +0.0055 / +0.0240 /
  +0.0216, i.e. depth-3 routing is at CHANCE on a random library while depth-1
  clusters strongly. This also supplies the first MEASUREMENT of the clustering
  mechanism that has explained J1c since it was run and had only ever been
  argued. `reports/n1_anchor_difficulty_gate.json`, 6 tests.
  **N1 is now freeze-ready pending PI approval**, with both audits answered.
  **Amendment 3 (2026-09-22)** adds two pre-launch findings. (a) "Matched
  positions" is VACUOUS in this trainer: `train_stage` pools every task's
  examples and draws minibatches uniformly, so there are no stream positions.
  Restated as matched POOL COMPOSITION (188 tasks, 24,064 examples for both
  `INTERLEAVED` and `SHAM`) plus an identical minibatch index stream, which
  makes the contrast cleaner - `STAGED` vs `INTERLEAVED` isolates ORDER (three
  sequential stages against one pooled call over the same 188 tasks),
  `INTERLEAVED` vs `SHAM` isolates ANCHOR EASINESS. (b) Cells are SIZED by a
  measured performance pass: 24.56 ms/update at length 3 and 16.63 ms at length
  1, so a cell is ~25-27 min and **12 cells are ~5.26 h serial, ~1.75 h at pool
  3** - one local batch, no remote workers.
  **HOST PRECONDITION NOT MET:** free memory measured 4.3 GiB of 27.8, and SO4's
  first attempt died on paging exhaustion at about 5 GiB free. A registered
  launch precondition of 8 GiB free for a pool of 3 is added, failing closed.
  N1 is freeze-ready but **NOT launch-ready, and the blocker is the host**.
- **N2, `N2_FORMATION_IDENTIFIABILITY_PLAN.md` — is identifiability graded ALONG
  FORMATION?** SG6 died because the twelve held libraries are bimodal: good or
  broken, nothing between, effective n = 2. A formation run passes through the
  whole range BY CONSTRUCTION, so the trajectory is the graded axis SG6 lacked,
  and it would make SG0's "inference is hard only on immature libraries" reading
  testable with the instrument SG0 already validated. Preflight verified two
  things: the per-stage checkpoints EXIST (`stage1/2/3` under every staged J1c
  cell, up to 18 trajectory points), and the naive load FAILS - stage-1 and
  stage-2 models carry their own task codes and the strict loader rightly
  refuses them. Step 0 is therefore a LIBRARY-ONLY loader with an argued partial
  reconstruction, and a kill condition if it cannot reproduce stage 3 bitwise.
  STATUS: **not freeze-ready and deliberately NOT in `check_adequacy`'s scope** -
  its sampler rates are not yet measured and the plan says so. Registered in
  advance: a BIMODAL result at three coarse points escalates to finer
  checkpointing rather than closing the question, so a sampling artifact cannot
  be read as a substantive negative.

**World budget: RULED (2026-09-22, Claude under delegated judgement; the PI may
reverse).** Worlds 0-2 MAY be reused for N1 and N2, under three binding
conditions: (1) DEVELOPMENT ONLY, permanently - nothing measured there may ever
be reported as confirmatory, and a confirmatory version needs a fresh band;
(2) the reason is that `AGENTS.md` defines seeds 0-9 as the development
partition, which is repeated use by construction, and "spent" was recorded about
the ONLINE staged protocol where each world's outcome is now known - N1 and N2
are offline, ask a different estimand, and their new arms have never been run;
(3) `STAGED` is not blind, so it is an ANCHOR whose failure voids the run, never
evidence - the registered estimand is the CONTRAST between arms. Decision 5 is
answered for these two rungs and remains open for any ONLINE rung, which still
needs a fresh band.

# Candidate rungs from review 85 (2026-09-22, external suggestions)

Assessment in `reviews/reviewer-feedback-85.txt`. An external reviewer working
from the README alone proposed five directions. Two survive as candidates, one
is a vote on an open decision, one is largely already run and negative, and one
is a refinement of a closed line. None is a plan; none is approved.

- **C1, confidence-gated commitment (online).** Commit a route only when its
  margin clears a per-world-normalized threshold; otherwise defer. Predicts J1c's
  staging emerges online without being told program length. NECESSITY and
  OPPORTUNITY are satisfied by J1c (0.005-0.007 against 0.92-0.96 offline);
  DISCRIMINATION needs the sham-versus-real separation checked at the intended
  sample size. **Blocking design objection, to be answered IN the plan:** J0
  measured that gradient routing fails above about 0.6 median NMSE and that
  early in joint training every library is on the failing side, so a gate on a
  random library defers everything and nothing ever becomes confident - a
  DEADLOCK. The candidate fix is to withhold only the HARD commitment while soft
  routes still update the library. Also **blocked on PI decision 5**: this is an
  online rung and worlds 0-9 are spent.
  - **THE CANDIDATE FIX IS THE FAILING BASELINE** (2026-09-22). "Withhold only
    the HARD commitment while soft routes still update the library" is what the
    online protocol ALREADY does: `DiscreteLibraryLearner` is
    "Overcomplete operator library with relaxed training and hard evaluation
    routes", and `RotatedDiscreteLibraryLearner` inherits it unchanged, so
    SO2/SO3/SO4 trained through a temperature-annealed softmax and hardened only
    to score. The deadlock fix as stated therefore escapes nothing; it names the
    arm that failed three times. J1 is the exception that proves it - its cells
    carry `pinned_one_hot: True`, which is why J1 and the ordinary online
    protocol are different failures, not the same one.
  - **Three points on the commitment axis now fail, and the one that works is
    not on it** (J1 commitment census, 2026-09-22, read-only over
    `reports/j1_search_loop.json`; regenerate with
    `reports/j1_commitment_census_2026-09-22/derive_census.py`):
    soft throughout (ordinary online) fails in SO2/SO3/SO4; hard commitment (J1)
    freezes after round 2-3 - of 64 tasks only 1, 1 and 5 change in round 1 and
    0, 1 and 2 in all later rounds combined - at terminal 0.757-0.774; random
    hard reassignment (SHAM) churns 60-63 of 64 every round and is WORSE at
    0.975-0.982. J1c's length curriculum works at 0.005-0.007 and changes WHAT
    TASKS ARRIVE WHEN, not how or when the learner commits. C1 must show its
    gate is not interpolating among known failures.
  - **C1's gating signal is not recorded anywhere.** J1 round records carry only
    `round`, `argmin_ok`, `changed` and `slots_used` - no margin, gap, posterior
    or entropy - so whether margin at commitment time is informative about
    whether that commitment is RIGHT cannot be answered on held artifacts. C1's
    DISCRIMINATION gate needs instrumentation before it can be checked, which
    the three-gate protocol requires before freezing.
- **C2, depth recurrence / looping as sharing across position.** Manipulate how
  often the hidden program reuses the SAME operator across positions, and
  compare a weight-tied learner against an untied one at matched compute. This
  is a genuinely new axis - V1's `rho` is recurrence ACROSS TASKS, not within a
  program - and it is the cleanest bridge ROW has to an outside debate (looped
  transformers). **Admissible only on the ROTATED family:**
  `ITERATION_WORLD_SPEC.md` was withdrawn before freezing because repeated
  application in the canonical family is a CONTRACTION, failing its necessity
  gate at 0.026-0.123 against a required 0.25; the rotated family passes at
  1.67-1.73. **What may be registered is the CROSSOVER LOCATION and linearity in
  MEASURED positional recurrence**, never the sign: "tied beats untied at high
  recurrence" is true by construction and would be an implementation check.
  First step is the necessity gate on the new knob, not the learner comparison.
- **C3 = PI decision 7.** The reviewer's control-flow proposals (per-task
  iteration count; learned predicate selecting sub-routes) are PX2 and PX3 of
  the program ladder, reached independently from the README. Treat as a vote for
  the offline scope. It contributed one registrable clause the ladder lacked;
  see `PROGRAM_LADDER_PLAN.md` Revision 3.
- **C4, two-timescale plasticity: LARGELY ANSWERED.** This is SO2-P, which swept
  the shared learning rate down 2x/4x/10x and did not improve on the unchanged
  protocol anywhere; the committed learning is that MOVEMENT MAGNITUDE IS NOT
  INTERFERENCE - the best arm drifted more. Genuinely untested: the RATIO to the
  route rate, in both directions and across the whole run rather than stage 3.
  Low priority, and needs a stream-only control arm.
- **C5, macro create/refuse/wait: deprioritized** below everything else. The E6
  line is closed and E7 refuted the parameterized-macro rationale (for this
  testbed's corpus) against the learner's own unplanted-structure null, which any successor must beat; the
  prize is 40-55 bits on dead patterns.

The reviewer's recommendation to drop the amortized writer is agreed and already
executed by SG0. Their observation that C1 would pass through the
immature-library regime by construction - supplying for free the setting SG6
could not obtain on held artifacts - is the most useful connection in the review
and is the main reason C1 ranks first.

# Queued for another agent: five Tier 0 follow-ups (2026-09-22)

Full specification in `FOLLOWUP_AUDITS.md`. Written after the three gates were
separated (`DESIGN_ADEQUACY.md`) and applied retrospectively to what the
repository already holds. **No experiment needs re-running**: the retrospective
pass found no wrong numbers, only labels, unverified constructions, and plans
not yet written. None of the five opens a world, starts a run, or touches a
sealed band, and all are implementation audits rather than experiments.

| item | priority | blocked on |
|---|---|---|
| **A1 arm-provenance retrofit** - the only item that could surface a real defect | highest | nothing |
| **A2 claims audit** - necessity failure vs genuine refutation | high | nothing |
| **P1** record SG3's necessity problem in `notes/identifiability-sketch.txt` | high | nothing |
| **A3** re-read H39's two-slot claim against `capacity_forces_structure` | optional | nothing |
| **P2** re-specify ladder gates L0a-L0c under the new protocol | deferred | PI decision 7 |

- **A1 is the finding.** `src/row/arm_provenance.py` was built after E5's defect
  - three modules with an arm called `S`, one of which built it from the TRAINED
  library, making a fine-tuning arm mislabelled as scratch (0.23-0.44 log units;
  no verdict flipped). Every such arm is a REFUSAL ARM under the new vocabulary,
  and `refusal_cost` is only as good as the arm that refuses. **26** experiment
  modules build a scratch, fresh or deep-copied arm; **2** use the harness;
  **24** do not. Most sites will be legitimate probe copies, which are required
  practice - the value is in the few that are not, and in leaving the 24
  checked rather than assumed. Where a committed report depends on the code, the
  LABEL is corrected and the code stays as it was run.
- **A2 matters now** because the negative-results paper is becoming a primary
  output, and "reuse did not pay" versus "our task never required reuse" is the
  difference between a finding and a mistake.
- **P1** records that SG3 would MANUFACTURE the ambiguity it measures, so it
  would need to pass its own necessity gate. Worth recording whichever way
  decision 6 goes.

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
  - **Four consecutive L0d gates (called opportunity gates in their plans;
    NECESSITY failures under `DESIGN_ADEQUACY.md`) have now failed to produce route
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
  (on this testbed macros pay; retrospective accounting alone cannot time
  creation but a gated criterion can (E6F); compilation works only at ~4x slot
  capacity and does not pay there); the E7 census (no real
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
5. Development-world budget. **ANSWERED.** Worlds 0-2 reusable by the OFFLINE
   rungs N1/N1b/N1c as development evidence only (Claude, delegated, 2026-09-22).
   **Worlds 10-19 allocated by the PI on 2026-09-23** as development band 2 for
   the ONLINE anchor-supply line; recorded in `AGENTS.md`'s partition list and
   verified unused beforehand.
6. (Answered by SO4: the re-test ran and failed.)
7. ANSWERED: the cross-run world census ran, along with the library-geometry and
   route-margin normalization censuses. The Track B mechanism hunt is closed
   with all three candidates withdrawn.
8. ANSWERED: SG0 was approved, implemented, run and scored. Verdict
   `NO-HEADROOM`; see the top section.
9. **LIVE, and the measurement is now IN (ladder decision 6):** SG0 returned
   NO-HEADROOM and the amended, scored sub-triage returns
   `NO-HEADROOM-SATURATED` with staged `j = 0` of 36 and 0 of 576 programs
   carrying any near-tie, robust across a tenfold tolerance widening, with
   non-vacuity passing on the control arm. SG3's first prerequisite gate FAILS
   on these artifacts. Pursue the identifiability generator anyway, or bank the
   economics and negative-results papers and stop opening rungs on this
   substrate? **Claude's recommendation: BANK.** The amendment asked for in the
   previous wording is done; what remains is the PI's call, not more
   measurement. Note the scope: this says the usable staged vocabulary is
   functionally separated at depths 3-4, not that no substrate has ambiguity -
   the controls in the same grid have plenty.
10. **LIVE (ladder decision 7):** the L1-L8 learner gate is unreachable - SO2,
    SO3 and SO4 all failed and worlds 0-9 are spent. Amend stop rule 2 for
    offline scope on frozen J2A libraries, or close L1-L8. They cannot remain
    pending.
11. **LIVE (2026-09-24): freeze O2?** `O2_ONLINE_ANCHOR_RELIABILITY_PLAN.md`
    (Tier 2, worlds 13-19, ~3.6 h, one evening). Sub-question: keep `MIXED_L1`
    (+0.7 h)? Claude's recommendation: freeze with `MIXED_L1`, because it is the
    cheapest direct test of the offline/online discrepancy.

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
