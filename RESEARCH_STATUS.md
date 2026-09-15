# ROW research status: what is live, blocked, pending, closed

This is an INDEX, rewritten in place whenever a line of research changes state,
and committed with the change that caused it. The scientific record lives
elsewhere and is append-only: verdicts, hypotheses and corrections in
`PREDICTIONS.md`; completed steps in `PROGRESS.md`; results in `reports/`. If
this file disagrees with those, they win and this file is stale.

Last rewritten: 2026-09-15, after the SO3 verdict. Completeness was checked
against every top-level plan/spec status line, every `PROGRESS.md` H1 heading,
and the review index (latest review: 84).

# Awaiting a PI decision (nothing running)

## Track B: online learnability of the strong (rotated) substrate

- **State:** SO3 CLOSED as `SO3_FAILS`, but the unchanged protocol passed on
  fresh worlds. No run is active.
- **SO3 result:**
  - Neither stage-3 candidate (LR_HALF, STORE_8) improved on the unchanged
    protocol in 2/3 worlds, so the one-world Tier 1 rescue does not replicate.
  - BASE, SO2's protocol with no change, passed the terminal criterion in 3/3
    fresh worlds (world medians 0.028 / 0.014 / 0.012; median 52-58 of 64 tasks
    at or below 0.05).
  - Terminal error varies 0.009-0.066 across replay streams within one world.
  - SO2's failure is therefore not a stable property of the protocol. SO3
    registered no BASE gate and no G5R margin, so the B2 statement is NOT
    licensed and stop rule 2 is not lifted by SO3.
- **Path:**

| step | result | commit |
|---|---|---|
| rotated substrate spec | rotation makes iteration AND branching necessary | (ROTATED_SUBSTRATE_SPEC, Amendments 1-2) |
| G5 / G5R | learnability fails 0/3 online (mismatched, then matched family) | 2026-08-31 / 09-01 |
| G5R diagnosis + LBFGS correction | failure is downstream of isolated operator fitting | 2026-09-01 |
| G5R Stage D | `BUDGET_LIMITED` | 8b9b802 |
| SO0 | no existing pair controls any budget axis | 1042878 |
| SO1 | oracle routes pass, learned routes fail | dd57e24 |
| SO1R | routes recoverable, so the wall is co-formation | 9c8d058 |
| J0 / J1 | routing tracks library quality; search in the loop locks in | f9bcc30 / 01305d4 |
| J1c / J1c-R | length curriculum acquires the substrate offline, at 2 seeds | d4e6808 / 3bf6a59 |
| J2A | offline staged library exports 64/64 unseen programs | 686e4d3 |
| curriculum cost | same compute, +124 tasks of data | aa79aa3 |
| SO2 | `SO2_FAILS`: export 3/3, terminal 1/3 (one stream, worlds 0-2, seed 5000) | 6a4f707 |
| interference census | post-acquisition, recency-weighted loss in stage 3 | 3d87a0f |
| SO2-P (Tier 1) | LIVE on world 1 (one stream); drift is not the lever | afb1ac6 |
| SO2-P correction | replay arms gradient-matched but stream-confounded | b250973 |
| SO3 | `SO3_FAILS`; BASE passes 3/3 fresh worlds; stream spread straddles threshold | (this commit) |

- **Development-world use** under the rotated staged protocol: worlds 0-2 (SO1
  through SO2-P), worlds 3-5 (SO3); worlds 6-9 unused.
- **Proposed next (PI decision 6):** `SO4_B2_RETEST_PLAN.md`, a DRAFT that is
  not frozen and has no code. It re-tests the UNCHANGED online staged protocol:
  - worlds 6-9, model seed 7000, 3 replay streams per world, a PLAIN control;
  - a world-level criterion over streams, with a sub-clause that at least 2 of
    3 streams must pass in each passing world;
  - the G5R margin on each world's pre-specified stream-0 library;
  - labels PASSES / ACQUIRES_ONLY / STREAM_FRAGILE / FAILS;
  - about 4 h local.

  It needs PI answers to D1-D4 (worlds, margin coverage, thresholds, compute)
  before it is frozen.

# Drafted, awaiting PI decisions

## Program ladder: compositional operators -> control flow -> higher-order programs

- **State:** DRAFT plan `PROGRAM_LADDER_PLAN.md` with hypotheses PX1-PX9 in
  `PREDICTIONS.md` (committed `fbb547a`).
- **Extends:** Track C (C0 typed IR, C1 CHAIN/COMPOSE, C2 IF/REPEAT, C3
  discovery) and Track D of `POST_E6_RESEARCH_PROGRAM.md`. None of C0-C3 has
  started.
- **Prior evidence it must respect:**
  - The 2026-08-31 loop opportunity census: on straight-line E6 routes a loop
    construct is net negative at depth 6 and pays only where routes have
    collapsed.
  - Its same-day CORRECTION: that census measured the absence of iteration in a
    generator with no iteration, so the necessity gate moved into generator
    design.
  - The E6 line: macros pay but cannot be timed or compiled.
  - The E7 census: a parameterized-macro family was not real against the
    learner's own grams.
- **Can run now without breaking stop rule 2:** L0d, the program-inference
  census on J2A's frozen straight-line libraries (PX7).
- **Needs a PI answer first:** L0a-L0c (teacher-side and oracle-executor gates).
- **Gated:** learner rungs L1-L3 wait on online learnability being established,
  now expected from the proposed B2 re-test rather than from SO3.

# Paused

## H28 closure / coordinate reuse (H28-C, then H28-Q)

- **State:** provisional development checks only; no scientific verdict.
  - T0 instrument checks and a bounded artifact inventory.
  - Oracle coordinate fixture validated.
  - Oracle-core adapter pilot transfers.
  - The learner harness has oracle, independent and random-core controls,
    reconstruction, and parameter-byte accounting (`11c9471`).
- **Missing:** amortized economic endpoint; frozen
  `H28_C_LEARNER_OPPORTUNITY_PLAN.md`; independent scorer; restart test. H28-Q not
  started.
- **Plans:** `H28_CLOSURE_RESEARCH_PLAN.md` (CL1-CL8), `H28_T0_INSTRUMENT_PLAN.md`,
  `H28_C_COORDINATE_GATE_PLAN.md`, `H28_C_ADAPTER_PILOT_PLAN.md`,
  `H28_C_LEARNER_OPPORTUNITY_PLAN.md`.
- **Next:** add the economic endpoint and freeze the learner plan. It can resume
  now: no run holds HEAD.

# Available in parallel (no dependency on Track B)

## Track A: role-filler semantics (review 84)

- **State:** RF0a done (raw semantics survive; role adds about 0). RF0b
  UNRESOLVED (`71a4411`). RF1 not planned in detail; RF2 not started.
- **Plans:** `RF_ROLE_FILLER_PLAN.md` (draft program),
  `RF0_ROLE_FILLER_PROTOCOL.md`, and the RF0a/RF0b plans (frozen, run).
- **Next:** an RF1 plan, if prioritized. The convergence gate (Track A RF1 plus
  Track B) is required before typed higher-order combinators; PX6 depends on
  it.

## Budgeted execution economy

- **State:** `BUDGETED_EXECUTION_ECONOMY_PLAN.md`, a PI-merged DRAFT marked
  do-not-freeze. B0/B1 are Tier 0/1 and queued; not started.

## Infrastructure backlog

- `CONCURRENCY_PLAN.md` is still marked DRAFT (with Amendment 1). The pool module
  and bounded-pool practice are in use.
- Non-bitwise speedups (batched `forward_tasks`, batched slots, about 1.95x) are
  versioned candidates in `notes/performance_audit.txt`, admissible only at a
  new plan boundary with an equivalence gate.

# Design only

## Track D: self-hosted program synthesis (SH0-SH4)

- **State:** design in `POST_E6_RESEARCH_PROGRAM.md`. SH0 waits on C0's typed
  IR.

# Closed (do not reopen without a new plan)

- **V1-V5 sealed programs:** V5 closure includes review-55 withdrawals.
- **V6 / V6R:** adaptation geometry (development).
- **H39:** sealed (seeds 700-729). The formation line H47-H53 closed negative.
- **Export branch:** E0-E3 and E8 development; the sealed export block is
  confirmed (seeds 800-829). E5's writer failed on quality. E5.1: search cost is
  logarithmic in space size. The E6 macro line closed: macros pay but cannot be
  timed or compiled. The E7 census refuted the parameterized-macro rationale.
  E8D and E9 were withdrawn before freezing.
- **Iteration world spec:** withdrawn; superseded by the rotated substrate.
- **SO2:** `SO2_FAILS` (6a4f707). SO3 shows it does not generalize to fresh
  worlds.
- **SO2-P and SO3 stage-3 consolidation candidates:** SO3_FAILS; the rescue does
  not replicate.

# Hypothesis families and where they live (all in `PREDICTIONS.md`)

| family | line | status |
|---|---|---|
| CF1-CF7 | staged formation (Track B) | working hypotheses; J1c supported staging; CF4 refuted by J1 |
| CL1-CL8 | H28 closure / coordinate reuse | working hypotheses, untested |
| PX1-PX9 | program ladder | working hypotheses, recorded 2026-09-15 |
| B0-B3 | H47 membership / opportunity | closed |
| SO0-SO3, J-rungs | Track B plans | registered inside each frozen plan |

# Decisions pending from the PI

1. Program ladder: may teacher-side and oracle-executor gates (L0a-L0c) run
   before online learnability is established? The plan reads "yes".
2. Program ladder: run L0d (inference census on existing libraries) first?
3. (Superseded by SO3's outcome; now folded into 6.) If online learnability
   fails again: close the ladder's learner rungs, or authorize an
   offline-scoped variant?
4. Should L1-L3 target remote workers from the start?
5. Development-world budget for later rotated Tier 2 rungs: worlds 6-9, or a new
   development band?
6. NEW: register the B2 re-test of the unchanged online staged protocol on
   worlds 6-9, with several streams and the G5R margin? If it passes, the B2
   statement and the ladder's learner rungs open. It would use up 6-9, which
   bears on decision 5.

# Housekeeping owed

- **`SPEC_AUDIT.md` full re-audit still owed.** A PARTIAL record-consistency
  re-audit (rotated substrate through SO3) was appended 2026-09-15.
  - Two items were verified in code: the SO2 terminal estimand and the replay
    RNG.
  - It found that J2A's zero route gap on its failure controls is not J0's
    threshold prediction; corrected in PROGRESS and PREDICTIONS.
  - Still owed: a code-level audit of runner and scorer against plan, with at
    least one recomputed cell per milestone.
- **Stop-rule numbering.** Documents written 2026-09-15 (this file,
  `PROGRAM_LADDER_PLAN.md`, `SO4_B2_RETEST_PLAN.md`) call the online-failure
  rule "stop rule 2". In `POST_E6_RESEARCH_PROGRAM.md` it is the THIRD Track-B
  stop rule ("Online SO2 fails: do not run branching or iteration"); rule 2 is
  "oracle routes pass but learned routes do not". The intended rule is the
  online-failure rule everywhere.
- **`PROGRESS.md` is not strictly chronological.** The 2026-09-14/15 SO2-line
  entries sit before the 2026-09-14 H28 entries; see the placement note.
- **Paper draft** is current through SO3. The copy in
  `~/Downloads/ROW_paper_draft_2026-09-15.md` predates the SO3 addendum and
  does not auto-update.
