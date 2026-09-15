# ROW research status: what is live, blocked, pending, closed

This is an INDEX, rewritten in place whenever a line of research changes state,
and committed with the change that caused it. The scientific record lives
elsewhere and is append-only: verdicts, hypotheses and corrections in
`PREDICTIONS.md`; completed steps in `PROGRESS.md`; results in `reports/`. If
this file disagrees with those, they win and this file is stale.

Last rewritten: 2026-09-15, while SO3 runs. Completeness was checked against
every top-level plan/spec status line, every `PROGRESS.md` H1 heading, and the
review index (latest review: 84).

# Live and running

## Track B: online learnability of the strong (rotated) substrate

- **State:** SO3 RUNNING, launched from `a185df1` at 11:28 UTC, pid 29964.
  Check it with `Get-Content artifacts/so3_consolidation/status.json`.
- **Question:** does a stage-3 consolidation setting (halved shared LR, or 8
  stored replay examples per task) make SO2's online staged protocol pass the
  terminal criterion on fresh worlds 3-5, with 3 replay streams per arm and a
  stream-only control?
- **Plan:** `SO3_STAGE3_CONSOLIDATION_PLAN.md`, frozen at `462e5cd`, hashed at
  `eaf2bd6`.
- **Path so far:**

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
| SO2 | `SO2_FAILS`: export 3/3, terminal 1/3 | 6a4f707 |
| interference census | post-acquisition, recency-weighted loss in stage 3 | 3d87a0f |
| SO2-P (Tier 1) | LIVE: LR/2 33/64, STORE_8 53/64 on world 1; drift is not the lever | afb1ac6 |
| SO2-P correction | replay arms gradient-matched but stream-confounded | b250973 |

- **Binding rule:** Track-B stop rule 2 ("online SO2 fails: do not run branching
  or iteration") holds until SO3 or a successor clears it.
- **Development-world use** under the rotated staged protocol: worlds 0-2 (SO1
  through SO2-P), worlds 3-5 (SO3); worlds 6-9 unused. A later Tier 2 rung needing
  fresh development worlds has at most 6-9, unless the PI allocates a new band.
- **Next:** when SO3 exits, commit the waiting documents (below), then run the
  independent scorer, `check_prereg.py` and `check_invalid.py`; copy logs to
  `reports/so3_consolidation_<date>/`; record the verdict in PREDICTIONS,
  PROGRESS, this file and the paper.
- **What each SO3 outcome licenses** (from its plan):
  - `SO3_PASSES`: the B2 statement, pending a targeted G5R-margin rung on the
    passing arm.
  - `SO3_PARTIAL`: report only.
  - `SO3_FAILS`: the stop rule stands.

# Drafted, awaiting PI decisions

## Program ladder: compositional operators -> control flow -> higher-order programs

- **State:** DRAFT plan `PROGRAM_LADDER_PLAN.md` with hypotheses PX1-PX9 in
  `PREDICTIONS.md`. Both are uncommitted until SO3 exits, so the remaining SO3
  lifetime folders keep the launch commit.
- **Extends:** Track C (C0 typed IR, C1 CHAIN/COMPOSE, C2 IF/REPEAT, C3
  discovery) and Track D of `POST_E6_RESEARCH_PROGRAM.md`. None of C0-C3 has
  started.
- **Prior evidence it must respect:**
  - The 2026-08-31 loop opportunity census: on straight-line E6 routes a loop
    construct is net negative at depth 6 and pays only where routes have
    collapsed (depth 8-10).
  - Its same-day CORRECTION: that census measured the absence of iteration in a
    generator with no iteration, not the value of loops. Hence the
    opportunity/necessity gate moved into generator design (the rotated
    substrate).
  - The E6 line: macros pay but cannot be timed or compiled.
  - The E7 census: a parameterized-macro family was not real against the
    learner's own grams.
- **Can run now without breaking stop rule 2:** L0d, the program-inference
  census on J2A's frozen straight-line libraries (PX7).
- **Needs a PI answer first:** L0a-L0c, the teacher-side PX1 world census,
  count-world gates and oracle-executor count horizon.
- **Gated on SO3 or a successor:** learner rungs L1 (PX1 economics), L2 (supplied
  ITERATE) and L3 (promoted ITERATE).

# Paused

## H28 closure / coordinate reuse (H28-C, then H28-Q)

- **State:** provisional development checks only; no scientific verdict.
  - T0 instrument checks and a bounded artifact inventory (six selected
    world-0 directories lack history/snapshot files).
  - Oracle coordinate fixture validated.
  - Oracle-core adapter pilot transfers.
  - The learner harness has oracle, independent and random-core controls,
    reconstruction, and parameter-byte accounting (`11c9471`).
- **Missing:** amortized economic endpoint; frozen
  `H28_C_LEARNER_OPPORTUNITY_PLAN.md`; independent scorer; restart test. H28-Q
  (nontrivial coarse-graining and autonomy) not started.
- **Plans:** `H28_CLOSURE_RESEARCH_PLAN.md` (CL1-CL8), `H28_T0_INSTRUMENT_PLAN.md`,
  `H28_C_COORDINATE_GATE_PLAN.md`, `H28_C_ADAPTER_PILOT_PLAN.md`,
  `H28_C_LEARNER_OPPORTUNITY_PLAN.md` (all development or draft).
- **Next:** add the economic endpoint and freeze the learner plan. It was paused
  behind SO2/SO3 because both need clean-HEAD launches; it can resume between
  Track B runs.

# Available in parallel (no dependency on Track B)

## Track A: role-filler semantics (review 84)

- **State:** RF0a done (raw semantics survive; role adds about 0). RF0b
  UNRESOLVED (`71a4411`): world 1's strong cell failed its permutation null.
  RF1 (execution of a held-out filler-role binding) not planned in detail; RF2
  (causal factorized learner) not started.
- **Plans:** `RF_ROLE_FILLER_PLAN.md` (draft program),
  `RF0_ROLE_FILLER_PROTOCOL.md`, and the RF0a/RF0b plans (frozen, run).
- **Next:** an RF1 plan, if prioritized. The convergence gate (Track A RF1 plus
  Track B SO2/SO3) is required before typed higher-order combinators. The
  ladder's PX6 bindings rung depends on it.

## Budgeted execution economy

- **State:** `BUDGETED_EXECUTION_ECONOMY_PLAN.md`, a PI-merged DRAFT marked
  do-not-freeze. B0/B1 are Tier 0/1 and queued; not started.

## Infrastructure backlog

- `CONCURRENCY_PLAN.md` is still marked DRAFT (with Amendment 1). The pool module
  and bounded-pool practice it describes are in use.
- Non-bitwise speedups (batched `forward_tasks` across distinct tasks, batched
  slots, about 1.95x) are recorded in `notes/performance_audit.txt` as versioned
  candidates. They are admissible only at a new plan boundary with an
  equivalence gate. `rotated_discrete_fast` (`batched_rotation_v1`) is the one
  already adopted.

# Design only

## Track D: self-hosted program synthesis (SH0-SH4)

- **State:** design in `POST_E6_RESEARCH_PROGRAM.md`. SH0 waits on C0's typed IR;
  no meta-writer is trained before a native search runs through that IR.

# Closed (do not reopen without a new plan)

- **V1-V5 sealed programs:** V5 closure includes review-55 withdrawals
  (`V5_CLOSURE.md`).
- **V6 / V6R:** adaptation geometry (development).
- **H39:** sealed (seeds 700-729). The formation line H47-H53 closed negative,
  including H47 B1/B2, H48b, H49, H50, H51 and H53 outcome D.
- **Export branch:** E0-E3 and E8 development; the sealed export block is
  confirmed (seeds 800-829). E5's writer failed on quality. E5.1: search cost is
  logarithmic in space size. The E6 macro line (E6A-E6F, E6.2) closed: macros
  pay but cannot be timed or compiled. The E7 census refuted the
  parameterized-macro rationale. E8D and E9 were withdrawn before freezing.
- **Iteration world spec:** withdrawn (its operators contract); superseded by the
  rotated substrate.
- **SO2:** `SO2_FAILS` (6a4f707); its successors are above.

# Hypothesis families and where they live (all in `PREDICTIONS.md`)

| family | line | status |
|---|---|---|
| CF1-CF7 | staged formation (Track B) | working hypotheses; J1c supported the staging idea, CF4 refuted by J1 |
| CL1-CL8 | H28 closure / coordinate reuse | working hypotheses, untested |
| PX1-PX9 | program ladder | working hypotheses, recorded 2026-09-15 (uncommitted until SO3 exits) |
| B0-B3 | H47 membership / opportunity | closed |
| SO0-SO3, J-rungs | Track B plans | registered inside each frozen plan |

# Decisions pending from the PI

1. Program ladder: may teacher-side and oracle-executor gates (L0a-L0c) run
   before online learnability is established? The plan reads "yes".
2. Program ladder: run L0d (inference census on existing libraries) first?
3. If SO3 fails: close the ladder's learner rungs, or authorize an
   offline-scoped variant by amendment?
4. Should L1-L3 target remote workers from the start?
5. Development-world budget for later rotated Tier 2 rungs: use worlds 6-9, or
   allocate a new development band?

# Waiting on the SO3 exit (do not commit earlier)

- `PREDICTIONS.md`: PX1-PX9 entry.
- `PROGRAM_LADDER_PLAN.md`.
- This file; the `POST_E6_RESEARCH_PROGRAM.md` status appendix; the
  `PROGRESS.md` placement note.
- `CLAUDE.md`: the `RESEARCH_STATUS.md` pointer, and the OptMem policy. OptMem is
  a situational working log; durable learnings go to repo files. The policy is
  a section after the tool-managed block plus matching in-block edits. `memo
  setup` rewrites text between the markers, so only the outside section is
  guaranteed to survive.
- `AGENTS.md` and `notes/learnings.txt`: the SO2-line implementation learnings.

# Housekeeping owed

- **`SPEC_AUDIT.md` re-audit is overdue.** The last section is the sealed
  export-confirmation audit (2026-08-27). Since then: the rotated substrate and
  G5/G5R, Stage D, SO1/SO1R, J0/J1/J1c/J1c-R/J2A and SO2. CLAUDE.md requires a
  re-audit after major milestones.
- **SO2-line implementation learnings: done (uncommitted).** Added to `AGENTS.md`
  and `notes/learnings.txt`: terminal versus end-of-task with the last-task
  anchor, probe codes in strict reloads, replay storage sharing the sampling
  RNG, no commits during runs, and movement magnitude versus interference.
- **`PROGRESS.md` is not strictly chronological.** The 2026-09-14/15 SO2-line
  entries sit before the 2026-09-14 H28 entries; see the placement note at its
  end.
- **Paper draft** is current through the SO3 freeze (`6cae33a`). A copy is in
  `~/Downloads/ROW_paper_draft_2026-09-15.md` and does not auto-update.
