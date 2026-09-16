# ROW research status: what is live, blocked, pending, closed

This is an INDEX, rewritten in place whenever a line of research changes state,
and committed with the change that caused it. The scientific record lives
elsewhere and is append-only: verdicts, hypotheses and corrections in
`PREDICTIONS.md`; completed steps in `PROGRESS.md`; results in `reports/`. If
this file disagrees with those, they win and this file is stale.

Last rewritten: 2026-09-16, after the SO4 verdict and the world-quality census.
Nothing is running.

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
| SO4 | `SO4_FAILS`: terminal 2/4, margin 3/4; world 6 negative margin | (this commit) |

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
- **Successor options (PI DECISION 7, now informed):**
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
  - Claude's recommendation: option 2 first (minutes, no new world, targets the
    one counterexample), then option 1 only if it survives.
- **Any further online lifetime needs a new development band** (decision 5):
  worlds 0-9 are spent.

# Drafted, awaiting PI decisions

## Program ladder: compositional operators -> control flow -> higher-order programs

- **State:** DRAFT `PROGRAM_LADDER_PLAN.md`, hypotheses PX1-PX9 in
  `PREDICTIONS.md` (committed `fbb547a`).
- **Learner rungs (L1-L3) remain gated** by the online-failure stop rule, which
  SO4 did not lift.
- **Runnable now, no learner and no new world:** L0d, the program-inference
  census (PX7). Plan drafted: `L0D_INFERENCE_CENSUS_PLAN.md`.
  - Twelve frozen J2A libraries; depths 3-5; support sizes 128/32/8.
  - Mechanisms: exhaustive enumeration, gradient route optimization, beam
    {1,4,16}, posterior averaging, a commit-late hybrid, plus random and teacher
    controls.
  - Registered additions: a variable-depth forward gated bitwise at depth 3, and
    an eligibility gate separating search failure from execution drift.
  - Its decisions: run now or after the world census; include depth 5 or cap at
    4; any mechanism to add or drop.
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
2. Ladder: run L0d first, or after the world census?
3. (Folded into 7.)
4. Should ladder rungs L1-L3 target remote workers when they open?
5. Development-world budget: worlds 0-9 are spent for the rotated staged
   protocol. Allocate a new development band, or stop opening worlds for it?
6. (Answered by SO4: the re-test ran and failed.)
7. NEW, recommended first: run the Tier 0 cross-run world census (10 worlds, 22
   staged cells, existing artifacts) before any further Track B intervention?

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
