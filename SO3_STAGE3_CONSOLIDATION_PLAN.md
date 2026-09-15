# SO3: does a stage-3 consolidation setting make online staged formation pass? (Tier 2 DRAFT)

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any SO3 code or cell exists. The PI approved the
draft unchanged on 2026-09-15 ("continue"), including every choice marked
**[PI]** below; those marks now record which defaults Claude proposed and the
PI accepted. The runner and independent scorer are committed together before
launch.

# Question

SO2 (`SO2_FAILS`, 6a4f707) found that online staged formation acquires and
exports the rotated substrate but misses the terminal threshold. The Tier 0
census (3d87a0f) localized the failure: stage 3 loses recently learned tasks
after they are learned. The Tier 1 exploration SO2-P (afb1ac6, corrected in the
following commit) found, on world 1 only with one stream per arm, that two
stage-3 settings rescue it:
- shared learning rate halved (0.0005): 33/64 tasks at or below 0.05;
- 8 stored replay examples per task instead of 4: 53/64, gradient-matched but
  confounded with a different replay sampling stream.

It also found that less drift alone is not the lever.

SO3 asks, on development worlds SO2 did not use, from scratch through all three
stages, with several replay streams per arm and a stream-only control: does
either setting pass SO2's terminal criterion online, and is REPLAY's gain more
than stream luck?

# Construction

- **Protocol:** SO2's online staged protocol unchanged (`audit_so2_online_gate`
  construction). Three consecutive lifetimes per world (60 length-1, 64
  length-2, 64 length-3 tasks), library-only transfer, score before update,
  `rotated_discrete_fast`.
- **Stage 1-2 settings:** identical in every arm, the canonical config.
- **Stage 3:** each arm changes exactly one stage-3 field.
- **Worlds [PI]:** development seeds 3, 4, 5. They are unused by SO2, SO2-P and
  the census. They were used in V1-era Continuous/Dense work on a different
  substrate, never with `rotated_discrete_fast` or any staged protocol.
- **Model seed [PI]:** 6000. It is fresh for this protocol; SO2 and SO2-P used
  5000.
- **Replay streams [PI]:** 3 per arm per world.
  - A stream is the replay buffer's generator seed, applied in ALL THREE
    stages.
  - Stream 0 is the canonical `seed + 1`; streams 1-2 are
    `SeedSequence([7300, world, stream])`.
  - This needs one registered, additive, default-off change to
    `learned_lifetime.run`: an optional `replay_seed=` that replaces
    `seed + 1`. When omitted, it must reproduce every existing caller bitwise,
    verified on SO2's saved STAGED_w1 stage 3 (G0 below).

| arm | stage-3 change | role |
|---|---|---|
| BASE | none | SO2's construction on fresh worlds |
| LR_HALF | `global_learning_rate` 0.001 -> 0.0005 | Tier 1 candidate |
| STORE_8 | `replay_examples_per_task` 4 -> 8 | Tier 1 candidate (rehearsal diversity) |

- **Stream-only control:** BASE at streams 1-2 IS the stream-only control for
  STORE_8. It keeps 4 stored examples per task and changes only the replay
  stream. STORE_8's effect is read against BASE's across-stream spread, not
  against one BASE stream.
- **Dropped from Tier 1:** LR_1/4 and LR_1/10 (they removed backward transfer),
  and STORE_16 (it failed). This follows the scoring-economy rule: drop
  curve-refinement cells, never causal samples.

Cells: 3 arms x 3 worlds x 3 streams = 27. Stages 1-2 depend only on world,
model seed and stream, not on the arm. So they are shared, not recomputed: one
stage-1-2 prefix per (world, stream), 9 prefixes, each fanned out to 3 stage-3
arms. A registered gate checks that sharing is bitwise-equivalent to
recomputing.

# Gates (a failed gate makes the run HARNESS_FAILED)

- **G0 additive change:** with `replay_seed` omitted, `learned_lifetime.run`
  reproduces SO2's saved STAGED_w1 stage 3 exactly (library SHA, and every
  per-task terminal within 1e-6). With `replay_seed = seed + 1` passed
  explicitly, the result is identical too.
- **G1 prefix sharing:** for one (world, stream) chosen before launch (world 3,
  stream 0), a stage-3 BASE cell run from a freshly recomputed prefix equals the
  shared-prefix cell bitwise.
- **G2 non-vacuity:** within each (world, stream), the three arms' terminal
  libraries differ. Within each (world, arm), different streams give different
  terminal libraries.
- **G3 anchor:** in every stage of every cell, the last task's terminal NMSE
  equals its end-of-task NMSE within 1e-6.
- **G4 transfer:** stages 2-3 start from the previous stage's library SHA; task
  ids never collide across stages.

# Estimands

Per cell (world, stream, arm), on the 64 canonical length-3 tasks:
- `M` = terminal median query NMSE;
- `K` = count of tasks with terminal NMSE <= 0.05.

A **world** passes for an arm when the median of `M` over its 3 streams is
<= 0.05.

Reported beside, per cell: end-of-task median; lost and gained threshold
counts; recency Spearman; library drift from the stage-2 library; stage and
whole-stream prequential cost; example-gradients (equal by construction; the
resolved training fields are recorded); J2A's 64-program export diagnostic.

**[PI] G5R held-out margin: NOT computed.** It costs about 107 min per cell (27
cells is about 48 h), and SO2-P showed the scratch arm is weak enough that the
margin passed 3/3 where terminal quality failed. SO2's registered margin clause
is therefore not re-tested here; a SO3 pass cannot be called an SO2_PASSES
analogue. See Limits.

# Registered classification (evaluated in order, per arm)

0. **HARNESS_FAILED:** any gate fails, any missing or non-finite cell.
1. For each candidate arm A in {LR_HALF, STORE_8}:
   - **A_PASSES:** A's world-level median `M` <= 0.05 in at least 2 of 3
     worlds, AND in each of those worlds A's median `M` is below BASE's median
     `M`.
   - **A_STREAM_ROBUST:** (STORE_8 only, reported with its label) in at least
     2 of 3 worlds, A's median `M` is below the MINIMUM of BASE's three streams
     in that world.
   - **A_PARTIAL:** A's world-level median `M` <= 0.5 x BASE's in at least 2 of
     3 worlds, without passing.
   - **A_FAILS:** otherwise.
2. **Program label:**
   - **SO3_PASSES** if at least one candidate is A_PASSES. For STORE_8, it must
     also be A_STREAM_ROBUST, otherwise STORE_8 is reported as
     `STORE_8_STREAM_CONFOUNDED` and does not count.
   - **SO3_PARTIAL** if at least one candidate is A_PARTIAL or a confounded
     pass.
   - **SO3_FAILS** otherwise.

BASE enters only as the contrast. Its own terminal pass rate is reported. If
BASE passes 2/3 worlds on fresh worlds, that is reported prominently: SO2's
failure would then not generalize, and the candidates' contrasts carry the
claim.

# Registered predictions [PI: probabilities are Claude's; revise before freezing]

- BASE fails the world-level terminal criterion in at least 2/3 worlds: 0.6.
  SO2 failed 2/3, and fresh worlds may differ.
- LR_HALF_PASSES: 0.35. It was one task over a Tier 1 cutoff on one world and
  one stream, with a non-monotone dose.
- STORE_8_PASSES: 0.5. Tier 1 had a large margin but a confounded stream.
- STORE_8_STREAM_ROBUST given STORE_8_PASSES: 0.6.
- SO3_PASSES: 0.55.
- In candidate arms, stage-3 lost counts fall relative to BASE in at least 2/3
  worlds: 0.75.

# Registered consequences

- **SO3_PASSES:** "online staged formation of the rotated substrate passes the
  terminal criterion at this protocol with a registered stage-3 consolidation
  setting", stated with the supplied curriculum and the untested G5R margin.
  The margin clause (a targeted Tier 2 on the passing arm only) must be run
  before the B2 statement and any C2 control-flow rung can be opened.
- **SO3_PARTIAL:** report which arm and whether it was stream-confounded. No
  B2 statement.
- **SO3_FAILS:** the Tier 1 rescue does not replicate. SO2's stop rule stands,
  and the destructive-versus-consolidating movement hypothesis is weakened.

# Cost and run discipline

- **Per prefix:** stages 1-2 take about 5-6 min single-process (SO2 measured
  about 8 min for stages 1-3 under 3-way contention).
- **Per stage-3 cell:** about 3 min plus about 3 s export diagnostic.
- **Total:** 9 prefixes (about 50 min) plus 27 stage-3 cells (about 80 min),
  about 130 min single-process, about 45-60 min in a 3-worker pool. Well
  inside one overnight batch.
- **Order:** prefixes first, then stage-3 cells world by world, world 3 first.
  G0 and G1 run before any scored cell.
- **Early stop:** none; every cell is causal-sample, not curve refinement.
- **Discipline:** clean committed code; `check_prereg.py` hash; scorer
  committed with the runner; dry run and performance pass; durable hashed
  records per prefix and per cell; relaunch resumes; `run.log` / `status.json`
  (listing actually running cells, fixing SO2's cosmetic defect) /
  `exit.json`; detached launch; independent recomputation from saved models;
  logs committed to `reports/`.

# What this cannot establish

- **Not a discovered curriculum:** stage boundaries are supplied.
- **Not full SO2_PASSES:** the margin clause is untested.
- **Not a mechanism claim:** why a setting works is not identified. Lower
  learning rate and more diverse rehearsal are candidates, not explanations.
- **No other protocols or lengths:** nothing about program lengths beyond 3,
  batch sizes, or confirmatory bands.
- **Development worlds only.**
- **Single development protocol for a vocabulary claim:** the rotated family
  only.
