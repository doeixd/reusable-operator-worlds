# SO2-P: is stage-3 shared-library plasticity the online failure? (Tier 1, exploratory)

Status: FROZEN at this commit (PI approved the draft unchanged, 2026-09-15),
before any SO2-P code exists. Tier 1 plans are not hashed in
`tools/check_prereg.py`; this commit is the freeze record. The runner and
independent scorer are committed together before launch.

EXPLORATORY TIER 1. One development world, reduced scope. It produces no
verdict, does not revisit or relabel `SO2_FAILS` (6a4f707), and a null here
does not close the question (AGENTS.md: small versions can give different
answers). Its only job is to decide whether a plasticity intervention is worth
a preregistered Tier 2 plan, and to size that plan's thresholds against a
measured baseline.

# Question

The Tier 0 census (`SO2_INTERFERENCE_CENSUS_PLAN.md`, 3d87a0f) found:
- terminal-over-end-of-task degradation only in STAGED stage 3 of worlds 1-2;
- that degradation weighted toward the latest-arriving tasks;
- earlier-stage computation largely retained;
- stage-3 library drift about twice that of the passing world.

Working hypothesis (PREDICTIONS, 2026-09-15): each stage-3 task is fit partly by
moving shared operators, and later tasks undo those moves. If that is the
mechanism, making the shared library less plastic during stage 3 should reduce
drift and the terminal-over-end-of-task loss, and lower terminal error. If
drift falls but terminal error does not, the hypothesis is disfavoured.

# Construction

- **World:** development world 1 only. It has the largest stage-3 loss (21 tasks
  lost the threshold) and near-largest drift (0.52).
- **Start state:** SO2's saved STAGED_w1 stage-2 model
  (`artifacts/so2_online_gate/cells/STAGED_w1/stage2`), restored with all of its
  state, including the probe code, as the SO2 scorer does. Stage 3 is continued
  exactly as SO2's runner does: `carry_library` (library only, fresh task codes,
  rebuilt optimizer), then `learned_lifetime.run` with kind
  `rotated_discrete_fast` on the canonical world-1 length-3 tasks, model seed
  5000. Stages 1-2 are NOT rerun.
- **Arms:** stage 3 only. Each arm changes exactly one field of the resolved
  stage-3 config; everything else is identical.

| arm | change | purpose |
|---|---|---|
| BASE | none | reproduction gate |
| LR_1/2 | `discrete_model.global_learning_rate` 0.001 -> 0.0005 | plasticity dose |
| LR_1/4 | -> 0.00025 | plasticity dose |
| LR_1/10 | -> 0.0001 | plasticity dose |
| REPLAY_2x | `discrete_model.replay_examples_per_task` 4 -> 8 | interference via rehearsal |
| REPLAY_4x | -> 16 | interference via rehearsal |

- The task-code learning rate (0.05) is unchanged in every arm, so each task can
  still fit its route.
- The replay arms raise example-gradients as well as rehearsal, so they are a
  budget intervention as much as an interference one. Their gradient counts are
  reported, and they are not read as pure interference tests.
- Not included: interleaving earlier-stage tasks into stage 3 (a schedule
  change), freezing the library (a separate question), and any change to stages
  1-2.

# Gates (a failed gate makes the run uninterpretable)

- **G0 reproduction.** BASE must reproduce SO2's stage 3 exactly: library SHA-256
  equal to `STAGED_w1` stage 3's recorded `library_sha256`, and every per-task
  terminal NMSE equal to the recorded `terminal_per_task` within 1e-6. If it
  does not, stop; no arm is read.
- **G1 non-vacuity.** Every non-BASE arm's terminal library SHA differs from
  BASE's, and the resolved configs differ in exactly the one registered field.
- **G2 anchor.** In every arm, the last task's terminal NMSE equals its
  end-of-task NMSE within 1e-6.

# Measures (per arm; all reported, none a verdict)

- Stage-3 terminal median NMSE and count at or below 0.05 (of 64).
- End-of-task median and count at or below 0.05.
- Tasks losing or gaining the 0.05 threshold (end-of-task -> terminal).
- Spearman correlation of `log10(terminal/end_of_task)` with arrival position.
- Library drift from the stage-2 library: median and maximum per-slot relative
  functional change, with the census probe (`SeedSequence([2026, 1])`, 512
  states).
- Stage-3 prequential Gaussian log loss, example-gradients, and wall seconds.
- J2A's 64-program export diagnostic on the terminal library (about 3 s). The
  G5R margin is NOT computed (about 107 min per arm, and its scratch arm is weak).

# Exploratory triage rule (decides only whether to write a Tier 2 plan)

Read against BASE. Registered here so that it is not chosen after seeing arms.
BASE's values are known from SO2: terminal 0.1264, end-of-task 0.0770, 6/64 at
or below 0.05, 21 lost, drift 0.52.

- **LIVE:** some arm has terminal median <= 0.05 AND at least 32/64 tasks at or
  below 0.05, with drift below BASE's, AND its end-of-task median <= 2x BASE's
  (so it did not just trade the loss into slower acquisition). Write a Tier 2
  plan on fresh development worlds (see Limits) that registers this arm family.
- **PARTIAL:** some arm roughly halves BASE's terminal median (<= 0.063) or
  lost count (<= 10) without meeting LIVE, with drift below BASE. Report it; a
  Tier 2 plan may be considered with the PI, with thresholds sized from these
  measurements.
- **DISFAVOURED:** the LR arms reduce drift monotonically, but no arm reaches
  PARTIAL. Plasticity as measured by drift is not the lever. Record it, and do
  not pursue a Tier 2 LR plan.
- **UNINFORMATIVE:** drift does not fall with learning rate, or a gate fails.

# Cost and run discipline

Six stage-3 lifetimes of about 5 min each run in a 3-worker pool, so about
10-15 min wall-clock. Every rule applies:
- clean committed code;
- runner and independent scorer committed together before launch;
- dry run first, and G0 checked on the dry path where possible;
- performance pass;
- durable hashed per-arm record, with relaunch resuming;
- `run.log`, `status.json` and `exit.json`, launched detached;
- logs copied to `reports/` and committed;
- clearly labelled EXPLORATORY in PROGRESS.

# Limits

- **One world, one initialization, no margin.** It cannot establish that any arm
  helps online staged formation in general.
- **The start state is fixed.** Every arm inherits SO2's stage-2 library, so the
  question is only "given this library, does stage-3 plasticity matter?". A
  lower learning rate in earlier stages could change what stage 3 inherits;
  that is not tested.
- **World 1 is now used for the hypothesis and for Tier 1.** Development worlds
  0-2 have all been read in SO2. Any Tier 2 follow-up must be registered on
  development worlds not used by SO2, at a fresh model seed, with its own
  thresholds, and must rerun all three stages. No confirmatory band is touched.
- **A learning-rate change also changes acquisition speed.** The end-of-task
  clause guards against reading slower acquisition as less interference, but
  does not separate the two completely.
