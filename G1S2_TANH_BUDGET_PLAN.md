# G1S2: sleep length and wake updates for the `tanh` learner (Tier 1, EXPLORATORY, sizing)

Status: FROZEN 2026-10-17.

Runner `row.experiments.g1s2_tanh_budget`; independent scorer `row.experiments.score_g1s2_tanh_budget`; tests
`tests/test_g1s2_tanh_budget.py`. Development band 6, world 70, stream 0. Exploratory: this rung CHOOSES a budget for
the next probe and produces no verdict.

# Why

G1S (`G1S_TANH_SIZING_PLAN.md`, `reports/g1s_tanh_sizing.json`) labelled `IMPROVES`: at learning rate 0.003 and initial
gain 1 the slept depth-1 error is 0.025, under the formation bar, but depth 2 is 0.068 and depth 3 0.184 (x2.7 per
step). The improvement over G1T came from sleep, not wake: the chosen setting's wake terminal was 0.497 against G1T's
0.510, while sleep took it to 0.184 against 0.405. Its plan's pre-stated next step: "a second sizing axis on world 70
at the chosen setting: wake updates per example and sleep length". This rung asks two things at once, in one cheap
design: does longer sleep keep lowering error, and does the compounding with depth shrink as per-operation error does
(a budget effect) or stay (a property of the `tanh` form, which has no identity skip to damp each step's error)?

# Construction (arms are constructions; world 70, stream 0, lr 0.003, gain 1)

| arm | construction |
|---|---|
| `WAKE_u1` | `o2.run_single('SHUFFLED')` verbatim at G1S's chosen setting; `updates_per_example = 1` (G1S's) |
| `WAKE_u2` | the same with `updates_per_example = 2` in every stage config (twice the wake updates; the replay stream changes with it, as it shares the sampling generator) |
| `SLEEP_u{u}` | on that wake terminal, `o2c.consolidate`'s loop verbatim (AdamW groups, batch 2, reservoir of 64 per task, seed `[1941, w, stream, 64]`) for 65,536 updates, scored at 8,192, 16,384, 32,768 and 65,536 |

Four cells and eight scored points. Each sleep is ONE trajectory: its 8,192 point is G1S's sleep, and each later point
continues the same minibatch stream, so the curve is a dose curve without new sampling noise between its points.
Scoring is `@torch.no_grad` in eval mode and changes no parameter, optimizer or RNG state.

**Gates (before any cell).** B1: under each budget all three stage configs carry lr 0.003 and the set
`updates_per_example`, a fresh learner starts at gain 1, and the `u = 1` learner is bitwise G1S's chosen setting's. B2:
`restore()` returns the defaults. B3: the checkpointed loop reproduces `o2c.consolidate` bitwise (24 updates on a small
pool, checkpoints at 8 and 24). **Per cell:** the wake's own resolved `config.yaml` must show lr 0.003 and its
`updates_per_example`; the sleep must read lr 0.003, and the library must change between every pair of checkpoints.

**Anchor A0.** `WAKE_u1`'s per-task terminals must equal G1S's `WAKE_lr0.003_g1`, and `SLEEP_u1` at 8,192 must equal
G1S's `SLEEP_lr0.003_g1`, EXACTLY.

# Measurements

Per point (wake updates x sleep length): slept terminal median on the canonical 64, per-task terminals, per-depth
medians over the 188 stream tasks (60 / 64 / 64), slept depth-1 identity ARI, and the COMPOUNDING RATIO, depth-3 median
over depth-1 median. Per wake: terminal, depth-1 identity ARI, seconds. Per sleep: seconds to each checkpoint.

# Reading (written before any cell; exploratory, chooses and never confirms)

Among points whose slept vocabulary keeps identity (`d1_ari >= 0.8`), CHOOSE the lowest slept terminal median:
- A0 fails -> `ANCHOR_MISS`: nothing is read;
- no point keeps identity -> `IDENTITY_LOST`;
- chosen `< 0.05` -> `FITS`. Next: a new frozen formation probe at the chosen setting and budget with the confirmed
  pipeline and G1T's other arms, on worlds 71-72 (paired with G1T) and fresh band-6 worlds, never world 70;
- `0.05 <= chosen < 0.1` -> `IMPROVES`. Next: if the chosen wake budget's curve is still falling at 65,536 (its last
  doubling lowered the terminal by more than 10%: `last_doubling_ratio < 0.9`, computed by runner and scorer), one
  further doubling; otherwise the compounding census below;
- chosen `>= 0.1` -> `NO_GAIN`: budget is not the lever. Next: a Tier 0 compounding census on world 70 ("freeze one
  half"): with the teacher's routes given, does this slot family fit `tanh2` compositions under the lifetime's optimizer?

The compounding ratio is read at every label: if it falls as depth-1 error falls, compounding is a budget effect; if it
stays near G1S's 7.4 (0.184 / 0.025) while depth-1 error falls, it belongs to the form and more budget will not reach
depth 3.

# Necessity

The behaviour is reaching the formation bar at depth 3, given operations that already fit at depth 1. The arm that
refuses more budget is G1S's chosen point (wake `u = 1`, sleep 8,192), part of this run as the anchor; its refusal cost
is measured: 0.184 against the 0.05 bar (scale: a factor of 3.7) and against 0.009 for the rotated learner on this world.
The expected impostor is a point that lowers terminal error by collapsing operations onto fewer slots; the identity
filter excludes it from the choice. A second impostor is over-consolidation on the retained 64 examples per task
(memorizing the reservoir): the terminal is scored on held-out evaluation examples, so memorization cannot pass the
bar. Difficulty band: the target is the rotated learner's regime, slept terminal 0.009-0.03 (G1); a point below 0.005
would be suspicious and checked before use.

# Discriminating power

**The rule:** chosen = argmin slept terminal median over the eight points with `d1_ari >= 0.8`; `FITS` if `< 0.05`,
`IMPROVES` if `< 0.1`, else `NO_GAIN`.

Simulated (200,000 draws, seed 20261016) with the seven non-anchor points log-normal around their true values and drawn
independently, which overstates their spread since points on one trajectory are correlated:
- null (no budget helps: all seven centred on G1S's 0.184): with log-sd 0.25, `FITS` false-fire 0.0000, `IMPROVES`
  false-fire 0.0504; with log-sd 0.15 (still above the 0.06 of G1T's three slept worlds), 0.0000 and 0.0001;
- full effect (one point centred on 0.02, log-sd 0.5): `FITS` detected 0.968;
- partial effect (one point centred on 0.06, log-sd 0.5): `IMPROVES` or better detected 0.924 (`IMPROVES` 0.547,
  `FITS` 0.377).

**What the checks do not cover:**
- one world, one stream, fitted to world 70, which is why the next probe excludes it;
- two wake budgets and four sleep lengths; nothing between them;
- longer sleep on 64 retained examples per task is a memory budget the online learner has already paid; the plan does
  not count it against a cost bar, and the compute of each point is reported, not judged;
- doubling wake updates also changes the replay stream (shared generator), so `u = 2` against `u = 1` is the budget and
  the draw together;
- re-routing is not run, as in G1S;
- the thresholds are judgements: 0.05 is the formation bar, 0.1 about half G1S's chosen point.

# Operational

4 cells, pool 2 (two lifetimes at ~340 MB each): `WAKE_u2` and `WAKE_u1` first, each sleep submitted when its wake
exists. Timing from G1S on this host: wake ~1,000-1,120 s per update-per-example, sleep ~275 s per 8,192 updates, so
`WAKE_u2` ~37 min and each sleep ~37 min plus eight scorings; expected about 75-80 minutes. Each cell is written
atomically to `artifacts/g1s2_tanh_budget/cells/`, stamped with the protocol hash (this plan, config, the G1S report,
every reused module, commit) and reused on relaunch only under that stamp; the parent writes every `run.log` line. NEW
(G1S learning): at exit, success or failure, the runner itself copies `run.log`, `status.json`, `exit.json`,
`gates.json`, `precondition.json`, `manifest.json`, `run.pid`, any `error.json`, and every cell record into
`reports/g1s2_tanh_budget_<date>/`, so the record no longer depends on the untracked `artifacts/` surviving. Dry run:
world 99, scale 1/16, sleep checkpoints 64 and 128, all four cells through `validate`, with a restart test. Launched
detached from clean committed code. The scorer recomputes the reading from the cell records and G1S's committed cells
and refuses a report whose inputs changed.

**Write-time check (2026-10-16), item by item against the implementation learnings.** Opportunity: budget can move the
depth-3 error or not, and the compounding ratio can fall or stay. Discrimination: the rule was run against null and
effect samplers (above). Threshold against baseline: the anchor point is itself eligible and sits at 0.184, so "no
budget helps" resolves to `NO_GAIN`. Denominators: eight points, per-depth counts 60 / 64 / 64 checked in `validate`.
Estimands against code: the terminal is the final model on the canonical 64; the depth-3 median over the stream's 64
depth-3 tasks is that same set. Fit and score: world 70 chooses and is excluded from the next probe. Arms are stated as
constructions. Non-vacuity: per-cell learning-rate and updates checks, the library must change between checkpoints,
gate B3 compares against `o2c.consolidate` itself, A0 can fail. Found one gap, fixed before freezing: the `IMPROVES`
branch names a "still falling" criterion that the code did not compute; `last_doubling_ratio` is now computed by both
runner and scorer and tested.

**Dry run disclosure.** World 99, scale 1/16, sleep checkpoints 64 and 128, all four cells. The first attempt failed
before any cell because this plan file did not exist yet (the protocol hashes it). After the doubling-ratio fix the dry
directory was deleted and rerun: stopped after one cell, two orphan workers killed by `parent_pid`, relaunched; the
finished cell was reused byte-identical (sha256 `c13f2f8e...`), the other three ran, exit 0, and the archive step copied
the logs and all four cell records. Dry terminals (1.12-1.31) mean nothing at that budget. No point on world 70 beyond
G1S's has been run.

