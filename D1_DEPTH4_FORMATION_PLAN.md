# D1: does the confirmed fully online protocol form the substrate when programs are four steps long? (Tier 1, exploratory)

Status: FROZEN 2026-10-04, before any D1 cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** Decision 17 (the depth rung) was approved by the PI
("Ok continue", 2026-10-04) with Claude's defaults: a depth-4 world
configuration (a new testbed, not comparable with existing artifacts) and
development band 30-49, allocated here. D1 uses worlds 30-36; world 48 was used
for the pre-plan timing cell (disclosed below) and world 49 is the dry-run
world; 37-47 stay unused.

# Why

O11 (sealed) confirmed at depth 3 that wake + in-stream re-routing of earlier
tasks + O3's sleep forms the rotated substrate reliably online (45/45, and
45/45 paired over sleep alone). Every claim in the online-formation line is at
depth 3. The open question is whether the protocol survives longer programs,
where routes have more positions to go stale and anchors are a smaller share
of the stream.

**A premise corrected before planning (Tier 0, 2026-10-04).** Decision 17 was
written assuming in-stream exhaustive re-routing at depth 4 would cost ~1.6 h
per cell and need O7's gradient re-router. A structural probe on world 30 (a
scale-16 lifetime) measured exhaustive search at **0.27 s** per depth-4 task
on 64 examples and O7's 500-step gradient search at **1.06 s**. Only 64 of the
252 stream tasks are depth 4, so in-stream exhaustive re-routing was projected
at about 40 minutes per cell (a full-scale cell then measured 47 minutes; see
Operational), and the gradient re-router is slower, not faster, at this
depth. D1 therefore tests the confirmed protocol itself at depth 4, with
exhaustive search; the gradient re-router is deferred to depth 5.

# The depth-4 stream (construction)

`row.experiments.d4_stream`: one stream-seeded random order (order seed
`[1930, w]` at stream 0, else `[1930, w, s]`) over 252 tasks:
- 64 canonical length-4 tasks: `generate_rotated_world` with
  `program_length=4` (distinct programs; cap 6^4 = 1,296), the scored set;
- 60 length-1, 64 length-2, 64 length-3 tasks: `generate_curriculum_world`
  (programs with replacement), exactly the depth-3 line's anchor construction.

The learner is `PlannedDepthRotatedLearner` with `task_steps=4`; each task runs
at its own depth (the parent class supports per-task depth up to
`task_steps`). Everything else is O2-O11's: model seed 5000, `configs/v1.yaml`,
`rotated_discrete_fast`, LEAN diagnostics, replay seed `None` at stream 0 else
`SeedSequence([7500, w, s])`, 64 retained examples per task (seed
`[1940, w, s, i]`).

# Arms, as constructions

| arm | construction |
|---|---|
| `SHUFFLED4` | wake alone: `o2.run_single`'s construction on the depth-4 stream and learner |
| `RW4` | `SHUFFLED4` plus O8's `Rerouter` hook verbatim (after each task, exhaustive search over `12^d` routes for every earlier task on its 64 retained examples, O5's logit swap; the arriving task is never touched) |
| `RW_SLEEP4` (primary) | O3's sleep calls verbatim (`o2c.consolidate`, 8,192 updates, sampling `[1941, w, s, 64]`) on this run's `RW4` terminal |
| `SLEEP4` (reference) | the same sleep on this run's `SHUFFLED4` terminal |

# Cells

Worlds 30-36, streams 0-2: 21 cells per arm, 84 cells. `RW4` and `SHUFFLED4`
first; each sleep cell is submitted when its parent terminal exists.

# Rule (registered)

`k` = `RW_SLEEP4` cells with terminal median `< 0.05`, denominator 21.
`h` = cells that `SLEEP4` passes and `RW_SLEEP4` fails, denominator = the
number of cells `SLEEP4` passes (counted, at most 21). A non-finite value never
passes.

| condition (first matching row) | label |
|---|---|
| `h >= 3` | `HARMS` |
| `k >= 18` | `TRANSFERS` |
| `12 <= k <= 17` | `PARTIAL` |
| `k <= 11` | `DOES_NOT_TRANSFER` |

The labels partition every outcome. Also reported: `k` for every arm,
collapses (`>= 1.0`) per arm, `RW_SLEEP4` strictly below `SLEEP4` (of 21),
medians, routes changed and terminal stale routes per `RW4` cell, re-route
seconds, end-of-task median over the canonical tasks.

# Necessity

- **Target behaviour:** reliable online formation at program length 4 by the
  protocol confirmed at length 3.
- **Refusal arm:** `SLEEP4`, the same memory and consolidation without
  in-stream re-routing; and `SHUFFLED4`, wake alone.
- **Measured refusal cost and its scale:** at depth 3 (O11, sealed), sleep alone
  left 7 of 45 cells above the registered 0.05 threshold and wake alone 31 of
  45; the combined protocol left none. The depth-4 refusal cost is measured
  here, and its scale is the threshold.
- **Impostors:** extra compute (re-routing trains nothing; the sleep budget is
  identical across the two sleep arms); world luck (7 worlds x 3 streams); an
  executor that cannot reach the threshold at depth 4 at all: L0d's depth-4
  execution gate reached median NMSE 0.0087 with a frozen staged library on
  depth-4 programs, so 0.05 is reachable in principle.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. The depth-3 protocol sits at 0.004-0.019.

# Discriminating power

**The decision rule, as it will be applied:** `TRANSFERS` if `k >= 18` of 21
and `h <= 2`; `HARMS` if `h >= 3`.

Source `reports/d1_design/rates.py`, output `rates_output.txt`, exact
binomials.

| true per-cell pass rate | `TRANSFERS` (`k >= 18`) fires |
|---|---|
| 0.43, depth-3 order-free wake alone (null) | **false-fire 0.0001** |
| 0.6 | 0.011 |
| 0.7, partially reliable (null) | **false-fire 0.086** |
| 0.8 | 0.370 |
| 0.9 | 0.848 |
| 0.95, near the depth-3 sealed rate (effect) | **detection 0.981** |

Harm clause, at a per-cell break probability of 0.05: `HARMS` fires 1.2% of
the time over 10 sleep passes and 8.5% over 21.

**What the checks do not cover:**
- 21 cells on 7 worlds, one development band: a Tier 1 sizing, not a verdict;
  a true rate of 0.8 lands in `PARTIAL` most of the time.
- The 0.05 threshold is carried over from depth 3; deeper programs compound
  per-step error (E5.1's drift), so a library can be good per step and still
  miss 0.05 at depth 4. If `PARTIAL` or `DOES_NOT_TRANSFER`, the per-cell
  values and the depth-3 sub-scores of the same library must be read before
  concluding formation failed.
- The timing cell on world 48 (full scale, stream 0) ran before this plan was
  frozen and its results were SEEN: wake alone 2.06 (collapsed; 128 stale
  routes at the terminal), wake + in-stream re-route 0.091 (1 stale route),
  re-route + sleep **0.0149**, sleep alone 1.50 (collapsed). One world, one
  stream; it informed the runtime estimate and was not used to set the rule
  (the thresholds and labels are the depth-3 line's). World 48 is not in D1's
  cells.
- One sleep budget and memory size, tuned at depth 3.

# What it decides (for planning)

- **`TRANSFERS`:** the protocol survives depth 4. Next: depth 5, where
  in-stream exhaustive search becomes costly (~3 s per depth-5 task) and the
  gradient re-router or a sparser re-route schedule is needed; then a sealed
  band (PI decision).
- **`PARTIAL`:** read the per-cell values: per-step quality against
  composition drift, near-misses against collapses.
- **`DOES_NOT_TRANSFER`:** the depth-3 protocol does not form a depth-4
  substrate at this budget; diagnose before any scaling claim.
- **`HARMS`:** in-stream re-routing hurts at depth 4; diagnose first.

# Operational

84 cells on a pool of 3. Measured on world 48 at full scale (two cells
concurrent): `RW4` 3,820 s (2,833 s of it in-stream re-routing, more than the
~40 min projected from the per-task probe), `SHUFFLED4` 1,133 s, each sleep
148 s. Total about 21 x (3,820 + 1,133 + 2 x 148) / 3 = 36,700 s, about 10
hours. Durable stamped cells; protocol fingerprint (this
plan, config, the d4/D1/O8/O3/O2/O2C/O2D/census/lifetime modules, commit);
`run.log`, `status.json`, `exit.json`, `precondition.json`; detached launch;
scale-16 dry run of every arm on world 49 with a restart test; reserve 2.0 GiB
(PI 2026-10-03); logs archived to `reports/d1_*`.
