# O9: sealed confirmation that a wake-only learner which re-derives earlier routes during the stream forms the rotated substrate reliably

Status: FROZEN 2026-10-04, before any world in 930-959 was generated.
Decision 15 (a new sealed band, its arms and thresholds) was put to the PI on
2026-10-04 with Claude's recommendation, and the PI answered "Ok continue";
the recommended defaults are adopted. **SEALED: worlds 930-944.** Worlds
945-959 are allocated to this band but stay sealed and unused; a later sealed
test may use them only under its own frozen plan.

This is the online-formation line's third confirmatory rung. O6 confirmed (45
of 45) wake, then exhaustive re-route of every task AFTER the stream, then
consolidation. O8 (Tier 1, exploratory, development worlds 20-26) found that
re-deriving every EARLIER task's route during the stream, after each arrival,
with NO consolidation, passes 20 of 21 cells against 8 for wake alone
(`PREVENTS`), and removes every collapse. O9 tests that fully online protocol
on worlds nobody has seen.

# The claims

1. **Reliability (primary, registered):** a wake-only online learner forms the
   rotated substrate in >= 90% of (world, stream) cells. It receives an
   order-free task stream containing single-operation anchors, retains 64
   examples per task, and after each arriving task re-derives the route of
   every earlier task by exhaustive search on that task's retained examples
   against its current library. It has no consolidation phase and no
   end-of-stream repair.
2. **Attribution (secondary, registered):** in-stream re-routing, not world
   luck, is responsible. Across paired cells, `REROUTE_WAKE` ends strictly
   below the same stream's plain wake terminal more often than chance allows.

Named assumptions, not results:
- single-operation tasks are present in the stream;
- the per-task memory holds 64 examples;
- routes are short enough to enumerate after every arrival (depth <= 3).

One substrate family, one learner, one model seed, one budget.

# Arms, as constructions

Model seed 5000. Stream `s` in {0, 1, 2}: replay seed `None` at 0, else
`SeedSequence([7500, w, s])`; order seed `[1920, w]` at 0, else
`[1920, w, s]`. These are O2-O8's recipes, verbatim, applied to worlds 930-944.

| arm | construction | cells |
|---|---|---|
| `REROUTE_WAKE` (primary) | O8's `run_cell('REROUTE_WAKE', ...)` verbatim: `o2.run_single`'s `SHUFFLED` stream and learner with O8's `Rerouter` prospective hook. After task `t`, every earlier task `i < t` is re-routed by exhaustive search over `12^d_i` routes on its 64 reservoir examples (seed `[1940, w, s, i]`), installed by O5's minimal logit swap; task `t` is never touched at step `t`. No consolidation; the hook makes no random draws. | 45 |
| `SHUFFLED` (refusal arm; reference for claim 2) | `o3.run_cell('SHUFFLED', ...)`, i.e. `o2.run_single`, LEAN; exactly O4's and O6's | 45 |
| `REROUTE_SLEEP` (comparator, descriptive) | O6's `run_cell('REROUTE_SLEEP', ...)` verbatim on the same cell's saved `SHUFFLED` terminal: exhaustive re-route of all 188 tasks, then 8,192 consolidation updates | 45 |
| `PLAIN` (floor) | `o3.run_cell('PLAIN', ...)`, no anchors, stream 0 | 15 |

`REROUTE_WAKE` and `SHUFFLED` share the stream, the learner, the replay
stream and every update; they differ only by the hook. O8's gate E1 showed the
hook switched off reproduces `SHUFFLED` bitwise; gate E2 here reproduces O8's
committed `REROUTE_WAKE` cell bitwise. Every arm is scored on the TERMINAL
model over the 64 canonical length-3 tasks.

# Gates (before any sealed cell; on DEVELOPMENT worlds only, except E4b)

- **E1:** O9's `SHUFFLED` reproduces O3's committed `SHUFFLED_w20_s0`
  bitwise (library sha and per-task terminal).
- **E2:** O9's `REROUTE_WAKE` reproduces O8's committed
  `REROUTE_WAKE_w20_s0` bitwise (library sha, per-task terminal, and total
  routes changed).
- **E3:** O9's `REROUTE_SLEEP`, applied to the E1 terminal, reproduces O5's
  committed `O3_w20_s0` bitwise (library sha, per-task terminal, routes
  changed).
- **E4b:** after the freeze commit, every sealed stream interleaves depths
  within its first 20 tasks.
- **E5:** a scale-16 dry run on DEVELOPMENT world 27, covering every arm, is
  interrupted after one cell and relaunched, and must reuse that cell.

# Rules (registered)

**Primary.** `k_RW` = `REROUTE_WAKE` cells with terminal median `< 0.05`.
**Denominator: all 45 cells, always.** A non-finite terminal counts as not
passing. A crashed cell is rerun, never dropped.

| condition (first matching row) | label |
|---|---|
| any `PLAIN` cell passes | `FLOOR_FAILED` |
| `k_RW >= 41` | `CONFIRMED` |
| `k_RW <= 40` | `NOT_CONFIRMED` |

**Secondary (attribution).** `n_better` = cells where the `REROUTE_WAKE`
terminal median is STRICTLY below the same cell's `SHUFFLED` terminal median.
**Denominator: all 45 paired cells.** A tie, or a non-finite `REROUTE_WAKE`
terminal, counts as not better.

| condition | label |
|---|---|
| `n_better >= 30` | `ATTRIBUTED` |
| `n_better <= 29` | `NOT_ATTRIBUTED` |

Each table partitions every outcome. The two labels are reported together, and
neither changes the other.

Descriptive only, with no rule attached:
- `k_SHUFFLED` and `k_RS` (`REROUTE_SLEEP` passes);
- prevention against repair: cells where `REROUTE_WAKE` ends below
  `REROUTE_SLEEP`, and the median per-cell ratio (O8 development: 5 of 21,
  1.15x);
- collapse counts (`>= 1.0`) per arm;
- terminal stale routes per `REROUTE_WAKE` cell; routes changed per cell;
  re-route seconds; end-of-task median over canonical tasks against
  `SHUFFLED`'s.

No early stop: the attribution clause needs every `SHUFFLED` cell, and
`REROUTE_SLEEP` depends on them. `REROUTE_WAKE` cells are submitted first, so
the primary count is known earliest.

# Necessity

- **Target behaviour:** reliable online formation by a learner that keeps its
  task-to-part assignments current as it learns, with no batch repair.
- **Refusal arm:** `SHUFFLED`, the same stream and learner with routes left
  as committed. On sealed O6 it passed 13 of 45 with 6 collapses; on O8's
  development cells 8 of 21.
- **Measured refusal cost and its scale:** in O8, all 13 development cells
  that wake alone left above the registered 0.05 threshold were brought under
  it by in-stream re-routing, with no consolidation; the median terminal moved
  from 0.0742 to 0.0112. The scale is the registered threshold.
- **Impostors:**
  - extra optimisation: `REROUTE_WAKE` adds no updates; route search trains
    nothing;
  - extra data: the reservoir holds examples already trained on, the 64-per-task
    memory assumption shared with O3-O8;
  - end-of-stream repair in disguise: nothing runs after the last task's row,
    and the last-task anchor (terminal equals end-of-task) is enforced on every
    `REROUTE_WAKE` cell;
  - world luck: 15 fresh sealed worlds x 3 streams;
  - tuning on test worlds: the protocol was designed on 13-26 and 900-929, none
    of which is in the sealed set.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. O8's `REROUTE_WAKE` cells lie at 0.0067-0.151, inside it,
  and `PLAIN` sits above it (~1.9).

# Discriminating power

**The decision rules, as they will be applied:** `CONFIRMED` if `k_RW >= 41`
of 45 and no `PLAIN` cell passes; `ATTRIBUTED` if `n_better >= 30` of 45.

**Primary: null and effect samplers.** Source `reports/o9_design/rates.py`,
output `rates_output.txt`, 20,000 draws, seed 21, O6's sampler: each world
draws a logit offset `N(0, sd)` and its streams pass independently given it.

| true per-cell rate | `CONFIRMED` fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.43, wake alone | 0.000 / 0.000 / 0.000 |
| 0.80, partially reliable (null) | **false-fire 0.039 / 0.029 / 0.020** |
| 0.844, sealed O4 sleep-alone level | 0.147 / 0.119 / 0.067 |
| 0.90 | 0.528 / 0.446 / 0.292 |
| 0.952, O8's development 20/21 (effect) | **detection 0.938 / 0.908 / 0.778** |
| 0.98 | 0.998 / 0.997 / 0.985 |

**Secondary: null and effect samplers.** Source
`reports/o9_design/attribution.py`, output `attribution_output.txt`, 20,000
draws, seed 22, same clustering.

| per-cell probability RW < SHUFFLED | `ATTRIBUTED` fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.5, re-routing adds nothing (null) | **false-fire 0.017 / 0.023 / 0.037** |
| 0.7 | 0.745 / 0.688 / 0.569 |
| 0.8 | 0.989 / 0.977 / 0.921 |
| 0.952, O8's development 20/21 (effect) | **detection 1.000 / 1.000 / 1.000** |

**What the checks do not cover:**
- **The development rate is optimistic.** O8's 20/21 comes from the worlds the
  protocol was designed after. At a true rate of 0.90, `CONFIRMED` fires only
  29-53%; at heavy world heterogeneity (sd 1) detection at 0.952 is 0.78.
  O9 can therefore fail on a real but smaller effect, and a `NOT_CONFIRMED`
  with k near 40 should be read beside the per-cell values.
- **Middle ground.** If the true rate is 0.844, `CONFIRMED` still fires 7-15%
  of the time; attribution is the guard against reading that as a re-routing
  effect.
- **Prevention against repair is descriptive.** O9 does not test whether
  `REROUTE_WAKE` matches `REROUTE_SLEEP`; O8 says it is slightly worse per
  cell (median 1.15x) while passing as often.
- **Scaling.** Exhaustive in-stream re-routing costs ~2 minutes per cell at
  depth 3 and is projected at ~1.6 hours at depth 4. Nothing here speaks to
  longer programs.
- **The logit swap leaves the lifetime optimizer's Adam moments for the swapped
  entries unswapped** (disclosed in O8); this is part of the construction.

# What it decides

- **`CONFIRMED`:** the line's fully online result. A learner that keeps its
  route assignments current while it learns forms the rotated substrate
  reliably, with no batch phase, within the named assumptions. With
  `ATTRIBUTED`, stale early commitments are confirmed as the cause of
  online-formation failure, not only a symptom. Next: the scaling rung, where
  in-stream re-routing needs the length-linear re-router (O7) from depth 4, on
  a new development band (30-49 recommended, PI decision).
- **`NOT_CONFIRMED`:** prevention alone does not reach the registered bar on
  fresh worlds. The per-cell values decide between residual near-misses (add
  consolidation; O8's interleaved arm) and residual collapse.
- **`FLOOR_FAILED`:** the substrate is formable without anchors on these
  worlds, and the comparison is uninterpretable.

# Operational

- **Cells:** 150 (45 `REROUTE_WAKE` + 45 `SHUFFLED` + 45 `REROUTE_SLEEP` + 15
  `PLAIN`). `REROUTE_WAKE` first, then `SHUFFLED`, then `PLAIN`; each
  `REROUTE_SLEEP` is submitted when its `SHUFFLED` cell completes. Pool of 3,
  about 7-8 h.
- **Operational contract:**
  - durable stamped cells, and relaunch resumes;
  - a fingerprint over this plan, `configs/v1.yaml`, the O3, O5 and O8
    reports, the runner, the O2/O3/O5/O6/O8/O2C/O2D modules, the census
    staleness function, and learner, lifetime and search code, plus the commit;
  - `run.log`, `status.json` and `exit.json`; a detached launch that runs the
    gates and the run in one process; no commits mid-run;
  - memory precondition 2.0 GiB, under the PI's 2026-10-03 instruction that low
    memory is acceptable, recorded in `precondition.json`;
  - the independent scorer is committed before the sealed band is opened;
  - logs are archived to `reports/o9_*`.
