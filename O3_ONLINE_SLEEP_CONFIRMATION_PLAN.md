# O3: does online order-free anchor supply plus a sleep phase form the substrate reliably on unseen worlds, and is the phase structure what matters? (Tier 2, development)

Status: FROZEN 2026-09-26, re-frozen the same day BEFORE ANY CELL RAN. The
first freeze `0707e95` omitted two items `tools/check_adequacy.py` requires: the
measured refusal cost and its scale, and a restatement of the decision rule in
`# Discriminating power`. Both were added. No threshold, arm or seed changed.
(PI decision 12 allocated development band 3, seeds
20-29; the PI directed Claude to continue with its judgement). Worlds **20-26**;
27-29 are held back. Development evidence, never confirmatory. Drafted earlier
the same day; this version supersedes that draft. Its candidate rollback-and-
retry note is carried forward to "What it decides".

# Why

- **O2 (registered):** no online protocol is reliable. Order-free `SHUFFLED`
  passes 9 of 21 cells, mostly near-misses.
- **O2C-O2E (exploratory, on O2's own models):** a post-stream sleep phase over
  retained examples repairs every near-miss at 64 retained examples per task,
  giving `SHUFFLED` 20/21, and breaks nothing.

Two things were never tested:
1. whether that holds on worlds its settings were not chosen on;
2. whether it is the PHASE STRUCTURE (consolidate after the stream) or just the
   extra updates on retained memory.

O3 tests both, and nothing else. `STAGED` and rollback-and-retry are left out on
purpose: order-free plus sleep already reached 20/21, collapses are rare in it
(1/21), and each extra arm would dilute the run.

# Arms, as constructions

Model seed 5000. Stream `s` in {0, 1, 2} sets the replay seed (`None` at 0,
else `SeedSequence([7500, w, s])`) and the order seed (`[1920, w]` at 0, else
`[1920, w, s]`), all as O2 (`o2_online_reliability.build_stream`,
`replay_seed_for`). Every arm is scored on the TERMINAL model over the 64
canonical length-3 tasks.

| arm | construction | cells |
|---|---|---|
| `SHUFFLED` | O2's single online lifetime, verbatim (`o2.run_single`, LEAN) | 21 |
| `SLEEP` | the SAME cell's saved `SHUFFLED` terminal, then O2D's `RES64` sleep verbatim: `o2c.consolidate`, 8,192 updates of 2, on `o2d.reservoir` (64 examples per stream task, `SeedSequence([1940, w, s, i])`), sampling `[1941, w, s, 64]`, temperature at its final value | 21 |
| `INTERLEAVED` | a separate `SHUFFLED` lifetime with the SAME stream, seeds, learner and LEAN config, plus a per-task hook (`learned_lifetime`'s `prospective_hook`, unmodified runner). After each completed stream task `t`, its 64 reservoir examples join the pool and `n_t` consolidation updates of 2 run on the pool so far, with `n_t = round(8192(t+1)/188) - round(8192 t/188)`, so exactly **8,192 in total**. It uses one persistent AdamW (library plus codes of completed tasks, the LRs of `consolidate`), sampling `SeedSequence([1950, w, s])`, and the routing temperature where the lifetime left it. | 21 |
| `PLAIN` | the 64 canonical tasks, one fresh lifetime, stream 0 (floor) | 7 |

`SLEEP` and `INTERLEAVED` get the SAME retained memory (the same reservoir
examples) and the SAME number of extra updates. They differ only in WHEN the
updates happen: after the stream, or spread through it. Both spend +34% over
the stream's 24,064 updates, and that cost is charged to any claim.

# Gates (before any scored cell)

- **E1:** the `SHUFFLED` construction reproduces O2's committed
  `SHUFFLED_w13_s0` cell bitwise (library sha and per-task terminal).
- **E2:** the `SLEEP` construction, applied to O2's `SHUFFLED_w13_s0` terminal,
  reproduces O2D's committed `RES64_w13_s0` cell bitwise.
- **E3:** the `INTERLEAVED` hook with all `n_t = 0` reproduces the plain
  `SHUFFLED` lifetime bitwise (scale 16, world 20, stream 0). With the
  registered schedule, it changes the library (non-vacuity), and it performs
  exactly 8,192 updates over the stream.
- **E4:** per world, the three streams' `SHUFFLED` terminal libraries differ
  (checked by the scorer on the real cells). **E4b:** every stream interleaves
  depths within its first 20 tasks (checked before launch).
- **E5:** a scale-16 dry run interrupted after one cell and relaunched reuses
  the cell bitwise.

# Estimands and rules (registered)

Denominator: 21 cells for every three-stream arm, always. A non-finite
terminal counts as not passing.

**Primary, `SLEEP`:** `k_SLEEP` = cells with terminal median `< 0.05`.

| `k_SLEEP` | label |
|---|---|
| `>= 19` | `WAKE_SLEEP_RELIABLE` |
| `17-18` | `WAKE_SLEEP_INTERMEDIATE` |
| `<= 16` | `WAKE_SLEEP_UNRELIABLE` |

**Secondary, the phase contrast:** per cell,
`r = log10(M_SLEEP / M_INTERLEAVED)`, paired within the cell (same world,
stream, order and initialisation).

| condition | label |
|---|---|
| `>= 16` of 21 cells `r < 0`, AND `>= 6` of 7 worlds with `>= 2` of 3 streams `r < 0`, AND `median(r) <= -0.15` | `PHASE_MATTERS` (sleep better) |
| the mirror image | `INTERLEAVED_BETTER` |
| `7 <= #(r < 0) <= 14` AND `abs(median r) <= 0.10` | `EQUIVALENT` |
| otherwise | `INDETERMINATE` |

A non-finite terminal enters the contrast as `r = +inf` when it is in
`SLEEP`, `r = -inf` when it is in `INTERLEAVED`, and `r = 0` when both are
non-finite. The median is taken over these values.

`INTERLEAVED`'s own `k` gets the primary's partition, reported as a secondary
label. Descriptive only, with no rule attached:
- `k_SHUFFLED`, and the sleep effect `log10(M_SLEEP / M_SHUFFLED)` per cell;
- the collapse count (terminal `>= 1.0`) per arm;
- the first-16 canonical end-of-task median.

**Floor clause:** if any `PLAIN` cell passes, every label is `FLOOR_FAILED`.

# Necessity

- **Target behaviour:** online formation made reliable by consolidation.
- **Refusal arm:** `SHUFFLED`, the same lifetimes without sleep. Its expected
  rate from O2 is 9/21, about 0.43.
- **Measured refusal cost and its scale:** on O2's worlds, going without sleep
  cost 11 of 21 cells: 9/21 without sleep against 20/21 with `RES64` (O2D).
  In terminal error, the 8 near-misses sat at 1-4x the 0.05 threshold before
  sleep and at 0.45-0.93x it after. The scale is the registered 0.05 threshold,
  not output variance.
- **Impostors:**
  - extra compute and retained memory: `INTERLEAVED` matches both;
  - world luck: 7 fresh worlds x 3 streams;
  - the sleep setting having been tuned on O2's worlds: these worlds are
    unseen.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. O2's order-free cells lie in it apart from one collapse, and
  `PLAIN` sits above it.

# Discriminating power

**The decision rules, as they will be applied:**
- primary: `k_SLEEP >= 19` of 21 is `RELIABLE` (with `17-18` and `<= 16` as
  above);
- contrast: the four-way rule on `r` above.

**Primary** (`reports/o3_design/rates.py`, output `rates_output.txt`). Each
world draws a logit offset `N(0, sd)`, and the three streams pass independently
given it. 20,000 draws, seed 11.

| true per-cell rate of `SLEEP` | `RELIABLE` fires, sd 0 / 1 |
|---|---|
| 0.43, sleep does not transfer (null) | **0.000 / 0.000** |
| 0.80, partial transfer | 0.179 / 0.123 |
| 0.95, as on O2's worlds (effect) | **0.913 / 0.797** |

**Contrast** (`contrast_rates.py`, output `contrast_rates_output.txt`).
`r = u_world + e_cell + delta`, 20,000 draws, seed 11.

| world sd / cell sd | null `delta = 0`: `PHASE` / `INTER_BETTER` fire | `delta = -0.3` (2x) | `delta = -0.5` (3x) |
|---|---|---|---|
| 0.0 / 0.3 | **0.005 / 0.005** | 0.859 | **0.999** |
| 0.2 / 0.3 | 0.018 / 0.019 | 0.657 | 0.967 |
| 0.4 / 0.3 | 0.039 / 0.039 | 0.393 | 0.746 |
| 0.2 / 0.6 | 0.015 / 0.017 | 0.283 | 0.661 |

**What the checks do not cover:**
- The contrast reliably detects only a LARGE phase effect, about 3x in terminal
  error. A 2x effect is often `INDETERMINATE`, and `INDETERMINATE` is not
  equivalence.
- A pass-count contrast was considered and rejected: at moderate rates it
  false-fires 16-22%.
- One model seed and one budget.
- `INTERLEAVED` is one of many ways to spend the same updates online. It
  matches memory and compute, not every conceivable online schedule.

# What it decides

| primary | contrast | reading and next step |
|---|---|---|
| `RELIABLE` | `PHASE_MATTERS` | Online formation made reliable by a WAKE-then-SLEEP learner, and the phase structure is what matters (development). Candidate for a sealed confirmation (PI decision). |
| `RELIABLE` | `EQUIVALENT` / `INDETERMINATE` | Reliable, but the evidence does not separate sleep from more updates on retained memory. The claim is "retained memory plus consolidation compute", not "sleep". |
| `RELIABLE` | `INTERLEAVED_BETTER` | Reliable, and consolidating during the stream is better. The successor is an interleaved design. |
| `INTERMEDIATE` / `UNRELIABLE` | any | The O2C-O2E rescue was partly specific to O2's worlds or settings. Next: the collapse and rollback-and-retry line on worlds 27-29. |

# Operational

- **Cells:** 70 (21 `SHUFFLED` + 21 `SLEEP` + 21 `INTERLEAVED` + 7 `PLAIN`).
  Each `SLEEP` cell is submitted when its `SHUFFLED` cell completes.
- **Estimated cost:** ~6 h on a pool of 3, from the measured costs: ~19.5 min
  per `SHUFFLED` lifetime, ~5.5 min per sleep, ~25 min per `INTERLEAVED`
  lifetime, ~6 min per `PLAIN`.
- **Operational contract:**
  - durable stamped cells, and relaunch resumes;
  - a fingerprint over the plan, config, O2/O2D reports, runner, the O2, O2C
    and O2D modules, learner and lifetime code, and the commit;
  - `run.log`, `status.json` and `exit.json`, an 8 GiB precondition never
    lowered, a detached launch, and no commits mid-run;
  - the independent scorer is committed before launch;
  - logs are archived to `reports/o3_*`.
