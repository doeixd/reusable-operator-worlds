# O6: sealed confirmation that wake, then re-route, then sleep forms the rotated substrate reliably online

Status: FROZEN 2026-09-27, before any world in 915-929 was
generated. Decision 13 (sealed band 900-929, and thresholds) was delegated by
the PI to Claude. **SEALED: worlds 915-929**, the untouched half of that band.
Worlds 900-914 were consumed by O4 and then used as DESIGN evidence by O5, so
they are contaminated for this protocol and are not re-used. After O6 the
band 900-929 is exhausted. Any further sealed test needs a new band approved
by the PI.

This is the online-formation line's second confirmatory rung. O4 confirmed
nothing (38/45 against 41): consolidation repaired near-misses, but not
COLLAPSE. The Tier 0 census and O5 (Tier 1, exploratory) then found that a
collapse is mostly STALE ROUTES on a usable library, and that re-inferring every
task's route by search on its retained examples before consolidating passed
87/87 saved terminals. That includes all 8 collapses and all 45 O4 cells, and
it improved the terminal over sleep alone in 87/87 cells. O6 tests that
protocol on worlds nobody has seen.

# The claims

1. **Reliability (primary, registered):** an online learner forms the rotated
   substrate reliably, in >= 90% of (world, stream) cells. The learner receives
   an order-free task stream containing single-operation anchors, retains 64
   examples per task, and at the end of the stream re-infers each task's route
   by exhaustive search on those examples and then spends 8,192 consolidation
   updates on them.
2. **Attribution (secondary, registered):** the re-routing step, not
   consolidation alone, is responsible. Across paired cells, re-route + sleep
   ends below sleep alone more often than chance allows.

Named assumptions, not results:
- single-operation tasks are present in the stream;
- the per-task memory holds 64 examples;
- routes are short enough to enumerate (depth <= 3, at most 1,728 routes).

One substrate family, one learner, one model seed, one budget.

# Arms, as constructions

Model seed 5000. Stream `s` in {0, 1, 2}: replay seed `None` at 0, else
`SeedSequence([7500, w, s])`; order seed `[1920, w]` at 0, else
`[1920, w, s]`. These are O2-O4's recipes, verbatim, applied to worlds 915-929.

| arm | construction | cells |
|---|---|---|
| `SHUFFLED` (wake) | `o3.run_cell('SHUFFLED', ...)`, i.e. `o2.run_single`, LEAN; exactly O4's | 45 |
| `REROUTE_SLEEP` (primary) | From the same cell's saved `SHUFFLED` terminal (`o3.load_shuffled_terminal`), O5's `run_cell` construction verbatim: `o2d.reservoir` (64 per task, seeded `[1940, w, s, i]`); then `o5.reroute`, where for every one of the 188 stream tasks of depth `d` and with the library frozen, the route is the argmin over all `12^d` routes of support MSE on its 64 reservoir examples, installed by a minimal logit swap; then `o2c.consolidate`, 8,192 updates of 2, sampling `[1941, w, s, 64]` | 45 |
| `SLEEP` (reference for claim 2) | `o3.run_cell('SLEEP', ...)`: the same terminal, then `o3.run_sleep` (same reservoir, same 8,192 updates, same sampling seed); exactly O4's | 45 |
| `PLAIN` (floor) | `o3.run_cell('PLAIN', ...)`, no anchors, stream 0 | 15 |

`REROUTE_SLEEP` and `SLEEP` differ ONLY in the route swap. Gate E3b checks
this, and it can fail. Every arm is scored on the TERMINAL model over the 64
canonical length-3 tasks.

# Gates (before any sealed cell; on DEVELOPMENT worlds only, except E4b)

- **E1:** O6's `SHUFFLED` reproduces O3's committed `SHUFFLED_w20_s0`
  bitwise (library sha and per-task terminal).
- **E2:** O6's `SLEEP`, applied to that terminal, reproduces O3's committed
  `SLEEP_w20_s0` bitwise.
- **E3:** O6's `REROUTE_SLEEP`, applied to that terminal, reproduces O5's
  committed cell `O3_w20_s0` bitwise (library sha, per-task terminal, and
  routes changed).
- **E3b (non-vacuous construction check, replacing O5's vacuous G1):**
  `REROUTE_SLEEP` with the swap disabled, applied to that terminal,
  reproduces O3's `SLEEP_w20_s0` bitwise. So the only difference between the
  two arms is the swap, and this check fails if anything else differs.
- **E4b:** after the freeze commit, every sealed stream interleaves depths
  within its first 20 tasks.
- **E5:** a scale-16 dry run on DEVELOPMENT world 27, covering every arm, is
  interrupted after one cell and relaunched, and must reuse that cell bitwise.
- **E4:** per sealed world, the three `SHUFFLED` terminal libraries differ.
  The scorer checks this.

# Rules (registered)

**Primary.** `k_RS` = `REROUTE_SLEEP` cells with terminal median `< 0.05`.
**Denominator: all 45 cells, always.** A non-finite terminal counts as not
passing. A crashed cell is rerun, never dropped.

| condition | label |
|---|---|
| any `PLAIN` cell passes | `FLOOR_FAILED` |
| otherwise, `k_RS >= 41` | `CONFIRMED` |
| otherwise, `k_RS <= 40` | `NOT_CONFIRMED` |

**Secondary (attribution).** `n_better` = cells where the `REROUTE_SLEEP`
terminal median is STRICTLY below `SLEEP`'s. **Denominator: all 45 paired
cells.** A tie, or a non-finite `REROUTE_SLEEP` terminal, counts as not
better.

| condition | label |
|---|---|
| `n_better >= 30` | `ATTRIBUTED` |
| `n_better <= 29` | `NOT_ATTRIBUTED` |

Each table partitions every outcome. The two labels are reported together, and
neither changes the other.

Descriptive only, with no rule attached:
- `k_SLEEP` and `k_SHUFFLED`;
- collapse counts (`>= 1.0`) per arm, and how many `SHUFFLED` collapses
  `REROUTE_SLEEP` rescues;
- routes changed per cell;
- the paired per-cell `log10(M_RS / M_SLEEP)`.

# Necessity

- **Target behaviour:** reliable online formation, including recovery from
  collapse.
- **Refusal arm:** `SLEEP`, the same memory and updates without re-routing.
  On sealed O4 it passed 38/45 and rescued 1 of 6 collapses.
- **Measured refusal cost and its scale:** without re-routing, 7 of 45 sealed
  cells stayed above the registered 0.05 threshold, 5 of them collapses. In
  O5, re-routing moved those collapses from 0.05-1.4 to 0.008-0.016 and
  lowered the terminal in 87/87 cells (median 2.5x). The scale is the
  registered threshold.
- **Impostors:**
  - extra optimisation compute: the sleep budget is identical, and search is a
    one-off inference that trains nothing;
  - re-routing alone, without sleep: 0 of 8 collapses passed in the census;
  - world luck: 15 fresh sealed worlds x 3 streams;
  - tuning on test worlds: the protocol was designed on 13-26 and 900-914,
    none of which is in the sealed set.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. O5's `REROUTE_SLEEP` cells lie at 0.005-0.03, inside it,
  and `PLAIN` sits above it (~1.9).

# Discriminating power

**The decision rules, as they will be applied:** `CONFIRMED` if `k_RS >= 41`
of 45 and no `PLAIN` cell passes; `ATTRIBUTED` if `n_better >= 30` of 45.

**Primary: null and effect samplers.** Source `reports/o6_design/rates.py`,
output `rates_output.txt`, 20,000 draws, seed 12, O4's sampler. Each world
draws a logit offset `N(0, sd)`, and the streams pass independently given it.

| true per-cell rate | `CONFIRMED` fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.43, no consolidation | 0.000 / 0.000 / 0.000 |
| 0.80, partially reliable (null) | **false-fire 0.038 / 0.031 / 0.022** |
| 0.844, re-routing adds nothing over sealed `SLEEP` | 0.152 / 0.114 / 0.066 |
| 0.90 | 0.528 / 0.454 / 0.290 |
| 0.95 | 0.927 / 0.890 / 0.761 |
| 0.98, near O5's 87/87 (effect) | **detection 0.998 / 0.997 / 0.982** |

**Secondary: null and effect samplers.** Source
`reports/o6_design/attribution.py`, output `attribution_output.txt`, 20,000
draws, seed 13, with the same clustering. The independent-cell exact
threshold would be 29, but at sd 1 it false-fires 6.2%, so 30 is registered.

| per-cell probability RS < SLEEP | `ATTRIBUTED` fires, sd 0 / 0.5 / 1 |
|---|---|
| 0.5, re-routing adds nothing (null) | **false-fire 0.018 / 0.025 / 0.038** |
| 0.7 | 0.751 / 0.689 / 0.572 |
| 0.8 | 0.991 / 0.978 / 0.920 |
| 0.95, near O5's 87/87 (effect) | **detection 1.000 / 1.000 / 1.000** |

**What the checks do not cover:**
- **Middle ground on the primary.** If re-routing adds nothing and the true
  rate is sealed `SLEEP`'s 0.844, `CONFIRMED` still fires 7-15% of the time.
  That is a reliability claim at a rate between the null (0.80) and the
  target (0.90). Claim 2 exists to catch it: under that null the attribution
  rule fires at most 3.8%.
- **The O5 design evidence is optimistic.** The 87/87 is on the worlds the
  protocol was designed after, so the effect row is a best case. At a true
  rate of 0.90, `CONFIRMED` fires only 29-53%.
- **Scaling.** Exhaustive route search is feasible at depth 3 only. Nothing
  here speaks to longer programs.
- **The comparison is SLEEP, not INTERLEAVED.** Interleaved consolidation had
  0 collapses in O3 but is not an arm here.

# What it decides

- **`CONFIRMED`:** the online-formation line's first confirmatory positive.
  Online, order-free formation of the rotated substrate is reliable with
  retained memory, end-of-stream route re-inference and consolidation, within
  the named assumptions. It goes to the paper as confirmatory. With
  `ATTRIBUTED`, stale routes are confirmed as the residual failure that sleep
  alone leaves.
- **`NOT_CONFIRMED`:** the O5 repair does not generalise at the registered
  bar. The per-cell values say whether the shortfall is residual collapse
  (library damage beyond routes) or near-misses.
- **`FLOOR_FAILED`:** the substrate is formable without anchors on these
  worlds, and the comparison is uninterpretable.

# Operational

- **Cells:** 150 (45 `SHUFFLED` + 45 `REROUTE_SLEEP` + 45 `SLEEP` + 15
  `PLAIN`). When a `SHUFFLED` cell completes, its `REROUTE_SLEEP` cell is
  submitted before its `SLEEP` cell. Pool of 3, about 8 h.
- **Operational contract:**
  - durable stamped cells, and relaunch resumes;
  - a fingerprint over this plan, `configs/v1.yaml`, the O3 and O5 reports,
    the runner, the O2/O3/O5/O2C/O2D modules, and learner, lifetime and
    search code, plus the commit;
  - `run.log`, `status.json` and `exit.json`, an 8 GiB precondition never
    lowered, a detached launch, and no commits mid-run;
  - the independent scorer is committed before the sealed band is opened;
  - logs are archived to `reports/o6_*`.
