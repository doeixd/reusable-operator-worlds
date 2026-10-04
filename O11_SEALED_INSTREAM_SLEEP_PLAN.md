# O11: sealed confirmation that wake + in-stream re-routing + consolidation forms the rotated substrate reliably online

Status: FROZEN 2026-10-04, before any world in 945-959 was generated. The PI
approved decision 16 ("Ok continue", 2026-10-04). **SEALED: worlds
945-959**, the last 15 sealed worlds of band 930-959; after O11 the band is
exhausted. The Tier 0 attribution sizing (`reports/o11_attribution_sizing.json`,
on O9's opened worlds) was run before freezing; its numbers are below.

O9 (sealed, 930-944) showed in-stream re-routing alone removes every collapse
but leaves near-misses (37/45, `NOT_CONFIRMED`; `ATTRIBUTED` 38/45 against
wake alone). O10 (exploratory, O9's opened worlds) showed O3's sleep on those
in-stream terminals repairs all 8 near-misses and breaks nothing (45/45,
`REPAIRS`). O11 tests the combined protocol on worlds nobody has seen.

# The claims

1. **Reliability (primary):** wake + in-stream re-routing of earlier tasks +
   O3's sleep forms the rotated substrate in >= 90% of (world, stream) cells,
   with NO end-of-stream route search.
2. **Attribution (secondary):** in-stream re-routing contributes beyond
   consolidation: the combined protocol ends strictly below wake + sleep
   (no re-routing) in more paired cells than chance allows.

Named assumptions: single-operation anchors in the stream; 64 retained
examples per task; depth <= 3 (exhaustive search after every arrival).

# Arms, as constructions

Model seed 5000; O2-O9 stream recipes verbatim on worlds 945-959.

| arm | construction | cells |
|---|---|---|
| `REROUTE_WAKE` | O9's / O8's `run_cell('REROUTE_WAKE', ...)` verbatim | 45 |
| `RW_SLEEP` (primary) | O10's construction on this run's own `REROUTE_WAKE` terminal: `o3.run_sleep` verbatim (64 per task, 8,192 updates, sampling `[1941, w, s, 64]`), no route search | 45 |
| `SHUFFLED` | `o3.run_cell('SHUFFLED', ...)`, O4/O6/O9 | 45 |
| `SLEEP` (attribution reference) | `o3.run_cell('SLEEP', ...)` on this run's `SHUFFLED` terminal, O4/O6 | 45 |
| `PLAIN` (floor) | `o3.run_cell('PLAIN', ...)`, stream 0 | 15 |

`RW_SLEEP` and `SLEEP` share the sleep call, memory and budget; they differ in
whether routes were kept current during the stream.

# Gates (development worlds, before any sealed cell)

- **E1:** `SHUFFLED` reproduces O3's `SHUFFLED_w20_s0` bitwise.
- **E2:** `REROUTE_WAKE` reproduces O8's `REROUTE_WAKE_w20_s0` bitwise (and
  its total routes changed).
- **E3:** `SLEEP` on the E1 terminal reproduces O3's `SLEEP_w20_s0` bitwise.
- **E3b:** `RW_SLEEP` applied to O9's saved `REROUTE_WAKE_w930_s0` terminal
  (an opened world, not one of 945-959) reproduces O10's committed `w930_s0`
  cell bitwise. A gate comparing the sleep call with itself would be vacuous;
  this one compares against a committed result and can fail.
- **E4b:** after the freeze, every sealed stream interleaves depths in its
  first 20 tasks.
- **E5:** a scale-16 dry run of every arm on development world 27, interrupted
  after one cell and relaunched; the cell must be reused.
- **E4 (scorer):** per sealed world, the three `SHUFFLED` libraries differ;
  each `RW_SLEEP` and `SLEEP` cell starts from this run's own parent terminal
  (library hash before sleep equals the parent's terminal hash).

# Rules (registered)

**Primary.** `k` = `RW_SLEEP` cells `< 0.05`, denominator 45; a non-finite
value never passes. `FLOOR_FAILED` if any `PLAIN` cell passes; else
`CONFIRMED` if `k >= 41`; else `NOT_CONFIRMED`.

**Secondary.** `n_better` = cells where `RW_SLEEP` is strictly below
`SLEEP`, denominator 45; ties and non-finite count as not better.
`ATTRIBUTED` if `n_better >= 30`, else `NOT_ATTRIBUTED`.

**Sizing (Tier 0, measured before freezing, O9's opened worlds 930-944):**
`o3.run_sleep` on O9's 45 saved `SHUFFLED` terminals against O10's committed
`RW_SLEEP` cells on the same streams. `RW_SLEEP` was strictly below `SLEEP`
in **45 of 45** cells (41 of 41 among cells sleep alone passes), at a median
per-cell ratio of **0.41x**; it rescued 4 of the 4 cells sleep alone failed.
The per-cell rate is far above 0.8, so the clause has room on all 45 cells and
is kept as drafted.

**The sizing also shows why the attribution clause is necessary.** Sleep
alone passed **41 of 45** on those worlds, exactly the primary bar. So
`CONFIRMED` alone cannot separate the combined protocol from sleep alone; the
paired clause can (sizing 45/45 against a 0.5 null). Read the two labels
together: `CONFIRMED` without `ATTRIBUTED` is not evidence for in-stream
re-routing.

Descriptive (computed in the summary or stored per cell): `k` for every arm;
collapses per arm; `RW_SLEEP` rescues among cells `SLEEP` fails; the median
per-cell `RW_SLEEP / SLEEP` ratio; per `REROUTE_WAKE` cell, routes changed,
terminal stale routes and re-route seconds. Staleness after sleep is not
recorded (O10 measured it: 14 routes over 45 cells).

# Necessity

- **Target behaviour:** reliable, fully online formation with no
  end-of-stream route search.
- **Refusal arm:** `SLEEP`, the same memory and consolidation without in-stream
  re-routing: 38/45 (O4) and 36/45 (O6) on sealed bands, with collapses it
  cannot repair.
- **Measured refusal cost and its scale:** 7-9 of 45 sealed cells above the
  registered 0.05 threshold without re-routing; the combined protocol passed
  45/45 on O9's worlds (O10). Scale: the threshold.
- **Impostors:** extra compute (re-routing trains nothing; sleep budget equal
  across arms); end-of-stream search in disguise (none runs); world luck (15
  fresh worlds x 3 streams); tuning on test worlds (designed on 13-26 and
  900-944).
- **Difficulty band, higher-is-harder:** terminal median NMSE `(0.0005, 0.5)`;
  O10's cells at 0.004-0.014, `PLAIN` ~1.9.

# Discriminating power

**The decision rules, as they will be applied:** `CONFIRMED` if `k >= 41` of
45 and no `PLAIN` pass; `ATTRIBUTED` if `n_better >= 30` of 45.

Primary: `reports/o11_design/rates.py` (O9's rule, sampler and seed, 20,000
draws): null 0.80 false-fire 0.020-0.039; effect 0.952 detection 0.78-0.94;
effect 0.98 detection 0.985-0.998. Secondary:
`reports/o11_design/attribution.py`: null 0.5 false-fire 0.017-0.037;
effect 0.8 detection 0.92-0.99.

**What the checks do not cover:** the primary bar does not discriminate the
combined protocol from sleep alone on bands where sleep alone does well (41/45
on O9's worlds; 36-38/45 on O4/O6), which is why the labels are read together;
the attribution sizing and the O10 design evidence come from worlds the
protocol was assembled on and are optimistic; depth 3 only; this exhausts band
930-959.

# What it decides

- **`CONFIRMED`:** the online-formation line's fully online result:
  prevention (in-stream re-routing) + consolidation, no batch route search.
  Next: the depth rung, where in-stream re-routing needs the length-linear
  re-router (O7) from depth 4, on a new development band.
  With `ATTRIBUTED`, in-stream re-routing is confirmed to contribute beyond
  consolidation.
- **`CONFIRMED` with `NOT_ATTRIBUTED`:** reliability holds, but the band does
  not show in-stream re-routing adding to sleep; the claim reduces to
  consolidation.
- **`NOT_CONFIRMED`:** the per-cell values say which mode remains.
- **`FLOOR_FAILED`:** uninterpretable.

# Operational

195 cells (45 `REROUTE_WAKE` ~13 min, 45 `RW_SLEEP` ~2.5 min, 45 `SHUFFLED`
~12 min, 45 `SLEEP` ~2.5 min, 15 `PLAIN` ~3 min) on a pool of 3: about 8
hours. One detached orchestrator (dry run, gates, checks, run); durable
stamped cells; fingerprint over this plan, `configs/v1.yaml`, the O3, O8
and O10 reports, the runner and the O2/O3/O8/O2C/O2D/census/lifetime/learner
modules, plus the commit; reserve 2.0 GiB (PI 2026-10-03); scorer committed before the band is
opened; no commits mid-run; logs archived to `reports/o11_*`.
