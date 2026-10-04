# O11: sealed confirmation that wake + in-stream re-routing + consolidation forms the rotated substrate reliably online

Status: **DRAFT, NOT FROZEN (2026-10-04).** Opening worlds 945-959, the last
15 sealed worlds of band 930-959, waits for the PI (decision 16). Before
freezing: run the Tier 0 attribution sizing below and fill in its numbers.

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

- E1 `SHUFFLED` = O3 `SHUFFLED_w20_s0` bitwise; E2 `REROUTE_WAKE` = O8
  `REROUTE_WAKE_w20_s0` bitwise; E3 `SLEEP` = O3 `SLEEP_w20_s0` bitwise;
  E3b `RW_SLEEP` on E2's terminal = the same call made directly (bitwise);
  E4b sealed streams interleave depths in their first 20 tasks (after the
  freeze); E5 scale-16 dry run of every arm on world 27 with a restart test.

# Rules (to register)

**Primary.** `k` = `RW_SLEEP` cells `< 0.05`, denominator 45; a non-finite
value never passes. `FLOOR_FAILED` if any `PLAIN` cell passes; else
`CONFIRMED` if `k >= 41`; else `NOT_CONFIRMED`.

**Secondary.** `n_better` = cells where `RW_SLEEP` is strictly below
`SLEEP`, denominator 45; ties and non-finite count as not better.
`ATTRIBUTED` if `n_better >= 30`, else `NOT_ATTRIBUTED`. **Open design point:**
`SLEEP` passes 36-38 of 45 on sealed bands, so this clause has room only if
in-stream re-routing also lowers the terminal of cells sleep already passes.
Composition suggests it does (end-of-stream re-route + sleep beat sleep in
44/45 on O6; `RW_SLEEP` is at 0.90x of that protocol on O9's worlds), but that
is an argument, not a measurement. **Tier 0 sizing before freezing:** run
`o3.run_sleep` on O9's 45 saved `SHUFFLED` terminals and count cells where
O10's `RW_SLEEP` is below it. If that rate is under ~0.8, re-design the clause
on the cells the effect can reach (rescues among `SLEEP` failures), per the
O2C learning.

Descriptive: `k` for every arm; collapses per arm; `RW_SLEEP` against
`REROUTE_WAKE`; stale routes after sleep; cost of in-stream re-routing.

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

**What the checks do not cover:** the attribution effect size is unmeasured
(see the open design point); the O10 design evidence is optimistic (45/45 on
worlds the protocol was assembled on); depth 3 only; this exhausts band
930-959.

# What it decides

- **`CONFIRMED`:** the online-formation line's fully online result:
  prevention (in-stream re-routing) + consolidation, no batch route search.
  Next: the depth rung, where in-stream re-routing needs the length-linear
  re-router (O7) from depth 4, on a new development band.
- **`NOT_CONFIRMED`:** the per-cell values say which mode remains.
- **`FLOOR_FAILED`:** uninterpretable.

# Operational

195 cells (45 `REROUTE_WAKE` ~13 min, 45 `RW_SLEEP` ~2.5 min, 45 `SHUFFLED`
~12 min, 45 `SLEEP` ~2.5 min, 15 `PLAIN` ~3 min) on a pool of 3: about 8
hours. One detached orchestrator (dry run, gates, checks, run); durable
stamped cells; fingerprint over this plan, the O3/O5/O8/O9/O10 reports and
modules; reserve 2.0 GiB (PI 2026-10-03); scorer committed before the band is
opened; no commits mid-run; logs archived to `reports/o11_*`.
