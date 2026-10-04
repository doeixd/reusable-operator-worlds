# O10: does consolidation alone repair the near-misses that in-stream re-routing leaves? (Tier 1, exploratory)

Status: FROZEN 2026-10-04, before any O10 cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It runs on O9's 45 saved `REROUTE_WAKE` terminals
(worlds 930-944). Those worlds were opened by O9's sealed run, so everything
here is DESIGN evidence. Worlds 945-959 stay sealed and are not touched. No new
stream lifetimes run.

# Why

O9 (sealed) found that wake plus in-stream re-routing of earlier tasks, with no
consolidation, removes every collapse but leaves 8 of 45 cells as near-misses
(0.051-0.232, at most one task of 64 at or above 1.0). O9's registered branch
for `NOT_CONFIRMED` with near-misses says the remedy is consolidation. The
question is whether the fully online protocol needs anything at the END of the
stream beyond consolidation: if O3's sleep alone, on the in-stream terminal,
repairs the near-misses without breaking the passing cells, then the protocol
is in-stream re-routing + consolidation, with no end-of-stream re-route.

# Arm, as a construction

| arm | construction |
|---|---|
| `RW` (reference, not re-run) | O9's committed `REROUTE_WAKE` cell. |
| `RW_SLEEP` (new) | O9's saved `REROUTE_WAKE` terminal (`artifacts/o9_sealed_online/work/REROUTE_WAKE_w{w}_s{s}/lifetime/model.pt`), loaded with `o3.load_shuffled_terminal` (the in-stream learner is the same planned-depth learner on the same `SHUFFLED` stream; only routes were swapped), then `o3.run_sleep` verbatim: `o2d.reservoir` (64 per task, seed `[1940, w, s, i]`), `o2c.consolidate` with 8,192 updates of 2, sampling `[1941, w, s, 64]`. No route search. |

`RW_SLEEP` is O3's/O6's `SLEEP` arm applied to a different terminal; gate E2
checks the sleep call is unchanged.

# Cells

All 45 O9 `REROUTE_WAKE` terminals:
- **Target set `F`, 8 cells:** the failing ones: 931 s0, 931 s2, 932 s2,
  937 s1, 937 s2, 938 s1, 938 s2, 944 s2.
- **Harm set `H`, 37 cells:** the passing ones.

# Rule (registered)

`r` = cells of `F` whose `RW_SLEEP` terminal median is `< 0.05`, denominator 8.
`b` = cells of `H` whose `RW_SLEEP` terminal median is `>= 0.05` or non-finite,
denominator 37. A non-finite value never passes.

| condition (first matching row) | label |
|---|---|
| `b >= 3` | `HARMS` |
| `r >= 7` | `REPAIRS` |
| `4 <= r <= 6` | `PARTIAL` |
| `r <= 3` | `NO_REPAIR` |

The labels partition every outcome. Also reported: per-cell `RW_SLEEP / RW`
ratio, `RW_SLEEP / REROUTE_SLEEP` ratio against O9's end-of-stream protocol on
the same stream, the number of `RW_SLEEP` cells below 0.05 over all 45, and the
terminal stale-route count after sleep (the census function; sleep trains
through routes, so this checks that consolidation does not re-create staleness).

# Necessity

- **Target behaviour:** repair of the in-stream learner's residual near-misses
  by consolidation, without an end-of-stream route search.
- **Refusal arm:** `RW` itself, no consolidation: 8 of 45 cells above the
  registered threshold.
- **Measured refusal cost and its scale:** the 8 cells sit at 0.051-0.232;
  O9's end-of-stream re-route + sleep on the same streams sits at 0.009-0.014.
  The scale is the registered 0.05 threshold.
- **Impostors:**
  - route repair in disguise: `RW_SLEEP` performs no route search; terminal
    staleness after `RW` is 6 routes in total across 45 cells;
  - extra data: the reservoir is the same 64-per-task memory as every O-line arm;
  - selection regression: `F` was chosen from its outcome, but the comparison
    is the same terminal before and after sleep, not a re-draw.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. `RW` on `F` sits at 0.051-0.232, inside it.

# Discriminating power

**The decision rule, as it will be applied:** `REPAIRS` if `r >= 7` of 8 and
`b <= 2` of 37; `HARMS` if `b >= 3`.

Source `reports/o10_design/rates.py`, output `rates_output.txt`, exact
binomials.

| per-cell rescue probability | `REPAIRS` (`r >= 7`) fires |
|---|---|
| 0.1, sleep's rate on collapses (null) | **false-fire 0.0000** |
| 0.3 | 0.0013 |
| 0.5 | 0.0352 |
| 0.8 | 0.503 |
| 0.9, sleep's near-miss repair rate in O4 (21/23) and O2E (12/12) (effect) | **detection 0.813** |
| 0.95 | 0.943 |

Harm clause: with a per-cell break probability of 0.01 `HARMS` fires 0.6% of
the time, at 0.02 3.8%, at 0.05 28%, at 0.1 73%.

**What the checks do not cover:**
- Only 8 target cells; a true repair rate of 0.7-0.8 usually lands in
  `PARTIAL`.
- The worlds were opened by O9; this is design evidence, and a sealed test of
  the combined protocol needs worlds 945-959 under a new frozen plan.
- One sleep budget (8,192 updates) and one memory size (64 per task).
- Depth 3 only.

# Gates (before any cell)

- **G0 (reload):** for every target cell and three harm cells, the reloaded
  `RW` terminal reproduces O9's recorded per-task terminal exactly.
- **E2 (sleep call unchanged):** `o3.run_sleep` on O3's saved
  `SHUFFLED_w20_s0` terminal reproduces O3's committed `SLEEP_w20_s0` bitwise.

# What it decides (for planning)

- **`REPAIRS`:** the fully online protocol is in-stream re-routing +
  consolidation, with no end-of-stream route search. Next: a sealed
  confirmation of exactly that protocol on worlds 945-959 under a new plan;
  then the depth rung, where in-stream re-routing needs the length-linear
  re-router.
- **`PARTIAL` or `NO_REPAIR`:** the in-stream learner's near-misses are not the
  near-misses sleep repairs; examine the failing cells before any sealed test.
- **`HARMS`:** consolidation re-creates a failure the in-stream learner had
  avoided; diagnose before anything else.

# Operational

45 cells of about 2-3 minutes each on a pool of 3: about 40 minutes. Durable
stamped cells, protocol fingerprint (this plan, the O9 report and its runner,
O3/O2C/O2D, the lifetime module), `run.log`, `status.json`, `exit.json`;
detached launch; independent scorer committed before launch; reserve 2.0 GiB
(PI instruction 2026-10-03); logs archived to `reports/o10_*`.
