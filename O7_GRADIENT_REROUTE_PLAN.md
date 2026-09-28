# O7: does a length-linear gradient re-route, followed by sleep, do what exhaustive re-routing did? (Tier 1, exploratory)

Status: FROZEN 2026-09-28, before any O7 cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It runs on saved order-free (`SHUFFLED`) terminals only:
O2 (development worlds 13-19), O3 (development worlds 20-26) and O4 (worlds
900-914). The O4 worlds are sealed-band worlds, and are already contaminated
for the re-route protocol by O5, so they enter as DESIGN evidence exactly as in
O5. There are no new worlds and no new stream lifetimes.

# Why

O6 (sealed) confirmed wake, then EXHAUSTIVE re-route, then sleep: 45 of 45
cells. Exhaustive search costs `12^d` route evaluations, so the claim is
limited to depth <= 3. The O7 Tier 0 censuses (`960fbb1`, descriptive, on the
saved development terminals) asked which re-router could replace it:
- **Coordinate descent from the stale route fails:** it recovers 17% of stale
  depth-3 routes. Stale routes need 2-3 positions changed together.
- **A slot map derived from the anchors fails:** staleness is many-to-one and
  position-specific.
- **A fresh-relaxation gradient re-route recovers 76% of stale routes.** This is
  SO1R's `optimize_route`: the code is reset to uniform, Adam 0.05,
  T 1.0 -> 0.1. Its cost is linear in depth. The "safe" variant is within ~10%
  of exhaustive query quality in 12 of 14 cells, but one collapse lags badly
  (0.54 against 0.38).

Routes alone are not the endpoint. O5 showed that sleep after re-routing is what
reaches the threshold. O7 asks whether gradient re-routing followed by sleep
reaches the same place as exhaustive re-routing followed by sleep.

# Arms, as constructions

Each starts from the saved `SHUFFLED` terminal (`o3.load_shuffled_terminal`),
with O5's reload gate G0.

| arm | construction |
|---|---|
| `RS` (reference, not re-run) | The committed O5 cell: exhaustive re-route of all 188 stream tasks, then O3's sleep verbatim. |
| `SLEEP` (reference, not re-run) | The committed sleep cell: O2D `RES64`, O3 `SLEEP` or O4 `SLEEP`. |
| `GRS` | For EVERY stream task of depth `d`, with the library frozen and the task's 64 reservoir examples (`o2d.reservoir`, O5's pool). **Depth 1:** exhaustive over the 12 slots, which is itself linear. **Depth >= 2:** `optimize_route(library, x, y, 500)` from SO1R, verbatim. It is deterministic (zero-initialised code, full-batch Adam), so it needs no seed. Then SAFE: keep the gradient route only if its support MSE is strictly below the current route's. Install it with O5's minimal logit swap. Then O3's sleep verbatim: `o2c.consolidate`, the same reservoir, 8,192 updates, sampling `[1941, w, s, 64]`. |

`GRS` differs from `RS` only in how the new route is chosen. Gate E1 checks
that the pipeline around the choice is O5's: a `GRS` whose chooser is replaced
by exhaustive search must reproduce O5's `O3_w20_s0` cell bitwise.

# Cells

- **Target set `F`, 10 cells.** Every saved order-free cell where the terminal
  collapsed (`>= 1.0`) or sleep alone failed (`>= 0.05`):
  - O2 w14 s0;
  - O3 w22 s1;
  - O4 901 s0, 907 s2, 908 s1, 909 s2, 911 s0, 911 s2, 914 s1, 914 s2.

  Exhaustive `RS` rescued 10 of 10, and `SLEEP` 1 of 10.
- **Harm set `H`, 12 cells.** Development cells that sleep passes:
  - O2 w13, w15, w16, w17, w18, w19, all stream 1;
  - O3 w20, w21, w23, w24, w25, w26, all stream 0.

  `RS` passed 12 of 12.

# Rule (registered)

`r` = cells of `F` whose `GRS` terminal median is `< 0.05`, denominator 10.
`b` = cells of `H` whose `GRS` terminal median is `>= 0.05` or non-finite,
denominator 12. A non-finite value never passes.

| condition | label |
|---|---|
| `b >= 2` | `HARMS` |
| `r >= 8`, `b <= 1` | `MATCHES_EXHAUSTIVE` |
| `4 <= r <= 7`, `b <= 1` | `PARTIAL` |
| `r <= 3`, `b <= 1` | `NO_RESCUE` |

The labels partition every outcome. Also reported:
- per-cell `GRS / RS` terminal ratio;
- stale routes that `GRS` chose differently from exhaustive search;
- routes changed per cell;
- seconds for re-routing against exhaustive search.

# Necessity

- **Target behaviour:** repair of online libraries by a re-router whose cost is
  linear in program length.
- **Refusal arm:** `SLEEP`, the same memory and updates with no re-routing. It
  rescues 1 of the 10 target cells.
- **Measured refusal cost and its scale:** 9 of 10 target cells stay above the
  registered 0.05 threshold without re-routing. Exhaustive re-routing brings all
  10 to 0.008-0.016. The scale is the registered threshold.
- **Impostors:**
  - extra optimisation: the sleep budget is identical, and route search trains
    nothing shared;
  - exhaustive search, i.e. a cost that is not linear: that is the reference
    arm, not the tested one.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)`. `RS` sits at 0.008-0.016, and `SLEEP` on `F` sits at
  0.047-1.42.

# Discriminating power

**The decision rule, as it will be applied:** `MATCHES_EXHAUSTIVE` if
`r >= 8` of 10 and `b <= 1` of 12.

Source `reports/o7_design/rates.py`, output `rates_output.txt`, exact
binomials.

| per-cell rescue probability | `MATCHES_EXHAUSTIVE` (`r >= 8`) fires |
|---|---|
| 0.1, i.e. `SLEEP`'s observed rate on `F` (null) | **false-fire 0.0000** |
| 0.2 (harder null) | false-fire 0.0001 |
| 0.5 | 0.055 |
| 0.8 | 0.678 |
| 0.9 (effect) | **detection 0.930** |
| 0.95 | 0.989 |

Harm clause: with a per-cell break probability of 0.02, `HARMS` fires 2.3% of
the time; at 0.1, it fires 34%.

**What the checks do not cover:**
- Only 10 target cells, so a true rate near 0.5-0.8 lands in `PARTIAL`.
- The target set was chosen from its outcome. That is the intended target, not
  a regression artefact: both reference arms start from the same terminals.
- Re-using the O4 sealed-band worlds makes this design evidence only.
- Linear cost is shown at depth 3 only. Depth >= 4 is not tested here.
- The gradient chooser has a single deterministic initialisation, so its
  sensitivity to initialisation is unmeasured.

# What it decides (for planning)

- **`MATCHES_EXHAUSTIVE`:** gradient re-routing is admitted as the scalable
  re-router. The next rung is a depth-4+ stream design, where exhaustive search
  is infeasible, then a registered development test, then a sealed test on a
  new band that the PI must approve.
- **`PARTIAL` or `NO_RESCUE`:** gradient re-routing is not yet a substitute.
  Next, examine which cells fail, as in the w22 s1 census lag. Candidates are
  more steps, multiple restarts, or re-routing during the stream.
- **`HARMS`:** SAFE is not safe after sleep. Diagnose before anything else.

# Operational

22 cells at about 6-9 minutes each (re-routing plus sleep), on a pool of 3:
about 1 h.
- durable stamped cells, a protocol fingerprint, `run.log` and `status.json`;
- a 4.5 GiB memory precondition (the PI's 2026-09-27 instruction), recorded
  in `precondition.json`;
- G0 and E1 run before any cell;
- the independent scorer is committed before launch;
- logs are archived to `reports/o7_*`.
