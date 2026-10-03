# O8: does re-deriving earlier tasks' routes DURING the stream prevent what end-of-stream re-routing repairs? (Tier 1, exploratory)

Status: FROZEN 2026-10-03, before any O8 cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It runs new stream lifetimes on O3's development worlds
20-26 (band 3), three streams each, paired cell for cell with O3's committed
`SHUFFLED` and `INTERLEAVED` cells. No sealed world. Those worlds have already
carried O3, O5 and O7, so everything here is DESIGN evidence and must be
confirmed on a fresh band before any claim.

# Why

O6 (sealed) confirmed wake, then exhaustive re-route of every task, then sleep:
45 of 45. The repair is a batch step after the stream. The O8 Tier 0 census
(`reports/o8_staleness_position_census.json`, 63 development terminals) found
that staleness has a time structure:
- 33.0% of tasks that arrived in the first third of the stream have stale
  routes at the terminal, against 9.2% and 10.3% in the second and last thirds;
  the shape holds in every cell;
- depth-1 anchors are essentially never stale (1.1%); depth-3 tasks are (37.3%);
- in-stream consolidation (`INTERLEAVED`) reduces staleness only modestly
  (median 24 against 30.5 per cell), because it trains through routes rather
  than re-deriving them.

So stale routes are mostly routes committed while the library was immature:
J1's early-commitment lock-in seen from the population side. That suggests a
learner that re-derives EARLIER tasks' routes against its CURRENT library as it
goes, instead of once at the end. If that works, the learner has no terminal
batch step and the formation claim becomes fully online. If it does not, the
reason is informative either way: routing alone cannot repair convergence (O2C),
or early re-routing thrashes (J1).

The census also corrected a premise. One end-of-stream exhaustive pass costs
1.1 s at depth 3 and a projected ~47 s at depth 4 (measured 15.9x per depth), so
O6's protocol is feasible through depth 5. IN-STREAM re-routing multiplies the
cost by the number of arrivals (about 188 x 94 route searches per cell), which
is ~2 minutes at depth 3 and ~1.6 h at depth 4. Which protocol the scaling claim
is about depends on O8's outcome, which is why O8 precedes any depth rung.

# Arms, as constructions

All four arms use O2's `SHUFFLED` stream construction verbatim
(`o2.build_stream('SHUFFLED', w, s)`: 60 depth-1, 64 depth-2 and 64 canonical
depth-3 tasks in the stream-seeded order; `planned_model`; the stream's replay
seed; `LEAN` diagnostics; 65,536 updates). The learner is the planned-depth
rotated discrete learner, `rotated_discrete_fast`.

| arm | construction |
|---|---|
| `SHUFFLED` (reference, not re-run) | O3's committed cell: the stream, nothing else. 8 of 21 pass. |
| `INTERLEAVED` (reference, not re-run) | O3's committed cell: the stream plus the `Interleaver` hook, which after each task adds that task's 64 reservoir examples (seed `[1940, w, s, i]`) to a pool and runs `n_t` consolidation updates on the pool, 8,192 in total. 17 of 21 pass. |
| `REROUTE_WAKE` (new) | `SHUFFLED` plus a `Rerouter` hook. After task `t` has trained and its summary row is written, for every EARLIER stream task `i < t`, with the library frozen for the duration of the search: exhaustive search over `12^d_i` hard routes on task `i`'s 64 reservoir examples (the same `[1940, w, s, i]` draw), then O5's minimal logit swap into task `i`'s code if the argmin differs from the recorded hard route. Task `t` itself is never re-routed at step `t`. No consolidation updates. The hook performs no random draws. |
| `REROUTE_INTERLEAVED` (new) | `INTERLEAVED` with the `Rerouter` applied inside the hook BEFORE that task's `n_t` consolidation updates, same pool, same 8,192 updates, same sampling generator `[1950, w, s]`. |

`REROUTE_WAKE` differs from `SHUFFLED` only by the hook; `REROUTE_INTERLEAVED`
differs from `INTERLEAVED` only by the re-route call. Gate E1 checks the first
claim bitwise on a real cell; gate E2 checks that the re-route call is a
switch that recovers `INTERLEAVED` exactly when disabled.

Why exhaustive and not O7's gradient re-router: at depth 3 exhaustive search is
the cheaper and exact chooser (0.021 s per depth-3 task), so the in-stream
question is asked with the chooser that cannot be blamed. The gradient chooser
is the scaling candidate for a later depth.

Why only earlier tasks: a task's own route at arrival was just inferred by
training, and re-deriving it in the same step would break the last-task anchor
(terminal == end-of-task on the last stream task), which `REROUTE_WAKE` must
satisfy exactly because nothing trains after its last task.

# Cells

21 paired cells: worlds 20-26, streams 0-2, for each of the two new arms, 42
new lifetimes. `REROUTE_WAKE` cells are submitted first (they decide rule A),
then `REROUTE_INTERLEAVED`. No early stop is registered; the arms are
independent and both complete.

# Rules (registered)

Terminal median NMSE over the 64 canonical depth-3 tasks; a cell passes if it
is finite and `< 0.05`. A non-finite value never passes and never counts as
"better".

**Rule A, `REROUTE_WAKE` against `SHUFFLED` (8 of 21):**
- `k` = passing `REROUTE_WAKE` cells, denominator 21;
- `n_better` = cells whose `REROUTE_WAKE` terminal median is strictly below the
  committed `SHUFFLED` terminal median, denominator 21;
- `h` = cells that `SHUFFLED` passes and `REROUTE_WAKE` fails, denominator 8.

| condition (first matching row) | label |
|---|---|
| `h >= 3` | `HARMS` |
| `k >= 14` and `n_better >= 16` | `PREVENTS` |
| `k >= 14` or `n_better >= 16` | `PARTIAL` |
| otherwise | `NO_EFFECT` |

**Rule B, `REROUTE_INTERLEAVED` against `INTERLEAVED` (17 of 21):**
- `f` = failing `REROUTE_INTERLEAVED` cells, denominator 21;
- `n_better_B` = cells strictly below the committed `INTERLEAVED` terminal
  median, denominator 21;
- `h_B` = cells that `INTERLEAVED` passes and `REROUTE_INTERLEAVED` fails,
  denominator 17.

| condition (first matching row) | label |
|---|---|
| `h_B >= 3` | `HARMS_B` |
| `f <= 1` and `n_better_B >= 16` | `REACHES_CEILING` |
| `n_better_B >= 16` | `IMPROVES` |
| otherwise | `NO_EFFECT_B` |

Both tables partition every outcome. Also reported, per cell:
- routes changed per re-route pass (a list of 187 counts) and their total;
- terminal staleness, measured by the census function on the finished model:
  stale count, by tercile and by depth (prediction OB1: fewer than 10 per cell
  for the re-routed arms, against the committed medians 30.5 and 24);
- end-of-task median over the canonical tasks and the first-16 canonical
  end-of-task median (online quality, as O3 records them);
- cumulative prequential Gaussian log loss;
- seconds spent in re-routing and total;
- the last-task anchor error (`REROUTE_WAKE`: must be 0 within O3's tolerance);
- library hash before and after.

# Necessity

- **Target behaviour:** an online learner keeps its task-to-part assignments
  current while its parts are still forming, so that no end-of-stream repair is
  needed.
- **Refusal arm:** `SHUFFLED`, the same stream with routes left as committed.
  It passes 8 of 21, and its terminals carry a median 30.5 stale routes.
- **Measured refusal cost and its scale:** 13 of 21 cells fail the 0.05
  threshold without re-routing; end-of-stream re-route + sleep (O5) passes 21
  of 21 on these same cells at 0.008-0.016. The scale is the registered
  threshold.
- **Impostors:**
  - extra optimisation: `REROUTE_WAKE` adds no updates; route search trains
    nothing; `REROUTE_INTERLEAVED` keeps `INTERLEAVED`'s 8,192 updates exactly;
  - extra data: the reservoir holds examples the learner has already trained on,
    the same 64-per-task memory assumption O3, O5 and O6 name; nothing unseen
    enters;
  - end-of-stream repair in disguise: the hook never runs after the last task's
    row for that task, and the last-task anchor is enforced on `REROUTE_WAKE`.
- **Difficulty band, higher-is-harder:** terminal median NMSE, band
  `(0.0005, 0.5)` for the pass/fail reading. `SHUFFLED` sits at 0.024-2.04 and
  `INTERLEAVED` at 0.013-0.84 on these cells, so the band contains cells on
  both sides of the threshold and neither arm is at a ceiling or floor
  everywhere.

# Discriminating power

**The decision rules, as they will be applied:** rule A labels `PREVENTS` if
`k >= 14` of 21 and `n_better >= 16` of 21, `HARMS` if `h >= 3` of 8; rule B
labels `REACHES_CEILING` if `f <= 1` of 21 and `n_better_B >= 16` of 21,
`HARMS_B` if `h_B >= 3` of 17.

Source `reports/o8_design/rates.py`, output `rates_output.txt`, exact binomials.

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| A: `k >= 14` | per-cell pass 0.381 (`SHUFFLED`'s 8/21) | **0.0075** | 0.86 (sleep level) | **0.9945** |
| A, B: `n_better >= 16` | sign 0.5 | **0.0133** | 0.9 | **0.9856** |
| A: `h >= 3` of 8 | break rate 0.05 | 0.0058 | 0.5 | 0.8555 |
| B: `f <= 1` | fail rate 0.19 (`INTERLEAVED`'s 4/21) | 0.0709 | 0.02 | 0.9347 |
| B: `h_B >= 3` of 17 | break rate 0.05 | 0.0503 | 0.5 | 0.9988 |

Joint `PREVENTS` under the null is below either clause's rate. At an
intermediate effect (pass 0.7, better 0.8) the clauses fire 0.72 and 0.77, so a
genuine but partial effect lands in `PARTIAL` rather than being missed.

**What the checks do not cover:**
- 21 cells on 7 worlds that already carried O3, O5 and O7: design evidence.
- Rule B's ceiling clause has little room: `INTERLEAVED` fails only 4 cells,
  which is why the paired sign test, not the pass count, carries rule B.
- The paired comparisons share the committed arms' streams and replay seeds;
  the new arms' hooks add no random draws, so the pairing is exact, but a
  single stream per cell means the sign pattern across cells is the evidence,
  not any one cell.
- Re-routing after EVERY task is one schedule; sparser schedules are untested.
- The logit swap moves the code entries but not the lifetime optimizer's Adam
  moment estimates for those entries (O5 swapped on a finished model and then
  built a fresh optimizer, so this is new here). The moments decay within a
  few dozen updates; the effect is part of the construction being tested and
  is disclosed, not controlled.
- Depth 3 only.

# Gates and dry run (before any cell)

- **E1 (switch recovers the baseline, real cell):** `REROUTE_WAKE` with the
  hook's re-route disabled reproduces O3's committed `SHUFFLED_w20_s0` bitwise
  (library hash and per-task terminal).
- **E2 (switch recovers `INTERLEAVED`, dry scale):** at `DRY_SCALE` on world
  20 stream 0, `REROUTE_INTERLEAVED` with re-route disabled matches
  `run_interleaved` bitwise, and with re-route enabled changes at least one
  route and the library hash.
- **E3 (the hook moves routes, dry scale):** `REROUTE_WAKE` at `DRY_SCALE`
  reports a positive total of routes changed and a library hash different from
  the base.
- **Dry run:** one cell of EACH new arm at `DRY_SCALE`, validated with the real
  run's `validate`, then the restart test: interrupt after one cell, relaunch,
  verify the completed cell is reused and the rest run.

# What it decides (for planning)

- **`PREVENTS`:** staleness is a cause of wake failure, not only a symptom, and
  prevention replaces repair for most of it. Next: a sealed confirmation of the
  fully online protocol on a NEW band (PI decision), and the scaling rung is
  about in-stream cost, where a linear re-router is needed from depth 4.
- **`PARTIAL` or `NO_EFFECT` with `REACHES_CEILING`:** routing and convergence
  are separate failure modes and both must be handled in-stream; the fully
  online protocol is `REROUTE_INTERLEAVED`.
- **`NO_EFFECT` and `NO_EFFECT_B`:** staleness is a symptom; end-of-stream
  repair stands (O6), and the depth rung is about one-pass cost, which binds
  only near depth 6-7.
- **`HARMS` or `HARMS_B`:** early re-routing thrashes (J1). Next, a schedule
  that re-routes only after a warm-up or only tasks older than `m` arrivals.

# Operational

42 new lifetimes at about 20-30 minutes each (the wake lifetime; re-routing
adds ~2 minutes at depth 3), on a pool of 3: about 6 h.
- durable stamped cells, a protocol fingerprint that includes this plan, the
  stream runner, the lifetime module and the committed O3 report;
  `run.log`, `status.json`, `exit.json`, `precondition.json`;
- a 4.5 GiB memory precondition, as O7 registered under the PI's 2026-09-27
  instruction, recorded in `precondition.json`;
- gates E1-E3 and the dry run with restart test before launch;
- the independent scorer is committed before launch;
- no commits while the run is in flight (the fingerprint includes the commit);
- logs are archived to `reports/o8_instream_reroute_<date>/`.
