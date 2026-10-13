# SC0: how many single-operation anchors does the confirmed online learner need? (Tier 1, exploratory)

Status: FROZEN 2026-10-12, before any SC0 cell ran. **Tier 1, EXPLORATORY: no verdict.** Development band 5 (seeds
50-69, allocated 2026-10-10 for the control-flow track, which paused the same day after the B3 census): world 64,
streams 0-4; dry-run world 66 (a dry-run world, as for B2-depth2). World 64 has never been generated (verified: no
runner, report or artifact references it). Using a band-5 world for Track S is disclosed here; it is development
evidence only.

# Why

Track S (`SUCCESSOR_LADDER.md`, issue #4) asks whether an adaptive task sampler (SC1) can find the curriculum that
makes the vocabulary form, against uniform sampling and the hand-designed anchor supply. Whether SC1 has anything to
win depends on one number the record does not have: the anchor dose the CONFIRMED online baseline needs.
- Offline (no sleep, no re-route; N1b/N1c): 6 length-1 anchors fail, 18 pass in 2 of 3 worlds, 32 pass.
- Online, the confirmed construction (wake + end-of-stream exhaustive re-route + sleep; O6, sealed 45/45) has only
  been run at 60 anchors, and with none (O9 `PLAIN` 0/15).

A sampler's effect on the anchor count is arithmetic. Over a 188-task stream drawn uniformly over PROGRAMS of length
1..D (6 operations), the expected number of length-1 tasks is `188 x 6 / sum 6^k`: **4.37 at D = 3**, 0.73 at D = 4,
0.12 at D = 5. Drawn uniformly over LENGTHS it is `188 / D`: 62.7, 47.0, 37.6. So:
- if the baseline forms with about 4 anchors, uniform-over-programs at depth 3 already suffices and an adaptive
  sampler has no terminal-endpoint opportunity there (only beyond depth 3, against uniform-over-programs);
- if it needs 8-16, uniform-over-programs fails at every depth while uniform-over-lengths supplies 37-63, so the
  sampler's only opportunity is against the program-uniform baseline, and it amounts to raising one mixture weight.

The SC0 census as first written (marginal learning value per task class per stage of saved lifetimes) cannot be run
read-only: the O-line lifetimes were deleted on 2026-10-04 and the surviving D1/D2 cells record no per-arrival probe
of other tasks. The dose is the quantity that decides SC1's opportunity, so SC0 measures it instead.

# Stream and protocol (constructions)

`row.experiments.sc0_anchor_dose` (runner), `build_dose_stream(n, w, s)`:
- O2's SHUFFLED stream for (w, s): 60 length-1, 64 length-2 and the 64 canonical length-3 tasks in one random order
  (`o2.build_stream('SHUFFLED', w, s)`, the stream O6 confirmed).
- All but `n` of the 60 length-1 tasks removed; everything else and its order unchanged. The kept anchors are the
  first `n` of a permutation of the 60 (sorted by task id) seeded `[6100, w, s]`, so doses are NESTED within a stream.
  Dose 60 is O2's stream exactly (unit test).
- Protocol, O6's `REROUTE_SLEEP` construction: lifetime on the stream (`o2.lifetime`, O2's planned-depth learner, the
  stream's replay seed); a reservoir of 64 examples per task (`o2d.reservoir`); exhaustive re-route of every task
  (`o5.reroute`); `o2c.consolidate` for 8,192 updates seeded `[1941, w, s, 64]`. Program length is given as the
  planned depth, as in O6. Scored on the canonical 64: terminal median query NMSE.

| cells | dose `n` | streams |
|---|---|---|
| decision | 4, 8, 16 | 0-4 (15 cells) |
| confirmed-construction control | 60 | 0 |
| floor control | 0 | 0 |

17 cells. Dose 4 is the expected count under uniform-over-programs at depth 3; 16 sits just under the offline
threshold region (18 passed 2/3), 8 between.

# Rules (registered)

Per cell: the terminal canonical median; passing means `< 0.05`. `k_n` = passing cells at dose `n`, of 5.
Controls: `top` = the dose-60 cell passes; `floor` = the dose-0 cell fails.

| condition (first matching row) | label |
|---|---|
| not (`top` and `floor`) | `UNINTERPRETABLE` |
| `k_4 = 5` | `NO_OPPORTUNITY` |
| `k_4 <= 1` and `k_16 >= 4` | `OPPORTUNITY` |
| otherwise | `GRADED` |

**What each label decides (stated before data):**
- `NO_OPPORTUNITY`: about four anchors suffice under the confirmed baseline; an adaptive sampler has no terminal
  opportunity at depth 3 against either uniform baseline. SC1 would need a different endpoint (cumulative cost, time to
  formation) or depth >= 4, where uniform-over-programs supplies under one anchor, and then only against that baseline.
- `OPPORTUNITY`: the baseline needs a quota between 4 and 16; uniform-over-programs fails at every depth, so SC1 has
  a measurable terminal opportunity against it, and none against uniform-over-lengths (>= 37 anchors). The interesting
  SC1 question is then whether a label-free sampler finds the quota without being given program length.
- `GRADED`: read the per-stream curve; no SC1 plan until the threshold is located.

**Reported, not registered:** per dose the wake, re-route-only and terminal canonical medians; anchors' own errors;
routes changed; operations covered by the kept anchors; per-stream monotonicity in dose; seconds per cell.

**Non-vacuity (fails the cell):** stream length `128 + n`, depth histogram `{1: n, 2: 64, 3: 64}`, `n` anchor ids,
reservoir `64 x (128 + n)`, 8,192 sleep updates, 64 canonical tasks scored, and sleep changed the library.

# Necessity

- **Target behaviour:** vocabulary formation online at a reduced anchor supply.
- **Refusal arm:** the dose-0 floor (no anchors; O9's `PLAIN` was 0/15 with no length-2 tasks either).
- **Measured refusal cost and its scale:** no anchors 0/15 at ~1.9 terminal median against 45/45 at ~0.0105 with 60
  (O9, O6 sealed): a ~180x gap; the scale for every dose is that interval and the 0.05 bar.
- **Impostors:** length-2 tasks alone (N1b offline: 64 do nothing; here they are present at every dose, so the floor
  measures them); stream length (doses change the stream from 128 to 144 tasks; N1's matched-pool sham showed extra
  tasks do not explain formation); coverage of operations (N1c refuted it offline; recorded per cell).
- **Difficulty band, higher-is-harder:** terminal canonical median, band `(0.005, 2.5)`: the confirmed construction
  sits near 0.0105, the floor near 1.9.

# Discriminating power

**The decision rules, as they will be applied:** `NO_OPPORTUNITY` iff both controls behave and `k_4 = 5`;
`OPPORTUNITY` iff both controls behave, `k_4 <= 1` and `k_16 >= 4`.

Source `reports/sc0_anchor_dose_design/rates.py`, output `rates_output.txt` (exact binomials, streams independent):

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `NO_OPPORTUNITY` (`k_4 = 5`) | per-cell 0.5 at dose 4 | **0.031** | per-cell 0.98 | **0.904** |
| `OPPORTUNITY` | no dose effect, worst p (0.5) | **0.035** | p_4 0.1, p_16 0.95 | **0.898** |
| controls both behave | | | per-cell 0.98 each | 0.960 |

**What the checks do not cover:**
- `NO_OPPORTUNITY` fires at a dose-4 rate of 0.7 with probability 0.17 and at 0.9 with 0.59: it is a sizing on one
  world, not a verdict.
- `OPPORTUNITY` detection falls to 0.68 at (0.2, 0.9) and 0.39 at (0.3, 0.8); a softer threshold lands in `GRADED`.
- The doses are varied inside O2's frame: 64 length-2 tasks stay at every dose, whereas uniform-over-programs at depth
  3 would carry ~26 length-2 and ~157 length-3 tasks. Fewer length-2 and more length-3 tasks could raise the need.
- Depth 3 only; D1/D2's depth-4 and depth-5 lifetimes ran at 60 anchors and their dose is not measured here.
- One world, five streams; streams share the world and its 64 canonical tasks, so they are not independent worlds.
- The terminal endpoint only; an adaptive sampler might still lower cumulative cost or time to formation.

# Operational

17 cells on a pool of 3, the dose-60 control first, then doses 16, 8, 4 (largest streams first), the floor last.
Restartable, durable stamped cells, protocol fingerprint (this plan, config, the runner, lifetime, O2, O2C, O2D, O3,
O5 modules, commit); `run.log`, `status.json`, `exit.json`, `precondition.json`; detached launch; scale-16 dry run of
doses 60, 4 and 0 on world 66 with a restart test; reserve 2.0 GiB (PI 2026-10-03). Independent scorer
`row.experiments.score_sc0_anchor_dose` recomputes the label from per-task records and must agree with the runner.
Tests `tests/test_sc0_anchor_dose.py` (labels, nested doses and dose 60 equal to O2's stream, scorer agreement,
disagreement on a tampered summary, refusal on a changed plan). Expected ~25 min per cell, ~2.5 h.
