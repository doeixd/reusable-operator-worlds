# B2-online v2: with the halting predicate reading the detached state, does a learner form the vocabulary AND learn data-dependent iteration online? (Tier 1, exploratory)

Status: FROZEN 2026-10-10, before any B2-online v2 cell ran. **Tier 1, EXPLORATORY: no verdict.** Development band 5 (seeds 50-69,
allocated 2026-10-10, verified unused): worlds 50, 51, 52, streams 0-2; dry-run
world 69. These worlds have never been generated.

# Why

B2-online v1 (`reports/b2_online.json`, worlds 44-46) labelled
`FORMATION_BROKEN`: LOOP's terminal canonical median passed in 4/9 cells (0.065;
controls 0.013-0.016), although wake learned state-dependent halting (`n_wake` 9/9)
and LOOP's vocabulary was no worse after wake. The localization census
(`reports/b2o_localize_census.json`, Tier 0 on the 9 saved wake models, gates G0
and G1 9/9) rebuilt the end of stream five ways: evaluating the halting predicate
on the DETACHED state during sleep restored formation 9/9 (0.014) and improved the
loops (0.264 against 0.327); freezing the predicate's parameters did not (4/9). The
damage is gradient through the state-reading predicate into the shared library.

That census was post hoc and on reused cells. This rung builds the fix into the
learner (wake and sleep), runs it on fresh worlds, and keeps v1's construction as a
paired arm, so the mechanism is tested prospectively.

# Stream and learner (constructions)

- **Stream:** `row.experiments.loop_stream`, unchanged from v1: O2's SHUFFLED
  depth-3 stream (60 length-1, 64 length-2, 64 canonical length-3 tasks) plus 48
  loop tasks (`z = x; repeat at most 6 times: stop if w.z <= 0, else z = P(z)`, P
  one teacher primitive, w a random unit hyperplane), one random order over 236
  tasks. 128 training examples per task. Loop identity GIVEN (planned depth 1).
- **Learner:** `row.models.loop_gated.LoopLearner` with the new switch
  `detach_halt_input`. When True the predicate reads `z.detach()`: its gradient
  trains `halt_w`, `halt_b` (unchanged) but never reaches the library or the body
  through the states. Evaluation is identical. Default False is v1 bitwise.
  Test (`tests/test_loop_gated.py::test_detach_halt_input`): same evaluation output,
  a different library gradient, an unchanged halting gradient.
- **End of stream:** v1's construction, unchanged and imported from
  `row.experiments.b2_online`: `deep_reroute.reroute` of the 188 straight-line
  tasks; support-only loop re-fit on the 64 retained examples per task
  (`refit_loops`, seed `[5820, w, i]`, installed only if it beats the current
  hardened support MSE); O3 sleep (8,192 updates, sampling `[1941, w, s, 64]`,
  64-per-task reservoir of all 236 tasks) on the library, all task codes and the
  halting parameters.

# Arms

| arm | construction |
|---|---|
| `LOOP` | `LoopLearner(state_halt=True, detach_halt_input=True)`; wake on the stream; v1's end of stream (re-fit as v1's LOOP) |
| `LOOP_ATTACHED` | v1's `LOOP` exactly: `LoopLearner(state_halt=True, detach_halt_input=False)` (the mechanism control) |
| `CONSTCOUNT` | v1's `CONSTCOUNT` exactly: `state_halt=False`, re-fit restricted to the best (s, L) (capacity control; its halting cannot read the state, so detaching is moot) |

`REFUSAL` is not re-run: its refusal cost is measured (census 1.02, v1 0.90) and
`CONSTCOUNT` carries the formation control.

# Rules (registered)

Per cell: canonical median (64 straight-line tasks) and loop median (48 loop
tasks), query NMSE with hard routes and hard halting, after wake (`wake_`) and on
the terminal.

- `k_formation` = `LOOP` cells with terminal canonical median `< 0.05`, of 9.
- `n_wake` = cells where `LOOP`'s wake loop median is `< 0.5 x` `CONSTCOUNT`'s wake
  loop median (paired by world and stream), of 9.
- `n_terminal` = `LOOP` cells with terminal loop median `< LOOP_BAR`, of 9, with
  `LOOP_BAR = 0.38` (see "The loop bar").
- `n_mechanism` = cells where `LOOP`'s terminal canonical median is below
  `LOOP_ATTACHED`'s (paired), of 9.
- `n_control` (construction check, reported) = cells where `LOOP`'s terminal loop
  median is below `CONSTCOUNT`'s.

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `n_terminal >= 8` and `n_wake >= 8` | `LOOPS_ONLINE` |
| `n_terminal >= 8` | `LOOPS_AFTER_SLEEP` |
| otherwise | `NO_LOOPS` |

Mechanism label, reported beside it: `DETACH_PROTECTS` if `n_mechanism >= 8`,
else `DETACH_NOT_SHOWN`.

**Which clauses carry the question (stated before data).**
- `k_formation` and `n_mechanism` test the fix. The census makes both LIKELY but
  not guaranteed: it detached only in sleep, on reused worlds, post hoc, and here
  wake also changes (detached halting may learn halting differently in wake, and
  the library forms under different gradients). If `LOOP_ATTACHED` happens to form
  on these worlds, `n_mechanism` can fail while `k_formation` passes.
- `n_wake` is the online-learning clause, as in v1. Detaching removes one gradient
  path from wake, so wake halting could get WORSE; v1 passed 9/9 with the path.
- `n_terminal` carries the absolute question: is the loop learned to the bar the
  64-example re-fit can reach? Given the baseline check below it is near-guaranteed when formation holds, so it is not decisive; reported against the 64-support loop form (0.214) as well.
- `n_control` is near-guaranteed (v1 9/9) and is a construction check only.

# The loop bar

`LOOP_BAR = 0.38`, set by v1's rule before any v2 data: v1's bar 0.25 was the
geometric midpoint between the loop form with a support-learned predicate (0.126 at
128 support) and the best impostor (0.49), sqrt(0.126 x 0.49) = 0.248. The
end-of-stream re-fit sees the 64 RETAINED examples per task, not 128, so the bar is
re-derived at 64 support (`reports/b2_support64_census.json`, `census_b2_support64`,
the B2 census unchanged on the same exactly rebuilt final B1-online libraries, gates
G0/G1 3/3 and the 128/512 rows reproduced exactly): loop form **0.214** (p10-p90
0.137-0.323), input-gated impostor **0.675**, one route 1.026, true predicate 0.041;
sqrt(0.214 x 0.675) = 0.380.

This also accounts for v1's terminal loops: the localization's DETACH_HALT reached
0.264 against a 64-support ceiling of 0.214, not the 128-support 0.126 the v1 plan
compared with.

**Baseline check of the bar.** v1's `LOOP` loops (0.28-0.37) and the census's
DETACH_HALT (0.23-0.28) both sit below 0.38, so `n_terminal` is close to guaranteed
whenever formation holds. It separates the loop form from the impostors, not
attached from detached halting. It is NOT the clause that carries the question.

# Also reported

Canonical passes and medians for all arms; loop medians and loop tasks below 0.05;
count accuracy (fraction of query inputs with the teacher's iteration count) after
wake and on the terminal for `LOOP` and `LOOP_ATTACHED`; re-fits installed; routes
changed; wake halting norms.
**Non-vacuity (fails the cell):** the median norm of each loop task's halting
parameters after wake must be above zero in every arm (wake trained them), and
sleep must change the library.

# Necessity

- **Target behaviour:** online learning of data-dependent iteration (a body and a
  state-dependent halting rule) while forming the vocabulary.
- **Refusal arm:** none re-run; refusal is measured: one route per task 1.02 (census),
  0.90 (v1 `REFUSAL`, terminal loop median).
- **Measured refusal cost and its scale:** census: one route 1.02, loop form with a
  support-learned predicate 0.126 at 128 support (0.214 at 64), true predicate
  0.041; the scale is the gap between the loop median and `LOOP_BAR`, which lies
  between the loop form and the best impostor (0.214 and 0.675 at 64 support).
- **Impostors:** extra capacity and compute (`CONSTCOUNT`: same body, iteration and
  sleep, without state access); deciding the count from the input (census impostor,
  not an arm); the predicate leaking (only the teacher uses w).
- **Difficulty band, higher-is-harder:** loop median NMSE, band `(0.0005, 0.5)`.
  Expected: `LOOP` terminal near the census's 64-support value (0.214-0.26); `LOOP_ATTACHED` near
  v1's 0.33; `CONSTCOUNT` near 1.0.

# Discriminating power

**The decision rules, as they will be applied:** `LOOPS_ONLINE` if `n_terminal >= 8`
of 9, `n_wake >= 8` of 9 and `k_formation >= 7` of 9; `LOOPS_AFTER_SLEEP` if
`n_terminal >= 8` and `n_wake <= 7`; `FORMATION_BROKEN` if `k_formation <= 6`;
`DETACH_PROTECTS` if `n_mechanism >= 8` of 9.

Source `reports/b2_online_design/rates.py` (v1), output `rates_output.txt`, exact
binomials; the same per-cell rules and counts apply.

| per-cell clause probability | `>= 8 of 9` fires |
|---|---|
| 0.5 (null) | **false-fire 0.0195** |
| 0.7 | 0.196 |
| 0.9 | 0.775 |
| 0.95 (effect) | **detection 0.929** |

Formation guard (`>= 7 of 9`): holds 99.9% at a per-cell rate of 0.98, 86% at 0.85,
and a broken rate of 0.5 passes only 9%. For `n_mechanism` the null (detaching does
nothing) is a per-cell coin flip between two arms of equal canonical distribution:
false-fire 0.0195; in the census DETACH_HALT beat FULL in 9/9 cells.

**What the checks do not cover:**
- 9 cells on 3 fresh development worlds: a sizing, not a verdict.
- Loop identity is GIVEN; one body operator, linear halting, K = 6, 128 examples
  (64 retained); half of all inputs iterate zero times.
- The detached predicate trains in wake on states it cannot shape; whether that
  slows wake halting is part of what `n_wake` measures.

# What it decides (for planning)

- **`LOOPS_ONLINE`:** loops are learned online with a vocabulary that forms. Next:
  loop discovery without identity (B2-hard), then longer bodies.
- **`LOOPS_AFTER_SLEEP`:** read wake medians and count accuracy for why wake fails.
- **`NO_LOOPS` with `DETACH_PROTECTS`:** the formation problem is solved and the loop
  quality is the remaining gap; next is the support limit (the re-fit sees 64).
- **`FORMATION_BROKEN`:** the census's mechanism does not carry to wake + fresh
  worlds; read `LOOP_ATTACHED` and the wake canonical medians first.

# Operational

27 cells on a pool of 3. Runner `row.experiments.b2_online_v2` (v1's runner with
the arms above; it imports v1's re-fit and count accuracy rather than copying
them). Restartable, durable stamped cells, protocol fingerprint (this plan,
config, the v2 and v1 runners, loop-stream, loop-learner, B2 census, B1-online,
deep_reroute, O2, O2D and lifetime modules, commit); `run.log`, `status.json`,
`exit.json`, `precondition.json`; detached launch; scale-16 dry run of every arm on
world 69 with a restart test; reserve 2.0 GiB (PI 2026-10-03). Independent scorer
`row.experiments.score_b2_online_v2` recomputes every clause from the per-task cell
records and must agree with the runner's summary. Tests: `tests/test_b2_online_v2.py`
(labels, arm constructions, scorer agreement and refusal) and
`tests/test_loop_gated.py`.
