# B2-hard: does the learner discover WHICH tasks iterate, without being told, while forming the vocabulary? (Tier 1, exploratory)

Status: FROZEN 2026-10-11, before any B2-hard cell ran. **Tier 1, EXPLORATORY: no verdict.** Development band 5
(seeds 50-69, allocated 2026-10-10): worlds 53, 54, 55, streams 0-2; dry-run world 68. These worlds have never been
generated (verified: no runner, report or artifact references them; the only textual hit for 53 is the seed
component `[5300, w, i]` of the B0 census, not a world).

# Why

B2-online v2 (`reports/b2_online_v2.json`, `LOOPS_ONLINE` with `DETACH_PROTECTS`, all clauses 9/9) learned
data-dependent iteration online while forming the vocabulary, but it was TOLD which tasks loop. This rung removes
that, as B1-hard did for branches.

The Tier 0 census `reports/b2h_data_test_census.json` (076f6bd, recorded 38acc1c) found two things that shape it:
1. On a formed library an end-of-stream data test (best LOOP form against best fixed-COUNT form on the 64 retained
   examples) separates loop tasks from straight depth-1 tasks perfectly (AUC 1.000 in 9/9 cells; recall 0.998 and
   specificity 1.000 at ratio 0.5). As in B1-hard, that test is near-guaranteed and cannot carry the question.
2. A linear state predicate alone cannot express "apply once and stop": forced on a straight task it costs 1.36
   query NMSE against 0.011. So loop machinery attached to every candidate needs halting that NESTS the count form.

# Stream and candidates

`row.experiments.loop_stream` unchanged (O2's SHUFFLED depth-3 stream: 60 length-1, 64 length-2, 64 canonical
length-3 tasks, plus 48 loop tasks, one random order over 236 tasks; 128 examples per task). Program LENGTH sets the
planned depth, as in every rung of this line, and loop bodies have planned depth 1, so the CANDIDATES are the 108
depth-1 tasks: 48 loop tasks and 60 straight length-1 tasks. Disclosed: the planned depth narrows the search from
236 tasks to 108; within them the learner gets no identity.

# Learner change (constructions)

`row.models.loop_gated.LoopLearner`, two new switches, defaults bitwise unchanged (tested):
- `step_bias=True`: the predicate is `h.z_j + b_j`, one bias per step; with `h = 0` it is any fixed count L, in
  particular L = 1, the single route. Test `test_step_bias_nests_count_and_decay_group`: with biases `[+60, -60, ...]`
  the output is the single route in evaluation and in training.
- `halt_decay`: h in its own optimizer group with that weight decay (a usage cost on reading the state, as B1-hard's
  gate decay 0.1).

# Arms

| arm | construction |
|---|---|
| `ALLLOOP` | `LoopLearner(loop_plan = all 108 candidates, state_halt=True, step_bias=True, detach_halt_input=True, halt_decay=0.1)`; wake on the stream; end of stream: re-route the depth-2/3 tasks (`deep_reroute.reroute`); per candidate, on its 64 retained examples with the library frozen, best COUNT form (body slot, fixed count 0..6) and best LOOP form (body slot, state predicate fitted as `b2_online.refit_loops` fits it, seed `[5820, w, i]`); keep the loop iff LOOP support MSE `< 0.5 x` COUNT support MSE, else install the count form (h = 0, biases +/-10); the chosen form is always installed (GIVEN, v2's construction, installs a re-fit only when it beats the current setting); then O3 sleep (8,192 updates, sampling `[1941, w, s, 64]`) on the library, all task codes and biases, and h with decay 0.1 |
| `ALLCOUNT` | the same 108 candidates in `loop_plan`, `state_halt=False` (per-step biases only; halting cannot read the state: the capacity control); same re-route; every candidate gets its best COUNT form; B2-online v2's sleep (`b1_online.sleep` with the biases) |
| `GIVEN` | B2-online v2's `LOOP` exactly (loop identity given: the 48 loop tasks only, single bias, detached input, no decay; v2's re-route, re-fit and sleep): the paired reference for what discovery costs |

All three arms start every halting parameter at zero: a candidate starts as a 50/50 continue mixture at every step.
For the 60 straight length-1 tasks this is new: B2-online v2 gave them none. They are the vocabulary's anchors
(N1b: length-1 tasks are the active ingredient of formation), so whether loop machinery on them damages formation is
a real question here, asked by `k_formation` and by the `ALLCOUNT` control.

# Rules (registered)

Per `ALLLOOP` cell: the wake state-weight AUC (the L2 norm of each candidate's `h` after wake; loop tasks positive,
straight candidates negative; ties half); end-of-stream decisions; terminal medians (query NMSE, hard routes and
hard halting).

- `k_formation` = `ALLLOOP` cells with terminal canonical median `< 0.05`, of 9.
- `AUC_med` = median over the 9 cells of the wake state-weight AUC.
- `recall` = loop tasks kept as loops / 432; `specificity` = straight candidates given the count form / 540 (pooled).
- `n_control` = cells where `ALLLOOP`'s terminal loop median `<` `ALLCOUNT`'s, of 9 (a construction check).

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `AUC_med >= 0.9` and `recall >= 0.8` and `specificity >= 0.95` and `n_control >= 8` | `DISCOVERS` |
| otherwise | `NOT_DISCOVERED` |

**Which clauses carry the question (stated before data).** `recall` and `specificity` are near-guaranteed on a
formed library (census above): they check that `ALLLOOP`'s own library supports the test. The question is carried by
`k_formation` (does loop machinery on every candidate, including the anchors, damage formation?) and by `AUC_med`
(does WAKE already put state-reading where it is needed?). A `DISCOVERS` read with `AUC_med` near its bar is to be
reported as resting on that one clause. `n_control` is near-guaranteed given formation (v2: 9/9).

**Reported, not registered:** the cost of discovery, `ALLLOOP` against `GIVEN` (paired terminal loop and canonical
medians, and cells where `ALLLOOP`'s loop median is at or below `GIVEN`'s); `n_terminal` (`ALLLOOP` loop median
below v2's bar 0.38); the anchors' median (the 60 straight candidates) in every arm, after wake and terminal; count
accuracy for `ALLLOOP` and `GIVEN`; wake state-weight norms by task type; per-task decisions with both forms'
support errors; routes changed.

**Non-vacuity (fails the cell):** wake moved the halting parameters (median norm > 0) in every arm; sleep changed the
library; `ALLCOUNT` kept no loop; every cell has 108 candidate decisions (`ALLLOOP`, `ALLCOUNT`).

# Necessity

- **Target behaviour:** discovering which tasks need data-dependent iteration, without being told, while forming
  the vocabulary.
- **Refusal arm:** `ALLCOUNT`: the same candidates and iteration machinery with no state access. Measured refusal
  cost in v2 (its `CONSTCOUNT`, same construction on the loop tasks only): terminal loop median 1.07 against 0.255
  with state halting; one route per task 1.02 (census).
- **Measured refusal cost and its scale:** loop median 1.07 (refusal) against 0.255 (identity given); the scale is
  that gap and v2's bar 0.38. The reverse error is measured too: a loop form forced on a straight task costs 1.36
  against 0.011 (census), so over-discovery is not free either.
- **Impostors:** extra capacity (`ALLCOUNT` has the iteration machinery and per-step biases without state access);
  identity leaking (the plan is every candidate; identity enters only the post-hoc AUC, recall and specificity);
  a decision that keeps loops everywhere (specificity clause) or nowhere (recall clause); the planned depth (it
  narrows to 108 candidates; disclosed).
- **Difficulty band, higher-is-harder:** loop median NMSE, band `(0.0005, 0.5)`; `GIVEN` is expected near v2's
  0.255.

# Discriminating power

**The decision rule, as it will be applied:** `DISCOVERS` if `AUC_med >= 0.9`, `recall >= 0.8` of 432,
`specificity >= 0.95` of 540, `n_control >= 8` of 9, and `k_formation >= 7` of 9.

Source `reports/b2_hard_design/rates.py`, output `rates_output.txt` (seed 41; 2,000 draws of 9 cells for the AUC
clause, exact binomials otherwise). Null and effect samplers per clause:

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `AUC_med >= 0.9` (48 vs 60 per cell) | state norms uninformative (d' = 0) | **0.000** | d' = 2 (AUC ~0.92) | **0.986** |
| `recall >= 0.8` of 432 | per-task 0.75 | **0.007** | per-task 0.85 | **0.998** |
| `specificity >= 0.95` of 540 | per-task 0.93 | **0.036** | per-task 0.97 | **0.996** |

Formation guard (`>= 7 of 9`): as in B2-online v2, holds 99.9% at a per-cell rate of 0.98 and a broken rate of 0.5
passes 9%.

**What the checks do not cover:**
- Pooled recall and specificity treat tasks as independent; tasks within a cell share a library, so the effective
  sample is smaller. Per-cell values are reported.
- The AUC clause is narrow around its threshold: d' = 1.5 (AUC ~0.86) passes 0.1%.
- 9 cells on 3 fresh development worlds: a sizing, not a verdict.
- The planned depth is given; one body operator, linear halting, K = 6, 128 examples (64 retained).
- `LOOP_RATIO = 0.5` and `HALT_DECAY = 0.1` are single registered values (B1-hard's), not tuned.
- The census's separation was measured on libraries formed WITH loop identity; `ALLLOOP`'s may differ.

# What it decides (for planning)

- **`DISCOVERS`:** the learner finds which tasks iterate and forms the vocabulary. Next: longer loop bodies (depth
  2), then the absolute bar at adequate sample size.
- **`NOT_DISCOVERED`:** read the clause that failed: wake AUC (wake does not localize state-reading; the end-of-stream
  test may still find loops), recall or specificity (the test fails on this library). Each points at a different fix.
- **`FORMATION_BROKEN`:** loop machinery on every candidate damages formation; compare `ALLCOUNT` (machinery without
  state) and the wake anchor medians first.

# Operational

27 cells on a pool of 3. Runner `row.experiments.b2_hard`; it imports v2-and-v1's construction pieces
(`b2_online.refit_loops`, `set_body`, `count_accuracy`; `b1_online.sleep`) rather than copying them, and its
`fit_forms` is tested equal to the census's `forms` on a loop and a straight task. Restartable, durable stamped cells,
protocol fingerprint (this plan, config, the runner, B2-online v1, loop-stream, loop-learner, B2 census, B1-online,
deep_reroute, O2, O2D and lifetime modules, commit); `run.log`, `status.json`, `exit.json`, `precondition.json`;
detached launch; scale-16 dry run of every arm on world 68 with a restart test; reserve 2.0 GiB (PI 2026-10-03).
Independent scorer `row.experiments.score_b2_hard` recomputes every clause from the per-task records (wake state
norms, decisions, terminal per-task errors) and must agree with the runner's summary. Tests: `tests/test_b2_hard.py`
(labels, arm constructions, `fit_forms` against the census and the installs, scorer agreement and refusal) and
`tests/test_loop_gated.py`.
