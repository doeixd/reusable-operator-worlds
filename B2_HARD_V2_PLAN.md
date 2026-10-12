# B2-hard v2: with candidates starting at the one-step form, does the learner discover WHICH tasks iterate, without being told, while forming the vocabulary? (Tier 1, exploratory)

Status: FROZEN 2026-10-12, before any B2-hard v2 cell ran. **Tier 1, EXPLORATORY: no verdict.** Development band 5
(seeds 50-69, allocated 2026-10-10): worlds 57, 58, 59, streams 0-2; dry-run world 67. These worlds have never been
generated (verified: no runner, report or artifact references them; band 5 has so far used 50-56 and 68).

# Why

B2-hard (`reports/b2_hard.json`, `B2_HARD_PLAN.md`) labeled `FORMATION_BROKEN` (k_formation 6/9) while every
discovery clause would have passed (wake AUC 0.983, recall 0.905, specificity 1.000). Three post-hoc Tier 0 censuses
and one Tier 1 sizing located the failure:
- `reports/b2h_wake_census.json`: a ROOTED operation at the end of wake (every straight anchor of one teacher operation
  best fit by L = 2 or 3 iterations of one slot, i.e. a slot Q with Q^L ~ P) marks EXACTLY the 6 failing candidate
  cells (0 of 12 passing, 0 of 9 GIVEN). The zero-initialized halting start every candidate shared is a 50/50
  continue mixture whose dominant terms are the identity and repeated steps.
- `reports/b2h_root_census.json`: restricting count forms to one step at the end of stream repairs 3 of 6 failing
  cells and breaks none; deep or multiple roots stay broken, so the fix belongs in wake.
- `reports/b2h_init_sizing.json` (world 56, stream 0, one cell per setting): with per-step biases starting at
  `[+c, -c, ...]`, c = 2 and c = 4 both had no roots, wake AUC 1.0 and terminal canonical ~0.01, where the zero start
  formed three roots and collapsed (1.30). c = 4 is registered (closer wake vocabulary to GIVEN's); not tuned further.

These are post hoc on the B2-hard cells and one sizing world; this plan tests the repair prospectively on fresh worlds
and registers the root mechanism as a separate prediction.

# Stream and candidates

`row.experiments.loop_stream` unchanged (O2's SHUFFLED depth-3 stream: 60 length-1, 64 length-2, 64 canonical
length-3 tasks, plus 48 loop tasks, one random order over 236 tasks; 128 examples per task). Program LENGTH sets the
planned depth, and loop bodies have planned depth 1, so the CANDIDATES are the 108 depth-1 tasks (48 loop, 60 straight
length-1). Disclosed, as in B2-hard: the planned depth narrows the search from 236 tasks to 108; within them the
learner gets no identity. Also disclosed: the ONE_STEP end of stream uses the planned depth a second time (a straight
length-1 task's count form is one application, L in {0, 1}; L = 0 is kept so a loop task's usual best count, zero,
stays available).

# Learner change (constructions)

`row.models.loop_gated.LoopLearner(halt_init=c)` (committed 34ec3d2, tested, default bitwise): per-step biases start at
`[+c, -c, ..., -c]` instead of zero, so a candidate begins as a single application. h starts at zero as before.

# Arms

| arm | construction |
|---|---|
| `ONESTEP` | `LoopLearner(loop_plan = all 108 candidates, state_halt=True, step_bias=True, detach_halt_input=True, halt_decay=0.1, halt_init=4)`; wake on the stream; end of stream: re-route the depth-2/3 tasks (`deep_reroute.reroute`); per candidate, on its 64 retained examples with the library frozen, the best COUNT form with L in {0, 1} (body slot) and the best LOOP form (body slot, state predicate fitted as `b2_online.refit_loops` fits it, seed `[5820, w, i]`); keep the loop iff LOOP support MSE `< 0.5 x` COUNT support MSE, else install the count form; the chosen form is always installed (`census_b2h_root.one_step_end_of_stream`); then B2-hard's sleep (`b2_hard.sleep_decay`: 8,192 updates, sampling `[1941, w, s, 64]`, library, all task codes and biases, h with decay 0.1) |
| `ZERO` | identical except `halt_init=None` (B2-hard's ALLLOOP wake, zero start): the paired control for the start; both arms share the ONE_STEP end of stream, so they differ only in the halting initialization |
| `GIVEN` | B2-online v2's `LOOP` exactly (B2-hard's `GIVEN`: loop identity given, the 48 loop tasks only, single bias, detached input, no decay; v2's re-route, re-fit and sleep): the paired reference for what discovery costs |

# Rules (registered)

Per cell: the wake state-weight AUC (`ONESTEP`, `ZERO`: the L2 norm of each candidate's `h` after wake; loop tasks
positive, straight negative; ties half); end-of-stream decisions; terminal medians (query NMSE, hard routes and hard
halting); the wake library's per-operation best fixed counts for the 60 straight anchors (every arm), from which a cell
is ROOTED if some teacher operation has all its anchors best fit at L >= 2 (teacher programs enter only this record).

**Primary label (`ONESTEP`):**
- `k_formation` = `ONESTEP` cells with terminal canonical median `< 0.05`, of 9.
- `AUC_med` = median over the 9 cells of the wake state-weight AUC.
- `recall` = loop tasks kept as loops / 432; `specificity` = straight candidates given the count form / 540 (pooled).
- `n_terminal` = `ONESTEP` cells with terminal loop median `< 0.38` (B2-online v2's bar), of 9.

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `AUC_med >= 0.9` and `recall >= 0.8` and `specificity >= 0.95` and `n_terminal >= 8` | `DISCOVERS` |
| otherwise | `NOT_DISCOVERED` |

**Secondary label (mechanism), from `r_one` and `r_zero` = rooted cells of 9 in `ONESTEP` and `ZERO`:**

| condition (first matching row) | label |
|---|---|
| `r_zero <= 1` | `NO_ROOTS_IN_CONTROL` (the control did not reproduce the failure mode; uninformative about the start) |
| `r_one == 0` | `ROOTS_PREVENTED` |
| otherwise | `ROOTS_NOT_PREVENTED` |

**Which clauses carry the question (stated before data).** B2-hard's discovery clauses passed; its question was
carried by formation, and so is this one: `k_formation` decides whether the one-step start lets the vocabulary form
with loop machinery on every candidate. `AUC_med` asks whether WAKE still localizes state-reading when every candidate
starts as a plain step (the sizing says yes in one cell). `recall`, `specificity` and `n_terminal` are near-guaranteed
on a formed library (B2-hard: 0.905, 1.000, 9/9; v2: 9/9). The secondary label tests the root mechanism found post hoc.

**Prospective check of the wake census (reported, registered as a prediction, not a label):** every `ZERO` cell that
fails formation is rooted (the census found 6 of 6). Under the ONE_STEP end of stream `ZERO` is expected to form
in about 8 of 9 cells (root census: 8/9 and 7/9 on the B2-hard cells), so few failures may be available to test it.

**Reported, not registered:** the cost of discovery (`ONESTEP` against `GIVEN`: paired terminal loop and canonical
medians, cells where `ONESTEP` is at or below `GIVEN`); `ZERO`'s formation count, AUC and recall; wake, re-route-only
and terminal canonical medians in every arm; anchors' medians; count accuracy; routes changed; wake state-weight norms
by task type; per-task decisions.

**Non-vacuity (fails the cell):** wake MOVED the halting parameters away from their initial values (median distance
> 0; with `halt_init=4` a nonzero norm would be vacuous); sleep changed the library; every candidate arm cell has 108
decisions; the anchor record covers the 60 straight anchors; each arm's `halt_init` is the registered one.

# Necessity

- **Target behaviour:** discovering which tasks need data-dependent iteration, without being told, while forming the
  vocabulary.
- **Refusal arm:** none run here: B2-online v2's `CONSTCOUNT` and B2-hard's `ALLCOUNT` (same iteration machinery, no
  state access) measured the refusal cost twice (terminal loop medians 1.07 and 1.074 against 0.255 and 0.305). Dropped
  to keep the run at 27 cells; the `n_control` clause it supported was near-guaranteed (9/9 both times).
- **Measured refusal cost and its scale:** loop median 1.07 (refusal) against 0.26-0.31 (state halting); the scale is
  that gap and the bar 0.38. A loop form forced on a straight task costs 1.36 against 0.011 (data-test census).
- **Impostors:** the ROOT impostor (a slot iterated L times standing in for a primitive) is the one this plan targets;
  extra capacity (measured by `ALLCOUNT` in B2-hard); identity leaking (identity enters only the post-hoc AUC, recall,
  specificity and the root record); a decision that keeps loops everywhere or nowhere (specificity, recall); the planned
  depth (narrows to 108 candidates and fixes one step for straight count forms; disclosed).
- **Difficulty band, higher-is-harder:** loop median NMSE, band `(0.0005, 0.5)`; `GIVEN` expected near 0.26.

# Discriminating power

**The decision rules, as they will be applied:** `DISCOVERS` if `k_formation >= 7` of 9, `AUC_med >= 0.9`,
`recall >= 0.8` of 432, `specificity >= 0.95` of 540 and `n_terminal >= 8` of 9. `ROOTS_PREVENTED` if `r_zero >= 2`
and `r_one == 0` (of 9 each).

Source `reports/b2_hard_v2_design/rates.py`, output `rates_output.txt` (seed 41; 2,000 draws of 9 cells for the AUC
clause, exact binomials otherwise):

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `AUC_med >= 0.9` (48 vs 60 per cell) | d' = 0 | **0.000** | d' = 2 (AUC ~0.92) | **0.990** |
| `recall >= 0.8` of 432 | per-task 0.75 | **0.007** | per-task 0.85 | **0.998** |
| `specificity >= 0.95` of 540 | per-task 0.93 | **0.036** | per-task 0.97 | **0.996** |
| `n_terminal >= 8` of 9 | per-cell 0.5 | **0.020** | per-cell 0.98 | **0.987** |
| formation guard `>= 7` of 9 | per-cell 0.5 | 0.090 | per-cell 0.98 | **0.999** |
| `ROOTS_PREVENTED` | start has no effect (both arms root at the zero-start rate 7/19 = 0.368 per cell: B2-hard 6 of 18 candidate cells, sizing 1 of 1) | **0.014** | one-step start roots at 0 / 0.02 per cell | **0.900 / 0.750** |

**What the checks do not cover:**
- The formation guard passes a broken rate of 0.5 with probability 0.09; as in B2-hard it is a guard, and a broken
  ONESTEP would most likely show as 0.6-0.7 per cell (B2-hard), which passes 20-46% of the time. `ZERO` is the paired
  reference for reading it.
- `ROOTS_PREVENTED` detection is capped at 0.90 by the chance (0.10) that `ZERO` roots in at most one cell, and drops
  below 0.80 once the one-step start roots in more than about 2% of cells. It is a secondary label; a single rooted
  ONESTEP cell reads `ROOTS_NOT_PREVENTED`, deliberately strict.
- Pooled recall and specificity treat tasks as independent; per-cell values are reported.
- The AUC clause is narrow around its threshold (d' = 1.5 passes 0.1%).
- 9 cells on 3 fresh development worlds: a sizing, not a verdict. `halt_init = 4` comes from one sizing cell.
- The root definition uses the best fixed count on 64 retained examples; "rooted" means "fit only by L >= 2", not a
  measured functional root.
- The planned depth is given; one body operator, linear halting, K = 6, 128 examples (64 retained).

# What it decides (for planning)

- **`DISCOVERS`:** with the plain step as the starting point, the learner finds which tasks iterate and forms the
  vocabulary; with `ROOTS_PREVENTED` the mechanism is the one located. Next: longer loop bodies (depth 2), and the
  generalization of the lesson (start new freedoms at the plain form) to the branch line's gates.
- **`NOT_DISCOVERED`:** read the failing clause; wake AUC failing would mean the one-step start suppresses wake
  localization (then compare `ZERO`'s AUC).
- **`FORMATION_BROKEN`:** the one-step start does not suffice; read `r_one` (roots despite the start, or a different
  failure mode) and `ZERO` before any further design.

# Operational

27 cells on a pool of 3, decisive arm first (`ONESTEP`, then `ZERO`, then `GIVEN`). Runner `row.experiments.b2_hard_v2`;
it imports B2-hard's construction pieces (`build('GIVEN')`, `sleep_decay`, `auc`), the root census's
`one_step_end_of_stream` and the wake census's `best_count` rather than copying them. Restartable, durable stamped
cells, protocol fingerprint (this plan, config, the runner, the loop learner, B2-hard, B2-online v1, B1-online, the B2
census, root and wake censuses, deep_reroute, lifetime, loop-stream, O2, O2D modules, commit); `run.log`,
`status.json`, `exit.json`, `precondition.json`; detached launch; scale-16 dry run of every arm on world 67 with a
restart test; reserve 2.0 GiB (PI 2026-10-03). Independent scorer `row.experiments.score_b2_hard_v2` recomputes every
clause (including the root counts) from the per-task records and must agree with the runner's summary. Tests:
`tests/test_b2_hard_v2.py` (labels, root rule, arm constructions and starts, scorer agreement, disagreement on a
tampered summary, and refusal) and `tests/test_loop_gated.py` (`halt_init`).
