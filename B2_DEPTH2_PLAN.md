# B2-depth2: with loops whose body is TWO steps, does the one-step-start learner discover which tasks iterate, without being told, while forming the vocabulary? (Tier 1, exploratory)

Status: FROZEN 2026-10-12, before any B2-depth2 cell ran. **Tier 1, EXPLORATORY: no verdict.** Development band 5 (seeds 50-69, allocated
2026-10-10): worlds 61, 62, 63, streams 0-2; dry-run world 66. These worlds have never been generated (verified: no
runner, report or artifact references them; band 5 has so far used 50-60 and 67-69).

# Why

B2-hard v2 (`reports/b2_hard_v2.json`) labeled `DISCOVERS` with `ROOTS_PREVENTED` for loops whose body is ONE
operator: with every candidate starting at the one-step form, the learner found which tasks iterate while forming the
vocabulary (9/9), at a cost of about 6% on loops against being told. The ladder's next step is longer bodies.
- `reports/b2d2_necessity_census.json` (Tier 0, on B2-hard v2's rebuilt final libraries, gates 3/3): a two-step-body
  loop beats one straight route (1.02), an input-picked unrolled count (0.51) and the best one-slot loop (1.69) in
  108/108 tasks (0.164 at 128 support); the body is found by search in 108/108 (runner-up ~52x worse); execution
  drift sets the ceiling (true-predicate loop 0.083; 0.07 at one body application, 0.42 at four).
- `reports/b2d2_sizing.json` (Tier 1, world 60, one cell per arm): both arms formed (canonical 0.018 not told, 0.011
  told), wake AUC 0.979, 43/48 loops kept, 64/64 straight kept straight, no length-2 anchor fit as an iterated body;
  the cost of discovery was larger than at depth 1 (terminal loops 0.368 against 0.286; wake vocabulary 0.24 against
  0.027, mostly stale routes, 0.047 after re-routing alone).

The body search is near-guaranteed on a formed library (census), so the question rests on formation and on wake
localizing state-reading when the candidates are two-step tasks.

# Stream and candidates

`row.experiments.loop_stream.stream(w, s, 48, body_depth=2)` (committed 622e784, tested; default bitwise): O2's
SHUFFLED depth-3 stream (60 length-1, 64 length-2, 64 canonical length-3) plus 48 loop tasks
`z = x; repeat at most K = 6 times: stop if w.z <= 0, else z = P2(P1(z))` (P1 != P2 teacher primitives, ids from
stream 49, draws [5810, world, i]), one random order over 236 tasks, 128 examples per task. Program LENGTH sets the
planned depth, so the CANDIDATES are the 112 depth-2 tasks (48 loop, 64 straight length-2). Disclosed: the planned
depth narrows the search from 236 tasks to 112, and the ONE_STEP end of stream uses it again (a straight candidate's
count form is one application of a two-step body, L in {0, 1}).

# Learner change (constructions)

`row.models.loop_gated.LoopLearner(body_depth=2)` (committed 622e784, tested; default 1 keeps the depth-1 check and
is bitwise): a loop task's body is its own route at its planned depth, one iteration being the whole route.

# Arms

| arm | construction |
|---|---|
| `ONESTEP2` | `LoopLearner(loop_plan = all 112 candidates, state_halt=True, step_bias=True, detach_halt_input=True, halt_decay=0.1, halt_init=4, body_depth=2)`; wake on the stream; end of stream: re-route the other tasks (`deep_reroute.reroute`); per candidate, on its 64 retained examples with the library frozen (`b2d2_sizing.end_of_stream_onestep`): the best COUNT form over all 144 two-step bodies with L in {0, 1}, and the LOOP form on the body with the lowest inferred-count support error (each example stopped at its best count), with a linear state predicate fitted as `b2_online.refit_loops` fits it (seed `[5830, w, i]`); keep the loop iff LOOP support MSE `< 0.5 x` COUNT support MSE, else install the count form; then `b2_hard.sleep_decay` (8,192 updates, sampling `[1941, w, s, 64]`, library, all task codes and biases, h with decay 0.1) |
| `GIVEN2` | loop identity given: `LoopLearner(loop_plan = the 48 loop tasks, state_halt=True, detach_halt_input=True, body_depth=2)` (B2-online v2's halting); end of stream: re-route the other tasks; per loop task the same LOOP form (`b2d2_sizing.refit_given`), installed iff it beats the current support MSE; `b1_online.sleep` with the loop parameters. The paired reference for what discovery costs |

Body choice by inferred-count error replaces B2-hard's per-slot predicate fit (144 fits per candidate would cost about
an hour per cell); the census found it picks the matched body in 108/108 tasks.

# Rules (registered)

Per cell: the wake state-weight AUC (`ONESTEP2`: the L2 norm of each candidate's `h` after wake; loop tasks positive,
straight negative; ties half); end-of-stream decisions; terminal medians (query NMSE, hard routes and hard halting);
the length-2 anchors' best fixed count over all two-step bodies on the wake library (teacher programs enter only this
record).

**Label (`ONESTEP2`):**
- `k_formation` = `ONESTEP2` cells with terminal canonical median `< 0.05`, of 9.
- `AUC_med` = median over the 9 cells of the wake state-weight AUC.
- `recall` = loop tasks kept as loops / 432; `specificity` = straight candidates given the count form / 576 (pooled).
- `n_terminal` = `ONESTEP2` cells with terminal loop median `< 0.5`, of 9. The bar is the input-picked-count
  impostor's median at 128 support on formed vocabularies (census: 0.508): a loop below it beats the best non-loop
  alternative measured. B2-online v2's 0.38 was sized for one-step bodies, whose ceiling is about half as high.

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `AUC_med >= 0.9` and `recall >= 0.8` and `specificity >= 0.95` and `n_terminal >= 8` | `DISCOVERS` |
| otherwise | `NOT_DISCOVERED` |

**Which clauses carry the question (stated before data).** Formation (`k_formation`) and wake localization
(`AUC_med`). `specificity` is near-guaranteed on a formed library (sizing 64/64; B2-hard v2 1.000), and so is the
body search. `recall` is not: the sizing kept 43/48 (0.896), so per-task recall near 0.9 passes the 0.8 bar with
probability ~1, but recall near 0.75 does not. `n_terminal` is sized from one cell (0.368 against 0.5).

**Reported, not registered:** the cost of discovery (`ONESTEP2` against `GIVEN2`: median paired ratio of terminal loop
medians, paired canonical medians, cells where `ONESTEP2` is at or below `GIVEN2`); `GIVEN2`'s formation count; wake,
re-route-only and terminal medians in both arms (canonical, loop, length-2 and length-1 anchors); count accuracy;
routes changed; wake state-weight norms by task type; per-task decisions; length-2 anchors best fit at L >= 2 (a root
analogue) per arm.

**Prediction (registered, not a label):** the cost of discovery is larger than at depth 1: the median paired ratio
of `ONESTEP2` to `GIVEN2` terminal loop medians lies in `[1.1, 1.6]` (sizing 1.29; B2-hard v2 1.06).

**Non-vacuity (fails the cell):** wake MOVED the halting parameters away from their initial values (median distance
> 0); sleep changed the library; every `ONESTEP2` cell has 112 decisions; the anchor record covers the 64 length-2
anchors; each arm's `halt_init` and `body_depth` are the registered ones.

# Necessity

- **Target behaviour:** discovering which tasks need data-dependent iteration over a two-step body, without being
  told, while forming the vocabulary.
- **Refusal arm:** none run; the census measured the refusal cost on formed vocabularies.
- **Measured refusal cost and its scale:** query NMSE 1.02 (one straight route) and 0.51 (input-picked unrolled count)
  against 0.164 (learned two-step loop) at 128 support, 108/108 tasks (census). The scale is that gap and the bar 0.5.
- **Impostors:** a one-step-body loop (census: 1.69, worse than refusal); an input-picked count (0.51, the bar);
  a ROOT analogue (a straight length-2 task fit as an iterated body; recorded per arm, 0/64 in the sizing); extra
  capacity (measured by B2-hard's `ALLCOUNT`, not rerun); identity leaking (identity enters only the post-hoc AUC,
  recall, specificity and the anchor record); a decision that keeps loops everywhere or nowhere (specificity, recall);
  the planned depth (narrows to 112 candidates; disclosed).
- **Difficulty band, higher-is-harder:** loop median NMSE, band `(0.0005, 0.6)`; `GIVEN2` expected near 0.29
  (sizing), true-predicate ceiling 0.083.

# Discriminating power

**The decision rules, as they will be applied:** `DISCOVERS` if `k_formation >= 7` of 9, `AUC_med >= 0.9`,
`recall >= 0.8` of 432, `specificity >= 0.95` of 576 and `n_terminal >= 8` of 9.

Source `reports/b2_depth2_design/rates.py`, output `rates_output.txt` (seed 41; 2,000 draws of 9 cells for the AUC
clause, exact binomials otherwise):

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `AUC_med >= 0.9` (48 vs 64 per cell) | d' = 0 | **0.000** | d' = 2 (AUC ~0.92) | **0.987** |
| `recall >= 0.8` of 432 | per-task 0.75 | **0.007** | per-task 0.85 | **0.998** |
| `specificity >= 0.95` of 576 | per-task 0.93 | **0.023** | per-task 0.97 | **0.995** |
| `n_terminal >= 8` of 9 | per-cell 0.5 | **0.020** | per-cell 0.98 | **0.987** |
| formation guard `>= 7` of 9 | per-cell 0.5 | 0.090 | per-cell 0.98 | **0.999** |

**What the checks do not cover:**
- The formation guard passes a broken rate of 0.5 with probability 0.09 and a rate of 0.7 with probability 0.46;
  `GIVEN2` is the paired reference for reading it.
- `n_terminal` at a per-cell pass rate of 0.9 passes only 77% of the time; the sizing's one cell sat at 0.368 against
  the 0.5 bar, so its per-cell rate is not known.
- Pooled recall and specificity treat tasks as independent; per-cell values are reported.
- The AUC clause is narrow around its threshold (d' = 1.5 passes 0.1%).
- 9 cells on 3 fresh development worlds: a sizing, not a verdict. `halt_init = 4` comes from the depth-1 sizing.
- The planned depth is given; two-step bodies, linear halting, K = 6, 128 examples (64 retained).

# What it decides (for planning)

- **`DISCOVERS`:** the one-step start carries to two-step bodies. Next: the branch line's gates with the plain start,
  or a self-curriculum census; the absolute loop bar at adequate sample size stays deferred.
- **`NOT_DISCOVERED`:** read the failing clause; recall failing would mean the data test loses loops at depth 2 (read
  its per-task supports); AUC failing would mean wake does not localize state-reading over two-step bodies.
- **`FORMATION_BROKEN`:** read `GIVEN2` (does the vocabulary form at all with two-step-body loops?) and the anchor
  record (an iterated-body impostor) before any further design.

# Operational

18 cells on a pool of 3, decisive arm first (`ONESTEP2`, then `GIVEN2`). Runner `row.experiments.b2_depth2`; it imports
the sizing's construction pieces (`build`, `end_of_stream_onestep`, `refit_given`, `anchor_best_counts`,
`count_accuracy`) and B2-hard's `sleep_decay` and `auc` rather than copying them. Restartable, durable stamped cells,
protocol fingerprint (this plan, config, the runner, the loop learner, the sizing, B2-hard, B1-online, both censuses,
deep_reroute, lifetime, loop-stream, O2, O2D modules, commit); `run.log`, `status.json`, `exit.json`,
`precondition.json`; detached launch; scale-16 dry run of both arms on world 66 with a restart test; reserve 2.0 GiB
(PI 2026-10-03). Independent scorer `row.experiments.score_b2_depth2` recomputes every clause from the per-task records
and must agree with the runner's summary. Tests: `tests/test_b2_depth2.py` (labels, arm constructions, scorer
agreement, disagreement on a tampered summary, and refusal) and `tests/test_loop_gated.py` (`body_depth`).
