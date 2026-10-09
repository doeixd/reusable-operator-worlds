# B1-hard: does the learner discover WHICH tasks branch, and is the gain state-conditioning rather than capacity? (Tier 1, exploratory)

Status: FROZEN 2026-10-06, revised 2026-10-09 (pre-launch data-test check, record fields, independent scorer) before any B1-hard cell at scale 1 ran. **Tier 1, EXPLORATORY: no
verdict.** It answers the PI's critique of B1-online ("Too easy?", 2026-10-06):
B1-online's paired clause was close to guaranteed, the learner was TOLD which tasks
branch, and there was no capacity-matched control. Development worlds 44-46,
streams 0-2: the SAME worlds, streams and branch tasks as B1-online, opened by it,
so every cell pairs with B1-online's committed `GATED` (branch identity given) and
`REFUSAL` cells. Band 30-49 has no unopened worlds left. Dry-run world 49.

# What changes from B1-online

1. **No branch identity.** Every one of the 236 stream tasks gets a second route
   code and a gate. The learner must find which tasks need state-dependent routing.
2. **A usage cost on state dependence.** The gate's state weights carry weight
   decay 0.1 (their own optimizer group, in wake and sleep); a gate pays for using
   the state.
3. **A capacity-matched control.** `CONSTMIX`: the same two routes and a gate on
   every task, but the gate is a learned bias only (state-independent).
4. **A data test for branching at the end of the stream.** Per task, on its 64
   retained examples, the best single route (exhaustive) against the best branch
   (B1g's gradient fit, 4 restarts); the branch is kept only if it at least halves
   the hardened support error (`BRANCH_RATIO = 0.5`).

**Not changed, and why (disclosed):** 128 examples per task. Giving branch tasks
512 examples means `examples_per_task = 512` for every task, about 4x every
lifetime (well over 20 hours for this rung). The absolute branch bar at adequate
sample size is deferred to its own rung. Decisions are still on the input with a
linear predicate (B1's INPUT variant); intermediate-state and nonlinear decisions
are later rungs.

# Arms, as constructions

| arm | construction |
|---|---|
| `ALLGATE` | `BranchGatedLearner(branch_plan = every stream task, state_gate=True, gate_decay=0.1)`; wake on the 236-task stream; end of stream per task: exhaustive single route vs B1g branch fit (seed `[5900, w, i, r]`); branch kept iff its hardened support MSE < 0.5 x the single route's, otherwise the task is made single-route (route installed by O5's swap, gate pinned constant to route 1); then sleep (O3's construction, 8,192 updates, sampling `[1941, w, s, 64]`) on the library, all route codes and gates, state weights with decay 0.1 |
| `CONSTMIX` | `BranchGatedLearner(branch_plan = every stream task, state_gate=False)`; same wake; end of stream: every task single-route (exhaustive); same sleep |
| `GATED`, `REFUSAL` (references, not re-run) | B1-online's committed cells on the same worlds and streams |

Construction checks: `branch_gated` defaults reproduce B1-online exactly (its unit
tests pass after the change); the lifetime's list branch groups plain-parameter
lists exactly as before and adds dict items as their own groups.

# Rules (registered)

Per `ALLGATE` cell: wake gate-norm AUC (the L2 norm of each task's gate state
weights after wake; branch tasks positive, straight tasks negative; ties half);
end-of-stream decisions; terminal canonical and branch medians.

- `k_formation` = `ALLGATE` cells with canonical median `< 0.05`, denominator 9.
- `AUC_med` = median over the 9 cells of the wake gate-norm AUC.
- `recall` = branch tasks kept as branches / 432; `specificity` = straight tasks
  made single-route / 1,692 (pooled over cells).
- `n_control` = cells where `ALLGATE`'s branch median < `CONSTMIX`'s, denominator 9
  (a construction check, as the critique showed for refusal).

| condition (first matching row) | label |
|---|---|
| `k_formation <= 6` | `FORMATION_BROKEN` |
| `AUC_med >= 0.9` and `recall >= 0.8` and `specificity >= 0.95` and `n_control >= 8` | `DISCOVERS` |
| otherwise | `NOT_DISCOVERED` |

Reported beside it: each clause's value; `CONSTMIX` canonical and branch medians;
B1-online's `GATED` and `REFUSAL` medians on the same cells; wake-only scores; gate
norms by task type.

# Necessity

- **Target behaviour:** discovering which tasks need state-conditional routing,
  without being told, while forming the vocabulary.
- **Refusal arm:** `CONSTMIX`: the same parameters with no state dependence; and,
  as a reference, B1-online's `REFUSAL` (branch median 1.02).
- **Measured refusal cost and its scale:** B1-online: one route per task left branch
  tasks at 1.02 against 0.23 with gates, and tripled straight-line error (0.0306
  against 0.0106). The scale is the 0.05 threshold and the paired medians.
- **Impostors:** extra capacity (`CONSTMIX` has it without state access); branch
  identity leaking (the plan is every task; identity enters only the post-hoc
  AUC, recall and specificity); a decision rule that keeps branches everywhere
  (specificity clause) or nowhere (recall clause).
- **Difficulty band, higher-is-harder:** branch median NMSE, band `(0.0005,
  0.5)`; B1-online's oracle-identity arm sat at 0.23.

# Discriminating power

**The decision rule, as it will be applied:** `DISCOVERS` if `AUC_med >= 0.9`,
`recall >= 0.8`, `specificity >= 0.95`, `n_control >= 8`, and `k_formation >= 7`.

Source `reports/b1_hard_design/rates.py`, output `rates_output.txt` (seed 41).

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `AUC_med >= 0.9` | gate norms uninformative (d' = 0) | **0.000** | d' = 2 (AUC ~0.92) | **0.993** |
| `recall >= 0.8` of 432 | per-task 0.75 | **0.007** | per-task 0.85 | **0.998** |
| `specificity >= 0.95` of 1,692 | per-task 0.93 | **0.0004** | per-task 0.97 | **1.000** |

**What the checks do not cover:**
- The pooled recall and specificity treat tasks as independent; tasks within a cell
  share a library, so the effective sample is smaller than 432 or 1,692. Per-cell
  values are reported.
- The AUC clause is narrow around its threshold: d' = 1.5 (AUC ~0.86) never passes.
- The worlds were opened by B1-online: design evidence only.
- 128 examples per task (see "Not changed"); input-only linear decisions.
- `BRANCH_RATIO = 0.5` and `GATE_DECAY = 0.1` are single registered values, not tuned;
  sensitivity is a later question.

# Pre-launch check of the end-of-stream data test (Tier 0, 2026-10-09, before any real cell)

Written into this plan before any B1-hard cell at scale 1 ran (only the scale-16 dry run on world 49 existed, and it
is re-run after this revision). The concern: the data test compares in-sample errors on 64 examples, so a two-route
fit might halve a straight task's error by overfitting and make the specificity clause unreachable.
`row.experiments.census_b1h_data_test` applied the test exactly as `b1_hard.end_of_stream` does to B1-online's saved
GATED WAKE terminals (worlds 44-46, stream 0, all 236 tasks; reload gate G0 reproduces the committed wake medians in
3/3). Report `reports/b1h_data_test_census.json`:

- support-error ratio branch/single, median: branch tasks 0.047, straight tasks 1.000 (the fit collapses to the
  single route); AUC 1.000;
- at `BRANCH_RATIO = 0.5`: recall 1.000, specificity 1.000; the same at 0.75; at 0.9 specificity 0.996;
- on held-out query data the separation also holds (query ratio medians 0.29 and 1.00; at 0.5 recall 0.965).

**Consequence for reading this rung, registered now:** on a FORMED library the recall and specificity clauses are
close to guaranteed: they test an offline search component that already works, not discovery. They stay in the rule
as checks that `ALLGATE`'s own library supports the test (its library is not B1-online's), but the clauses that
carry the question are `k_formation` (do gates on every task damage formation?) and the wake gate-norm `AUC_med`
(does wake itself put state dependence where it is needed?). A `DISCOVERS` read with `AUC_med` near its bar is to be
reported as resting on that one clause. The census is descriptive, used B1-online's development worlds, and changes
no threshold.

# What it decides (for planning)

- **`DISCOVERS`:** the learner finds which tasks branch and the gain is state-
  conditioning. Next: the absolute bar at adequate sample size (its own rung),
  then intermediate-state and nonlinear decisions, then B2 (iteration).
- **`NOT_DISCOVERED`:** read the clause that failed: wake gate norms (online
  discovery), recall (the data test misses branches), or specificity (it invents
  branches). Each points at a different fix.
- **`FORMATION_BROKEN`:** gates on every task damage the vocabulary; compare
  `CONSTMIX` and B1-online's references first.

# Operational

18 cells on a pool of 3 (each cell: a 236-task wake with two routes per task, a
236-task end-of-stream decision with B1g fits, sleep); timed on the dry run.
Restartable, durable stamped cells, protocol fingerprint (this plan, config, the
B1-hard/B1-online/branch-learner/B1g/branch-stream/deep_reroute/lifetime modules,
commit); `run.log`, `status.json`, `exit.json`, `precondition.json`; detached
launch; scale-16 dry run of both arms on world 49 with a restart test; reserve
2.0 GiB (PI 2026-10-03). Independent scorer `row.experiments.score_b1_hard` recomputes every clause from the
per-task records (wake gate norms, decisions, per-task terminal errors) and must agree with the runner's summary.
