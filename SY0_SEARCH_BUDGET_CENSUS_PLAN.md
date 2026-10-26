# SY0: does a fixed search budget FAIL, or only cost? (Tier 0 census, synthesis cycle)

Status: FROZEN on commit (2026-10-26). Tier 0, descriptive, no new computation of any model: a re-analysis of the
192 committed per-task E5.1 cache records (`reports/e5_1_cache/*.json`, protocol `f945fe71aec9f1bf`). Source:
`SYNTHESIS_CYCLE_PROPOSAL.md` (reviewer 86's library-growth cycle). Decision 22's delegation ("use your best
judgement to continue") and the PI's "continue with the research" (2026-10-26) cover running it.

# Registration order (read first)

Before this plan was written, Claude had seen the WORLD-LEVEL E5.1 report: geometric-mean oracle gaps per
(depth, world, budget) and the registered `k*` by depth (1000 at depths 3-5, 2000 at 6 and 8-10, >2000 at 7). The
TASK-LEVEL quantities below (per-task minimal budget, the fraction of tasks no budget solves) had not been computed.
The rule is therefore registered before its own statistic, not before all related data. Why the libraries are not
rebuilt: `artifacts/e1_disc` was deleted with the rest of `artifacts/` (2026-10-16), and the question can be
answered from the records.

# Question

The cycle (extract recurring subprograms as macros, re-solve) can change WHAT gets solved only if the incumbent search
FAILS on tasks that a shorter program would let it solve. If it merely COSTS more with depth, a macro buys the constant
factor E6B already measured (x0.78 of the E5.1 cost-law forecast), and running the cycle re-measures a known law.

# Records and definitions

- Tasks: E5.1's 8 programs per (world 0-2, depth 3-10), 24 per depth; library = the E1 discrete library, oracle route
  through the E0.1 assignment; every depth was ELIGIBLE in E5.1 (scratch over oracle >= 0.75 log units).
- Search arm OPT_k, k in {250, 500, 1000, 2000} route-optimization steps; each k is a SEPARATE run, so solved-at-k is
  not guaranteed monotone in k.
- A task is SOLVED at k iff `log(nmse_OPTk) - log(nmse_O) <= 0.15` (E5.1's own GAP, per task instead of per world).
- `b*(task)` = smallest k that solves it; HARD = solved at no k.
- `U(D)` = fraction of the 24 depth-D tasks that are HARD. `F(D, k)` = fraction solved at k.

# Decision rule (one label)

Macro ceiling proxy: a length-3 macro turns a depth-D program into a depth-(D-2) program for search. Treating the
macro-augmented search as exactly as easy as depth D-2 is OPTIMISTIC (the alphabet grows), so the rise of `U` over
two depths bounds from above what one length-3 macro could rescue.

Fit the weighted least-squares slope `s` of `U(D)` on depth over depths 3-10 (192 tasks), with pooled binomial
standard error `sqrt(pbar (1 - pbar) / Sxx)`.

- `OPPORTUNITY` iff `U_deep = mean(U(8), U(9), U(10)) >= 0.25` AND `2 s >= 0.10` AND `s / se >= 1.645`.
- `NO_OPPORTUNITY` otherwise. This label means: on straight-line programs on this generator, fixed-budget failures are
  cost (recovered by more steps), so the cycle's value is E6B's constant factor; the synthesis cycle moves to
  control-flow structure search (the proposal's alternative), not run here.

Rule history (before any task-level statistic was computed): the first draft fired on any single depth with
`U(D) >= 0.25` and `U(D) - U(D-2) >= 0.15`. Its discrimination check gave false-fire 0.26-0.57 on flat nulls at
0.15-0.35 (five noisy 24-task differences), so it was replaced by the pooled trend above.

Denominators: every fraction is over the 24 tasks of a depth (3 worlds x 8); a task missing its record fails the run
closed rather than shrinking a denominator.

Recorded beside the label (descriptive, no clause): `F(D, k)` for every D and k; median `b*` by depth (HARD counted as
>2000); the scaling exponent of the budget needed for `F >= 0.75` against depth (`row.necessity_gate.incumbent_degrades`,
depth 3 -> 10); non-monotone tasks (solved at k, not at 2k); and per-world `U`.

# Discriminating power

The rule is simulated with 24 binomial tasks per depth (`census_sy0_search_budget.discrimination`, 20,000 draws,
seed 2610):

| sampler | fire rate |
|---|---|
| null, flat U = 0.05 / 0.15 / 0.25 / 0.35 | 0.0000 / 0.0000 / 0.0001 / 0.0005 |
| effect, U linear 0.05 (depth 4) -> 0.45 (depth 10) | 0.825 |
| effect, step 0.05 -> 0.45 at depth 8 | 0.962 |

False-fire <= 0.05 and detection >= 0.80: adequate. The report recomputes these numbers; if they ever fail, the label
is prefixed `UNDERPOWERED:`.

# Gate check (DESIGN_ADEQUACY)

- NECESSITY: tests exactly whether the incumbent fails rather than costs; the impostor is "more steps".
- OPPORTUNITY: the macro proxy is a ceiling, so `NO_OPPORTUNITY` is conservative for the cycle.
- DISCRIMINATION: above.
- Known limits: one library family (E1 discrete, canonical world), depth <= 10 (E5.1's sweep; C0 says execution stays
  clean to ~8 applications); gradient search only; the macro proxy ignores alphabet growth and occurrence rates.

Output: `reports/sy0_search_budget_census.json`. Scorer-independence: the label is recomputed from the raw cache by
`--verify`, a second code path that does not reuse the classification function.
