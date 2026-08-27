
# E5.1: which horizon binds first, execution or search?

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract; source is
review 81 (`reviews/reviewer-feedback-81.txt`). **Development worlds 0-2.** The
sealed band 800-829 is spent and is not reused as development data.

# The question

E5 measured findability at exactly two depths and found no degradation: the
discrete program space grew 1,728x between `D = 3` and `D = 6` while gradient
route optimization went from parity with the oracle to BEATING it in 3/3 worlds.
Review 81 reads that as the language being SMOOTHLY SEARCHABLE. That is a
hypothesis about the language, not yet a measured property, and this project has
been wrong before about properties inferred from two points.

Two horizons are conflated in the phrase "programs get hard at depth":

    D_execute   the depth at which the frozen operators stop composing well
                enough for ANY program to solve the task
    D_search    the depth at which finding a good program stops working, given
                that good programs still exist

Everything ROW has assumed since V1 says search binds first. E5 says the
opposite. This rung measures which.

# Design

One frozen library per world (the E1 discrete artifacts, `slots = 12`), the E8
variable-depth executor (bitwise identical to the shipped one at depth 3,
re-verified per world before any cell is scored), and held-out teacher programs
generated at each depth. Nothing is trained except a task's route.

**Depth sweep:** `D in {3, 4, 5, 6, 7, 8, 9, 10}`, run in ascending order and
stopped early only by the eligibility rule below, never by inspection of the
search results.

**Budget sweep:** `K in {250, 500, 1000, 2000}` Adam steps at lr 0.01, the
sealed protocol's optimizer. This is the `C_find(D, K)` surface review 81 asked
for; `K = 2000` is the sealed/E5 setting and is the anchor.

**Tasks:** 8 held-out programs per (world, depth), drawn from a stream seeded by
`(world, depth)` alone, so the same programs are used at every `K` and every arm.
Support 128 examples, query disjoint. Query labels never enter selection.

| arm | what it is | run at |
|---|---|---|
| **O** oracle | teacher program through the E0.1 assignment | every depth |
| **OPT** | support-only route optimization, `K` steps | every depth, every `K` |
| **ENUM** | exhaustive support-scored search over `slots^D` | only where `slots^D <= 250,000`, i.e. `D <= 4` |
| **S** scratch | fresh library and route, same `K` budget | every depth, `K = 2000` |

`ENUM` is not an omission at `D >= 5`; its cost is reported as the number it
would take (`12^5 = 248,832`, `12^6 = 2,985,984`, ... `12^10 ~ 6.2e10`).

# Eligibility, per depth

Registered, and the same form as E5 Amendment 1 and E1.0: a depth is ELIGIBLE
iff the oracle beats scratch by `>= 0.75` log units in `>= 2 of 3` worlds. An
ineligible depth is reported as UNINTERPRETABLE for the search question and no
search verdict is read from it, because a degraded executor and a failed search
are not distinguishable there.

# Estimands

Registered before any code, all in log NMSE, all with the 2-of-3-worlds rule.

    D_execute = the smallest swept depth that is INELIGIBLE
    D_search  = the smallest ELIGIBLE depth at which OPT at K = 2000 has an
                oracle gap > +0.15
    K*(D)     = the smallest swept K whose oracle gap at depth D is <= +0.15
                (reported as ">2000" if none, and only for eligible depths)

Secondary, reported and never decisive: device-seconds and executions per cell;
the support-loss reduction achieved by OPT; ENUM-versus-OPT where both run; and
exact-route agreement with the teacher, which the gauge results have made
uninformative about use.

# Decision rules

- **EXECUTION BINDS FIRST** iff `D_execute < D_search` (including the case where
  `D_search` does not exist within the sweep). The substrate's limit is
  compositional fidelity, not findability, and E6's `V_find` term is worth
  little at any reachable depth.
- **SEARCH BINDS FIRST** iff `D_search < D_execute`. The V1-era assumption is
  vindicated, "smoothly searchable" is scoped to shallow programs, and E6's
  search-economy currency becomes primary rather than secondary.
- **NEITHER BINDS** if the sweep ends with both horizons unreached. Reported as
  a lower bound (`D_execute > 10`, `D_search > 10`), not as a null.

**`K*(D)` decides the shape of the finding**, and is registered separately so it
cannot be read post hoc: if `K*` is FLAT in `D` while the space grows by ~10^7
between `D = 3` and `D = 10`, the language is smoothly searchable in the strong
sense; if `K*` grows with `D` but stays below the budget, findability degrades
gracefully; if `K*` exceeds 2,000 at some eligible depth, that depth IS
`D_search`.

# Non-vacuity, all required before any cell is read

1. **The executor is the shipped one.** At `D = 3` the variable-depth executor
   must reproduce the shipped artifact's predictions bitwise, per world.
2. **OPT actually optimizes.** Support-loss reduction must be `> 0` in every
   scored cell; a cell that fails is reported, not silently averaged in.
3. **The budget axis is not vacuous.** `K = 250` and `K = 2000` must differ
   materially at some eligible depth. If they never do, the optimizer had
   converged before the smallest budget and `K*` is reported as `<= 250`
   throughout rather than as a measured curve.
4. **The depth axis is not vacuous.** Oracle error must increase with depth,
   which the sealed drift law predicts; if it does not, the depth sweep is not
   varying what we think it varies.

# Registered predictions

**Ours.** `EXECUTION BINDS FIRST`. From the sealed drift `b = 0.581` and the
depth-4 oracle error `~0.006`, execution error grows as
`e_D ~ 0.006 exp(b(D - 4))`, giving `e_8 ~ 0.061` and `e_9 ~ 0.11` against a
scratch band of `~0.05-0.13` - so eligibility should fail around `D = 8-9`. We
predict `D_execute in {8, 9}` and `D_search` does not exist within the sweep. We
further predict `K*` is FLAT or nearly flat in `D`, because E5 already showed
`K = 2000` sufficed at both 1,728 and 2.99M programs; and we predict `ENUM` beats
`OPT` on cost at `D = 3` and `D = 4` and is already the wrong tool by `D = 5`.

This is the same forecast form that worked for `e_6 ~ 0.019`, applied one level
further out, and it is registered WITH the eligibility gate that makes a wrong
forecast legible rather than silent.

**Review 81's.** Expects the sweep to be worth doing precisely because the
1,728x growth produced no visible degradation, and flags `D_execute` versus
`D_search` as "a major architecture lesson" either way. It does not commit to a
direction, but notes current evidence suggests execution binds first.

# Cost

No lifetimes and no new worlds. 3 worlds x 8 depths x 8 tasks, with 4 budgets
for OPT and single runs for O, S and (where feasible) ENUM. The 2,000-step arm
dominates at ~16 s/task; the whole sweep is a few hours in the background behind
a protocol-fingerprinted per-cell cache, and stops early if `D_execute` is
reached below 10.

# Out of scope

Any change to a sealed verdict; any retraining of a library; the recognizer,
which review 81 postpones until search is expensive; and E6, which is drafted
AFTER this result so that its treatment of `V_find` is decided by measurement
rather than by assumption.
