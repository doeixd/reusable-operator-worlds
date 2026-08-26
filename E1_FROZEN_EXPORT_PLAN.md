# E1: does a frozen library execute a program it never trained on?

Status: DRAFT (freeze commit in `tools/check_prereg.py` before any code or any
new lifetime). Governed by `EXPORT_BRANCH_PROGRAM.md` (Amendments 1-2) and
`E0_PHASE0_AUDIT_PLAN.md`; this rung may not alter their decision trees,
thresholds, or terminology contract.

# The question, as Phase 0 left it

Phase 0 settled that task identity is carried by the routed library, not by
per-task patches — architecturally so on the substrates E1 concerns, since
neither has a residual channel. E1 therefore does not ask whether anything is
worth exporting. It asks:

> Does a library that fully specifies its TRAINED tasks retain stable semantics
> on a program it never trained on?

Live alternatives: **stable reusable operations** versus **a distributed routed
function basis that only works in familiar program contexts.**

# Substrate: DISC only, and why

Only the route-expressible substrate may carry this rung's verdict
(`EXPORT_BRANCH_PROGRAM.md` Amendment 1). E1.0 measured DISC at 1.02 and the
mixture family at 4.53 / 4.06 against a 2.0 gate, so MIX is UNINTERPRETABLE here
and is not run. A failure of E1 on DISC is the branch's STOP; a failure on MIX
would have been a statement about mixture routing.

**Three fresh DISC lifetimes** (worlds 0, 1, 2) are run from current committed
code at `configs/v1.yaml`, `--model discrete`, rather than reusing
`artifacts/discrete/seed_0`. That existing artifact predates learnable operator
scales and current provenance conventions; mixing eras inside one comparison is
the kind of thing this project has had to withdraw before. The old artifact is
retained and reported as a fourth, era-labelled row, never pooled with the three.

Gate re-check, mandatory: E1.0 is recomputed on each new artifact, and any world
whose oracle-route ratio exceeds 2.0 is excluded from the verdict and reported.

# Held-out programs (chosen before any arm is run)

Each world trains on 64 of the 216 depth-3 programs. The held-out set is drawn
from the remaining 152 under constraints checked in code:

- every primitive of a held-out program appeared in training, in that position;
- no held-out program appeared in training;
- support and query examples are disjoint (support = the task's `train_x`,
  query = its `eval_x`);
- selection is by a fixed seed, never by performance;
- **`H1` and `H2` are reported separately** — E2-feas showed the two strata
  trade off directly, so pooling them would hide which kind of novelty a result
  came from. 24 programs per world: the first 12 of each stratum by seeded
  order, or all of a stratum if fewer than 12 exist (reported).

# Arms

The interface is **E1-P** throughout (Amendment 2): frozen library, only the
route inferred. DISC has no private residual channel, so E1-PR and E1-R are
UNDEFINED on this substrate and are recorded as such rather than run.

| arm | library | route | budget |
|---|---|---|---|
| **O** oracle program | frozen | teacher program through the E0.1 functional assignment | none |
| **R** route inference | frozen | inferred from support only | `S_ADAPT` steps |
| **S** scratch | fresh random init, TRAINABLE | inferred | `S_ADAPT` steps |
| **F** full finetune | trainable | inferred | `S_ADAPT` steps |
| **D** dense frozen | frozen dense learner, same world | its task-local interface | `S_ADAPT` steps |
| **W** wrong library | frozen, slots permuted by a seeded derangement | inferred | `S_ADAPT` steps |

`S_ADAPT = 2000`, Adam, lr 0.01 — the project's standard task-local fit, the
same as the H49 re-fit and Phase 0's refit, so no new adaptation protocol is
introduced. Support = 128 examples (the task's own training set); query loss is
NMSE on the 256 held-out evaluation examples. No query label ever enters
selection or early stopping.

**D** is run only where a dense artifact exists for the same world and config
family; worlds without one report `D = absent` rather than substituting a
different era's artifact.

# Primary quantities

    G_export = (L_S - L_R) / (L_S - L_O)     reported only when L_S - L_O > 0
                                              and L_S > 2 * L_O
    C_repair = L_R - L_F

with raw `L_O, L_R, L_S, L_F, L_D, L_W` always beside them, per stratum, in
absolute NMSE. Geometric mean over held-out programs within a world; worlds are
the unit of replication.

# Decision rules

Registered before any arm runs. Margins are in log NMSE, and "in >= 2 of 3
worlds" is the replication requirement throughout.

- **E1a — VOCABULARY EXPORTS.** `L_O` beats `L_S`, `L_D` and `L_W` by
  >= 0.15 log units in >= 2 of 3 worlds. The frozen library contains
  computation that transfers to a program it never trained on.
- **E1b — THE ROUTE IS FINDABLE.** `L_R` beats `L_S` by >= 0.15 and
  `log L_R - log L_O <= 0.15` in >= 2 of 3 worlds.
- **E1a passes, E1b fails** — vocabulary exists, the writer/search is missing.
  Registered successor: E3/E5, and the library is NOT changed.
- **E1a fails** — the branch STOP. Licensed sentence, exactly: "ROW learned
  economically useful shared computation, but these objects are not exportable
  program primitives." The successor is E9 (export-constrained formation),
  already registered with its trigger.

Non-vacuity, all required before any verdict:

- every arm's adaptation reduces its own support loss by > 1%;
- `L_W` and `L_S` are finite and worse than `L_O` (if the wrong library matches
  the right one, the assignment is not doing the work the claim needs);
- the E1.0 ratio on each new artifact is <= 2.0;
- held-out programs are verified absent from the training set in code;
- `H1` and `H2` rows are both present, or the missing stratum is reported.

# Registered predictions

Ours. **E1a passes**: Phase 0's oracle-route ratio of 1.02 on TRAINED programs
says the discrete library executes teacher programs through matched slots
almost losslessly, and there is no obvious mechanism by which that would hold
for 64 programs and collapse on the other 152 built from the same six
primitives. We expect `L_O` close to trained-task NMSE (~0.010) and clearly
better than scratch. **E1b is the uncertain half** and we expect it to FAIL or
to sit near the boundary: V2 established that online route inference is hard,
and DISC's own lifetime showed weak online routing beside strong operator
recovery. Modal outcome therefore: **vocabulary exports, route inference lags** —
which is also review 75's modal prediction. On strata we predict `H2` (pair-novel)
to be no worse than `H1` for the ORACLE arm (the library executes primitives, and
adjacency is a property of the program, not of the objects) but worse for the
INFERENCE arm, since route search has less local context to exploit. If `H2`
fails for the oracle arm too, the objects' semantics are context-dependent and
the successor question is the operator interface.

Review 75's: reasonable chance the oracle passes at high reuse; lower confidence
in route inference; a clean E1a failure would be an important negative.

# Cost

Three discrete lifetimes (~15 min each, bounded pool). E1 scoring is
6 arms x 24 programs x 3 worlds x 2000 steps, plus the E1.0 re-check —
comparable to two H49 cells, a few hours in the background, resumable through a
protocol-fingerprinted per-cell cache.

# Explicitly out of scope

E2's constructed world; the program recognizer; primitive invention; any change
to the library; any claim from the MIX substrate; depth generalization.


# Amendment 1 (2026-08-25, before any E1 arm is run): a vacuous control and an unavailable one

Found by reading the frozen arm table against what each arm can actually
measure, before writing the scorer.

## 1. The wrong-library control cannot be a slot permutation

The plan defines **W** as "frozen, slots permuted by a seeded derangement,
route inferred". Under FREE ROUTE INFERENCE a permutation of slot indices is
vacuous: the set of available operators is unchanged, and inference simply finds
the permuted index. The control would measure nothing and would pass trivially.

Amended, before any arm runs. The wrong-library control is a library from an
INCOMPATIBLE WORLD — review 75's own alternative — and it is applied to both
interfaces:

    O-W   teacher program executed through ANOTHER world's frozen library,
          using that library's own E0.1 functional assignment
    R-W   route inferred, with the SAME budget, over another world's frozen
          library

The donor world is fixed by a seeded rotation (world `w` receives world
`(w + 1) mod 3`'s library), recorded per cell. This asks the question the claim
needs — is it THIS library that carries the program, or would any trained
library of the same architecture do — rather than testing index bookkeeping.

The slot-permutation control is retained only for the ORACLE arm, where it is
NOT vacuous (a permuted assignment sends each program step to the wrong
operator); it is exactly the `shuffled_library_route` figure E1.0 already
reports, and is cited from there rather than recomputed.

## 2. The dense control is not available in every world

`D` requires a compute-matched dense artifact from the same world and config
family. At exact reuse those exist for worlds 1 and 2
(`artifacts/rho_development/rho_1/world_*/dense`) and not for world 0, and
`E1_FROZEN_EXPORT_PLAN.md` forbids substituting another era's artifact.

Amended: **E1a's threshold is stated over the two controls that exist in every
world** — scratch (`S`, no library at all) and wrong library (`O-W` / `R-W`).
`D` is reported wherever an artifact exists, as an additional reference, and is
never part of the criterion. This keeps the decision rule well defined in every
world instead of silently varying with artifact availability.

Restated E1a, superseding the arm list in the body: `L_O` beats `L_S` and
`L_O-W` by >= 0.15 log units in >= 2 of 3 worlds. E1b is unchanged.

## 3. Gate re-check, completed

Run before this amendment and reported here because it is a precondition, not a
result: the three fresh DISC lifetimes score E1.0 ratios of 1.17 / 1.09 / 1.20
against the 2.0 gate, with random-assignment and shuffled-library controls at
7.1-13.4x, and all three pass E0.1 substitutability (matched 0.040-0.072 against
a null edit of 1.000). All three worlds are eligible; none is excluded.


# Amendment 2 (2026-08-25, after a first pass, BEFORE any verdict is recorded): a mode-mixed diagnostic, and what the non-vacuity clause actually asks

A first full pass completed and its verdict is NOT recorded. Two problems were
found by reading the per-cell diagnostics, and both are fixed before any number
from this rung enters the ledger.

## 1. The support-reduction diagnostic mixed model modes

`DiscreteLibraryLearner` is relaxed-in-training and hard-at-evaluation by
design: `train()` routes with a softmax at temperature, `eval()` routes by
argmax. The first implementation measured the INITIAL support loss in train mode
and the FINAL support loss in eval mode, so the reported reduction mixed two
different routing rules and is not a measurement of anything. The reported query
NMSEs are unaffected — every one of them is computed in eval mode, consistently,
for every arm — but the non-vacuity statistic built on the mixed pair is void.

Fixed: each adapting arm now reports TWO reductions, both mode-consistent:

    support_reduction_objective   train mode at both endpoints — did the
                                  optimizer reduce THE LOSS IT WAS MINIMISING
    support_reduction_eval        eval mode at both endpoints — did the
                                  hard-routed prediction improve

## 2. The clause as written conflates "adapted" with "improved"

The frozen clause reads "every arm's adaptation reduces its own support loss by
> 1%". Its purpose is to prove each adapting arm actually optimized, so that a
comparison is not between two un-adapted models. But two arms are EXPECTED not
to improve, and their failure to improve is the instrument working:

- **R-W** (wrong library) — a library from an incompatible world should not fit
  the task. Measured: -0.026 support change. That is the control succeeding.
- **F** (full finetune) — at the registered budget (2,000 Adam steps at lr 0.01
  on one task's 128 support examples) training the whole library DEGRADES it
  badly (-2.1, i.e. support loss roughly tripled). This is a real property of
  the budget, not a bug, and it means `C_repair = L_R - L_F` is NEGATIVE: at
  this budget finetuning does not repair, it destroys. `C_repair` is reported as
  measured and is NOT interpreted as repair. No registered threshold depends on
  it.

Registered reading, fixed now rather than argued after the verdict: the
non-vacuity clause is satisfied when **`support_reduction_objective > 0.01` for
every arm whose claim depends on having adapted** — that is `R` and `S`, the two
arms entering E1a and E1b. `R-W` and `F` report their numbers and are exempt,
because an arm designed to fail is not evidence that the instrument failed. Both
exemptions are stated in the report beside the numbers rather than hidden in the
pass/fail flag.

This is a correction to a clause of our own drafting, made before its data was
read into a verdict, and the first pass is preserved as an instrument dry run.
