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
