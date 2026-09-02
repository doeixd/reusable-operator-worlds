# Post-E6 research program: semantic structure and learnable computation

Status: DRAFT PROGRAM-LEVEL DECISION TREE. This document orders existing and
future subplans; it is not itself an executable preregistration. Every new rung
requires its own frozen protocol, code commit, `tools/check_prereg.py` entry,
tests, and validated artifact before producing evidence. Development worlds
0–2 are used for construction and first decisions; no new sealed seed band is
allocated here.

This plan was written on 2026-09-02 after the incomplete
`reports/rotated_g5r_interference.json` contained all three `C_lo` cells. Those
cells had already been inspected. Their values are therefore an observed design
input, not a prospective prediction, and no threshold or classification in the
frozen `G5R_INTERFERENCE_PLAN.md` is changed here.

# Program thesis

The recent results localize the open problem more sharply than “does reuse
work?” Frozen learned operators already export, compose in unseen programs,
execute at unseen lengths, quantize, and support program-structured search. The
remaining bottleneck is the acquisition and organization of reusable
computation:

> Can an online learner form stable computational fillers, preserve their
> identity across structural contexts, bind them into explicit schemas, and do
> so at a lifetime cost that remains competitive?

The candidate architecture separates five responsibilities:

    continuous learning       forms functional operators / fillers
    semantic factorization    stabilizes identity across contexts
    typed combinators         store program structure
    program writer            selects fillers, roles, and schemas
    economic lifecycle        decides what to retain or create

No rung is allowed to make all five claims at once.

# Evidentiary starting point

The program inherits completed results without reopening them:

1. **Execution is real.** E1, E2, and E8 established frozen export, unseen
   composition/position use, and variable-length execution on the ordinary
   exact-reuse substrate.
2. **Search is not yet the dominant combinatorial barrier.** E5.1 found
   `C_find` approximately linear in program length and logarithmic in nominal
   program-space size over its tested range. E6B causally reduced search cost by
   shortening a paired program.
3. **Literal semantic locality degrades with depth.** E6's planted learner-gram
   survival fell from 91.15% to 52.60%, 13.02%, and 7.29% over depths
   4, 6, 8, and 10, while deep routes became attractor-like.
4. **Same-sized neural compilation is the wrong macro.** E6.2 could not fit a
   three-operation composition into one ordinary slot in 2 of 3 worlds. Larger
   compiled operators fit but did not amortize.
5. **Fast arguments matter.** H39 showed that a shared schema becomes fertile
   when variation has an explicit task-local argument channel.
6. **Control flow needs stronger operators.** Ordinary near-identity operators
   made both branch choice and repeat count behaviorally unnecessary. Rotated
   operators repaired the opportunity gate.
7. **Strength created an acquisition problem.** G5/G5R failed online. The G5R
   diagnosis established exact representability and isolated operator
   findability; oracle-route joint learning passed only at the much larger
   Stage-C budget and only in 2 of 3 worlds.

The current incomplete Stage-D artifact adds a provisional design observation:
`C_lo`, offline IID learning with oracle routes at the lifetime budget, has
terminal median NMSE 0.900, 0.790, and 0.897 and fails 0/3. This is not accepted
as a result until the frozen run completes and validates. If it does, the
registered Stage-D ladder returns `BUDGET_LIMITED` regardless of the remaining
descriptive cells.

# The horizons to keep separate

Future work reports a vector of horizons rather than one “compositionality”
boundary:

- `D_execute`: greatest depth at which frozen operators execute accurately;
- `D_semantic`: greatest depth at which local computational identity remains
  recoverable and non-collapsed;
- `D_search`: greatest depth at which the registered writer/search budget finds
  an adequate program;
- `D_form`: greatest operator strength/complexity learnable under the registered
  online lifetime budget;
- `H_amortize`: reuse horizon at which a representation or schema repays its
  description cost.

Every horizon requires either persistent failure over subsequent settings or a
fitted monotone trend with uncertainty. A single crossing is reported as one
cell, not named a horizon.

# Ordered program

    Phase 0  Finish and validate G5R Stage D
       |
       +-------------------------+
       |                         |
       v                         v
    Track A                   Track B
    semantic identity         strong-operator acquisition
    RF0 -> RF1 -> RF2         SO0 -> conditional SO1 -> SO2
       |                         |
       +------------+------------+
                    v
              Convergence gate
                    |
           +--------+---------+
           |                  |
           v                  v
       C1 CHAIN/COMPOSE    C2 IF/REPEAT
       ordinary substrate strong substrate only
           |                  |
           +--------+---------+
                    v
              role discovery

Track A and Track B answer independent questions. They may share analysis code
but never share adaptive thresholds. Only one compute-heavy writer runs at a
time on the local Windows host.

# Phase 0 — close the in-flight G5R Stage D diagnostic

The first action is completion, not redesign.

1. Run the remaining `L_lo`, `L_hi`, and `O_or` cells exactly as frozen.
2. Accept no classification until `complete=true`, all expected cells exist,
   anchors/pinning/reload/finite checks pass, the report is fresh, and the full
   validation suite passes.
3. Append the result to `PREDICTIONS.md`, `notes/learnings.txt`, and
   `PROGRESS.md`; commit it before starting a successor experiment.
4. Do not rerun or tune a Stage-D cell because its intermediate value is
   disappointing.

The frozen classification determines Track B:

    BUDGET_LIMITED
        -> measure the acquisition dose-response before changing routing
    ONLINE_INTERFERENCE
        -> repair task ordering/replay while holding routes and budget fixed
    ROUTE_INFERENCE
        -> repair routing while holding schedule and budget fixed
    BOTH_INDEPENDENTLY_SUFFICIENT
        -> treat routing and schedule as separate required interventions
    INTERACTION_ONLY
        -> test a joint curriculum; neither marginal intervention is enough

The remaining Stage-D cells are scientifically useful even if `C_lo` fixes the
primary label: they determine whether route learning becomes feasible at the
large budget and quantify terminal forgetting.

# Track A — semantic identity and role–filler factorization

Track A is governed by `RF_ROLE_FILLER_PLAN.md`. This program does not alter its
metrics, predictions, or gates.

## A0 — RF0 existing-artifact census

Reconstruct the E6 routes and common-state functional traces. Fit role–filler
factorizations without teacher labels and score the frozen, collapse-safe
contrast:

    S_adj    = P(code agrees | same operation, different contexts)
             - P(code agrees | different operations, matched contexts)
    Delta_RF = S_adj(filler) - S_adj(raw)

RF0 passes only at the registered deep-depth materiality margin with synthetic,
same-author, role-shuffled, held-out-reconstruction, and non-collapse checks.
Cost: minutes to hours, no lifetime.

## A1 — RF1 unseen binding

Run only after RF0 support. Hold out complete filler-role combinations and
insert constructed `B(F_i,R_j*)` bindings into held-out programs. This is the
first rung that can distinguish a recombinable representation from post-hoc
clustering.

## A2 — RF2 matched-budget learner

Run only after RF1 passes. Compare flat and explicitly factorized online
learners under parameter-, retained-bit-, and compute-matched controls. Preserve
identical worlds, tasks, orders, examples, replay, route budgets, and evaluation
sets. RF2 is the first rung that can claim explicit factorization CAUSED a
larger `D_semantic`.

## Track-A stop rules

- RF0 negative: stop role–filler factorization as an explanation of E6. Do not
  build RF1/RF2 on the same substrate.
- RF0 unresolved: diagnose the instrument once. A second material redesign
  requires a new plan, not another amendment.
- RF1 negative: retain RF0 as descriptive structure only; do not call it a
  grammar or systematic representation.
- RF2 negative: factorization exists and recombines but does not earn its online
  cost. Do not promote it architecturally.

# Track B — make strong computation learnable

Track B is conditional on the validated Stage-D classification. It never uses
teacher routes in a non-oracle model, and it separates existence, acquisition,
and online retention.

## B0 — SO0 acquisition census

Before a new lifetime, create a read-only report from G5R Stages B–D containing:

- isolated-operator loss versus update and example-gradient count;
- joint oracle-route loss versus updates, batch size, task diversity, and total
  example-gradients;
- per-world transition location and whether the transition persists;
- terminal versus end-of-task error for online arms;
- orthogonality, operator usage, and gradient-norm diagnostics already present
  in artifacts.

SO0 exists to choose the manipulated variable. It cannot establish causality.
World 0 is retained as a named dissenting world, never averaged away.

## B1 — SO1 conditional mechanism experiment

Freeze one plan after Stage D and SO0, using the corresponding branch:

### If Stage D is `BUDGET_LIMITED`

Run oracle-route, offline-IID cells first. Bracket a persistent acquisition
crossing while separately varying:

- optimizer update count;
- examples per update / task diversity in a batch;
- total example-gradients.

The design must contain matched pairs that isolate at least two of these axes;
`4096×64` versus `8192×2` alone is insufficient because it moves all three.
Fit a monotone response surface or run a staged bracket, and report the lowest
tested resource envelope that passes `median query NMSE <= 0.05` in at least
2 of 3 worlds. If no tested cell passes, report a lower bound rather than extend
the grid post hoc.

Only after an oracle-route cell passes may the same resource envelope be used
with learned routes. This preserves existence-before-discovery.

### If Stage D isolates online interference

Hold total updates, example-gradients, routes, and optimizer fixed. Compare the
frozen online schedule with controlled interleaving/replay schedules. The causal
estimand is paired terminal-minus-end-of-task degradation and terminal query
NMSE. A schedule that simply sees more data is a budget intervention, not an
interference intervention.

### If Stage D isolates route inference

Hold data schedule, update count, batch, library architecture, and optimizer
fixed. Compare hard oracle routes, soft learned routes, posterior/beam routing,
and an oracle-to-learned curriculum whose teacher exposure ends at a frozen
time. Query labels never choose routes. The curriculum is labelled
teacher-assisted and cannot become the final non-oracle claim.

### If both factors or only their interaction bind

Use the smallest factorial design that reproduces each marginal result and adds
one joint intervention. Do not tune routing in the schedule arm or replay in the
routing arm.

## B2 — SO2 online strong-substrate gate

After SO1 identifies a mechanism, test one prospectively selected training
protocol in the original online lifetime with no oracle route exposure. It must:

- score before update and preserve the canonical train/evaluation separation;
- beat matched scratch on held-out programs by G5R's `>=0.75` log-NMSE margin
  in at least 2 of 3 worlds;
- achieve terminal median task NMSE `<=0.05` in at least 2 of 3 worlds;
- keep the rotated operator opportunity gates for iteration and branching;
- report total compute, example-gradients, retained bits, and wall/device time
  beside the ordinary substrate.

Passing SO2 licenses “a strong substrate is learnable under this registered
protocol and cost.” It does not license “as learnable as the ordinary
substrate” unless the original comparability gate also passes.

## Track-B stop rules

- No oracle-route cell passes within the frozen resource envelope: report
  `D_form` below the required operator strength and close the current rotated
  parameterization.
- Oracle routes pass but learned routes do not: the substrate exists, but the
  writer cannot acquire it; control flow remains closed.
- Online SO2 fails: do not run branching or iteration, even if offline cells
  are excellent.
- A protocol exceeding its frozen compute ceiling is not a pass. Learnability
  is always stated with its resource envelope.

# Convergence gate

The two tracks do not need to pass simultaneously for their own findings to be
valuable. They do need to pass before the full proposed architecture is tested.

| Track A | Track B | Licensed next step |
|---|---|---|
| fails | fails | close both mechanisms; retain the measured horizons |
| passes | fails | study straight-line semantic structure only; no control flow |
| fails | passes | strong flat language exists; test control flow without claiming factorized semantics |
| passes | passes | test bound fillers inside typed higher-order combinators |

“Passes” means RF1 for recombinable fillers and SO2 for a learnable strong
substrate. RF2 is additionally required before claiming that factorization is
economically preferable during online formation.

# Track C — typed structural combinators

Typed combinators store structure; learned fillers store computation. This is
not another compiled-macro capacity ladder.

## C0 — typed IR and executor equivalence

Implement the smallest typed representation capable of:

    APPLY(function, argument)
    CHAIN_3(first, second, third)

The executor's equivalence to the corresponding flat expansion is true by
definition and is an implementation check only. Record schema identity,
argument ordering, types, expansion, retained bits, and execution operations in
the artifact fingerprint.

## C1 — CHAIN/COMPOSE on the ordinary substrate

C1 may run after Track A reaches RF1; it does not require the rotated substrate.
Hold out filler tuples, argument positions, depths, and whole schemas from the
writer's training set. Compare a flat program writer with a schema-aware writer
under matched task-code and search budgets.

Evidence must come from held-out writing/search, semantic locality, or
description economics. The typed executor reproducing its registered expansion
is not evidence. Report `D*`, `C_find`, execution operations, and predictive loss
separately; do not add bits, seconds, and errors without a frozen exchange rate.

## C2 — IF/REPEAT on the strong substrate

C2 requires SO2. Re-run the behavioral opportunity gates on the exact SO2
artifact before training a writer:

    IF      no fixed branch may approximate the conditional task
    REPEAT  no fixed count may approximate the count-varying task

Then test fixed typed schemas first:

    IF(predicate, true_branch, false_branch)
    REPEAT(count, body)

Each filler and scalar argument must appear in multiple training roles; complete
filler-role and structural combinations are held out. Controls include wrong
predicate, swapped branches, wrong count, wrong body, flat depth-matched
programs, and a schema-free writer with matched budget.

The first control-flow claim is limited to systematic use of supplied schemas.
Schema discovery, nesting, recursion, `MAP`, and `FOLD` require later plans.

## C3 — role and schema discovery

Run only after supplied roles and schemas succeed. Learn a role/schema library
with fixed capacity and explicit description charges. Compare against fixed
correct, role-permuted, collapsed-role, and matched unstructured controls.
Discovery is scored by held-out execution and economics, not agreement with
teacher names or attractive clusters.

# Metrics and accounting contract

Every executable subplan reports, as applicable:

- cumulative prequential Gaussian log loss, per online example and per target
  scalar;
- query NMSE and raw MSE with the evaluation denominator fixed across compared
  conditions;
- common-state functional distance;
- `S_adj`, raw/filler motif survival, and collapse diagnostics;
- support examples, optimizer updates, example-gradients, batch task diversity,
  forward multiply-adds, and device/wall time;
- shared parameters, task-local parameters, retained quantized bits, schema
  bits, and program bits;
- end-of-task and terminal metrics where interference is possible.

Description, search, execution, and predictive accuracy are separate currencies.
A scalar objective is permitted only when its exchange rates are frozen before
the relevant data exist and the verdict is insensitive across a registered
range.

# Pairing, provenance, and artifact rules

1. Compared learners receive identical worlds, task order, examples, replay,
   evaluation sets, and model/world seed schedules.
2. Every online example is scored before it is used for updating.
3. Teacher programs, primitive identities, and oracle routes remain inaccessible
   to non-oracle learners.
4. Function comparisons use common on-trajectory states. Parameter coordinates
   are never treated as functional identity.
5. Every cell records the complete arm construction: initialization source,
   checkpoint hash, trainable/frozen counts, optimizer groups, steps, examples,
   and information seen.
6. Resume validates the full resolved protocol fingerprint and fails closed on
   mismatch. One writer owns a cell.
7. Results require expected cell counts, finite metrics and tensors, fresh
   artifacts, clean committed launch code, exit code 0, prereg/invalid checks,
   pairing checks, and the full test suite.
8. Completed milestones are committed. Negative, invalid, withdrawn, and
   unresolved outcomes are appended rather than rewritten.

# Prediction ledger and contamination statement

Existing registered predictions remain controlling:

- RF0: review 84 predicts strong recovery; the project predicts a positive
  depth-8 effect but failure of the strong depth-8-and-10 gate.
- G5R Stage D: its frozen predictions and decision ladder remain unchanged.

No new blind prediction is registered here for Stage D or for the location of
the strong-operator budget crossing. The three `C_lo` endpoints were visible
before this document. A future SO1 protocol may make prospective predictions
only about cells that do not yet exist, after Stage D and SO0 are closed.

Program-level expectations, not executable verdicts:

1. The ordinary substrate is likely to retain a higher `D_execute` than
   `D_semantic`; RF0 decides whether the gap contains recoverable structure.
2. Strong-operator formation likely has a sharp resource/task-diversity
   transition rather than a smooth small degradation, based on the already
   observed Stage-C trajectories. The crossing and its cause remain unknown.
3. Structural schemas are more capacity-plausible than same-sized compiled
   macros because they retain constituent fillers, but their execution
   equivalence alone will provide no scientific evidence.
4. The full architecture is unlikely to succeed through one end-to-end loss
   without staged formation, binding, and writing. This remains a design prior,
   not a registered result.

# Deliverables and commit sequence

1. **G5R-D closure commit:** complete report, scorer output, predictions,
   learnings, progress, and validation record.
2. **RF0 freeze commit:** exact split/grid/null/scorer plus tests and prereg hash.
3. **RF0 result commit:** report and honest branch decision.
4. **SO0/SO1 plan commit:** only after the G5R-D classification; include the
   contamination statement and resource ceiling.
5. **SO1 result and SO2 plan commits:** separate mechanism from online proof.
6. **RF1/RF2 commits:** only along Track A's gates.
7. **Typed-IR commit:** implementation checks only, no scientific claim.
8. **C1/C2 protocol and result commits:** only after their prerequisites.

No commit combines a frozen plan with the result it governs.

# Immediate next actions

1. Finish G5R Stage D without modifying its frozen design.
2. Record and commit the validated Stage-D classification.
3. Freeze and implement RF0, the cheapest remaining causal census.
4. Build SO0 from completed artifacts and draft the classification-specific SO1
   resource/interference/routing protocol.
5. Do not implement RF2, a new strong learner, or typed control flow until the
   corresponding existence gates pass.

# Explicitly out of scope

No new sealed worlds; no post-hoc rescue of G5R; no concurrent full lifetimes on
the local host; no teacher-supervised non-oracle factorization; no parameter
matching as semantic evidence; no same-sized compiled-macro retry; no claim that
a definitional schema learned its own execution rule; no loop or branch learner
without a fresh opportunity check on a learnable strong artifact; no unpriced
capacity or compute; and no single “compositionality” score that hides which
horizon failed.
