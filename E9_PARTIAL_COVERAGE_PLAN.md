
# E9: partial coverage, and whether `p + alpha + eps` closes the gap

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract. Source is
`notes/e7-sketch.txt` sections 10-11. **Development worlds 0-2.** Sealed bands are
spent and are not reused.

# Why a new generator is needed at all

E8D and its coverage reframe both died on the same measured fact: across E1's
held-out programs and all nine E1-R recurrence cells, **route-only inference is
within +/-0.07 log units of the best available arm in 8 of 9 cells**. Where the
library covers a task, `p` reaches the oracle; where it does not, nothing reaches
anything (at `rho = 0` the oracle, the inferred route and a from-scratch learner
all sit at ~0.038).

    `p` IS SUFFICIENT WHEREVER ANYTHING IS SUFFICIENT.

The existing generator makes tasks that are IN-LANGUAGE or OUT-OF-LANGUAGE with
nothing between, so the decomposition question cannot be asked on it. E9 builds
the missing middle.

# The generator, and its exact reduction

A task's teacher program is built from the world's primitives as before, and then
ONE step's operator is PERTURBED by a controlled amount `delta` in a controlled
direction. Everything else -- world seeds, program sampling, example generation,
opaque IDs -- is unchanged.

**This is an EXTENSION with an exact reduction, not a replacement.** At
`delta = 0` the construction must reproduce the existing tasks BITWISE, per
world, and that equivalence is asserted before anything is scored. This is what
keeps E9 comparable to E1/E5/E6 rather than breaking every artifact, and it is
the control the project's own rule about generator changes demands.

## Two perturbation directions, chosen to be interpretable

The argument channel modifies a slot's `U` matrix and nothing else:
`P(alpha)(z) = tanh(z + a . (U_0 + sum_k alpha_k U_k) tanh(V z + b))`.

    delta_U   perturb `U` WITHIN span{U_k}      -- alpha can express it exactly
    delta_V   perturb `V`                        -- alpha cannot express it at all

These are deliberately a matched pair: one novelty the argument channel is built
to represent, one it structurally cannot. Both are swept over the same `delta`
grid and the perturbation's realized functional magnitude is reported per cell,
so the two conditions are compared at equal DIFFICULTY rather than at equal
nominal `delta`.

# Arms

Frozen library, adaptation-time surgery, support only, query labels never used.

| arm | fitted | role |
|---|---|---|
| **P** | route only | the export branch's existing inference |
| **P+A** | route + `alpha` (K = 16 on slot 11) | identify + argument |
| **P+A+E** | route + `alpha` + `eps` | the full three-term claim |
| **A-RAND** | route + `alpha`, `U_k` frozen at random init | matched-budget control |
| **CEILING** | the TRUE perturbed operator substituted in | achievable if you knew the answer |

`CEILING` is the generating function itself, not a fitted arm. E8D's withdrawal
established that library fine-tuning is NOT a ceiling -- it is ~10x worse than
route inference because it overfits 128 support examples -- so the ceiling here is
the oracle-with-the-true-perturbation, which is exactly achievable performance
under full knowledge.

# Estimand

Per cell, in log query NMSE:

    share(X) = [ log L(P) - log L(X) ] / [ log L(P) - log L(CEILING) ]

**PRIMARY: `share(P+A+E)` -- does the full decomposition CLOSE the gap?** This is
the adequacy question and it is not an identity: the three terms together may
plateau below the ceiling, leaving a residual none of them can reach, and that is
the outcome that would matter most.

Secondary, reported per direction: `share(P+A)`, `share(A-RAND)`, and the
marginal `share(P+A+E) - share(P+A)`.

# Identity checks, declared in advance

Four estimands have been retired in this branch for being unable to fail. Checked
here before freezing.

- **`delta_U` is a POSITIVE CONTROL, not evidence.** `alpha` can represent that
  perturbation by construction, so `share(P+A)` being high there tests the
  implementation. It is reported as a control and may not be cited as support for
  the thesis.
- **`delta_V` being uncapturable by `alpha` is likewise structural.** The
  empirical content there is whether `eps` closes it and whether the total does.
- **Nested ordering is arithmetic.** `L(P+A+E) <= L(P+A) <= L(P)` holds by
  capacity; "adding a channel helps" is an implementation check.
- **What CAN fail, and is therefore the primary:** whether `p + alpha + eps`
  reaches the ceiling at all, in either direction, at any `delta`.

# Non-vacuity, all required

1. **Exact reduction.** At `delta = 0`, tasks and all arm outputs reproduce the
   existing construction bitwise, per world.
2. **The gap opens.** `log L(P) - log L(CEILING)` must grow monotonically with
   `delta` and exceed 0.5 at the largest `delta` in >= 2 of 3 worlds. If it does
   not, the perturbation is not creating partial coverage and the rung is
   UNSCOREABLE -- the E8D failure, now a gate.
3. **The two directions are matched on difficulty**, not on nominal `delta`:
   realized functional perturbation magnitude reported per cell, and the
   comparison made at matched magnitude.
4. **Every channel moves.** `||alpha||` and `||eps||` materially nonzero in every
   scored cell.
5. **`A-RAND` is matched** in parameter count, optimizer and budget, asserted via
   `src/row/arm_provenance.py`.

# Decision rules

Registered before any code; 2 of 3 worlds, at matched realized perturbation.

- **DECOMPOSITION ADEQUATE** iff `share(P+A+E) >= 0.8` in BOTH directions.
- **ARGUMENT CARRIES STRUCTURED NOVELTY** iff, additionally, `share(P+A) >= 0.7`
  under `delta_U` while `P+A` beats `A-RAND` by `>= 0.15` log units. (The first
  half is the positive control; the `A-RAND` clause is what makes it meaningful.)
- **PATCH-DOMINATED** iff `share(P+A) < 0.3` under `delta_V` while
  `share(P+A+E) >= 0.8` -- novelty outside the basis is real and only the
  unstructured patch reaches it.
- **DECOMPOSITION INCOMPLETE** iff `share(P+A+E) < 0.8` in either direction: the
  three terms together leave a residual, and the thesis is missing a term.

# Registered predictions

**Ours: DECOMPOSITION ADEQUATE under `delta_U`, DECOMPOSITION INCOMPLETE under
`delta_V`.**

Under `delta_U` the perturbation lies in the argument's span and `alpha` should
capture it, with `eps` adding little -- that is the positive control working.

Under `delta_V` we predict the three terms will NOT close the gap. `eps` is a
rank-2 output residual; a perturbation of `V` changes which features the operator
reads, and no low-rank additive correction at the output reproduces a change in
the input projection. We expect `share(P+A+E)` around 0.4-0.7 there, and we
register that as the more informative half: it would say the thesis's third term
is the wrong shape for novelty that is not a coordinate in the existing basis.

We also predict `A-RAND` performs materially worse than `P+A` under `delta_U`
(H39's analogue was essentially unused) and indistinguishably from it under
`delta_V`, where neither can represent the perturbation.

# Downstream re-read, as the generator rule requires

Changing the generator obliges re-reading everything downstream of it. For E9 the
affected objects are: the task builder (`_build_tasks`), the oracle/ceiling
construction, and the equivalence control. No existing artifact is regenerated,
no sealed verdict is touched, and E9 writes to its own report and cache paths.
`delta = 0` is included in every sweep so the reduction is visible in the same
figure as the informative cells.

# Cost

No lifetimes. Five arms x `|delta|` grid x 2 directions x ~12 programs x 3 worlds
at 2,000 adaptation steps, behind a protocol-fingerprinted per-cell cache. Under
a day; the `delta = 0` column is the existing construction and is cheap.

# Out of scope

Amortizing any of these inferences. The closure diagnostic and the
attractor-collapse rung. Any change to a sealed verdict or to an existing
artifact.

# Amendment 1 (2026-08-30, before any E9 code): the registered CEILING is degenerate

Found by reading `_build_tasks` before writing the generator.

## The defect

`CEILING` is registered as "the TRUE perturbed operator substituted in". But task
targets are NOISELESS -- `eval_y = executor.execute(task_library, eval_x)` -- so
executing the true program through the true perturbed library reproduces the
targets EXACTLY. `L(CEILING) = 0`, `log L(CEILING) = -inf`, and therefore

    share(X) = [log L(P) - log L(X)] / [log L(P) - log L(CEILING)] == 0

for every arm at every `delta`. The primary estimand could not have come out any
other way. This is the FIFTH estimand in this branch to be unable to fail (after
E6A's algebraic crossing, E6C's definitional substitutability, E6D's two
arithmetic refusal controls, and E6.2's output-relative tolerance), and the first
to survive into a plan that was actually frozen -- caught one hour later, before
any code.

## Registered correction

`CEILING` is redefined to live in the SAME FITTING REGIME as the arms it bounds:

    CEILING = route inference + a FREELY FITTED full operator at the perturbed
              step (all of U, V and b free), fit on SUPPORT ONLY

This strictly dominates both channels under test -- `alpha` moves `U` within a
16-dimensional subspace, `eps` is a rank-2 output residual, and the ceiling moves
all of `U`, `V` and `b` without constraint -- so it upper-bounds what any
structured correction of one step could achieve from the same data. It is
achievable, finite, and fitted from support like every other arm, so query labels
still never enter.

The true perturbed operator is retained and reported as `GENERATOR` -- the
zero-error reference that says what the task actually is -- and is explicitly NOT
used as a denominator.

## Consequence for the decision rules

The thresholds are unchanged in form and now measure something attainable:
`share(P+A+E) >= 0.8` asks whether the three registered terms recover most of
what an unconstrained one-step correction recovers from the same support. That
can fail, and the failure would be exactly the informative one -- the three terms
being the wrong shape for the novelty.

## A note for the successor

Noiseless targets mean any "oracle" defined as the generating function is a
degenerate ceiling for share-style estimands. Whenever a denominator is a
performance gap, check whether either endpoint is exactly zero BY CONSTRUCTION
before freezing the ratio.

# Amendment 2 (2026-08-30, before any E9 cell was scored): the manipulation was too weak to open the gate

Found in a structural dry run. Both changes below were chosen AFTER measuring
realized target shift in a pilot, and are disclosed as such.

## What the pilot showed

The plan left the `delta` grid and the program-selection rule unspecified. The
first implementation drew held-out programs freely and then perturbed the
most-used primitive, so MOST PROGRAMS DID NOT CONTAIN IT and the manipulation was
diluted. Measured mean relative target shift, world 0, direction `U`:

    delta      programs drawn freely     all programs use the primitive
     0.5              0.0020                      0.0043
     1.0              0.0078                      0.0166
     2.0              0.0329                      0.0692

Route inference's baseline query NMSE on these tasks is ~0.007, so a shift of
0.002-0.004 is at or below the error already present -- the perturbation was
invisible. The dry run confirmed it: at `delta = 0.5` the `P -> CEILING` gap was
`+0.21` in direction `U` and `-0.33` in direction `V`, against a registered gate
of `>= 0.5`.

## Registered corrections

**(1) Every held-out program must CONTAIN the perturbed primitive.** Otherwise
the arms are averaged over tasks the manipulation never touched, which halves the
realized shift and is simply a diluted experiment.

**(2) The delta grid is `(0.0, 0.5, 1.0, 2.0)`**, chosen so the largest value
produces a target shift (~0.07) an order of magnitude above the baseline error,
giving the registered gate a chance to fire. `delta = 0` remains in the grid as
the exact-reduction control.

## What is NOT changed

The gate itself is untouched: the `P -> CEILING` gap must still grow with `delta`
and exceed 0.5 at the largest `delta` in >= 2 of 3 worlds, and the rung is
UNSCOREABLE otherwise. Choosing a grid that CAN clear a threshold is not the same
as lowering the threshold, and the pilot measured only target shift -- a property
of the generator -- never any arm's performance.

If the gate still fails at `delta = 2.0`, the honest reading is that this
perturbation cannot create partial coverage in this substrate, and E9 is reported
as UNSCOREABLE rather than pushed to larger deltas until something moves.

# Amendment 3 (2026-08-30, before any E9 verdict): no fitted arm can be the ceiling

Found in the dry run. This is the THIRD ceiling in this line to fail, and the
three failures share one cause, so the fix is structural rather than another
substitution.

## The measurement

Amendment 1 replaced the degenerate generator-ceiling with "route + slot 11 fully
free (U, V, b), fit on support". The dry run shows it is WORSE than route-only:

    delta = 0     P 0.01737   CEILING 0.02434   gap -0.34
    delta_V = 2   P 0.03577   CEILING 0.05082   gap -0.35

## The cause, and why it is not fixable by choosing a different arm

Targets are NOISELESS and support is 128 examples. Any arm with more free
parameters fits that support better and generalizes worse. E8D's library
fine-tuning failed this way (~10x worse than route inference); Amendment 1's
free operator fails the same way. **In this regime a fitted arm cannot be an
upper bound**, because capacity buys overfitting before it buys reach.

## Registered correction: the denominator becomes a REFERENCE, not an arm

The estimand is redefined as RECOVERY OF DEGRADATION. The perturbation does a
measured amount of damage, and the question is how much of it each channel
repairs:

    degradation   = log L(P at delta) - log L(P at delta = 0)
    recovery(X)   = [ log L(P at delta) - log L(X at delta) ] / degradation

Both endpoints are MEASURED performances of the same arm, so nothing is fitted
into the denominator and nothing can overfit into it. `recovery = 1` means the
channel fully repairs what the perturbation broke; `recovery = 0` means it
repairs nothing. It is undefined at `delta = 0` by construction, where the
control column reports raw values only.

This also states the question more directly than the ceiling ever did: novelty
was INTRODUCED by a known amount, and the thesis says `p + alpha + eps` should
absorb it.

## What is retained

`CEILING` is kept and REPORTED as a diagnostic, because "a freely fitted operator
generalizes worse than a frozen library plus a route" is itself a result about
this regime -- it is the E8D finding reproduced under a different construction.
It is no longer a denominator.

The registered gate is restated in the new currency: `degradation >= 0.5` at the
largest `delta` in >= 2 of 3 worlds, or the rung is UNSCOREABLE. This is the same
requirement as before -- the perturbation must actually break something -- with
the damage measured against the unperturbed control instead of against a fitted
arm.

The decision thresholds carry over unchanged in form, now on `recovery`:
DECOMPOSITION ADEQUATE iff `recovery(P+A+E) >= 0.8` in both directions, and so
on.

## Note for the successor

THREE ceilings failed here: the generating function (exactly zero error, so the
share was identically zero), library fine-tuning, and a freely fitted operator
(both overfit). When targets are noiseless and support is small, define
improvement against a MEASURED REFERENCE POINT -- an unperturbed control, an
earlier checkpoint -- and never against something you fit.

# Amendment 4 (2026-08-30, run KILLED after two cells): the argument channel was on the wrong slot

Found by disbelieving a positive control that returned nothing. The run was
stopped, its cache deleted, and no number from it is reported.

## The defect

`delta_U` is registered as a perturbation "within span{U_k} -- alpha can express
it exactly". The implementation perturbed TEACHER PRIMITIVE `target_primitive`
and parameterized LEARNER SLOT 11. Those are different index spaces, and the
E0.1 assignment says they disagree in 2 of 3 worlds:

    world 0   teacher primitive 4  ->  learner slot 3     alpha was on slot 11
    world 1   teacher primitive 3  ->  learner slot 11    correct by luck
    world 2   teacher primitive 0  ->  learner slot 6     alpha was on slot 11

So in two worlds the argument channel was attached to an operator unrelated to
the perturbation, and `delta_U` was not in its span at all. The symptom was
visible immediately: at `delta_U = 0.5`, `recovery(P+A) = -0.01` -- the positive
control recovered NOTHING -- while the patch recovered 0.40 and a free operator
0.72.

This is `AGENTS.md`'s oldest recurring error: **learner slot indices and teacher
primitive indices live in different spaces**, and any route-agreement or
correspondence claim must map through the functional matching. It cost this
project a silently-zero metric once before.

## Registered correction

The parameterized slot is chosen PER WORLD as `assignment[target_primitive]` --
the learner slot functionally matched to the perturbed teacher primitive -- rather
than a fixed index. The argument basis `U_k` is unchanged; only which slot it is
attached to changes.

## A residual caveat that survives the fix, disclosed rather than hidden

Even with matched indices, the teacher primitive and its matched learner slot are
DIFFERENT MATRICES -- that is the gauge result. Perturbing the teacher's `U` by
`Delta` and letting `alpha` add `Delta` to the learner's `U` are therefore not
identical operations, only approximately so, and the approximation is as good as
the functional matching (measured at ~0.0005 normalized distance in E0.1).

Consequently `delta_U` is registered from here as a NEAR-positive control:
`alpha` should be able to express most of that perturbation, not exactly all of
it. If `recovery(P+A)` under `delta_U` is high, that confirms the implementation;
if it is moderate, the gauge gap is the first explanation to check before any
claim about the argument channel is made.

## Status of the killed run

Two cells were printed before the run was stopped (`delta_U` 0.0 and 0.5, world
0). Their cache is deleted and no value from them is reported. The `delta = 0`
anchor did reproduce E1's independently measured route-only performance
(0.00720 against 0.00735), which is recorded here only as evidence that the
exact-reduction control works.
