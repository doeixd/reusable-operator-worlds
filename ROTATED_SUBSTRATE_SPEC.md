
# The rotated substrate: a vocabulary in which control flow can matter

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Supersedes `ITERATION_WORLD_SPEC.md`, which is withdrawn: its world was sound and
its primitive family was not. Governed by `EXPORT_BRANCH_PROGRAM.md` and the
terminology contract.

**This spec defines a SUBSTRATE and its gates. It registers no learner comparison
and commits no lifetimes. Training a library on it is a separate go/no-go.**

# Why the existing substrate cannot host the question

Macros, loops and branching are constitutive of program synthesis. Four censuses
on the current testbed returned negatives, and the 2026-08-31 correction
established those were artifacts: the generator has no subroutine, iterative or
conditional structure, so each census measured the absence of the thing in the
world rather than the value of the construct.

Building an iteration-necessary world on the existing library then failed for a
deeper reason. Every operator has the form `tanh(z + small)` and CONTRACTS toward
a fixed point, so iterates converge -- `P^5` and `P^6` differ by 1.1% -- and one
fixed repeat count approximates any distribution of counts to NMSE <= 0.12.
**Iteration count is nearly unidentifiable in that operator family**, which is why
loops cannot be made to matter there no matter how the tasks are drawn.

# The primitive family

    P(z) = Q ( z + a . U tanh(V z + b) )

`U`, `V`, `b`, `a` exactly as in `world.py`'s `Primitive` -- independently
spectral-normalized `U` and `V`, the same `alpha`, the same seeding discipline.
`Q` is an orthogonal matrix drawn per primitive from the world's stream.

The residual structure, the rank bottleneck and the scale conventions are
unchanged. The ONLY addition is `Q`.

## Why the rotation, measured rather than argued

Three families were checked against the necessity gate below, on the teacher
alone, before this spec was written:

    current    tanh(z + a U tanh(Vz+b))   0.024 / 0.032 / 0.027   norm x0.58
    no-squash       z + a U tanh(Vz+b)    0.009 / 0.014 / 0.011   norm x1.35
    rotated    Q(z + a U tanh(Vz+b))      1.667 / 1.721 / 1.728   norm x1.03

The obvious candidate -- removing the outer squashing function so the state is not
compressed -- FAILS WORSE than the current family. Its norm grows but its iterates
converge in DIRECTION, so a fixed depth approximates them better. **Non-contraction
of the norm is not the property that matters; absence of a fixed point is**, and an
orthogonal map is what supplies it while keeping states bounded (`x1.03`).

# The world

Tasks are as in `world.py` -- programs over primitives, examples from independent
deterministic streams, opaque IDs -- with one addition: a task's target may apply
an operator a number of times that depends on the input,

    y = P^{k(x)}(x),   k(x) = clip( round( mu + sigma g(w . x) ), k_min, k_max )

`w` a fixed unit projection per task, `g` a zero-mean unit-variance shaping
function. `sigma` is the independent variable and `mu` is HELD FIXED, so the knob
varies the VARIABILITY of the computation and not its AMOUNT -- the V5.1
residual-rank rule.

**Exact reduction.** At `sigma = 0` every task is a fixed-depth composition and
the world is an ordinary program world over the rotated family. `sigma = 0` is
included in every sweep.

# Gates, all measured on the GENERATOR with no learner

**G1 NECESSITY.** `min_d NMSE(P^d) >= 0.25` at the largest `sigma`, growing
monotonically with `sigma`. A fixed-length program gets the true operator and the
best depth and still loses. Measured at 1.67-1.73 in the pilot, well clear.

**G2 ACHIEVABILITY.** `NMSE(loop oracle) <= 0.02`. True by construction and
labelled an IMPLEMENTATION CHECK, never evidence. Its purpose is to catch a world
that has become unreachable for an unintended reason -- the `rho = 0` failure,
where nothing worked and "fixed depth is bad" therefore carried no information.

**G3 ITERATION GAP.** `log NMSE(best fixed) - log NMSE(loop oracle) >= 2.0` at the
largest `sigma`, in >= 2 of 3 worlds. The room a loop construct would have to
earn.

**G4 BALANCE**, each within 10% across the `sigma` sweep: mean `k`, mean output
variance, mean per-task output norm, and the marginal distribution of `x`. A
sweep failing any of these is UNSCOREABLE rather than weak.

**G5 THE SUBSTRATE IS STILL LEARNABLE AT ALL.** Before any iteration rung, a
library must be trainable on the `sigma = 0` world to a lifetime cost comparable
to the existing substrate's. This is the gate that decides whether the rotation
has broken what already worked, and it is the one that requires LIFETIMES. It is
the first go/no-go and is NOT authorized by freezing this spec.

# What is knowingly given up

Every existing artifact is over the unrotated family. Nothing transfers: no
library, no sealed verdict, no cached route. The banked export results remain
true of their substrate and say nothing directly about this one.

This is the cost the 2026-08-31 analysis identified and it is stated here rather
than discovered later: **making control flow matter means spending the vocabulary
this project spent months building.** The rotated family is chosen to keep every
other convention identical so the two substrates remain COMPARABLE IN FINDINGS
even though they are incompatible in artifacts.

# What this spec does not claim

That a learner can discover iteration, infer `k(x)`, invent a loop symbol, or
learn the rotated family at all. Each is a rung with its own plan. This spec
claims only that a substrate exists in which fixed-length programs provably lose
and iteration provably wins, with every gate measurable before a learner is
trained.

# Registered predictions

**G1-G4 pass** -- G1 and G3 are already measured in the pilot at 1.67-1.73 and are
far from their thresholds.

**G5 is the real question and we are genuinely uncertain.** A rotation makes the
target harder to fit: the composition no longer sits near identity, so a learner's
residual operators must model a rotation as well as a residual. We put this near
even. If G5 fails, the finding is that control flow and learnability trade off in
this operator class -- which would be a substantive result about why substrates
like this one are built contractive in the first place.

# Cost

Gates G1-G4 are teacher-side arithmetic, minutes. G5 is a library-training rung
requiring lifetimes and is separately authorized.

# Out of scope

Branching, which needs a conditional generator and has its own necessity gate to
design. Any learner rung. Any change to an existing artifact or sealed verdict.

# Amendment 1 (2026-08-31, before any code): G3 is degenerate and is withdrawn

Found by reading the gates before implementing them.

## The defect

G3 is registered as

    gap = log NMSE(best fixed depth) - log NMSE(loop oracle) >= 2.0

But the loop oracle executes the TRUE primitive the TRUE `k(x)` times, and the
target IS `P^{k(x)}(x)`. Targets are noiseless, so `NMSE(loop oracle) = 0`
EXACTLY, `log 0 = -inf`, and the gap is `+inf` for any world whatsoever. **G3
cannot fail.**

This is the same defect as E9's Amendment 1 -- a denominator that is exactly zero
by construction -- and it is the sixth unable-to-fail estimand in this branch. The
rule was already recorded after E9: *whenever a denominator is a performance gap,
check whether either endpoint is exactly zero by construction before freezing the
ratio.* It was written down and then not applied to the next thing written.

## Registered correction

**G3 is WITHDRAWN, not repaired.** G1 already states what G3 was meant to state.
G3's purpose was "the room a loop construct would have to earn", and G1's absolute
threshold -- the best fixed-length program, given the true operator and the best
depth, scores `NMSE >= 0.25` -- says exactly that without a ratio. At the pilot's
measured 1.67-1.73 the room is total.

Adding noise to the targets to give the loop oracle nonzero error was considered
and rejected: it would change the world's noiseless convention, which every
existing artifact and every scorer in this project assumes.

The surviving gates are G1 (necessity, absolute), G2 (achievability,
implementation check), G4 (balance) and G5 (learnability, lifetimes, separately
authorized).

# Amendment 2 (2026-08-31, gates measured): the rotation also makes BRANCHING necessary

The spec justified the rotated family by ITERATION alone, and listed branching as
out of scope pending "its own necessity gate to design". That gate was designed
and run, and the result broadens the justification.

## Measured

    target  y = P_a(x) if w.x > 0 else P_b(x)
    gate    best fixed program (1-step or 2-step, any operators) NMSE >= 0.25

    current  0.0069  0.0071  0.0074   ->  0/3   fails
    rotated  0.9292  1.0390  0.9832   ->  3/3   PASSES

Alongside the iteration gate measured the same day:

              iteration necessity      branch necessity
    current      0.024 - 0.032            0.007
    rotated      1.08  - 1.58             0.93 - 1.04

**One substrate change supplies both constructs.** The spec's cost-benefit changes
accordingly: the vocabulary is not being spent for loops alone but for control
flow generally.

## The unified diagnosis, which is the substantive finding

A single FIXED operator approximates a CHOICE BETWEEN TWO DIFFERENT OPERATORS to
0.7% error. That is only possible if the operators are nearly identical in effect
-- and they are, because `tanh(z + a . small)` at `a = 0.35` is a weak
perturbation of the identity.

    THE EXISTING SUBSTRATE'S OPERATORS ARE TOO WEAK FOR CONTROL FLOW OF ANY KIND
    TO CARRY INFORMATION.

Repeating them converges, so a loop cannot be identified. Choosing between them
barely changes the output, so a branch is not needed. These are not two problems;
they are one property seen twice. It also re-reads E6's macro results: a macro
abbreviates a sequence of operators whose individual effects are small, so the
description saving was always going to be marginal.

## What this changes in the spec

Branching moves from OUT OF SCOPE to a registered second application of the same
substrate, with the gate above as its necessity criterion. No other clause
changes: G1, G2, G4 stand as written, G3 stays withdrawn, and G5 -- learnability,
requiring lifetimes -- remains the single go/no-go this spec does not authorize.

The registered prediction for G5 is unchanged and if anything sharpened: the same
operator strength that makes control flow informative is what makes the family
harder to fit, so G5 remains genuinely uncertain and near even.

# Amendment 3 (2026-08-31, before G5 code): G5's threshold, made numeric

G5 is registered as "trainable ... to a lifetime cost COMPARABLE to the existing
substrate's". That is not a number and cannot be scored. Made precise here,
before any lifetime is spent.

## Registered criterion

G5 passes iff BOTH, in >= 2 of 3 worlds:

**(a) LEARNABILITY, the substantive clause.** A library trained on the rotated
`sigma = 0` world beats a FROM-SCRATCH learner on held-out programs, by
`>= 0.75` log units of query NMSE under support-only route inference. This is
E1's export margin and E5.1's eligibility margin, unchanged, so the rotated
substrate is being held to the bar the existing one already cleared.

**(b) COMPARABILITY.** The rotated library's final training NMSE is within a
factor of 2 of the standard library's on its own world. This asks whether the
rotation has made the family harder to fit, not whether it is fittable at all.

Clause (a) decides G5. Clause (b) is reported alongside and a failure of (b)
alone is recorded as "learnable but harder", not as a failed gate.

## Matching

The rotated lifetimes use the IDENTICAL config to the existing artifacts -- same
seeds, slots, ranks, learning rates, schedule, task count -- so `Q` is the only
difference between the two substrates. The comparison is against the EXISTING
standard artifacts; no standard lifetime is re-run, and none is modified.

## Implementation constraint

The rotated generator is a NEW MODULE (`src/row/rotated_world.py`), mirroring
`world.py` the way `mixed_world.py` and `task_group_world.py` already mirror it.
`world.py`, `WorldConfig` and every existing fingerprint are untouched -- adding a
field to the shared config would invalidate every resolved-config fingerprint in
the project, which the standing rule forbids.

The rotated primitive must reduce to the standard one when `Q = I`, and that
reduction is asserted as a unit test rather than assumed.
