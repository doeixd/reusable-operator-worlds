
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
