
# WITHDRAWN BEFORE FREEZING (2026-08-31), SUPERSEDED BY ROTATED_SUBSTRATE_SPEC.md
# gate checked first, no hash was ever recorded
#
# The WORLD in this spec was sound; its PRIMITIVE FAMILY was not. The successor
# keeps the construction, the knob, the exact reduction and all four gates, and
# replaces the contracting family with a rotated one that passes the necessity
# gate at 1.67-1.73 against a required 0.25.

This spec was never frozen. Its own registered most-likely-failure -- "if `P^k`
converges the iteration gap collapses ... it should be checked FIRST" -- was
checked first, and it fails decisively.

    NECESSITY gate: best achievable NMSE by ANY fixed depth, given the true
    primitive, against a required >= 0.25

      k in [2,6]   0.031  0.032  0.026     FAILS
      k in [1,8]   0.083  0.082  0.066     FAILS
      k in [1,11]  0.119  0.123  0.093     FAILS

    iterating a CYCLE of operators, the only variant preserving the banked
    library: 0.084 / 0.105 / 0.092                FAILS

THE CAUSE. Every operator in this substrate has the form `tanh(z + small)` and is
therefore a CONTRACTION toward a fixed point. Iterates converge -- per-repeat
change decays 0.29, 0.05, 0.03, 0.02, and `P^5` differs from `P^6` by 1.1% -- so a
single fixed repeat count approximates any distribution of counts to NMSE <= 0.12.
**Iteration count is nearly unidentifiable in this operator family.**

WHAT THIS DOES AND DOES NOT ESTABLISH. It establishes that iteration cannot be
made NECESSARY over THIS library, for this primitive family. It does NOT establish
that iteration is useless for program synthesis -- that is precisely the category
error corrected on the same day for loops, and it is not repeated here.

THE CONSEQUENCE, which is larger than a generator change. An iteration-necessary
world needs a NON-CONTRACTIVE primitive family (rotation-like, norm-preserving),
which means a new library, which means new lifetimes and the abandonment or
rebuilding of the banked export vocabulary. That is a new SUBSTRATE, not a new
generator, and it is a decision that should be taken deliberately rather than
reached for.

PROCESS NOTE. E9 needed four amendments before its root construct was found
unbuildable. This spec named its own most likely failure, that failure was checked
BEFORE a freeze hash was recorded, and the cost was about ten minutes. That is the
E9 lesson applied.


# The iteration world: tasks a fixed-length program cannot express

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract. Source is
the 2026-08-31 correction in `PROGRESS.md` and `notes/e7-sketch.txt` section 13.

This is a GENERATOR SPEC, not a rung. It defines a world and the gates that must
pass before any learner is trained on it. No learner comparison is registered
here.

# Why a new world

Four censuses closed nothing about macros, loops or branching; they established
that the existing testbed cannot evaluate them. Its generator composes a FIXED
number of primitives in sequence, so it has no subroutine structure, no
iteration, and no conditionals. A loop census on that world measures the absence
of iteration in the world, not the value of loops.

Macros, loops and branching are CONSTITUTIVE of program synthesis -- a language
with none of them is a fixed-length pipeline notation. If the goal is a substrate
that writes programs, the world has to contain something worth writing.

# The construction

A task's target applies ONE operator a number of times that depends on the input:

    y = P^{k(x)}(x)

    k(x) = clip( round( mu + sigma * g(w . x) ), k_min, k_max )

`P` is a teacher primitive. `w` is a fixed unit projection drawn per task from the
world's stream. `g` is a fixed zero-mean, unit-variance shaping function
(registered as `tanh` scaled to unit variance under the input distribution), so
`mu` is the MEAN repetition count and `sigma` its SPREAD.

Everything else -- world seeds, primitive construction, example generation,
opaque task IDs -- is unchanged from `world.py`.

## The knob, and the confound it must not have

`sigma` is the independent variable: it controls how much the required depth
VARIES with the input. `mu` is held FIXED across the sweep.

This is the project's standing rule that a knob meant to vary one property must
not vary another: raising `sigma` at fixed `mu` changes the VARIABILITY of the
computation, not its AMOUNT. A sweep that also moved `mu` would confound "needs
iteration" with "needs more work", which is the V5.1 residual-rank error.

**Balance gates, each verified before the world is used**, all within 10% across
the `sigma` sweep: mean `k`, mean output variance, mean per-task output norm, and
the marginal distribution of `x`. A sweep failing any of these is UNSCOREABLE
rather than weak.

## The exact reduction

At `sigma = 0`, `k(x) = mu` for every input and the task is a fixed-depth
composition `P^mu` -- an ordinary program in the existing language. The
construction must reproduce the existing generator's task format BITWISE at
`sigma = 0`, and that is asserted per world before anything else runs.

`sigma = 0` is included in every sweep so the reduction is visible in the same
figure as the informative cells.

# The two gates, and why BOTH are required

A world is only useful here if iteration is both NECESSARY and ACHIEVABLE. The
existing testbed failed the second condition at `rho = 0`: nothing worked, so
"the fixed-depth arm is bad" carried no information. Both gates are measured on
the GENERATOR, with no learner involved.

## Gate 1 -- NECESSITY: no fixed-length program can win

For every depth `d` in `[k_min, k_max]`, compute the best achievable query NMSE
of the fixed-depth program `P^d` -- the ORACLE at that depth, using the true
primitive. Take the best over `d`.

    NECESSITY passes iff  min_d NMSE(P^d) >= 0.25  at the largest `sigma`,
    and grows monotonically with `sigma`.

A fixed-length program is being given the true operator and the best possible
depth, and still cannot match the target. If this fails, the world does not
require iteration and is not the world we set out to build.

## Gate 2 -- ACHIEVABILITY: a loop CAN win

Compute the query NMSE of the LOOP ORACLE: the true primitive iterated the true
`k(x)` times.

    ACHIEVABILITY passes iff  NMSE(loop oracle) <= 0.02  at every `sigma`.

This is near zero by construction -- it IS the generating function -- so it is an
IMPLEMENTATION CHECK, not evidence, and is labelled as such. Its purpose is to
catch a world where the target has become unreachable for some unintended reason,
which is exactly what `rho = 0` was.

**A third quantity decides whether the world is interesting**, and unlike the two
gates it can genuinely fail:

## Gate 3 -- the ITERATION GAP is large enough to measure

    gap = log NMSE(best fixed depth) - log NMSE(loop oracle)

registered to exceed 2.0 log units at the largest `sigma` in >= 2 of 3 worlds.
This is the room a loop construct would have to earn. If the gap is small, the
world is technically iteration-necessary and practically uninformative.

# What this world does NOT settle

It does not show that a learner can discover iteration, infer `k(x)`, or invent a
loop symbol. Those are rungs, and each needs its own plan. This spec's claim is
narrower and is the precondition for all of them: **there exists a world, exactly
reducing to the current one at `sigma = 0`, in which fixed-length programs
provably lose and iteration provably wins.**

The gate that four censuses could not supply is buildable here precisely because
the world is CONSTRUCTED rather than inherited.

# Traps this design is checked against

- **No estimand here is a ratio against a fitted arm.** E8D and E9 lost three
  ceilings to overfitting on noiseless small support; the gates above are
  absolute NMSE levels and a log gap between two ORACLE quantities, neither of
  which is fitted.
- **The loop oracle is a zero-error reference and is never a denominator**, for
  the reason E9 Amendment 1 records.
- **No claim here maps learner slots to teacher primitives.** The gates are
  computed entirely in teacher space; any later rung comparing a learner's loop
  to the teacher's must go through functional matching, since E9 measured that
  matched operators have essentially orthogonal parameterizations (cosine +0.047).
- **`sigma = 0` is a reduction control, not a result.**

# Registered predictions

**Ours.** NECESSITY passes: with `mu = 4` and `k` ranging over `[2, 6]`, a single
fixed depth must be wrong for most inputs, and the residual after the best fixed
depth should be large because the operators are contractive -- applying `P` two
extra times moves the state substantially.

ACHIEVABILITY passes trivially (it is the generating function).

**Gate 3 is where we are least confident.** The operators are residual and
tanh-saturating, so `P^k` may converge as `k` grows: if `P^4 ~ P^6` functionally,
the iteration gap collapses and the world is uninformative despite being formally
iteration-necessary. We register this as the most likely failure and as the first
thing to measure -- it is checkable on the teacher alone, before any world is
generated at scale, and it should be checked FIRST.

If `P^k` does converge, the fix is a non-contractive or rotation-like primitive
family for the iterated operator, which would be a further generator change and
must be re-read downstream accordingly.

# Cost

The gates are teacher-side arithmetic: no learner, no lifetimes, minutes. The
world generation itself is the existing pipeline with one substitution.

# Out of scope

Branching, which needs its own world and has no census available. Any learner
rung. Any change to a sealed verdict or an existing artifact. This world is NEW
and does not regenerate or invalidate anything.
