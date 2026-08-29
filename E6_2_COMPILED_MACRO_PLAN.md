
# E6.2: can a recurring program fragment be compiled into a reusable operator?

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract; sources are
review 82 (`reviews/reviewer-feedback-82.txt`) and review 83
(`reviews/reviewer-feedback-83.txt`). **Development worlds 0-2.** Sealed bands
600-629, 700-729 and 800-829 are spent and are not reused as development data.

# The question, and why it could not be asked before

E6 covered DEFINITIONAL macros: `M := (P_a, P_b, P_c)` is a symbol whose
execution expands to its constituents. E6C asked whether such an `M` stays
substitutable for its expansion in unseen contexts, and was RETIRED at
Amendment 2 because a definitional macro **is** its expansion, bitwise,
everywhere -- the test could only measure the expansion runtime.

A COMPILED macro has independent semantics. Train a single operator

    P_M(x) ~ P_c(P_b(P_a(x)))

and the question becomes empirical, because `P_M` is an approximation that can
be right in the contexts it was fitted on and wrong elsewhere. This is where
E6C's question is answerable and where failure is possible.

# What success would and would not license

Success would license the sentence: *a recurring fragment of the learned program
language was compiled into a single reusable operator that stands in for its
expansion in contexts it was never fitted on.* Under the terminology contract
that is the strongest claim in this branch so far, and the first that would bear
on PRIMITIVE INVENTION in a semantic rather than an operational sense.

It would NOT license "the learner invented a primitive": the fragment is supplied
by E6's economics, not discovered here, and the compilation is a fitting step we
perform. Discovery of WHICH fragment to compile remains E6E/E6F's question.

# Construction

One frozen library per world (the E1 discrete artifacts, `slots = 12`) and the E8
variable-depth executor. The macro is the E6A candidate at `D = 6, L = 3` -- the
top gram of the learner's own inferred corpus, in worlds where E6A found it
non-degenerate (depths 4-6 only, per Amendment 3's scope limit).

**`P_M` has the same architecture and parameter count as ONE library operator.**
This is the project's matched-budget requirement: a compiled macro that is
allowed to be larger than the objects it replaces would win for capacity rather
than for compilation.

**Fitting distribution, and why it is the whole experiment.** `P_M` is fitted to
reproduce the learner's own composition `P_c o P_b o P_a` on states drawn from
the positions the macro actually occupies in the OBSERVED corpus. No teacher
information and no task targets enter: this is pure linguistic compilation, and
`P_M` can at best match the composition it is distilled from.

# Rungs

## E6.2a - semantic substitutability (E6C's retired question)

Execute held-out programs containing the fragment, once with the expansion and
once with `P_M` substituted, and compare query NMSE. Contexts are graded by how
far they move the operator's INPUT DISTRIBUTION away from the fitting one:

    C0  same positions, held-out programs        on-distribution
    C1  new immediate neighbours                  mild shift
    C2  new prefixes and suffixes                 moderate shift
    C3  positions the fragment never occupied     strong shift
    C4  program depths the fragment never saw     strongest shift

Primary estimand: the substitution gap `log NMSE(P_M) - log NMSE(expansion)` per
context class, with the realized input-distribution shift reported beside it so a
failure can be attributed to shift rather than asserted.

## E6.2b - the economics, with the parameter cost charged

A definitional macro costs a symbol. A compiled one costs a symbol PLUS an
operator's parameters, so its crossing is strictly higher:

    H*_compiled = [ D*(P_M) + N D log2((K+1)/K) ] / [ (L-1) log2(K+1) + s_exec ]

where `D*(P_M)` is the BEHAVIOURAL rate of the new operator (the E3 measure:
smallest interpolated bits/scalar whose composed task NMSE ratio stays under
1.10) and `s_exec` is the per-use execution saving, three operator applications
becoming one. Both terms are MEASURED, not assumed.

Registered before any data: we expect `D*(P_M)` to dominate. A definitional macro
pays after ~7.4 uses; a compiled one must additionally amortize an operator's
description, and E6A's realized use counts (22-56 at `D = 6`) may or may not
clear it. **The economics may say do not compile even where the semantics say the
compilation works**, and that would be a result, not a disappointment.

# Identity checks, declared in advance

This plan has already lost two estimands to constructions that could not fail.
Every clause here is checked against that rule before freezing.

- **"One call replaces three" is TRUE BY CONSTRUCTION.** The execution-count
  saving is arithmetic and is reported as an IMPLEMENTATION CHECK, never as
  evidence.
- **On-distribution substitutability is nearly guaranteed** for a sufficiently
  smooth composition and an operator of matched capacity. Context class C0 is
  therefore a NON-VACUITY CHECK -- if it fails, the fit is broken and nothing
  downstream is readable -- and the evidence lives in C2-C4.
- **`D*(P_M)` must be measured behaviourally**, not read off the parameter count,
  or it would restate the architecture rather than the operator's content.

# Non-vacuity, all required

1. **The fit works on-distribution** (C0 gap within the E2 tolerance). A broken
   distillation makes every later class meaningless.
2. **An untrained `P_M` fails.** The same architecture at random initialization,
   substituted identically, must produce a large gap. Without this, a small C3
   gap could reflect an easy test rather than a good fit.
3. **The context classes actually shift the input distribution.** Reported per
   class as the distance between the states `P_M` sees and its fitting states; a
   class that does not move the distribution is reported as such and cannot
   support a generalization claim.
4. **The expansion arm is the same object E6 measured** -- the learner's own
   operators, unmodified.

# Decision rules

Registered before any code; margins in log NMSE, 2 of 3 worlds.

- **COMPILATION HOLDS** iff the substitution gap stays within the E2 tolerance
  through C3, with C0-C2 also within it and the untrained control failing.
- **COMPILATION IS CONTEXT-BOUND** iff C0-C1 hold and C2 or C3 fails: the
  operator is a cache of the states it was fitted on, which is a real and
  reportable negative.
- **COMPILATION FAILS** iff C0 holds and C1 fails.
- **UNSCOREABLE** if the non-vacuity checks do not pass.

The economics verdict is reported separately and never pooled with the semantic
one: `COMPILATION PAYS` iff realized uses exceed `H*_compiled` in 2 of 3 worlds.

# Registered predictions

**Ours: COMPILATION HOLDS, and COMPILATION DOES NOT PAY.**

On the semantics we follow review 80's standing prior update, which this project
earned by being wrong twice in the same direction: *do not infer a binding
generalization barrier from unseen internal-state distributions.* At E8 and again
at the sealed C4 we predicted composition would fail on distribution shift; both
times the effect was real, visible in the per-step trace, and an order of
magnitude too small to bind. We therefore predict C3 holds, and register that
predicting the SHAPE of a degradation and predicting whether it BINDS are
different skills.

On the economics we expect `D*(P_M)` to be large relative to the coding saving.
E6A found abstractions in this substrate to be roughly 4-8x overparameterized at
`1-2` bits/scalar against 8 stored, so `D*(P_M)` should be a few hundred bits
against a definitional macro's ~11. At E6A's realized use counts the compiled
crossing is unlikely to be cleared.

If both predictions hold, the finding is a clean dissociation: **the compilation
works and is not worth doing at this scale** -- which is exactly the kind of
statement the amortization law was built to make, and which no amount of
"the compiled operator reproduces its expansion" would establish on its own.

**Review 83's:** expects the compiled macro to be where primitive invention could
be licensed in the stronger semantic sense, and flags a fourth currency
`Delta e_execute` -- one compiled call may accumulate less error than three
approximate ones.

**We register a correction to that fourth currency before it is measured.**
`P_M` is distilled from the learner's OWN composition, so it can at best match
that composition and cannot beat it; `Delta e_execute` against the learner's
expansion is bounded at zero by construction and is an implementation check.
A compiled operator could only beat the composition by being fitted against
something else -- task targets, or the teacher -- which is a different experiment
with a different supervision budget, and is out of scope here. Measuring
`Delta e_execute` against the TEACHER's composition is reported as a diagnostic,
with the bound stated.

# Cost

No lifetimes. One distillation per world (small; minutes) plus execution of
held-out programs across five context classes. A few hours behind a
protocol-fingerprinted cell cache.

# Out of scope

Discovery of which fragment to compile (E6E/E6F). Compiling more than one macro,
or macros over macros. Any change to a sealed verdict. The interaction-net /
graph-rewrite substrate, which review 82 places after macro invention and which
would be tested on branching, recursion and loops rather than on contiguous
fragments.

# Amendment 1 (2026-08-29, before any E6.2 code): a units error in this plan's own crossing

Found while implementing E6.2b. The frozen crossing reads

    H*_compiled = [ D*(P_M) + N D log2((K+1)/K) ] / [ (L-1) log2(K+1) + s_exec ]

The numerator is in BITS. The denominator adds `(L-1) log2(K+1)`, which is bits,
to `s_exec`, which is operator applications. **They cannot be summed without an
exchange rate between bits and computation**, and no such rate is registered
anywhere in this project. Left as written, whatever value was chosen for that
rate would silently carry the verdict -- the failure mode registered at
`E6_MACRO_PLAN.md` Amendment 2 and in `AGENTS.md` as "never let a free
coefficient carry a verdict".

## Registered correction

**The crossing is evaluated in DESCRIPTION BITS ALONE**, i.e. with `s_exec = 0`:

    H*_compiled = [ D*(P_M) + N D log2((K+1)/K) ] / [ (L-1) log2(K+1) ]

This is dimensionally clean and directly comparable to the definitional macro's
`H* = 7.44`, since the two differ only by `D*(P_M)` replacing `L log2 K` in the
numerator. `COMPILATION PAYS` is decided on this quantity.

**The execution saving is reported SEPARATELY, in its own units** -- operator
applications per use, three becoming one -- and is never added to a bit count.
Whether a project should trade bits for operations is a real question and needs
its own registered exchange rate; it is not settled here by an implicit one.

This is the V5.1 separate-currencies rule applied to this plan's own arithmetic:
`V_desc`, `V_find` and `V_exec` are measured independently and combined only
under a rate frozen in advance.

## Consequence for the registered prediction

The prediction is unchanged in direction and becomes sharper: with `s_exec`
removed from the denominator, the compiled crossing is `H* = 7.44` scaled by
`D*(P_M) / (L log2 K)` = `D*(P_M) / 10.75`. A `D*(P_M)` of even 200 bits puts the
crossing near 138 uses against E6A's realized 22-56, so **COMPILATION DOES NOT
PAY** is predicted more firmly than before, and the semantic verdict is where the
interest lies.

# Amendment 2 (2026-08-29, before any E6.2 cell was scored): context class C3 is unconstructible

Found in a structural dry run, which hung and was killed rather than producing a
number. Disclosed here.

## The defect

C3 is registered as "positions the fragment never occupied". At `D = 6` with
`L = 3` a length-3 fragment can start at only 4 positions (0, 1, 2, 3), and the
macro is measured to occupy ALL FOUR in the observed corpus of every world:

    world 0  macro (11, 6, 4)   seen positions [0, 1, 2, 3]   unseen: none
    world 1  macro (8, 8, 5)    seen positions [0, 1, 2, 3]   unseen: none
    world 2  macro (8, 9, 6)    seen positions [0, 1, 2, 3]   unseen: none

The class is EMPTY by construction. The generator looped forever looking for a
position that does not exist.

This is the horizon-bracketing rule (H19, V5) applied to a context class rather
than to a sweep range: **verify that a registered condition can be CONSTRUCTED
before freezing it**, not merely that it is well defined.

## Registered correction

**C3 is redefined as position novelty in LONGER programs**: the fragment placed
at a start index `>= 4`, which exists only at depth `>= 7`. Those indices are
genuinely never occupied in the observed corpus, since that corpus is depth 6.

**The confound is disclosed rather than hidden.** C3 so defined varies position
AND depth together, and C4 varies depth alone, so the two are no longer
independent. Reported consequence: a C3 failure cannot be attributed to position
novelty without checking C4, and if C4 fails too the honest reading is DEPTH, not
position. The comparison C3-versus-C4 is reported as the disambiguating contrast.

We considered and rejected reducing `L` to 2 to free a position: it would change
the macro under test and break comparability with every E6 rung.

**Position novelty at fixed depth is not available in this testbed**, and that is
itself a small finding: a fragment frequent enough to be worth compiling in a
depth-6 corpus will have been seen at every position it can occupy.

## Implementation requirement

Every context class must report whether it was CONSTRUCTIBLE. A class that cannot
be built is reported as such and excluded from the verdict; the generator takes a
bounded number of attempts and never spins. A class that yields fewer than the
registered program count is reported with its realized count.

# Amendment 3 (2026-08-29, before any E6.2 verdict): the tolerance was output-relative, and an untrained operator passed it

Found in the structural dry run by the registered non-vacuity control, which is
what that control exists for.

## The defect

`agreement_nmse` divided the substitution error by the magnitude of the PROGRAM
OUTPUT. But the library's operators are residual perturbations, so a length-3
fragment changes the output by only a few percent. A randomly initialized `P_M`
therefore scored 0.050-0.078 against a 0.15 tolerance and **PASSED in 3/3
worlds** -- the null edit succeeded.

This is `AGENTS.md`'s V4.1 rule, verbatim: *a functional tolerance must be
normalized against the quantity whose loss it licenses, never against total
output scale.* V4.1 divided by total output variance while an abstraction
contributed ~0.2% of it, and every abstraction substituted for every other. The
stated diagnostic for the error class is that THE NULL EDIT PASSES. It did.

## Registered correction

The substitution error is normalized by the MACRO'S OWN CONTRIBUTION:

    error = || out(P_M) - out(expansion) ||^2  /  || out(expansion) - out(identity) ||^2

where `out(identity)` executes the same program with the fragment replaced by a
pass-through. The denominator is what the macro actually does to the state, so
the ratio answers "what fraction of the fragment's own effect did the compiled
operator get wrong", which is the quantity the tolerance is meant to license.

An error of 1.0 now means "no better than deleting the fragment". The E2
tolerance of 0.15 applies to this contribution-relative quantity.

## Consequence for the non-vacuity requirement

The untrained control becomes able to fail, which it must be for the rung to
carry evidence: a random operator should score at or above 1.0, since it
reproduces none of the fragment's specific effect. **If the untrained control
still passes under the corrected normalization, the rung is UNSCOREABLE and no
substitutability claim may be made from it.**

The dry-run numbers under the OLD normalization are recorded here so the
correction is legible, and are not reported as results: substitution 0.010-0.021,
untrained 0.050-0.078, both nominally "within tolerance".

# Amendment 4 (2026-08-29, before any E6.2 verdict): matched-budget compilation fails, and a capacity ladder to say by how much

Found by the registered non-vacuity check C0 ("the fit works on-distribution"),
which failed at matched budget. Diagnosed before scoring.

## The measurement

`P_M` at a library slot's own size cannot reproduce the fragment's contribution
even on the states it was fitted on, and this is REPRESENTATIONAL rather than
optimizational: the fit is converged (3,000 steps 1.236e-2, 12,000 steps
1.231e-2 -- identical), and capacity alone moves it.

    rank  multiple  scalars  fit loss   C0 (contribution-relative)
      8      x1       265    1.23e-2       0.180   fail
     16      x2       529    6.96e-3       0.158   fail
     32      x4      1057    3.50e-3       0.082   PASS
     64      x8      2113    8.76e-5       0.003   PASS

**A compiled macro needs roughly 4x a library slot's parameters to stand in for a
composition of three of them.**

## Why this is a result and not a blocked rung

Left as registered, C0's failure would make E6.2 UNSCOREABLE and the finding
would be lost in a label. The finding is specific and worth stating:
COMPILATION FAILS AT MATCHED BUDGET, and the budget it needs is measurable.

## Registered structure, replacing the single matched-budget run

The **primary** verdict stays at matched budget and is now expected to read
`COMPILATION FAILS AT MATCHED BUDGET` -- reported as a substantive negative, not
as unscoreable, because the reason is measured.

A **capacity ladder** is added as a registered secondary: ranks `x1, x2, x4, x8`,
with every class reported at every rank and CAPACITY CARRIED AS AN EXPLICIT
COVARIATE. Any substitutability that appears above `x1` is labelled BOUGHT WITH
CAPACITY and may not be reported as "compilation works" without that
qualification. This is the project's matched-budget rule preserved rather than
relaxed: the constraint still decides the headline; the ladder only says what
lifting it would cost.

The economics are recomputed at each rung of the ladder. `D*(P_M)` scales with
the scalar count, so the crossing rises with capacity, and the prediction
`COMPILATION DOES NOT PAY` becomes stronger at exactly the capacities where the
semantics start working. That dissociation -- **the capacity that makes
compilation correct is the capacity that makes it uneconomic** -- is the sharpest
form the result can take and is registered here as our expectation.

## Registered predictions, updated

- `COMPILATION FAILS AT MATCHED BUDGET` (x1), on C0, in 3/3 worlds.
- Substitutability holds from `x4` upward, including in the shifted classes
  C1-C4, per review 80's standing prior on distribution shift.
- `COMPILATION DOES NOT PAY` at EVERY rung, and by a widening margin: the
  matched-budget crossing already measured ~151 uses against realized 22-54, and
  a 4x operator quadruples `D*(P_M)`.
- The untrained control fails at every rung (it already fails 3/3 under
  Amendment 3's corrected normalization, at 0.55-0.81).
