
# E6: language growth by macro invention

Status: DRAFT (freeze commit recorded in `tools/check_prereg.py` before any code).
Governed by `EXPORT_BRANCH_PROGRAM.md` and the terminology contract; sources are
review 81 (`reviews/reviewer-feedback-81.txt`) and review 82
(`reviews/reviewer-feedback-82.txt`). **Development worlds 0-2.** Sealed bands
600-629, 700-729 and 800-829 are spent and are not reused as development data.

# The question

The export branch banked a program language: frozen learned operators execute
unseen programs, compose in unseen combinations and positions, stay executable
at unseen lengths, and each task's solution IS a short discrete sequence. E5
showed the language cannot yet be WRITTEN by an amortized recognizer, and E5.1
showed why that matters less than expected: `C_find` is linear in program LENGTH
and logarithmic in program-space SIZE, so search is not the bottleneck a bigger
space was supposed to create.

The remaining question is whether the language can GROW:

> When should a learner create a new program-language symbol that abbreviates
> recurring computation, and does creating it pay in the currencies it claims?

This is the V5 amortization law moved from neural abstractions to LINGUISTIC
ones, and it is the first rung in this project where the object created is a
symbol rather than a tensor.

# Terminology, per the contract

A DEFINITIONAL MACRO `M := (P_a, ..., P_l)` is a symbol whose execution expands
to its constituent operators. It creates no new neural object; `M` is a naming
and coding decision, nothing more. This plan covers definitional macros ONLY.
A COMPILED macro - a distilled operator `P_M ~ P_l o ... o P_a` - is deferred to
E6.2 for the reason review 82 gives: a positive result from a compiled macro
could not distinguish linguistic compression from neural distillation.

Nothing here licenses the word SYNTHESIS beyond its E5.1 scope (search-based,
sealed), and inventing a macro is LANGUAGE GROWTH, not PRIMITIVE INVENTION: the
substrate's primitives are unchanged.

# The registered coding scheme, and the naive one rejected in advance

Programs are coded with a uniform code over the CURRENT alphabet. With `N`
programs of length `D` over `K` symbols, the corpus costs `N D log2 K` bits.
Introducing one macro of expansion length `L`:

    cost     L log2 K                    (its definition, written as its expansion)
           + N D log2((K+1)/K)           (the ALPHABET TAX: every symbol in every
                                          program now costs more)
    saving   H (L-1) log2(K+1)           (each of H uses drops L-1 symbols)

giving the registered crossing

    H* = [ L log2 K + N D log2((K+1)/K) ] / [ (L-1) log2(K+1) ]

At `K = 12, D = 6, N = 64` this predicts `H* =` 13.92, 7.44, 5.29 for
`L =` 2, 3, 4.

**The naive accounting is rejected here, before any data exists.** Charging only
the definition (`H* = L/(L-1)`) gives 2.00, 1.50, 1.33 uses - a threshold almost
any recurring pattern clears, which would make E6A vacuous by construction. This
is the project's standing rule about checking a registered threshold against its
own baseline, applied at design time. The alphabet tax is what makes macro
creation an economic DECISION rather than a free lunch, and it produces the
falsifiable dependencies below.

Registered secondary predictions of the same formula, each testable:

    H* grows with corpus size N   (2.95, 4.45, 7.44, 13.44 at N = 16, 32, 64, 128)
    H* grows with program depth D (5.45, 7.44, 9.44, 11.44 at D = 4, 6, 8, 10)
    H* falls with macro length L

The `N` dependence is the counterintuitive one and is registered deliberately:
a LARGER corpus makes a macro HARDER to justify, because every program pays the
tax while only `H` of them collect the saving.

# Testbed

One frozen library per world (the E1 discrete artifacts, `slots = 12`) and the
E8 variable-depth executor, both already used by E5 and E5.1 and re-verified per
world. Programs are depth `D = 6` unless a rung sweeps `D`.

**Corpus construction.** Tasks are generated from TEACHER programs in which a
chosen contiguous teacher subsequence is planted in a controlled fraction of
tasks, giving a target recurrence count `H`. Each task's program is then INFERRED
by the sealed/E5.1 route optimization (`K = 2000` steps, support only, query
labels never used), yielding the corpus of LEARNER-SPACE programs on which all
macro accounting is done.

This distinction is load-bearing and is registered explicitly: **the macro is a
symbol of the LEARNER's language, defined over the learner's own inferred
routes, never over teacher primitive indices.** The gauge results (E0.1 margins
0.001-0.019; inferred routes beating the teacher's in 7 of 9 E2 cells) mean a
planted teacher subsequence need not appear as a clean recurring learner
subsequence, and whether it does is itself a measured quantity, not an
assumption. A rung whose planted recurrence does not survive into the inferred
corpus is reported as such rather than scored.

# Rungs

## E6A - oracle macro economics

The candidate is GIVEN, not discovered: the most frequent contiguous subsequence
of the inferred corpus. Replace every occurrence, recompute the two-part code,
and locate the observed crossing in `H` against the predicted `H*`.

Swept: `H` (by construction), `L in {2, 3, 4}`, `N in {16, 32, 64, 128}`,
`D in {4, 6, 8, 10}`. Primary estimand: the observed crossing `H_obs` where the
macro-bearing code becomes shorter, against the formula's `H*`.

## E6B - search savings, testing the E5.1 law causally

E5.1 measured `C_find ~ D` across a depth SWEEP. E6B asks whether shortening a
program by macro substitution moves `C_find` the same way - i.e. whether the
scaling law is causal in program length rather than a property of the task
distribution at each depth.

Registered prediction, with no free coefficient:

    C_find(D') ~ C_find(D) * (D - L + 1)/D ,    D' = D - L + 1

Measured by running route optimization over an alphabet containing `M` and
comparing device-seconds and steps-to-parity against the same tasks without it.

## E6C - does the macro improve the LANGUAGE, or cache a pipeline?

E2 for macros. After `M` is created, evaluate novel programs in which the same
subcomputation appears with new prefixes, new suffixes, new neighbouring
operators, and in positions the subsequence never occupied. `M` must remain
functionally substitutable for its expansion in those unseen contexts, to the
same tolerance E2 used. A macro that works only where it was created is a cached
pipeline and is reported as one.

## E6D - refusal controls, all four required

A creation rule that never refuses is not an economic rule.

1. **Too little recurrence.** `H < H*` by construction. Must REFUSE.
2. **Accidental pattern.** A subsequence over-represented in the observed corpus
   that does not recur prospectively. Must REFUSE.
3. **Wrong grouping.** Against the true `A B C`, the plausible competitors
   `A B`, `B C`, `A C`, `B A C`. Discovery must beat these, not merely beat
   having no macro - the project's standing requirement that a structural claim
   defeat plausible WRONG structures.
4. **Sham alias.** A new symbol that shortens nothing. Isolates whether savings
   come from abbreviation or from the mere act of adding a symbol. Per the H50
   sham rule, beating sham proves the mechanism, beating max(WRONG, RANDOM)
   proves the structure, and BOTH are required.

## E6E - the learner discovers the macro

Only after E6A-D establish an opportunity exists. The learner enumerates
contiguous subsequences of its own inferred corpus and scores each by the
registered two-part code plus the E6B-predicted `Delta C_find`. No neural
proposer. The registered success condition:

> The learner creates a macro exactly when prospective savings exceed creation
> cost, and refuses in all four E6D controls.

# Non-vacuity, required before any rung is read

1. **Planted recurrence survives inference.** The planted subsequence must
   appear in the inferred learner corpus at a rate materially above the rate of
   an equally-long random subsequence. Reported per condition; a condition
   failing this is UNSCOREABLE, not negative.
2. **The crossing is bracketed.** The swept `H` range must contain the predicted
   `H*`, verified before running - the horizon-bracketing rule from H19 and V5.
3. **`C_find` has dynamic range** at the depths used, per E5.1's measured curve.
4. **The refusal controls can fire.** Each E6D control must be shown to produce
   a CREATE decision under some setting, or it is untestable rather than passed.

# Decision rules

Registered before any code; margins over 3 development worlds, 2 of 3.

- **LANGUAGE GROWTH DEMONSTRATED** iff E6A's observed crossing matches `H*`
  within a factor of 2, E6B's search saving matches `(L-1)/D` within a factor of
  2, E6C's substitutability holds in unseen contexts, and E6E creates above the
  boundary while refusing in all four E6D controls.
- **ECONOMICS WITHOUT DISCOVERY** - E6A-C pass, E6E fails. The opportunity is
  real and the learner cannot find it; the successor is a better proposer.
- **DESCRIPTION WITHOUT SEARCH** - E6A passes, E6B fails. Registered in advance
  as an INTERESTING outcome per review 82: the macro compresses without helping
  synthesis, establishing that the description-optimal and search-optimal
  languages differ.
- **NO OPPORTUNITY** - E6A fails. Macro creation does not pay at this scale, and
  the honest reading is that the substrate's programs are already short enough
  that abbreviation buys nothing.

# Registered predictions

**Ours.** `DESCRIPTION WITHOUT SEARCH` is our modal outcome, at roughly even
odds against `LANGUAGE GROWTH DEMONSTRATED`. The description half should work:
the coding scheme is explicit and the crossing is arithmetic. We doubt the
search half, because E5.1 found `C_find` linear in depth with a shallow slope -
from 7.7 s at `D = 3` to 25.4 s at `D = 10`, about 2.5 s per step of depth - so
an `L = 3` macro on a `D = 6` program should save roughly `2/6` of search time,
around 5 s, which is close to the cell-to-cell noise we measured. We predict the
effect is real, small, and possibly not resolvable at n = 8 tasks per cell; the
plan therefore requires a factor-of-2 agreement rather than a significance test,
and E6B's sample size should be set from E5.1's observed variance before running.

We further predict E6C PASSES (E2 and E8 both found this substrate more
context-robust than intuition suggested, twice) and that E6D's WRONG-GROUPING
control is the one most likely to fail, because contiguous subsequences of a
6-symbol program overlap heavily and `A B` is a strict prefix of `A B C`.

**Review 82's.** Expects the `(L-1)/D` law to be load-bearing and macro
invention to yield "shorter description plus proportionally cheaper synthesis",
and states the sharper hypothesis this rung tests: ABSTRACTION IMPROVES SYNTHESIS
PRIMARILY BY REDUCING EFFECTIVE PROGRAM LENGTH, NOT BY REDUCING THE NOMINAL
CARDINALITY OF THE SEARCH SPACE.

# Cost

No lifetimes and no new libraries. Route inference dominates: one corpus is
`N` tasks x ~16-25 s. The full `H`/`L`/`N`/`D` grid is run behind a
protocol-fingerprinted per-cell cache with one writer per cell, serially, and is
resumable; E6A-D are a few hours, E6E is cheap once the corpora exist.

# Out of scope

Compiled macros (E6.2); any change to a sealed verdict; the recognizer, which
review 81 postpones until search is expensive; and the interaction-net /
graph-rewrite substrate, which review 82 places AFTER macro invention succeeds,
where the question becomes branches, loops and arbitrary reusable subgraphs
rather than contiguous linear subsequences.
