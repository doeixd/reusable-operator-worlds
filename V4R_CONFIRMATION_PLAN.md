# V4R Confirmation Plan — a NEGATIVE result, preregistered

**Status: FROZEN on the commit that adds its hash to
`tools/check_prereg.py`. No edits after that commit.**

Sealed worlds: **seeds 400-429**, untouched to date. They must not be
generated, inspected, or summarized until this plan is frozen.

This plan differs from `CONFIRMATION_PLAN.md`, `V2_...`, and `V3_...` in
one important way: **the registered prediction is that nothing pays.**
The development census found no structural edit worth making in the
canonical regime, and this block exists to test whether that negative
replicates out of sample rather than to confirm a mechanism. A negative
result is only credible if it was predicted in advance with the same
precision a positive one would require, so the thresholds below are
written as intervals and a miss is a failure even when the sign is right.

---

# 1. What is being confirmed

## O1 — COMPRESS dominates FACTORIZE

For each sealed world, score the ambition ladder KEEP < COMPRESS <
FACTORIZE with the V3 learner frozen, FACTORIZE required to beat
**matched-budget** private compression, not full-precision atoms.

    PREDICTION: COMPRESS wins in >= 27 / 30 worlds.
    Development: 15/15 regime-worlds across F in {2,4,8}, N in {64,128,200}.

Interval, not just sign: mean COMPRESS-minus-FACTORIZE net advantage in
**[1,000, 4,000] nats**. Development mean was 1,634 at N=64/F=2 rising
to 2,793 at N=200/F=4.

## O2 — no factorization threshold below the reachable library size

    PREDICTION: no world shows FACTORIZE winning at any realized library
    size M <= 16.

A single sealed world with a FACTORIZE win at M <= 16 falsifies the
development claim that `M*_factorize > 16` and must be reported as such.

## O3 — RETIRE and FORK have no oracle opportunity

    PREDICTION: FORK pays (actual A' refit, one extra abstraction
    charged) in <= 2 / 30 worlds. Development: 0/30 abstraction-cells.
    PREDICTION: the obsolescence refusal control continues to fail to
    discriminate; permanent and returning arms retire at statistically
    indistinguishable times.

## O4 — the retention amortization law

The one positive development result, and the only rung here with a
mechanism to confirm. Controlled protocol: gap fixed, library frozen at
the gap so `D_retain - D_delete = D(A)` exactly, verified by asserting
zero post-gap births before scoring.

    PREDICTION: V_retain is monotone increasing in H_R, and the crossing
    lies in H_R in [14, 22].
    Derived point estimate: H_R* = lambda * D(A) / s_bar.
    Development: predicted 17.1, observed 17.9.

Also registered, because it is the part most likely to fail out of
sample: the per-use saving `s_bar` lies in **[50, 75] nats** and is flat
in gap length (coefficient of variation < 10% across gaps).

## O5 — the coding frontier

    PREDICTION: mean behavioral cost of quantizing shared abstractions
    is <= 300 nats at 4 bits/scalar and <= 50 nats at 6 bits.
    Development: 127 and -7.

---

# 2. Scoring rules, fixed in advance

* **Paired worlds only.** Every comparison is within-world. World-level
  means, never abstraction-level pooling: abstractions inside one world
  are not independent, and the abstraction-level test on development
  data overstated significance (p = 0.0079 across 15 vs 10 abstractions
  where the world-level n was 3).
* **Both currencies, always.** Bits saved and held-out Gaussian nats
  paid, reported together. A structural gain with an unpriced behavioral
  cost is not a result — that error produced the retracted V4.1 H14
  number.
* **Matched-budget counterfactuals.** Every sharing claim is scored
  against private storage at equal bits. Every value claim is scored
  against the cheapest legitimate alternative the learner would actually
  have, never against raw ablation.
* **Contribution-relative tolerances.** Any functional tolerance is
  normalized against the quantity whose loss it licenses. The null-edit
  guard is mandatory: if deleting an abstraction outright falls within
  tolerance, the tolerance is void and the cell is unscoreable.
* **No single-arm verdicts** where an end-of-lifetime edit is free under
  the objective.
* **Dormancy configs validated before reading.** Count returning tasks
  first; a gap that closes at or after the final task has an empty
  return window and its numbers are void. This error produced false
  readings three times in development.
* **Return-window instrument for interventions.** A mid-lifetime
  deletion is not a paired comparison after the intervention, because
  library state is path-dependent. Score the return window, never an
  end-of-lifetime J difference.

# 3. What would falsify the development picture

Any of: COMPRESS winning in fewer than 27/30; a FACTORIZE win at
M <= 16; FORK paying in more than 2/30; the retention crossing falling
outside [14, 22]; or `s_bar` outside [50, 75]. Each is reported as a
failure of the registered claim, not reinterpreted.

# 4. Deliberately out of scope

No operator is implemented for this block. No V4 lifecycle rung is
revisited: `row_v4_experimental_spec.md` is closed with its gate-outcome
banner and its hypotheses are not carried forward. The V5 scale
benchmark is a separate question on a different world and cannot be
scored here.
