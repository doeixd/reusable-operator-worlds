# Neural Library Learning
## V4R Experimental Specification: When Does a Library Need a Lifecycle?

**Status: PROVISIONAL DRAFT — not authoritative until frozen.** Sealed
seeds 400-429 remain sealed and must not be generated, inspected, or
summarized until this document's development phase is complete and
`V4R_CONFIRMATION_PLAN.md` is frozen with its hash in
`tools/check_prereg.py`. Development worlds are 0-9, as in V1/V2/V3.

**Date:** August 19, 2026

This document supersedes the QUESTION of
[`row_v4_experimental_spec.md`](row_v4_experimental_spec.md) without
revising it. That spec is preserved unrevised as the record of a
development outcome in which no registered rung reached sealing
eligibility, and the failure history is part of why the question below
is credible. Its §A inventory, §0.2 resource model, §4A constants, and
§5 statistical plan are inherited unchanged; only the hypotheses and the
staging are replaced.

---

# 0. Why the question changed

V4 assumed that successful abstraction birth implies a library
maintenance problem. Three development gates say otherwise, at this
scale:

| Rung | Gate | Outcome |
| --- | --- | --- |
| Redundancy elimination | contribution-relative substitutability | no opportunity: compaction net -1,180 / -2,393 / -1,377 nats |
| True retirement | §2.2 dormancy refusal control | not instantiated: both arms retire identically |
| Higher-order factorization | matched-budget private compression | geometry yes, economics no: private quantization wins 9/10 |

The third result is the most informative, and it points somewhere
specific. When the private-compression counterfactual is made fair, the
cheapest available representation change is not *share more between
abstractions* but *represent each abstraction more cheaply*. The V3
library at the canonical operating point is close to

    KEEP + privately compress

and that is a positive statement about V3 rather than only a negative
one about V4.

So the productive question is not whether any particular operator can be
made to fire. It is:

> **Under what conditions does the optimal representation cease to be a
> static library of independently compressed abstractions and begin to
> require lifecycle operations?**

This turns three failures into the starting result.

## 0.1 The edit-economy principle

Two constitutional rules now govern every structural claim in this
project. The first is inherited from V3:

> **Structural claims require functional substitutability** — never
> parameter similarity or clustering, because gauge freedom makes a
> parameter mean destroy the function it averages.

The second is new, and comes from the V4 factorization failure:

> **Every sharing claim needs a matched-BUDGET non-sharing alternative.**
> "Shared beats unshared at full precision" and "shared beats unshared
> at equal bits" are different claims, and only the second is evidence
> of reuse.

Together they order the edit vocabulary by structural ambition:

    KEEP  <  COMPRESS  <  SHARE / FACTORIZE  <  CREATE / FORK

A more ambitious edit earns its place only by beating every simpler
counterfactual **at the same resource budget**. `COMPRESS(A)` — lower
precision, lower rank, pruning, or a cheaper realization of the same
function — is a first-class operator here. V4 omitted it, and that
omission is exactly what made the factorization result look positive.

---

# 1. The opportunity census

No new operator is implemented until an offline oracle shows that the
operation has somewhere to pay. The V3 learner stays frozen: no DELETE,
no MERGE, no FORK, no COMPRESS during the lifetime. Libraries are
generated under varying regimes and audited afterwards.

For each library, ask of each candidate edit whether it *could* pay,
always against the matched-budget frontier:

| Edit | Question | Baseline it must beat |
| --- | --- | --- |
| `KEEP` | — | (the reference) |
| `COMPRESS` | how far can `A_i` be quantized or rank-reduced before behavioral loss outweighs bits saved? | KEEP |
| `FACTORIZE` | does `C + B alpha_i + E_i` beat storing the atoms? | KEEP **and** matched-bit COMPRESS |
| `RETIRE` | does deletion produce an actual future opportunity-cost saving? | KEEP; never scored on a single arm |
| `RETAIN` | is carrying a dormant `A` cheaper than paying `C_reacquire` on return? | delete-and-relearn at matched gap |
| `FORK` | does `A'` beat private deltas forever? | private deltas |

`E_i` carries a specific loophole and a registered guard: if the
per-abstraction remainder retains the expressive capacity of the
original, the shared family is decorative. Report both the fraction of
behavioral contribution carried by the shared family and the ratio
`D(E_i) / D(A_i)`; a pass needs `D(E_i) << D(A_i)` **and** substantial
shared capture.

## 1.0 STATUS — census results, 2026-08-19

Executed with the V3 learner frozen. Recorded here in the same commit as
the measurements, per project convention.

| cell | verdict | evidence |
| --- | --- | --- |
| COMPRESS | **dominates** | wins 15/15 regime-worlds over FACTORIZE across F in {2,4,8}, N in {64,128,200}; margin does not close with scale |
| FACTORIZE | **negative** | existence gate passes (rank-2 excess over isotropic null +15.7/+26.2/+27.3 points) but the matched-budget economic gate fails 9/10 worlds; leave-one-abstraction-out recovers a mean 7.5% |
| RETIRE | **negative** | obsolescence oracle's refusal control does not refuse |
| RETAIN | **positive (oracle)** | frozen-library oracle: crossover between gap 8 (+1,594, 7/10) and gap 16 (-240); refusal control passes 0/9 in the permanent arm |
| FORK | **negative** | 0/30 abstraction-cells with an actual A' refit; best gain 670 nats against a 1,098 cost, five splits negative. The earlier 24/33 "pass" was a too-loose bound measuring un-promotion |

**Registered prediction 1.1 outcome: FALSIFIED.** "Many related atoms
favors FACTORIZE" is not supported anywhere sampled. Library size is not
monotone in family count — F=8 yields SMALLER libraries (3-4) than F=4
(6-9), because 64 tasks split eight ways leaves too few per family to
reach `minimum_cluster = 3`, so the scale axis must be driven by N. At
N=128 libraries reach 9-12 and COMPRESS's margin WIDENS rather than
closing. No `M*_factorize` exists below SIXTEEN abstractions: at N=200 libraries
reach 11-16 and COMPRESS still wins 3/3. Locating a sharing threshold
would need a library an order of magnitude larger than a 216-program
world can generate -- a ceiling of the testbed, not only of the learner.

**The economic survival rule does not discriminate.** With the correct
counterfactual — the task's own private residual, in a two-part
comparison — every abstraction repays its description cost: 15/15
structured and 10/10 control, by 3x to 20x. An earlier claim that the
carry cost separates PROMOTE's false positives is RETRACTED in
`PREDICTIONS.md`; it used raw ablation, an alternative the learner never
faces. PROMOTE is not creating uneconomic abstractions, and an
abstraction over noise still beats storing that noise n times.

**What does discriminate is the RATE.** Prediction nats given up per
description nat saved is 0.194/0.190/0.164 in structured worlds against
0.321/0.300/0.275 in controls, lower 3/3, mean paired difference +0.116.
Both trades are profitable — 0% reach the break-even rate of 1.0 — so
this is a quality diagnostic, not grounds for refusing promotions. Note
the standing tension: the principled threshold `T_A = lambda * D(A)` is
untuned but does not sort, while the rate sorts but any cutoff on it
would be a tuned hyperparameter.

**CENSUS COMPLETE.** Four of five cells negative; RETAIN positive only
as an oracle. At this scale, under stationary recurrence with cheap
reacquisition, the optimal representation IS a static library of
independently compressed abstractions. §1.2's possibility is the
realized outcome: lifecycle machinery costs more than the slack it
recovers, and no `M*` was located below twelve abstractions.

**Blocked.** The online route to RETAIN requires PROSPECTIVE value.
An at-birth filter fails because an abstraction has its minimum
dependent set exactly when the decision is made; implemented and
measured, it collapses libraries in both conditions.

## 1.1 The sweep

Worlds 0-2 are sufficient for an opportunity map; ten worlds are for
confirmation, not for locating a boundary.

    lifetime length      N in {64, 128, 200}   # 256 unreachable:
                                              # tasks cap at 6**3 = 216
    recurring families   F in {2, 4, 8}
    turnover             q in {0, low, high}
    return gap           g in {8, 16, 32, 64}

The registered output is a classification of the oracle-optimal edit per
regime:

| Regime | Predicted best structural response |
| --- | --- |
| small, stationary | KEEP / private compression |
| many related atoms | FACTORIZE |
| temporary disappearance, expensive reacquisition | RETAIN |
| permanent disappearance, priced live storage | RETIRE |
| persistent divergence | FORK |

**These are predictions, registered before the sweep runs.** The V4
history says they can be wrong in either direction, and a regime whose
oracle-optimal edit is KEEP everywhere is a result.

## 1.2 The possibility that nothing pays until scale

The census may show that no lifecycle operation matters below some
library size. That outcome is not a null; it would mean **lifecycle
complexity is itself subject to amortization** — a small library should
not pay for garbage collection, deduplication, hierarchy, retention
policy, or retrieval indexing, because the machinery costs more than the
slack it recovers. The quantity to report is then a set of thresholds,

    M*_compress,  M*_factorize,  M*_retain,  M*_retire,

each the library scale at which that edit's oracle advantage first
exceeds zero. A lifecycle phase transition would be a stronger V4
contribution than any single operator, and it is directly analogous to
V1's recurrence crossover and V2's allocation/compression split.

---

# 2. Worlds that must be built

## 2.1 Retrieval continuity (the RETAIN world)

The V4 dormancy pair failed to create option value because the learner
re-promotes across the gap as cheaply as it would reuse. A world that
instantiates retention must make reuse materially cheaper than
reacquisition:

- **Retained:** `A` is available as a candidate from the first task of
  the returning regime; the learner pays only recognition and argument
  cost.
- **Deleted:** there is no `A`; the learner must accumulate recurring
  private innovations until PROMOTE can reconstruct `A'`, paying
  `C_reacquire = L_extra + lambda * D_temporary + C_search`.

Then `V_retain ~= P(return) * C_reacquire - C_carry`.

**Registered failure mode, measured 2026-08-19.** Two attempts already
failed. A gap of `(32, 64)` leaves the returning arm byte-identical to
the permanent arm for the whole lifetime. A gap of `(32, 62)` leaving
two returners — below `minimum_cluster = 3` — still failed, because the
*other* task group continues promoting after the gap, so post-gap births
occur in both arms and the dormant family's option is not isolated. A
working design must suppress unrelated promotion during the return
window or use a single-family world.

**Gate before any operator:** establish an oracle crossover in gap
length, `g < g*` favors retain and `g > g*` favors delete-and-relearn.
Only then implement an online retention policy.

## 2.2 Everything else waits

No world is built for FACTORIZE, FORK, or COPY-ON-WRITE until the census
shows a regime where its oracle advantage is positive against the
matched-budget frontier.

---

# 3. The eventual experiment

If the census finds regimes with distinct oracle-optimal edits, the
flagship becomes a selection problem rather than an operator
demonstration. Give the learner a candidate vocabulary

    E = {KEEP, COMPRESS, FACTORIZE, RETIRE, RETAIN, FORK}

estimate `Delta J(e)` for each, and choose `e* = argmin_e J(e)`. The
decision may be oracle-computed at first; a learned controller is not
required to make the point. Then build worlds where the correct answer
is known independently, and ask:

> **Can one economic criterion choose the right representation class
> across regimes?**

That is a stronger claim than any single operator, and it is what the
project's thesis actually needs: the representation reorganizes
according to whichever structural description is cheapest, rather than
according to a mechanism an experimenter installed.

---

# 4. Inherited, unchanged

Resource model and `J` (V4 §0.2), constants (V4 §4A), statistical plan
and paired-world discipline (V4 §5), sealed protocol and the prohibition
on tuning against a failed gate (V4 §6, §7.1), diagnostics (V4 §8), and
execution notes including the 4-6 concurrent-lifetime ceiling (V4 §9).

# 5. Pre-run checklist

1. Census oracles implemented, each reporting against the matched-budget
   frontier, and each refusing a single-arm verdict where the objective
   makes an end-of-lifetime edit free.
2. Sweep of §1.1 run on worlds 0-2 with the V3 learner frozen.
3. Regime classification recorded and compared against the §1.1
   predictions, including any regime where KEEP wins everywhere.
4. Only then: build the world for whichever edit shows the largest
   oracle advantage, gate it, and tune.
5. `V4R_CONFIRMATION_PLAN.md` frozen and hashed before seeds 400-429 are
   touched.
