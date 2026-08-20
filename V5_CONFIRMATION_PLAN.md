# V5 Confirmation Plan — representation economics

**Status: FROZEN.** Its hash is registered in `tools/check_prereg.py`
and this file may not change after that commit. Every interval below was
re-read against `PREDICTIONS.md` immediately before freezing. Seeds
600-629 were untouched at the moment of freezing and may be opened only
against the thresholds recorded here.

Sealed worlds: **seeds 600-629**. Development used 0-9 and the
contaminated V5 band 500-509; 510-599 stay unused.

V5's development phase produced a mixture of confirmations and clean
failures, and this plan reflects that honestly. Two rungs earned a
sealed test. Three did not, and the reasons are recorded so the absence
is not mistaken for an oversight.

---

# 1. What is being confirmed

## C1 — the amortization law (from H19)

The one result strong enough to carry a mechanism out of sample. In
development, six operating points — three residual ranks and three
return-value gains — landed on

    H* = lambda * D*(A) / s_bar

with mean |chi - 1| = 0.012 and every point inside [1.000, 1.019],
where chi = H*_observed * s_bar / (lambda * D(A)).

    PREDICTION: on the sealed block, mean |chi - 1| <= 0.15 and no
    single operating point outside [0.7, 1.3].

Registered as an interval, not a sign. The arms are the same two:
residual rank in {1, 2, 4} (which moves cost and utility together) and
the return-value gain g in {0.5, 1.0, 1.5} (which holds the carried
tensor bit-identical and moves only payoff). The claim is NOT that D and
s are independently manipulable — they are not, and saying so was
falsified in V5.1. The claim is that the ratio predicts across both
interventions.

Mandatory protocol, inherited from development and asserted before
scoring rather than assumed:

* two-stage. Stage 1 measures `s_bar` at a fixed window that contains no
  crossing; Stage 2 estimates the crossing from SEPARATE runs. A cell
  may not serve as both predictor and outcome.
* the Stage-2 grid is a deterministic function of Stage 1 —
  `H_low <= H_hat* - 4`, `H_high >= H_hat* + 4`, plus the two nearest
  interior even horizons — and is committed before any Stage-2 outcome
  exists.
* carry invariance is verified by CHECKSUM: the abstraction tensors must
  be bit-identical across gains. Not "within 2%".
* zero post-gap births and an exactly zero pre-intervention delta in
  every scored cell; violations void the cell rather than widening the
  error bars.
* no off-grid interpolation. A crossing outside its grid is refused.

    FALSIFIED BY: mean |chi - 1| > 0.15; any point outside [0.7, 1.3];
    or a crossing that cannot be bracketed without moving the grid after
    seeing outcomes.

## C2 — the schema crossing (from H20a / H25 / H26)

Development, on exogenous teacher atoms with the schema frozen after a
calibration set: `M* = D*(S) / s_bar_schema` predicted the observed
COMPRESS-to-FACTORIZE crossing to 6.4% at r_meta 0.90 and 10.1% at 0.95,
and `M*` fell monotonically with relatedness (unreachable, 39.7, 8.0,
5.1, 1.6).

    PREDICTION: at r_meta in {0.90, 0.95}, the observed crossing is
    within 15% of D*(S)/s_bar_schema, in >= 4 of 6 sealed cells.
    PREDICTION: M*(r_meta) is monotone non-increasing across
    r_meta in {0.0, 0.5, 0.7, 0.9, 0.95, 1.0}.
    PREDICTION: at r_meta = 0 the schema has negative per-member saving
    and never pays — the V4R negative reproduced as a limit.

r_meta = 1.0 is EXCLUDED from the 15% test and reported separately. The
observed M is an integer, so at a predicted 1.6 the finest attainable
relative error is 25%; the criterion there is `observed == ceil(predicted)`.
This exclusion is registered in advance because development hit exactly
that resolution floor.

Preconditions, all asserted before any cell is scored:

* R_LOO within the isotropic null at r_meta = 0, monotone in r_meta, and
  above the null at r_meta = 1. Fitting and scoring on the same families
  is forbidden: in development the in-sample number read 0.730 where the
  truth was no structure at all.
* the three balance gates — per-abstraction D*, per-use saving,
  behavioral contribution — each within 10% across the sweep. Promotion
  rate is NOT gated; it is a learner response and is reported as an
  outcome.

    FALSIFIED BY: fewer than 4/6 cells inside 15%; a non-monotone
    M*(r_meta); a positive schema saving at r_meta = 0; or any balance
    gate exceeding 10%, which makes the sweep unscoreable rather than
    weak.

---

# 2. Registered as NEGATIVE, and expected to replicate as such

These are development failures. They are listed with predictions
because a negative is only credible if it was predicted in advance with
the precision a positive would need.

## C3 — the learned library does not realize the schema economy

    PREDICTION: FACTORIZE fails to beat matched-budget COMPRESS on the
    learned library in >= 5 of 6 sealed cells, at both r_meta 0 and 1.
    PREDICTION: a frozen schema leaves > 0.6 of a held-out learned
    atom's effect variance unexplained at r_meta = 1.0.
    Development: 0/6 wins; 0.873 unexplained at r_meta 1.0.

    PREDICTION: realized library size does not track family count.
    Specifically M > F in >= 4 of 6 cells, and M does not fall as
    r_meta rises. Development: M/F 1.5 to 1.75, flat in r_meta.

If C2 passes and C3 replicates, the licensed statement is precise: the
representation class has an economic region and this promoter does not
reach it. That is a claim about PROMOTE, not about schemas, and it is
the reason the exogenous and learned arms were separated (D16).

## C4 — the coding-geometry mechanism

    PREDICTION: the D* gap holds in sign — private residuals cost more
    per scalar than shared abstractions in >= 5 of 6 cells.
    PREDICTION: the spectral explanation stays falsified — the rank
    correlation between the D* gap and the spectral gap is within
    [-0.5, 0.5], and shared objects do NOT show faster spectral decay.
    Development: +0.253 bits/scalar; correlation -0.009; shared
    sigma_2/sigma_1 ~0.70 against private ~0.54, the opposite direction.

---

# 3. Deliberately NOT sealed

* **H29** (the P_0/P_1/P_2 causal decomposition). Blocked by a
  provenance gap, not by cost: nothing checkpoints the member residuals
  at the sleep that consumes them, so P_0 does not exist in any
  artifact. Sealing a rung whose instrument is not built would be
  theatre. The fix is a recorded field in the promoting learner's sleep
  path; until then H29 has no sealed test.
* **H21 below r_meta = 1.0.** Prospective reuse passed at exact
  meta-recurrence (1 example against 8) and failed at 0.9 for a
  capacity reason — a rank-2 schema cannot express a member only partly
  in the subspace, so its error plateaus above any fixed target. The
  honest sealed version needs a schema+leftover arm at matched bits,
  which was not run in development. Not sealed.
* **H22-H24** (selection, planning, learned policy). D18 makes these a
  side branch; no development evidence exists for them and a
  confirmation plan cannot precede a development result.
* **The slots=6 crossing grid and the D\* currency grids.** Named debts
  from V5.1. slots=6 changes `s_bar` by 24%, so absolute crossings are
  protocol-dependent; the sealed block fixes slots=12 and says so rather
  than pretending the number transfers.

---

# 4. Scoring rules, fixed in advance

* **Paired worlds only.** Every comparison is within-world; world-level
  means, never object-level pooling.
* **Both currencies, always.** Bits and held-out nats reported together.
  C1 is scored at the 8-bit proxy, which is what its grids bracket; the
  D* reading is reported where a grid brackets it and declared
  unscoreable otherwise.
* **Matched-budget counterfactuals.** Every sharing claim is scored
  against private storage at equal bits. In C3, a member the schema
  fails to cover is charged full private price; no leftover code is
  invented in FACTORIZE's favour.
* **Gauge discipline.** Any schema over LEARNED objects is fitted in
  effect space and charged in parameter space. Fitting a schema to
  parameter vectors is forbidden.
* **Denominators reported.** Every guard prints the population it ran
  over. A check with an empty denominator is a failure, not a pass.
* **Thresholds are not edited after the fact.** A miss inside a looser
  registered bound is reported as a miss with the tighter bound named.

# 5. What would falsify the development picture

Any of: mean |chi - 1| > 0.15 on C1; fewer than 4/6 schema crossings
inside 15% on C2; a non-monotone M*(r_meta); FACTORIZE winning on the
learned library in >= 2 cells (C3 reversing); or the D* gap changing
sign (C4). Each is reported as a failure of the registered claim, not
reinterpreted.
