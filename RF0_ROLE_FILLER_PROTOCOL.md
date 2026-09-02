# RF0: frozen unsupervised role–filler protocol

Status: FROZEN before RF0a scores, RF0 code, derived rows, fits, or outputs
exist. This document operationalizes the RF0 rung in
`RF_ROLE_FILLER_PLAN.md`. Development worlds 0–2 only. It allocates no sealed
seed band and launches no lifetime.

# Purpose and contamination boundary

RF0 asks whether a label-free factorization of frozen E6 learner occurrences
exposes teacher-semantic recurrence that raw route symbols obscure. Teacher
primitive identities and planted-site flags are unavailable to fitting,
initialization, model selection, convergence, and stopping. They are opened
once for scoring after every learner-side choice is fixed.

RF0a is an oracle information census whose result decides whether RF0 runs, but
must not redesign RF0. This protocol therefore freezes RF0's observations,
factor families, grids, splits, selection rule, scorer, controls, and decision
rule before any RF0a aggregate is produced or read.

# Frozen inputs and reconstruction

RF0 uses exactly the E1 discrete artifacts and 1,536 E6A cached routes declared
in `RF0A_SEMANTIC_RECOVERABILITY_PLAN.md`: worlds 0–2, depths 4/6/8/10, and
128 deterministic tasks per world/depth. Programs are regenerated from
`SeedSequence([970, world, depth])` through the existing `plant_corpus`
instrument. Routes are read from cache and never optimized again.

The audit must reproduce, cell by cell, E6A's protocol fingerprint, top
length-3 learner gram, planted-site hit count, and survival rate. It must also
reproduce the displayed world means 91.15%, 52.60%, 13.02%, and 7.29%. A
missing, duplicate, stale, or mismatched cell is fatal.

One occurrence is `(world, depth, task, position)`. Tasks, not occurrences, are
the split unit. No persistent row-level artifact containing teacher labels is
written.

# Frozen learner-side observations

The primary observation for occurrence `n` is the common-state functional
fingerprint `x_n` of its selected frozen learner slot. Construct it exactly as
in RF0a:

- take the first 32 support inputs of the first 16 depth-4 tasks;
- include their initial states and every learner post-step state, producing
  2,560 common states per world;
- evaluate all 12 frozen slots on the identical state bank;
- flatten output-minus-input, center across slots, and compute an SVD;
- retain singular components above `1e-8` of the largest, capped at 11.

The supplied role vector `r_n` is also identical to RF0a: boundary one-hot
`FIRST/INTERIOR/LAST`, normalized position, normalized remaining position, and
depth divided by 10. Append an intercept for decoding. No neighbours, native
activation statistics, route histograms, task IDs, teacher functions, targets,
motif flags, or query data enter RF0 fitting. Those richer observations remain
outside this registered test even if RF0a later favors them.

# Frozen cross-depth split

For each world independently:

    fit candidate factors       depth 4
    select family/grid/seed     depth 6
    refit selected form         depths 4 + 6
    infer frozen filler IDs     depths 8 and 10 separately
    open semantic scorer        only after all preceding choices are fixed

Depth-8/10 learner-side reconstruction may be reported but cannot select or
stop a fit. No deep teacher label is read before selection.

# Candidate factor models

All models use hard latent filler assignments `k_n in {0,...,K-1}` and squared
functional reconstruction. The frozen grid is:

    K                 2, 4, 6, 8, 10, 12
    ridge lambda      1e-6, 1e-4, 1e-2, 1
    initialization    0, 1, 2

`K` is deliberately not fixed to the six teacher primitives.

The arms are:

1. **FLAT:** `x_hat_n = f[k_n]`. This label-free clustering baseline receives
   no role.
2. **ADDITIVE:** `x_hat_n = f[k_n] + A r_n`. A shared role displacement is
   added to each filler.
3. **TPR:** `x_hat_n = W vec(onehot(k_n) outer r_n)`. This is a literal
   filler-by-role binding followed by a shared linear decoder; equivalently,
   each filler owns coefficients over the same declared role basis.
4. **ROLE-ONLY:** `x_hat_n = A r_n`. This non-factorized non-vacuity baseline
   has no filler and is never eligible as the winning semantic code.

There is no nonlinear decoder, neighbour extension, or post-RF0a arm in this
protocol. The broader research program may register one later only as a new
experiment, not as a repair to RF0.

# Deterministic fitting and selection

Standardize fingerprint coordinates from depth 4 only. Role columns are used
as declared and are not standardized. For each `(family,K,lambda,seed)`:

1. Initialize assignments by seeded k-means++ on standardized depth-4
   fingerprints using NumPy `default_rng(SeedSequence([981, world, seed, K,
   family_index, lambda_index]))`.
2. Alternate a closed-form ridge decoder fit and minimum-squared-error hard
   reassignment. Decoder intercept/filler intercept terms are unpenalized;
   role-dependent coefficients receive `lambda`.
3. Empty fillers are deterministically reseeded to the depth-4 occurrence with
   greatest current squared error, breaking ties by row order.
4. Stop when assignments do not change or after 100 updates. A nonfinite solve
   or failure to finish is an invalid candidate, never silently retried.
5. Fit uses an SVD pseudoinverse with relative cutoff `1e-10`.

Infer depth-6 assignments by the fitted candidate's minimum reconstruction
error, without updating it. Select the candidate minimizing

    BIC = N6 * log(SSE6 / (N6 * q) + 1e-12) + p * log(N6)

where `N6` is the number of depth-6 occurrences, `q` is fingerprint dimension,
and `p` is the exact number of free decoder coefficients plus `K*q` centroid
coordinates where applicable. Ties within `1e-12` choose, in order: smaller
`p`, smaller `K`, larger `lambda`, family order `FLAT`, `ADDITIVE`, `TPR`, then
smaller seed.

The selected family, K, lambda, and initialization are refit on depths 4+6
using the same algorithm and a refit RNG whose first SeedSequence component is
982. Deep filler assignments are then frozen by minimum reconstruction error.
For audit robustness, every initialization of the selected family/K/lambda is
also refit and scored, but teacher scores never choose among them.

# Frozen semantic scorer

For code `c` equal either to raw learner symbol or inferred filler ID, define at
each world/depth:

- `S_same`: macro-average over teacher primitives of the fraction of all
  unordered occurrence pairs with that same primitive and DIFFERENT boundary
  roles for which `c` agrees.
- `S_diff`: macro-average over `(unordered primitive pair, boundary role)` cells
  of the fraction of all cross-primitive occurrence pairs within that SAME
  boundary role for which `c` agrees.
- `S_adj = S_same - S_diff`.
- `Delta_RF = S_adj(filler) - S_adj(raw)`.

Every eligible pair is used exactly once; there is no pair subsampling. Empty
cells make that world/depth unscoreable. This contrast penalizes one-code
collapse and tests identity across roles without letting different role
frequencies define the negative baseline.

Also report effective filler count, occupancy, entropy, mutual information with
boundary role, common-state reconstruction SSE, and planted-motif filler-trigram
survival. These are secondary and cannot replace `Delta_RF`.

# Nulls and non-vacuity

All conditions are required for a positive verdict:

1. Complete cache/protocol validation and exact E6 reproduction pass.
2. All 12 slots use bitwise-identical common states; SVD rank is finite and in
   `[1,11]`.
3. Every teacher primitive and boundary role needed by the scorer is present.
4. The selected factor model beats ROLE-ONLY depth-6 SSE and is no worse than
   FLAT depth-6 SSE by more than 1%.
5. At depths 8 and 10, effective filler count exceeds one, every occupied
   filler has at least two occurrences, and `S_same > S_diff`.
6. A synthetic positive generated from role-conditioned manifestations of
   latent fillers yields `Delta_RF >= 0.40`; a role-shuffled synthetic negative
   yields `Delta_RF < 0.10`.
7. For 200 deterministic null draws per world/depth, permute teacher labels
   independently within `(absolute position, boundary role)` and rescore the
   already-frozen codes. A supporting observed `Delta_RF` must exceed the null
   p99. Nulls never refit or select a factorization.
8. The selected family/K/lambda's three registered initializations agree in
   verdict in at least two of three runs. The validation-selected seed is still
   the reported primary fit.

The same-author unplanted-site diagnostic reports the same scorer after
excluding every planted trigram position. It is descriptive because the
remaining programs still come from the E6 generator; it cannot rescue a failed
primary result.

# Decision rule

The world is the replication unit and depths 8 and 10 are scored separately.

- **RF0 SUPPORTS RECOVERABLE SEMANTIC RECURRENCE** iff all non-vacuity checks
  pass and `Delta_RF >= 0.20` at BOTH depths in the SAME at least 2 of 3 worlds.
- **RF0 DOES NOT EXPLAIN E6** iff the instrument is valid and
  `Delta_RF < 0.10` at both depths in the SAME at least 2 of 3 worlds.
- Every other valid pattern is **RF0 UNRESOLVED**.
- Fewer than two scoreable worlds is **RF0 UNSCOREABLE**, not a negative.

A supporting result licenses only that a learner-side, label-free
factorization exposes teacher-semantic recurrence hidden by raw symbols. It
does not establish recombination, causation, architectural advantage, or a
unique ontology. RF1 requires a separate preregistration.

# Registered prediction and RF0a gate

Review 84 predicts support. The project prediction is that RF0 will not reach
the 0.20 margin at both deep depths: each selected learner slot already has a
context-invariant function, so role-conditioned relabeling is more likely to
describe route compensation than reveal one latent local primitive.

RF0 runs only if RF0a returns **ROLE-CONDITIONED LOCAL SEMANTICS**. Every other
RF0a classification stops this exact protocol. A stop is recorded as NOT RUN,
not as an RF0 failure. RF0a cannot alter this gate.

# Implementation, artifacts, and acceptance

Intended implementation:
`src/row/experiments/audit_rf0_role_filler.py`. Tests:
`tests/test_rf0_role_filler.py`. Output:
`reports/rf0_role_filler.json`, written atomically after all worlds complete.

Accept only after clean committed launch code; exit code 0; exact expected
cells; finite fits and metrics; frozen-plan hash validation; artifact freshness;
synthetic controls; nulls; full unit suite; `git diff --check`;
`tools/check_prereg.py`; and `tools/check_invalid.py`. Preserve invalid,
unresolved, and not-run outcomes rather than retuning this protocol.

