# RF0a: is local semantic identity recoverable at depth?

Status: FROZEN before RF0a code, derived rows, fits, or aggregate outputs exist.
The freeze commit is recorded in `tools/check_prereg.py` before implementation.
Development worlds 0–2 only. No new lifetime, route optimization, world, or
sealed seed. Source: the RF branch in `RF_ROLE_FILLER_PLAN.md` and the post-E6
decision tree in `POST_E6_RESEARCH_PROGRAM.md`.

# Launch gate and contamination boundary

RF0a is an explicitly ORACLE information census: teacher primitive identities
are labels for its probes. Its result could influence the detailed unsupervised
factorization design if read too early. Therefore:

1. RF0a is a single-threaded, read-only audit rather than a full lifetime. The
   PI explicitly authorized it to run alongside the in-flight G5R Stage D on
   2026-09-02, before RF0a code or output existed. It must not read, write, or
   contend for any Stage-D artifact or process state.
2. The exact unsupervised RF0 protocol—factor families, grids, splits, nulls,
   selection rule, and scorer—must be frozen in its own protected plan before
   RF0a is run or its aggregate output is opened.
3. RF0a then decides whether that already-frozen RF0 protocol runs. It cannot be
   used to redesign RF0 after the fact.

No RF0a report exists at freeze time. Existing E6 teacher programs and inferred
routes have already been observed in E6 and are not newly sealed; the boundary
protects the new cross-depth semantic scores.

# Question and claim limit

E6 found that a planted teacher trigram survives as one literal learner trigram
at mean rates 91.15%, 52.60%, 13.02%, and 7.29% at depths 4, 6, 8, and 10.
RF0a asks a narrower question than whether a role–filler representation can be
learned:

> At depths 8–10, does a frozen learner occurrence still contain enough local
> information to identify which planted teacher computation occupies that
> position, when the probe was fitted only at depths 4–6?

Teacher labels enter the probe, so a positive result licenses only an
INFORMATION/OPPORTUNITY statement. It does not establish unsupervised discovery,
a TPR, recombination, causal architectural benefit, or that the teacher ontology
is the learner's unique natural basis.

# Frozen inputs

- E1 discrete artifacts: `artifacts/e1_disc/world_{0,1,2}`.
- E6A cached routes: `reports/e6a_cache/w{world}_d{depth}_{task}.json` for
  3 worlds × 4 depths × 128 tasks = 1,536 cells.
- E6A aggregate used only for exact reproduction:
  `reports/e6a_macro_economics.json`.
- Canonical config loader: `configs/v1.yaml`, resolved against each E1 artifact.
- Depths: 4, 6, 8, 10. Tasks: the exact 128 deterministic E6A programs per
  world/depth, regenerated from `SeedSequence([970, world, depth])` using
  `plant_corpus`; no resampling.
- Cached route protocol: the exact imported E6A fingerprint. Any missing,
  malformed, or mismatched cache cell is fatal. RF0a never recomputes a route.

Every regenerated task uses E6A's task ID and index offset. The learner library
is frozen and evaluated in `eval()` mode. Only support inputs (`train_x`) are
used to build occurrence features. Query inputs and targets are absent from
feature construction and probe selection; query data are not needed by RF0a.

# Unit of analysis and split

One row is one `(world, depth, task, position)` occurrence. Its label is the
teacher primitive at that position. All positions from a task stay together.

The cross-depth split is fixed:

    fit candidates       depth 4
    select ridge lambda  depth 6
    refit final probe    depths 4 + 6
    primary test         depth 8 and depth 10, separately

No depth-8/10 labels affect preprocessing, hyperparameters, arm choice, or
stopping. Worlds are fitted and scored separately; the world is the replication
unit. Balanced accuracy macro-averaged over the six teacher primitives is the
classification statistic.

# Structural roles

For an occurrence at zero-based position `j` in a depth-`D` route, the supplied
role vector contains only:

- boundary one-hot: `FIRST`, `INTERIOR`, `LAST`;
- normalized position `j / (D - 1)`;
- normalized remaining position `(D - 1 - j) / (D - 1)`;
- normalized depth `D / 10`.

There is no teacher identity, motif flag, planted site, task ID, world ID, or
query statistic in the role vector. A depth-one division case is implemented
and unit-tested although the registered depths are all greater than one.

# Learner-side occurrence features

The learner route is the cached hard route. Starting from each task's support
inputs, execute its selected frozen learner slots sequentially and record at
each position:

- incoming-state coordinate mean and standard deviation over support examples;
- outgoing-state coordinate mean and standard deviation;
- update `(outgoing - incoming)` coordinate mean and standard deviation;
- RMS incoming state, RMS outgoing state, and RMS update.

These are learner-side activations; task targets are never used. The full trace
feature contains, padded to depth 10:

- mean and standard deviation of every post-step state;
- a ten-entry valid-step mask;
- normalized 12-slot route histogram;
- normalized 12×12 adjacent-bigram histogram.

Left and right neighbour features are one-hot over 12 learner slots plus a
boundary token. They identify immediate learner syntax only.

# Common-state functional fingerprint

Native activations can confound operation with the state distribution arriving
at its role. RF0a therefore includes a functional fingerprint evaluated on one
COMMON bank per world.

The bank is constructed without labels from the first 32 support examples of
the first 16 depth-4 tasks and contains their initial states plus every learner
post-step state: 2,560 states total. Every one of the 12 frozen learner slots is
evaluated on exactly this same bank. Flatten each slot's output-minus-input,
center the 12 slot vectors, and compute an SVD. Retain every component whose
singular value exceeds `1e-8` times the largest, capped at 11. The resulting
coordinate for the selected slot is its label-free functional fingerprint.

The bank, SVD, and rank use depth-4 learner-side data only. Teacher functions,
program labels, and depth-6/8/10 data do not enter. Tests assert identical state
rows for all compared slots and deterministic fingerprints.

# Probe arms

All probes use the same rows, labels, split, standardization, class weights, and
ridge grid. Each arm adds a declared information source:

    R       supplied structural role only
    Z       raw learner symbol one-hot only
    ZR      Z + role + Z×boundary + Z×continuous-role interactions
    FR      functional fingerprint + the same role interactions
    ZRN     ZR + immediate left/right learner symbols
    LOCAL   FR + neighbours + native incoming/outgoing/update summaries
    TRACE   LOCAL + full learner trace + route histogram + bigram histogram

`ZR` is the primary role–symbol arm. `FR` tests whether continuous function
geometry exposes a relation hidden by categorical slot identity. `LOCAL` is the
broad local-dynamic ceiling. `TRACE` is the sequence-distributed ceiling.

No arm is added after depth-8/10 scores are opened. In particular there is no
opaque neural probe, task identifier, teacher-function distance, or query target
arm.

# Probe fitting

Use deterministic weighted one-versus-all ridge regression in closed form. For
each arm:

1. Standardize nonconstant columns using depth-4 means and standard deviations;
   categorical columns follow the same rule. Drop columns constant on depth 4.
2. Weight each depth-4 row by the inverse frequency of its teacher primitive,
   normalized to mean one.
3. Fit six simultaneous squared-loss targets with an unpenalized intercept for
   each `lambda` in `{1e-4, 1e-2, 1, 1e2}`.
4. Select the lambda with greatest depth-6 balanced accuracy; ties choose the
   LARGEST lambda.
5. Recompute preprocessing and class weights on combined depths 4+6, refit with
   the selected lambda, and score depths 8 and 10 exactly once.

Use a stable SVD/pseudoinverse solve with relative cutoff `1e-10`, not a matrix
inverse. Predicted class is `argmax`; ties choose the lowest primitive index.
Report per-class recall, confusion matrices, selected lambda, feature count, and
fit/validation balanced accuracy beside each deep score.

# Null and non-vacuity controls

All are required before a semantic verdict:

1. **Cache completeness:** exactly 1,536 unique route cells, each with the
   imported E6A protocol fingerprint and expected route length.
2. **E6 reproduction:** for every world/depth, the top length-3 learner gram,
   planted-site hit count, and survival rate reproduce
   `reports/e6a_macro_economics.json` exactly. The four world-mean rates must be
   91.15%, 52.60%, 13.02%, and 7.29% to displayed precision.
3. **Label coverage:** every teacher primitive occurs in depth 4, depth 6,
   depth 8, and depth 10 in every world. Otherwise that world is unscoreable.
4. **Role leakage:** `R` balanced accuracy must be `<= 0.30` at both deep
   depths. A larger value means the generator lets role predict the label and
   the role-conditioned comparison is confounded.
5. **Permutation null:** for 200 deterministic draws per world, permute depth-4
   and depth-6 teacher labels independently within each absolute position,
   rerun the complete `ZR` selection/refit procedure, and score against the real
   deep labels. The observed `ZR` score must exceed the null p99 at each depth
   used for a positive verdict.
6. **Synthetic positive:** a unit-test corpus where one raw symbol has different
   identities in `FIRST` versus `LAST` roles must give `ZR - Z >= 0.40` and
   `ZR >= 0.95` under a held-out-task split.
7. **Synthetic negative:** independently permuted labels must not exceed 0.30
   balanced accuracy under the same fitter.
8. **Function-bank identity:** every learner slot is evaluated on bitwise
   identical bank states; fingerprint rank is in `[1, 11]` and finite.

Null draws and probe fitting use `torch.set_num_threads(1)` and deterministic
NumPy/Torch seeds. No best seed is selected.

# Primary estimands

For world `w` and depth `D`:

    A_arm(w,D)       balanced teacher-primitive accuracy
    Delta_role(w,D)  A_ZR(w,D) - A_Z(w,D)

The registered materiality margin is 0.20, inherited from RF0. Absolute
recoverability requires `A_arm >= 0.60`, far above six-class chance 1/6 while
leaving room for gauge-equivalent learner programs.

Replication requires the stated condition at BOTH depths 8 and 10 in the SAME
at least 2 of 3 worlds. A world failing a non-vacuity check is unscoreable, not a
negative; fewer than two scoreable worlds makes the experiment UNSCOREABLE.

# Decision ladder

Apply in order after all non-vacuity checks:

1. **RAW LOCAL SEMANTICS SURVIVE** if `A_Z >= 0.60` at both deep depths in at
   least 2 of 3 worlds. Literal trigram survival understated position-wise
   semantic stability; RF0 may run descriptively but factorization is not needed
   to establish local recoverability.
2. **ROLE-CONDITIONED LOCAL SEMANTICS** if raw does not pass, and both
   `A_ZR >= 0.60` and `Delta_role >= 0.20` at both deep depths in at least 2 of
   3 worlds. This is the opportunity gate for the already-frozen RF0
   factorization.
3. **FUNCTION-GEOMETRIC LOCAL SEMANTICS** if steps 1–2 do not pass, and `FR`
   exceeds both `Z` and `ZR` by at least 0.20 while reaching 0.60 at both deep
   depths in at least 2 worlds. The opportunity is functional but not captured
   by the proposed raw-symbol role binding.
4. **DYNAMIC LOCAL SEMANTICS** if earlier steps do not pass, and `LOCAL` exceeds
   `max(ZR,FR)` by at least 0.20 while reaching 0.60 at both deep depths in at
   least 2 worlds. The relevant role includes incoming state or realized
   contribution; the fixed RF0 role vocabulary is insufficient and RF0 does not
   run as written.
5. **TRAJECTORY-DISTRIBUTED SEMANTICS** if no local arm passes and `TRACE`
   exceeds `LOCAL` by at least 0.20 while reaching 0.60 at both deep depths in
   at least 2 worlds. Redesign around program/trajectory objects rather than
   local fillers.
6. **TEACHER-LOCAL IDENTITY NOT RECOVERABLE** if no arm reaches 0.60 at both
   depths in at least 2 worlds. Stop RF0 as an explanation of E6. This does not
   negate execution or prove that no alternative learner ontology exists.

If multiple later arms meet their numeric rule, the earliest applicable rung
controls. Any pattern not classified above is **UNRESOLVED**.

# Registered predictions

The RF review predicts ROLE-CONDITIONED LOCAL SEMANTICS: raw syntax collapses,
but filler identity becomes stable when role is represented.

The project prediction remains skeptical. We predict `Delta_role > 0` at depth
8 but not `A_ZR >= 0.60` and `Delta_role >= 0.20` at both depths 8 and 10 in
2 of 3 worlds. Our modal classification is DYNAMIC LOCAL SEMANTICS or TEACHER-
LOCAL IDENTITY NOT RECOVERABLE, because each raw slot already has a context-
invariant function and deep route inference is attractor-dominated. `TRACE`
passing after every local arm fails is the most educational contrary result.

# Artifact, validation, and cost

Implementation: `src/row/experiments/audit_rf0a_semantic_recoverability.py`.
Tests: `tests/test_rf0a_semantic_recoverability.py`. Aggregate output:
`reports/rf0a_semantic_recoverability.json`, written atomically only after every
world is complete. No row-level teacher-labelled artifact is persisted.

The report records the frozen plan, launch commit, complete protocol, E1 model
checkpoint hashes, E6A report hash, cache fingerprint, feature dimensions,
selected lambdas, null distributions, per-world confusion matrices, all
non-vacuity checks, and the decision ladder result.

Accept only after: exit code 0; RF0 exact protocol frozen; clean committed RF0a
launch code; expected world/depth/arm counts;
finite features and scores; exact E6 reproduction; report freshness; atomic
completion; full unit suite; `git diff --check`; `tools/check_prereg.py`; and
`tools/check_invalid.py`.

Expected cost is minutes after Stage D: route optimization is reused, library
execution is CPU-only and read-only, and ridge probes are small. One process,
one report writer.

# Explicitly out of scope

No unsupervised factorization; no new learner; no route refit; no lifetime; no
query inputs or labels in features; no teacher function or parameter features;
no claim that teacher primitive identity is the only valid semantic basis; no
change to E6; no post-output feature engineering; and no RF0 run or redesign
until its exact protocol has been frozen independently.
