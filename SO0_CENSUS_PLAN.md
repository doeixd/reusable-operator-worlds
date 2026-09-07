# SO0: acquisition census over G5R Stages B-D

Status: FROZEN before any census code exists (hash to be recorded in
`tools/check_prereg.py` in the following commit). This is `POST_E6_RESEARCH_PROGRAM.md`
B0, opened by the validated Stage D classification `BUDGET_LIMITED`
(`PREDICTIONS.md` 2026-09-05, commit `8b9b802`). It is a READ-ONLY census over
existing reports and artifacts: no lifetime, no training, no new cell, no
learner update, no teacher identity outside the cells already labelled oracle.
It exists to choose SO1's manipulated variables; it cannot establish causality
and it carries no pass/fail verdict.

# Inputs (all existing, none modified)

    reports/rotated_g5r_diagnosis.json                 Stage A (existence), B (isolated
                                                       operators), C (joint oracle routes)
    reports/rotated_g5r_diagnosis_lbfgs_correction.json Stage B H-LBFGS correction
    reports/rotated_g5r_interference.json              Stage D six-cell grid
    reports/rotated_g5r.json                           G5R verdict and protocol
    artifacts/g5r_rotated/world_{0,1,2}/               O_lr lifetimes (metrics.jsonl,
                                                       hard_routes.json, model.pt)
    artifacts/g5r_interference/oracle_online/world_{0,1,2}/
                                                       O_or lifetimes (same files)

Every number the census prints is either copied from one of these files or
computed from them by a stated formula. The census records the git commit and
SHA-256 of every input file it reads.

# The five registered tables (program B0), with sources and formulas

T1. ISOLATED-OPERATOR ACQUISITION (Stage B). Per (world, primitive, arm):
    initial and final train/query NMSE and the pass flag, as stored. The
    example-gradient count per cell is `adam_steps x train_examples`
    (2,000 x 512 = 1,024,000; full-batch) for Adam arms and is reported as
    NOT COMPARABLE for the L-BFGS arm (line-search evaluations are not
    gradient steps). Stage B stores endpoints only, so T1 has no trajectory;
    the census says so rather than interpolating.

T2. JOINT ORACLE-ROUTE ACQUISITION versus budget. Rows: C_hi (4,096 x 64) and
    C_lo (8,192 x 2) per world; columns: updates, batch, distinct-task
    diversity per batch (expected distinct tasks in a batch of size B drawn
    with replacement from 64 tasks: `64 x (1 - (63/64)^B)`, i.e. 1.98 at B=2
    and 40.5 at B=64), total example-gradients (`updates x batch`), and the
    median query NMSE at every stored checkpoint. The census states explicitly
    that these two rows differ on ALL THREE axes and therefore isolate none.

T3. TRANSITION LOCATION AND PERSISTENCE. For every cell with a checkpoint
    trajectory (C_hi, C_lo, L_lo, L_hi) and every world: the first checkpoint
    at which median <= 0.05 (descriptive "first crossing", review 82's
    caveat applies), and PERSISTENCE, defined as: the crossing checkpoint AND
    at least two later stored checkpoints are all <= 0.05. A cell whose only
    sub-threshold checkpoint is its terminal one is recorded as
    `crossed, persistence unobservable`. Monotonicity of each trajectory is
    also recorded (a trajectory is non-monotone if any checkpoint median
    exceeds the previous one).

T4. ONLINE TERMINAL versus END-OF-TASK. For O_lr and O_or per world: the
    stored end-of-task median, terminal median, and their difference; plus,
    computed from metrics.jsonl, the per-task pair (end-of-task `final_nmse`
    from `task_summary`, terminal from the report's `final_per_task`), the
    fraction of tasks whose terminal error exceeds their end-of-task error,
    and the median per-task ratio terminal/end-of-task. No threshold.

T5. ROUTING AND OPERATOR DIAGNOSTICS already in artifacts. For every offline
    cell: the stored `routing` block (entropy, mean max coefficient, active
    operators, usage counts). For O_lr and O_or: `hard_routes.json` usage
    counts per slot and per position. Orthogonality is NOT reported as a
    learned diagnostic: the Householder parameterization is orthogonal by
    construction, so the check is definitional (review 83) and appears only
    as the Stage A/B stored `orthogonality_max_abs_error` implementation
    check.

# The one decision output: axis-isolation tabulation for SO1

The census lists every pair of existing oracle-route offline cells and marks
which of the three budget axes (update count, per-batch task diversity, total
example-gradients) the pair holds fixed. SO1's `BUDGET_LIMITED` branch requires
matched pairs isolating at least two axes; the census outputs the set of
contrasts the existing cells already provide and the set SO1 must add. It does
not propose the SO1 grid; that belongs to the SO1 plan.

# Registered descriptive expectations (not hypotheses; nothing passes or fails)

- E1. No existing pair of oracle-route offline cells isolates any single axis
      (C_hi and C_lo move all three).
- E2. C_hi's worlds 1 and 2 are `crossed, persistence unobservable` (the only
      sub-threshold checkpoint is the terminal 4,096); no other cell crosses.
- E3. Every L_hi trajectory is non-monotone; every C_lo, C_hi, and L_lo
      trajectory is monotone.
- E4. In both online arms, more than 90% of tasks have terminal error above
      end-of-task error on every world.
- E5. Stage B isolated operators pass on all worlds and arms (as stored), so
      the acquisition wall is in the JOINT problem, not the single-operator one.

If any expectation is wrong the census reports it as wrong; it changes nothing
about what the census may compute.

# What the census may not do

- Add, rerun, extend, or tune any cell.
- Interpolate a crossing between checkpoints or fit a response surface; SO1
  does that on cells designed for it.
- Average world 0 away; it is reported as a named world in every table.
- Read query labels for anything but the stored evaluation numbers.
- Emit a pass/fail verdict for Track B.

# Acceptance

Accept the census only after: exit code 0; every input file's hash recorded;
every number copied from a report reproduced exactly (a checker re-reads the
sources and compares); every computed quantity finite; the report written
atomically to `reports/so0_census.json` with the protocol fingerprint (input
hashes, formulas' constants, threshold 0.05, persistence rule); unit tests for
the diversity formula, the persistence rule, and the axis-isolation tabulation;
`check_prereg.py`; `check_invalid.py`; `git diff --check`. Record in
`PROGRESS.md` and `PREDICTIONS.md` (expectations E1-E5, resolved) before any
SO1 plan is written.

# Cost

Minutes. No lifetime runs.
