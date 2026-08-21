# H39 confirmation plan: two-slot in-basis arguments on sealed seeds 700-729

Status: FROZEN (2026-08-21, after PI approval). Seeds 700-729 must not be generated, inspected,
or summarized until this file is frozen with its commit hash in
`tools/check_prereg.py` and `python tools/check_prereg.py` passes. Licensed
by `H39D_CAPACITY_PLAN.md` verdict A (marginal; see the H39d RESULT entry in
`PREDICTIONS.md` for the caveats this plan exists to test).

# Claim under test

On tasks generated from recurring hidden operations whose family operators
share one functional subspace, a learner whose basis carries two routed
parameterized operators `P(alpha)` — each `tanh(z + a (U_0 + sum_k alpha_k
U_k) tanh(Vz + b))` with K = 32 learned argument directions — formed online
under the ordinary prequential objective with no family labels,

1. makes an unseen family member cheaper to express through its arguments
   alone than the ordinary learner can through its full private residual;
2. does so because the argument directions are LEARNED, not because a task
   gained 64 extra scalars on a fixed channel;
3. pays nothing in present-task cost, and in fact lowers it; and
4. leaves a full task-local interface that is better than ordinary's.

Development evidence (worlds 0-2) for each: alpha-only ratio 1.27 / 1.74 /
1.36 (mean 1.46); frozen-direction control at ~3.1 (single-slot G_8, H39c);
present-task gain -1,991 / -975 / -1,162 nats; full-interface ratio 0.71 /
0.82 / 0.76. Point 1 is the marginal one; points 2-4 were large and
consistent in development.

# Frozen protocol

- Worlds: seeds 700-729 (30), `configs/v5_h72.yaml`, generator arguments
  verbatim from `tools/run_v6_clean.sh`: `--r-meta 1.0 --meta-families 4
  --meta-tasks-per-family 16 --meta-subspace-rank 2 --family-onset 8
  --operator-slots 12 --sleeps 16 24 32 48 64 --lifecycle --arm ordinary
  --prospective-steps 8 --prospective-inner-steps 8`. Model seeds unchanged.
- Arms, every world, paired (identical world, task order, examples, replay
  sampling, evaluation sets):
  - **O** ordinary: `--model prospective` (the V6 ordinary learner).
  - **M** two-slot arguments: `--model pslot --pslot-count 2 --slot-args 32`.
  - **G** frozen directions: `--model pslot --pslot-count 2 --slot-args 32
    --freeze-matrices` (same parameter count and per-task scalars as M;
    `U_k` fixed at their random spectral-norm-1 initialization; alphas learn).
- 90 lifetimes, bounded pool of 3, one writer per cell, complete
  intervention record in `rho_profile.json`, resume refuses mismatches,
  launcher exits nonzero on any failure. Output
  `artifacts/h39_confirmation/{ordinary,m2k32,g2k32}/world_{s}/lifecycle`.
- Futures: the two `novel_family_tasks` per world (unseen families from the
  shared subspace), never trained on by any arm.
- Fits (H39b functions, unchanged numerics): on each future,
  - O: full fit of code + residual at k = 128, B1 (Adam 0.01, 2,000 updates,
    fixed checkpoints, support-only) — the V6R anchor construction;
  - M, G: alpha-only (code + both slots' alphas; residual frozen at its
    task-free init) at k = 128 under B1, with Adam 0.05 and LBFGS
    robustness; full (code + alphas + residual) at k = 128 under B1;
    alpha-only and full at k = 1 under B1 (secondary).
  - Scaled currency `MSE / (2 * 0.1^2)`. Query labels never select anything.
- Channel use per cell: alpha-zeroed family-NMSE ratio over all 64 family
  tasks (functional criterion); route mass reported only.
- Non-vacuity per cell, fail-closed: argument matrices moved (M) / bitwise
  at init (G); both slots' family alpha norms > 0; alpha moves in every
  alpha-only fit; support falls > 1%; k=0 and final query differ; finite.
  A cell that fails is reported as failed; no cell is excluded after any
  query value is seen. Fewer than 30 complete triples = sealed block
  incomplete, no verdict.

# Estimands, predicted intervals, and decision rules

All per-world quantities are paired within world. Intervals are 95% paired
bootstrap over worlds (10,000 resamples, NumPy `SeedSequence([700, 39])`).
As in V2-V5, an interval miss is reported as a failure even when the sign
passes.

    R_M(w)  = mean alpha-only k=128 B1 endpoint of M / O's full-fit endpoint
    R_G(w)  = same for G
    D(w)    = R_G(w) - R_M(w)
    L(w)    = lifetime loss(M) - lifetime loss(O)            [nats]
    F(w)    = M's full-fit k=128 endpoint / O's full-fit endpoint
    U(w)    = M's alpha-zeroed NMSE ratio

E1 (learned directions matter). mean D > 0, CI excludes 0.
    Predicted interval for mean D: [0.8, 2.2].
E2 (absolute fertility). Fraction of worlds with R_M <= 1.5 is at least
    0.5; and mean R_M in [1.2, 1.8].
    Predicted: fraction 0.55-0.75; mean 1.46 +/- 0.2.
E3 (present-task gain). mean L < 0, CI excludes 0.
    Predicted interval for mean L: [-2,000, -500] nats.
E4 (full interface). mean F < 1, CI excludes 1.
    Predicted interval for mean F: [0.65, 0.90].
E5 (the channel is used). U(w) >= 1.25 in at least 27 of 30 worlds.

Verdict (fixed):
- **CONFIRMED**: E1, E3, E4, E5 pass on sign and interval, and E2's
  fraction rule passes. The claim stands as stated, including point 1.
- **CONFIRMED-RELATIVE**: E1, E3, E4, E5 pass; E2's fraction rule fails.
  Learned in-basis arguments are fertile relative to a matched fixed
  channel and cost nothing now, but do not reach the 1.5x absolute bar in
  most worlds; point 1 is withdrawn, points 2-4 stand.
- **FAILED**: E1 fails, or E3 fails, or E5 fails. Any of these voids the
  sharing claim regardless of the others.
- Any interval miss with a passing sign is recorded as a PARTIAL on that
  estimand in the closure and in `PREDICTIONS.md`, never upgraded.

# Anti-fooling guards

- `check_prereg.py` must pass before any seed >= 700 is generated.
- O's full-fit endpoints are computed fresh in the scorer; no world-0-2
  number is reused.
- Identical support/query arrays, initializations, and optimizer budgets
  across arms; support-only optimization; fresh probe IDs removed after
  each fit; complete learner reconstruction (both slots' matrices, alphas,
  references, retirement); report written atomically with executed steps,
  learning rates, supports, seeds, sigma, commit, and every cell.
- The scorer prints the verdict from the table above; narrative cannot
  override it.

# Compute

90 lifetimes at ~25 min each on a pool of 3: about 13 hours. Scoring about
4 hours. Free memory must be checked against ~350 MB per `slots=12` cell
before launch.

# Not authorized

Seeds outside 700-729, any other K or slot count, any retuning after the
block is opened, gates, prospective pressure, or learner-discovered
grouping. Anything learned from this block that suggests a different
architecture goes back to development seeds 0-9.

# Amendment 1 (2026-08-21, review 61; before any sealed cell was read)

Registered while the lifetimes were running and before the scorer had been
executed or any seed-700-729 artifact opened. It changes the decision rule
only; the lifetimes, arms, fits, and E1/E3/E4/E5 are unchanged.

1. **E2 primary is the magnitude, not the 1.5x fraction.** Let
   `z_w = log R_M(w)` and `R~_M = exp(mean_w z_w)` (geometric mean). E2
   passes iff `R~_M` lies in the preregistered interval **[1.2, 1.8]** and
   its 95% paired bootstrap interval lies entirely below 1.8 (the claim is an
   upper bound on multiplicative opportunity loss). The arithmetic mean and
   its interval are reported for continuity with development (1.46).
2. **The fraction `Pr(R_M <= 1.5)` is a reported CONTINUITY STATISTIC against
   census C0, not verdict-determining.** Its development-derived prediction
   (0.55-0.75 from three worlds) is withdrawn as false precision.
3. **Verdict table, restated.** CONFIRMED: E1-E5 all satisfy their frozen
   intervals/gates including E2's magnitude interval. CONFIRMED-RELATIVE:
   E1, E3, E4, E5 pass and E2 misses its magnitude interval — learned
   directions are causally useful, improve present and full-interface
   behaviour, and are used, but alpha-only acquisition did not reach the
   predicted opportunity level. FAILED: E1, E3, or E5 fails. PARTIAL marks
   are unchanged.
4. **Scope sentence, frozen.** This block tests EXISTENCE AND USE, NOT
   DISCOVERY. The architecture (two parameterized slots, K = 32) is
   supplied. A positive result means a jointly formed parameterized
   representation can preserve future variation opportunity and improve
   acquisition; it does not mean the learner can autonomously discover that
   representation. Discovery is the next rung.

Reviewer predictions (review 61), registered for scorekeeping: E1 very
likely passes; E3 likely passes with world variability; E4 likely the
cleanest strong positive; E5 very likely passes; E2 geometric mean about
1.4-1.7; the 1.5x fraction the least stable quantity. Expected headline:
learned directions real and useful, alpha-only preserves substantial but
incomplete opportunity, the full schema + innovation interface clearly beats
ordinary.
