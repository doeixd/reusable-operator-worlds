# V2 confirmation plan (frozen before any seed in 200-229 exists)

Frozen 2026-08-18, while development steps 005-007 were still running.
Where a development verdict is pending, this plan pre-registers the
DECISION RULE that maps the verdict to the confirmatory design, so no
choice is made after seeing sealed data — or after seeing the development
data it depends on. No edits after the first confirmatory run starts;
forced deviations are documented in PROGRESS.md.

# Protocol

- Sealed worlds: seeds 200-229 (30 worlds), never inspected before this
  freeze. Configurations: the frozen `configs/v1.yaml` families, batch 2,
  forward order, fast-tuning evaluation mode — identical to the V1
  confirmatory protocol.
- Component A (always runs): the six-point homogeneous rho sweep,
  Continuous vs Dense-C, 360 paired lifetimes
  (`sweep_rho --worlds 200..229 --output artifacts/rho_confirmatory_v2`).
- Component B (conditional): the canonical mixed-profile Benchmark D run,
  shared-residual vs Dense-C vs Continuous, 90 lifetimes
  (`mixed_lifetime`, canonical profile, worlds 200-229). DECISION RULE:
  Component B runs iff the development H9a sign test on worlds 0-9
  rejects at p < 0.05 with a majority of negative per-world correlations.
  If development H9a fails, Component B is dropped and H9a is reported
  as a development-stage negative; no sealed compute is spent on it.
- Exclusions: a crashed run is retried once; a second failure excludes
  that world from all conditions and models of its component, reported.

# Pre-registered primary outcomes (Holm-corrected within each family)

Family A (runs always; 4 outcomes):
1. Slope of the pooled Dense-minus-Continuous effect against measured
   recurrence in [4,000, 7,500] nats per unit recurrence.
2. Pooled-fit zero crossing in [0.40, 0.60] measured recurrence AND mean
   per-world interpolated crossing in [0.42, 0.58].
3. Pooled R^2 in measured-recurrence coordinates >= 0.85 and exceeding
   the configured-rho R^2 by >= 0.15.
4. Sign reversal within worlds: Dense wins at rho = 0 and Continuous at
   rho = 1 in a majority of worlds, exact sign test p < 0.05.
(1-3 restate the intervals already frozen in the V2 spec, section 5;
interval misses are reported as parameter-replication failures even if
signs pass.)

Family B (runs iff Component B runs; 2 outcomes):
5. Allocation: the per-world Spearman correlation between per-primitive
   measured recurrence and per-primitive mean residual fraction
   (true-route-position attribution, the frozen H9a procedure) is
   negative in a majority of the 30 sealed worlds, exact sign test
   p < 0.05.
6. Envelope: shared-residual beats min(L_Dense, L_Continuous) on
   cumulative prequential Gaussian log loss in a majority of sealed
   mixed worlds, sign test p < 0.05 — reported alongside, and never
   without, the two-part-code accounting at lambda = ln 2, which is
   expected to reverse the raw result (a confirmed reversal is a
   CONFIRMATION of the development finding, not a failure).

# Not confirmed by this plan

H7 (advantaged bound), H8 (characterized negative), H6 (boundary
crossing), 002b (manifold inversion), and H10 remain development-stage
results, reported as such: H7/H8's verdicts are about instruments and
bounds rather than population effects; H6 and 002b would each need their
own sealed designs, deliberately deferred; H10's verdict (pending) is a
within-lifetime mechanism finding. The confirmatory currency is spent on
the two claims the eventual papers lean on hardest: the dose-response law
(Family A) and, conditionally, learned allocation (Family B).

# Reporting

Rules inherited verbatim from RELEASE_PLAN.md and V2 spec section 5.
Development and sealed numbers never mixed unlabeled. All six outcomes
reported regardless of direction, including interval misses and the
expected two-part-code reversal.
