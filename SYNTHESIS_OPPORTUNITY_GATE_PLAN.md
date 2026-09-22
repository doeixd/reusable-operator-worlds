# SG0: the synthesis opportunity gate

Status: DRAFT, 2026-09-22. Tier 0, existing artifacts only, no new world, no
learner, no lifetime. Requires PI approval before implementation.

This plan asks ONE question, one level above any rung:

> On this substrate, is there any headroom for a learned program proposer at
> all — and if so, where?

It exists because eight rungs have now failed for absence of opportunity rather
than learner failure, and because the opportunity-gate rule (review 68) has been
applied after building each rung instead of before. SG0 applies it to the GOAL.

# Why this is the right question now

Four consecutive L0d gates found no route ambiguity on a usable frozen
vocabulary: support 128/32/8 (preflight), support 4/2/1 (sparse-evidence gate),
depth four and depth five. Independently, E5 found a learned writer that cost
more than not having one (7.6-36.7 s/task against search at 0.30 s) and failed
on quality (+0.32 / +0.31 against a registered <= 0.15), and E5.1 found search
cost logarithmic in program count — the space grew 3.58e7-fold while route
optimization's seconds grew 3.30-fold, at oracle parity.

Working reading, to be tested here and not assumed: these are the SAME
observation. Support data in this substrate overdetermines the route, so the
incumbent (enumerate, or optimize the route code) is already at the ceiling, and
an amortized proposer has nothing to sell. If that is true, every writer,
proposer, discovery and uncertainty rung on this substrate is dead on arrival,
and SG0 says so in hours instead of one rung at a time.

# What the L0d gates measured, and what they did not

All four asked whether the support-selected route CHANGES under an
intervention. None asked the question that decides headroom:

- **is the support-optimal route also the query-optimal route, and if not, what
  does choosing it cost?**

That quantity — commitment regret — is the headroom. A proposer, a posterior, a
beam or a learned prior can only ever recover some of it. If it is ~0, no
inference mechanism can pay, however clever, and PX7(a) has nothing to measure.
The four negative gates are consistent with regret 0 and with regret large;
they do not distinguish these, which is why they did not settle anything.

# Estimands

On a frozen library L, a program tau, a support set S and a query set Q, with
all routes r enumerated:

- `r_hat(S)` = argmin over r of support loss — what any committed learner picks.
- `r_star(Q)` = argmin over r of query loss — hindsight best, an ORACLE, not
  attainable; it bounds every mechanism from above.
- **`regret = NMSE_Q(r_hat(S)) - NMSE_Q(r_star(Q))`**, the PRIMARY estimand.
- **`identifiability`** = the normalized support-loss gap between `r_hat(S)` and
  the best route functionally distinct from it.
- **`near-tie disagreement`** = among routes within a registered support-loss
  tolerance of `r_hat(S)`, the spread of query NMSE. This is regret's mechanism:
  ties that agree on query carry no headroom; ties that disagree do.
- `C_enum`, `C_opt` seconds, reported separately from quality, per E5's rule.

Report all four against each other, not pooled into one number.

# Selection-noise guard (mandatory)

`r_star(Q)` selected and scored on the same query set overfits it: with 1,728 to
248,832 routes against 256 examples, some route wins by noise. Never fit and
score a selection on the same objects (V5 lesson). Split Q into Q_a and Q_b by
a registered seed; select `r_star` on Q_a, score regret on Q_b. Register the
NULL CALIBRATION: recompute regret with `r_hat` selected on a random half of S
and a disjoint half — the sampling-noise floor. A regret not clearly above that
floor is ZERO for this plan's purposes and must be reported as such.

# Cells

Existing frozen J2A staged libraries and their controls, as used by the L0d
preflight. No new world, model or training.

| axis | values |
|---|---|
| library | STAGED5000 / STAGED6000 (usable) and their NONSTAGED controls |
| world | 0, 1, 2 |
| depth | 3, 4; depth 5 only if 3-4 show nonzero regret |
| support | 128, 8, 2 |

Depth 3 is 1,728 routes and depth 4 is 20,736; both enumerate fully with query
losses as well as support losses. Depth 5 is 248,832 and reuses the committed
chunked evaluator (`MEMORY_SAFE_SEARCH`, c72b02b) — it is included only if the
cheaper depths find headroom, per the scoring-economy rule.

Order cells cheapest- and most-decisive-first: depth 3, support 128, staged,
world 0 is the single cell most likely to make the outcome impossible, and is
run first.

# Registered triage

Computed before any interpretation, with the thresholds fixed here:

- **NO-HEADROOM** — median regret is within the null-calibration floor in every
  staged cell. Consequence: the amortized-proposer branch is CLOSED on this
  substrate. PX7 is retired as unmeasurable here, not as refuted; L0d's full
  census is withdrawn rather than left unfrozen; and any synthesis claim
  requires a different generator (see `notes/identifiability-sketch.txt`).
- **HEADROOM-LOCALIZED** — median regret clears the floor in some cells and the
  cells are ordered by an identifiability or near-tie-disagreement statistic.
  Consequence: that statistic is the MEASURED ambiguity source the L0d census
  has been missing; the census may then be frozen around it, with its
  independent variable being identifiability rather than support size.
- **HEADROOM-UNEXPLAINED** — regret clears the floor but tracks no measured
  statistic. Consequence: report it, and do NOT freeze a mechanism census; this
  is the route-margin failure repeating, and a census without a comparable
  independent variable produced three withdrawn candidates last time.

The triage must be able to come out any of three ways on the artifacts we hold.
If a dry run shows regret is identically zero by CONSTRUCTION — for example if
`r_hat(S) == r_star(Q)` is forced by how the evaluation set is generated — the
gate is an implementation check, not an experiment, and this plan is withdrawn
before it runs (review 83).

# Non-vacuity checks

- The NONSTAGED control libraries must show LARGE regret. A poor library whose
  routes change constantly under support reduction (65/96 at support 32, 93/96
  at support 8) must not read as zero headroom; if it does, the instrument is
  broken, not the substrate.
- Regret must be non-negative by construction on Q_a and may be negative on Q_b
  only within the noise floor; systematic negatives mean the split leaked.
- Reducing support must not reduce regret. If it does, the ordering is inverted
  and the metric is misread.
- The identifiability statistic must be COMPARABLE across cells before it is
  correlated with anything (the route-margin normalization lesson): register it
  as a library-intrinsic, within-cell-normalized quantity, and report the
  un-normalized version only as a diagnostic.

# Cost

Enumeration with query losses at depth 3 is milliseconds per cell and at depth 4
was 10.04 s for 16 programs. The whole grid excluding depth 5 is minutes. Depth
5 is about 4 minutes per program. Tier 0 throughout; no lifetime, no pool, no
host-memory precondition beyond the committed chunked evaluator.

# What this plan does NOT do

It does not test any inference MECHANISM (beam, posterior, commit-late,
amortized proposal). It measures the ceiling those mechanisms would compete for.
It is not a PX7 verdict under any outcome, and a NO-HEADROOM result is a
statement about this substrate, not about program synthesis.

# Acceptance

Independent scorer recomputing the triage from the raw report; protocol
fingerprint over plan, config, library hashes, seeds, splits and implementation;
restart reuses completed cells bitwise; `check_prereg.py` and `check_invalid.py`
pass; operational records archived to `reports/` and committed with the result.
The equivalence check named above is written as a TEST before launch, per the
depth-five lesson.
