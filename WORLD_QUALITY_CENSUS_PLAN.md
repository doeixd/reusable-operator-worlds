# World-quality census: does early-stage quality predict online staged success? (Tier 0)

Status: written before any census number exists, committed with its code and
tests, as the SO2 interference census was. Tier 0, descriptive: it reads only
committed report JSON, trains nothing, opens no world, and produces NO verdict.
It cannot change `SO2_FAILS`, `SO3_FAILS` or `SO4_FAILS`.

# Question

Three preregistered online runs reached the terminal criterion in 6 of 10
development worlds, at three learner seeds, and within a single world the replay
stream alone moved terminal error by up to an order of magnitude. World identity
dominates. The successor intervention depends on WHERE the failure originates:

- if a cell's stage-3 outcome is predictable from its stage-1/stage-2 prefix,
  the next intervention acts on the EARLY stages (the curriculum's own
  formation);
- if it is not, the failure arises in stage 3 itself, and the successor is a
  stage-3 reliability intervention.

Nothing in the three runs has asked this, and it is answerable with no compute.

# Data (committed reports only; no artifact directory is read)

| source | report | cells | seed | streams |
|---|---|---|---|---|
| SO2 | `reports/so2_online_gate.json` | `cells.STAGED_w{0,1,2}` stages 1-3 | 5000 | 1 (canonical) |
| SO3 | `reports/so3_consolidation.json` | `prefixes.w{3,4,5}_s{0,1,2}` stages 1-2 + `cells.w*_s*_BASE` | 6000 | 3 |
| SO4 | `reports/so4_b2_retest.json` | `prefixes.w{6..9}_s{0,1,2}` + `cells.w*_s*` | 7000 | 3 |

24 staged cells (3 + 9 + 12). SO3's LR_HALF and STORE_8 arms are EXCLUDED from
the primary analysis (they are not the unchanged protocol) and reported only as
a labelled sensitivity.

# Predictors, outcome, and statistics

Per cell, all already recorded:
- **Predictors:** stage-1 terminal median; stage-2 terminal median; stage-2
  end-of-task median; the stage-1-to-stage-2 improvement ratio.
- **Outcome:** stage-3 terminal median `M`, and the boolean `M <= 0.05`.
- **Statistics:**
  - Spearman rank correlation of each predictor with `M`, pooled over the 24
    cells and within each run;
  - the median of each predictor among passing versus failing cells, and their
    ratio;
  - a permutation null for the pooled Spearman: 2000 permutations,
    `SeedSequence([2029])`, reporting the one-sided fraction at or above the
    observed value;
  - the same analysis restricted to worlds with MIXED stream outcomes (some
    streams pass, some fail), where world identity is held constant.

No threshold is fitted and no classifier is trained. Any separating value is
reported as descriptive.

# Guards (the census refuses to write a report unless all pass)

- Cell counts are exactly 3 (SO2), 9 (SO3 BASE), 12 (SO4).
- Every predictor and outcome is finite.
- Each report's own recorded classification is present and matches the known
  labels (`SO2_FAILS`, `SO3_FAILS`, `SO4_FAILS`), so a report replaced by a
  different run cannot be read silently.
- Stage numbering is present for every cell (1, 2 and 3 for SO2; 1 and 2 in the
  prefix plus the stage-3 cell elsewhere).
- Pass/fail counts reproduce each run's published per-world result.

# Triage (decides only which successor plan is worth writing)

- **PREFIX-PREDICTIVE:** pooled Spearman `>= 0.5` with permutation fraction
  `<= 0.05`, AND the failing cells' median stage-2 terminal is at least 2x the
  passing cells'. Then write a successor acting on stages 1-2 (for example,
  budget or stopping criteria for the early stages), and use the prefix statistic
  as its registered pre-launch eligibility screen.
- **STAGE3-LOCALIZED:** pooled Spearman `<= 0.2` or permutation fraction `> 0.2`,
  AND no 2x separation. Then the prefix does not carry the outcome, and the
  successor is a stage-3 reliability intervention. World 6's failure would then
  need its own explanation.
- **MIXED:** anything else, including a pooled effect that vanishes within
  mixed-outcome worlds. Report and take no successor decision from it.

# Disclosed confounds

- **Seeds differ across runs** (5000, 6000, 7000), so a pooled correlation mixes
  three learner initializations. The within-run correlations are reported beside
  the pooled one.
- **SO2 contributes one stream per world**, SO3 and SO4 three.
- **SO3's BASE cells share their prefixes** with that run's candidate arms; the
  prefix is stage-1-2 only, so this does not leak stage-3 information.
- **The 0.05 threshold** is SO2's registered terminal threshold, reused here for
  labelling only.
- **24 cells is small**, and the permutation null is the only inferential claim
  the census makes.

# What this cannot establish

It cannot show that improving a prefix would improve stage 3: the relation is
observational across existing runs. It says nothing about worlds outside 0-9,
about other protocols, or about why any individual world fails. A
PREFIX-PREDICTIVE outcome licenses a plan, not a mechanism.

# Cost

Seconds. It reads three JSON files, uses no torch, and never runs beside a
lifetime.
