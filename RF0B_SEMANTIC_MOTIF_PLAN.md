# RF0b: does shallow semantic canonicalization restore motif recurrence?

Status: FROZEN before RF0b code, decoded routes, null fits, or aggregate output
exist. Development worlds 0–2 only. No new lifetime, route optimization, world,
or sealed seed. This protocol was designed after the complete RF0a result was
observed and therefore cannot confirm RF0a's raw-local-survival claim.

# Motivation and claim limit

RF0a found that a depth-4/6 oracle probe using only raw learner slot identity
predicts the teacher primitive at depths 8 and 10 with mean balanced accuracies
0.764 and 0.661. Structural role, functional geometry, local activations, and
the full trace added essentially nothing. E6 nevertheless found that one exact
learner trigram at the planted teacher-motif site survives only 13.02% and 7.29%
of the time at those depths.

RF0b asks whether these facts compose:

> If each deep learner symbol is mapped through the frozen shallow semantic
> decoder, does the entire planted three-operation motif become stable again?

This is an ORACLE CEILING. Teacher primitive labels train the shallow decoder
and score its deep output. A positive result establishes that a semantic
canonicalizer could repair E6 in principle. It does not establish a label-free
canonicalizer, macro discovery, compression, executable recombination, or an
architectural advantage.

# Frozen inputs

- `reports/rf0a_semantic_recoverability.json`, required to pass its independent
  scorer and classify `RAW LOCAL SEMANTICS SURVIVE`.
- The same E1 discrete artifacts, E6A aggregate, and exact 1,536 E6A route
  caches used by RF0a.
- Worlds 0–2, depths 4/6/8/10, 128 tasks per world/depth, and the exact E6A
  regeneration from `SeedSequence([970, world, depth])` through `plant_corpus`.
- Exactly 64 planted tasks at every world/depth cell; the planted motif has
  length three. Cached routes are observations and are never recomputed.

RF0b reconstructs only raw route symbols, teacher programs, carry flags, and
planted sites. It does not need support inputs, support targets, query inputs,
query targets, learner activations, or model execution. No row-level decoded
route, hidden program, or slot-to-teacher mapping is persisted.

# Frozen decoder

For each world independently, reproduce RF0a's `Z` arm exactly:

1. one-hot encode each raw learner slot over the 12-slot vocabulary;
2. fit weighted six-class one-versus-all ridge at depth 4;
3. select `lambda` from `{1e-4, 1e-2, 1, 1e2}` by depth-6 balanced accuracy,
   breaking ties toward the largest value;
4. recompute preprocessing and class weights and refit on depths 4+6;
5. freeze the resulting slot-to-class decoder and apply it once at depths 8
   and 10.

The SVD solver, `1e-10` relative cutoff, standardization, inverse-class-frequency
weights, unpenalized intercept, and lowest-index argmax tie rule are identical
to RF0a. RF0b must reproduce RF0a's selected lambda, feature count, depth-8/10
confusion matrices, per-class recalls, and balanced accuracies exactly before a
motif score is valid.

A ROLE-ONLY decoder is also reproduced from RF0a's `R` arm and refit under the
same split. It is a negative control, not an eligible canonicalizer.

# Primary estimands

For planted task `t`, let `s_t` be its registered planted site, let
`p_t[s_t:s_t+3]` be the teacher motif, and let `c_t` be the deep learner route
after every raw slot is decoded by the frozen shallow `Z` probe.

Define, separately for each world and depth:

    C_exact = mean_t 1[c_t[s_t:s_t+3] == p_t[s_t:s_t+3]]
    L_exact = E6 survival of its top raw learner trigram at the planted site
    Delta_sem = C_exact - L_exact

`C_exact` is joint three-position semantic recovery, not the mean of three
position-wise accuracies. `L_exact` is imported from the already-frozen E6A
aggregate and reproduced from the cached routes before comparison.

The materiality rule, frozen after RF0a and before RF0b output, is:

    C_exact >= 0.30  and  Delta_sem >= 0.20

at BOTH depths 8 and 10 in the SAME at least 2 of 3 worlds. The 0.30 absolute
bar requires a practically recurring joint object. The 0.20 contrast matches
the previously frozen RF materiality margin and prevents a numerically small
relabeling gain from motivating a new representation.

# Secondary diagnostics

Report without replacing the primary:

- accuracy at each of the three relative motif positions;
- the product of those three marginal accuracies;
- `C_exact - product`, showing positive or negative dependence among errors;
- pairwise phi correlations among the three correctness indicators, with zero
  denominator reported as `null` rather than coerced;
- the most frequent decoded trigram at planted sites and its recurrence rate;
- whether that modal decoded trigram equals the teacher motif;
- the fraction of unplanted decoded routes containing the teacher motif at
  least once;
- the corresponding joint score and primary contrast for the ROLE-ONLY decoder.

The modal sequence is stored in teacher-class coordinates because RF0b is an
oracle audit. It is not exported as a model or made available to a future
non-oracle learner.

# Nulls and non-vacuity

All are required for a positive classification:

1. RF0a's scorer passes and its recorded decision is raw local survival.
2. Exactly 1,536 valid E6A cache cells exist under the imported protocol.
3. RF0a `Z` and `R` arms reproduce exactly, including deep confusion matrices.
4. The E6 top gram, planted hits, survival rate, and displayed four-depth means
   reproduce exactly.
5. Every scored cell contains exactly 64 planted and 64 unplanted tasks, valid
   sites, six-class shallow coverage, and finite metrics.
6. The observed `Z` canonical score exceeds the ROLE-ONLY score by at least
   0.20 at each depth/world used for a positive verdict.
7. For 200 deterministic draws per world, permute depth-4 and depth-6 labels
   independently within absolute position, rerun the full `Z` lambda selection
   and refit, and score joint deep motif recovery against real programs. Seeds
   are `SeedSequence([986, world, draw])`. An observed `C_exact` must be
   STRICTLY greater than the depth-specific null p99 in each supporting cell.
8. A synthetic fixed slot-to-class decoder recovers every three-token motif;
   independently shuffled semantic labels do not exceed 0.10 joint recovery.

The best null draw is never selected or inspected as a model. Teacher labels
enter null fitting because this is a falsification control for an explicitly
oracle decoder; they do not enter any future label-free model.

# Decision ladder

Apply in order after all structural validity checks:

1. **SEMANTIC CANONICALIZATION CEILING EXISTS** if the absolute and materiality
   gates, role gap, and permutation-null gate pass at both depths in the same at
   least 2 of 3 worlds.
2. **SEMANTIC MOTIF HORIZON BETWEEN 8 AND 10** if the complete gate passes at
   depth 8 in at least 2 worlds but at depth 10 in fewer than 2.
3. **LOCAL DECODABILITY DOES NOT RESTORE MOTIFS** if `C_exact < 0.15` and
   `Delta_sem < 0.10` at both depths in the same at least 2 worlds.
4. Otherwise **RF0B UNRESOLVED**.

Fewer than two structurally scoreable worlds yields **RF0B UNSCOREABLE**, not a
negative. A failed role/null control makes that world ineligible for a positive
classification but does not rewrite its observed `C_exact`.

# Registered prediction and successor

Based on the already-observed RF0a means, independence would suggest joint
recovery near `0.764^3 = 0.447` at depth 8 and `0.661^3 = 0.289` at depth 10.
This calculation is an observed design input, not new evidence. The registered
prediction is **SEMANTIC MOTIF HORIZON BETWEEN 8 AND 10**: depth 8 will clear
the full gate in at least two worlds, while depth 10 will miss either the 0.30
absolute bar or the 0.20 gain in at least two worlds. Correlated route errors
could instead make the joint recovery substantially larger or smaller than the
independence estimate.

If the ceiling exists at both depths, the next experiment is a separately
frozen label-free canonicalizer built from functional substitutability among
the 12 frozen slots. If there is a horizon, that canonicalizer is tested only
through the passing depth and the horizon is preserved. If local decodability
does not restore motifs, stop local semantic-macro work and move to whole-
program or trajectory-level synthesis.

# Artifact and acceptance

Implementation:
`src/row/experiments/audit_rf0b_semantic_motif.py`. Independent scorer:
`src/row/experiments/score_rf0b.py`. Tests:
`tests/test_rf0b_semantic_motif.py`. Output:
`reports/rf0b_semantic_motif.json`, written atomically only after all cells and
null draws complete.

Accept only after committed launch code; exit code 0; expected world/depth/task
counts; exact RF0a and E6 reproduction; finite metrics; synthetic and null
controls; launch/plan/input hashes; report freshness; independent scorer; full
unit suite; `git diff --check`; `tools/check_prereg.py`; and
`tools/check_invalid.py`. Preserve a negative, horizon, unresolved, invalid, or
unscoreable outcome without changing the thresholds.

