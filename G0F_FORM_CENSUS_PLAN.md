# G0F: operator-form census for decision 20 (a) (Tier 0)

Status: FROZEN 2026-10-13. **Tier 0, descriptive census with a registered selection rule; no verdict.** Program:
`GENERALITY_PROGRAM.md`, Track G, decision 20 (a). The PI answered "Sure" to Claude's proposal to survey new operator
forms. A form that passes here is what would, if adopted, change the program rule that the learner does not change
(the learner must take the same form); this census only measures, and adopting a form is reported back before any
learner is built. Worlds 70-72, development band 6 (the same worlds G0 used; the form libraries are new draws on
stream 120, verified unused elsewhere in `src/`).

# Why

G0 (`NO_CANDIDATE`) found that no knob of the rotated form `Q (z + a U tanh(V z + b))` brings the linear
predictability of a computed state from the input below 0.958: identity skip, bounded residual and orthogonal map keep
every operator near an orthogonal map, so a decision on a computed state is never necessary (B1-mid). G3a and G3b then
closed the inference axis inside the same family. Both axes Track G set out to open are closed in the rotated family.
G0F asks, without training a vocabulary, whether a different operator FORM breaks the near-linearity while keeping the
two things formation needs: depth-1 tasks cluster by operation on an untrained library of the same form (N1's premise),
and a single operator of that form can be fitted by gradient.

# Forms (`row.form_world`, version `f1`, d = 16, K = 6)

- `rot0.35`: the current rotated generator (`general_library` at its defaults), the ANCHOR.
- `tanh` (no skip), gain `g` in {0.5, 1, 2, 4}: `P(z) = c_g Q tanh(g R z + b)`; `R`, `Q` Haar, `b ~ N(0, 0.2^2)`,
  `c_g = 1/sqrt(E[tanh(g x)^2])` so variance is kept. Small `g` is near a linear map; large `g` near a sign map.
- `gate` (multiplicative), gain `g` in {1, 2, 4}: `P(z) = Q (z * 2 sigmoid(g (W z + c)))`, `W ~ N(0, 1/d)`,
  `c ~ N(0, 0.2^2)`.
- `fold` (piecewise linear), `j` in {1, 2, 4} folds: `P(z) = Q f_j(...f_1(z))`, `f_i(z) = z - 2 min(0, w_i . z) w_i`
  with unit `w_i`; norm-preserving exactly.

Programs, inputs and task ids are drawn exactly as in the rotated worlds (streams 20/21/30/31, 40/41); only the library
differs. 11 candidates x 3 worlds = 33 jobs.

# Measurements (runner `row.experiments.g0f_form_census`)

| | statistic | denominator |
|---|---|---|
| S1 | G0's `s1_linear`, verbatim: held-out accuracy of the least-squares linear separator on the INPUT for a random unit hyperplane on the state after k teacher steps; k = 1 primary, k = 3 reported | 72 hyperplanes |
| S2 | G0's `s2_runner_up`, verbatim: teacher-library runner-up NMSE among all 216 routes (reported, not registered) | 96 tasks |
| S3 | ARI between teacher labels and support-argmin slot labels on an UNTRAINED SAME-FORM library: 12 slots drawn from the candidate's own form and strength with seed 5000 (the rotated anchor: 12 rotated draws at seed 5000), 60 length-1 tasks; and the mean per-position ARI over all 1,728 routes for the 64 canonical depth-3 tasks | per world (3) |
| S4 | G0's `s4_scale`, verbatim | per world |
| S6 | single-operator learnability: a same-form operator with free parameters (orthogonal `Q = Q0 expm(S - S^T)`; `tanh` free `A`, `b`, scale; `gate` free `A`, `b`; `fold` free normals; rotated free `U`, `V`, `b`, `alpha`), initialized from untrained slot 0, Adam lr 0.01, 2,000 full-batch steps on 128 inputs of one teacher primitive, held-out NMSE on 1,024 inputs | 18 fits (6 primitives x 3 worlds) |

The S3 library is a draw from the form's own prior, which is how a learner of that form would be initialized; for the
rotated anchor it is a teacher-prior draw rather than `FastRotatedDiscreteLibraryLearner`'s initialization, so it is
calibrated below against G0's learner-based S3.

# Baseline pre-measurement (anchor only, computed before any threshold below)

`reports/g0f_baseline_premeasure.json` (same code, anchor only; no new form was measured before freezing):
- S1 0.978 (k = 3: 0.980), S2 1.654, S4 variance 1.017 (max/min 1.175), norm ratio max 1.041: IDENTICAL to G0's
  `reports/g0_baseline_premeasure.json`.
- S3 same-form ARI 0.471 / 0.875 / 0.530 (depth 3: -0.001 / 0.007 / 0.004). G0's learner-initialized library gave
  0.957 / 0.407 / 0.538: same range (minimum 0.47 against 0.41), different per world. The proxy is calibrated in
  aggregate, not per world.
- S6 held-out median 0.00046, max 0.0011: the rotated form is fitted easily from a random same-form start.

# Rules (registered)

Per candidate:
- `s1_moved`: S1 median (k = 1) `<= 0.90` (anchor 0.978), as G0.
- `s3_usable`: same-form depth-1 ARI `>= 0.30` in EVERY world and above that world's depth-3 ARI (anchor minimum 0.47),
  as G0.
- `s4_usable`: all finite; output variance max/min `<= 4`; norm ratio max `<= 3`; median output variance in `[0.25, 4]`,
  as G0.
- `s6_learnable`: S6 median held-out NMSE `<= 0.01` (anchor 0.00046; 0.01 is roughly what a formed vocabulary reaches
  on whole depth-3 tasks, so a single operator that cannot be fitted to it would bottleneck formation).
- A candidate PASSES iff all four hold.

**Gate (A0):** the anchor's S1, S2 and S4 records equal G0's baseline pre-measurement exactly, and its whole record
(every field but seconds) equals this census's own pre-measurement.

**Label (first match):** gate fails -> `UNINTERPRETABLE`; some candidate passes -> `SELECTED`, the passing candidate
with the highest minimum depth-1 ARI (formation's premise is the property most at risk), then lower S1, then key;
otherwise `NO_FORM`. Also reported per form: which strengths pass, the best S1 mover, S2 and whether it falls in G0's
band `[0.01, 0.5]` (a hint that the inference axis could reopen in that family), S1 at k = 3, label balance, timings.

**What the labels mean (stated before data):**
- `SELECTED`: that form breaks near-linearity while keeping formation's premise and single-operator learnability.
  Claude's next step would be to report it and, with the PI's go, build a same-form learner (the program-rule change)
  and run the G1 formation probe on it. Only formation decides whether the form is usable; G0F cannot.
- `NO_FORM`: none of these three departures from the rotated form both breaks near-linearity and keeps the
  premise and learnability; recorded as a result about these forms, and decision 20 (a) returns to the PI.
- `UNINTERPRETABLE`: the census no longer reproduces its anchor.

# Necessity

The necessity gate does not apply to G0F itself, because no vocabulary is trained and no studied behaviour is
elicited: it measures properties of teacher worlds, of an untrained library and of single-operator fits. It is the step
that decides whether a later mid-state rung can have necessity at all: S1 is the refusal cost proxy for a mid-state
decision (the arm that refuses to read the computed state and reads the input instead; B1-mid measured it tied at
0.97-0.98 linear accuracy, which is the scale it is measured at), and the input-reading separator is the impostor that
every mid-state gate must beat. The band for S1 is one-sided here (`<= 0.90`); whether a moved S1 makes a mid-state
gate pay is decided by the re-run necessity census (G3) on a formed vocabulary of the selected form.

# Discriminating power

**The decision rule, as it will be applied:** `s1_moved` iff the median of 72 held-out accuracies `<= 0.90`; then
`s3_usable`, `s4_usable`, `s6_learnable` and the selection order above.

Source `reports/g0f_form_census_design/rates.py`, output `rates_output.txt` (20,000 draws, seed 13). Null: the anchor's
own 72 accuracies resampled. Effect: 72 i.i.d. units around a true median with a spread WIDER than the anchor's
(new forms may give very different accuracies per hyperplane), SD 0.03 / 0.08 / 0.15:

| true median | SD 0.03 | SD 0.08 | SD 0.15 |
|---|---|---|---|
| null (anchor, 0.978) | **0.000** false-fire | | |
| 0.94 | 0.000 | 0.000 | 0.036 |
| 0.92 | 0.000 | 0.047 | 0.184 |
| 0.91 | 0.011 | 0.195 | 0.326 |
| 0.89 | **0.988** | **0.807** | 0.677 |
| 0.88 | 1.000 | 0.957 | 0.825 |
| 0.86 | 1.000 | 1.000 | 0.965 |

So the false-fire of `s1_moved` is 0.000 under the anchor's null and at most 0.047 for a form whose true median is 0.92
with moderate spread; detection is at least 0.80 at a true median of 0.89 with SD up to 0.08, and at 0.86 for SD 0.15.
With a very wide spread, forms within 0.02 of the bar are close to a coin flip, and the reported S1 per form shows how
far each sits from it.

**What the checks do not cover:**
- the 72 accuracies are not independent (24 per world, shared programs within world); the rates assume independence;
- S3 is one ARI per world from one untrained draw, not resampled; S6 is a median of 18 single-start fits, not
  resampled; a candidate near either bar is decided by few values;
- the S3 proxy library is a draw from each form's prior; for the rotated form this differs from the learner's
  initialization, calibrated in aggregate above but not per world;
- the bars are judgements carried over from G0 (S1, S3, S4) or set against the anchor (S6), not measured necessity;
- the form set, strengths and parameterizations are design choices; a negative is about them;
- a passing form must still form a vocabulary (G1) and give a mid-state decision a cost on a FORMED vocabulary (G3).

# Operational

33 jobs, pool 3, under 15 minutes expected (the anchor took ~55 s for three worlds). Records written atomically to
`artifacts/g0f_form_census/full/cells/`, stamped with the protocol hash (this plan, config, the form and general world
modules, G0's runner and baseline, this runner, commit), reused on relaunch only under that stamp; `run.log`,
`status.json`, `run.pid` and `exit.json` in the run directory, archived to `reports/g0f_form_census_<date>/`. Restart
path: relaunching skips completed cells (tested on the dry run). Independent scorer
`row.experiments.score_g0f_form_census` recomputes every statistic from the per-world records, applies the rules,
checks gate A0 against both pre-measurements, and refuses a report whose inputs changed. Tests:
`tests/test_g0f_form_census.py` (form properties, rules, selection order, gate, refusal).

**Dry run disclosure.** One job per new form at its strongest setting (`tanh4`, `gate4`, `fold4`) on world 99 (band 6's
dry-run world, also G1's), printing only timings and finiteness: every path executed, all values finite, 37-59 s per
job; a completed cell was reused bitwise on relaunch and a mismatched stamp recomputed. No statistic of any new form
was looked at before freezing. The anchor pre-measurement is the only run on worlds 70-72.
