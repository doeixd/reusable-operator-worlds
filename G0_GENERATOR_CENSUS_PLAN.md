# G0: generator census for the generality program (Tier 0, no lifetimes)

Status: FROZEN 2026-10-13. **Tier 0, descriptive census with a registered selection rule.** Program:
`GENERALITY_PROGRAM.md`, Track G, step G0 (decision 19). Development band 6: worlds **70-72** (verified unused before
allocation; the baseline pre-measurement below is the only earlier use, made with the same code).

# Why

Every ROW result comes from one generator, and its recent rungs fail necessity because that generator is easy in two
measured ways: a decision on a computed state is ~98% a linear decision on the input (B1-mid), and a formed
vocabulary makes route inference trivial (SG0). G0 asks, without training anything, whether a variant of the
generator moves either property out of its saturated region while keeping the property that makes formation
possible at all (depth-1 tasks cluster by operation on an untrained library, N1's premise) and keeping the task
scale sane. It picks the generator G1 will run the confirmed pipeline on, or records that none qualifies.

# Generator

`row.general_world` (version `g1`): `P_k(z) = Q_k (z + a U_k tanh(g V_k z + b_k))`, with
- `a` = `alpha` (residual scale) and `K` = `teacher_primitives`, existing `WorldConfig` fields;
- `g` = hidden gain (new);
- `m` = shared rotation component (new): `Q_k = polar(G_k + m G_shared)` with `G_k` the current rotation draw
  (stream 12) and `G_shared` one draw per world (stream 50, unused elsewhere). At `m > 0` the rotation is the polar
  factor rather than QR (both are Haar for a Gaussian draw, so this is a different but equally random family); `m = 0`
  keeps the QR draw.

At `g = 1, m = 0` the module reproduces `generate_rotated_world` and `generate_curriculum_world` BITWISE on real worlds
(`tests/test_general_world.py`: worlds 57, 70, 71, 72; K = 8 and alpha = 1.0 pass-through). Every knob stays inside the
learner's operator family (`RotatedLearnedOperator` learns `U`, `V`, `b`, `alpha` and `Q` freely), so a later learner
failure cannot be a representational mismatch. State width `d = 32` and operators outside the family (a second
nonlinearity) are deliberately not candidates: the first doubles every cost before any reason to pay it, the second
would confound world difficulty with learner misalignment (the GELU lesson).

**Candidates (39):** `(a, g)` in {(0.35,1), (0.7,1), (1.0,1), (1.5,1), (0.35,2), (0.35,4), (0.7,2), (1.0,2), (0.7,4)}
x `K` in {6, 8, 12} at `m = 0` (27); and `m` in {2, 4, 8, 16} x `(a, g)` in {(0.35,1), (1.0,1), (0.7,2)} at `K = 6`
(12). The base `(0.35, 1, 6, 0)` is the current generator.

# Measurements (per candidate, worlds 70-72; runner `row.experiments.g0_generator_census`)

| | statistic | denominator |
|---|---|---|
| S1 | held-out accuracy of the least-squares linear separator ON THE INPUT for a random unit hyperplane on the state after k teacher steps (random program); fit on 2,048 inputs, scored on 2,048 others; k = 1 primary (B1-mid's case), k = 3 reported | 72 hyperplanes (24 per world) |
| S2 | teacher-library runner-up NMSE: support NMSE (128 training examples) of the best WRONG route among all `K^3` routes for a canonical depth-3 task (the true route is exactly 0; checked) | 96 tasks (first 32 canonical per world) |
| S3 | ARI between teacher labels and support-argmin slot labels on an UNTRAINED learner library (slots `2K`, model seed 5000) for 60 length-1 tasks; and mean per-position ARI on the 64 canonical depth-3 tasks | per world (3) |
| S4 | per-task output variance of the 64 canonical tasks (median, max/min); median state-norm ratio after five random steps (64 programs; median, max); finiteness | per world |
| S5 | seconds for 200 learner updates at slots `2K` (reported only) | one timing per K |

Teacher programs enter only these measurements. Nothing is trained.

# Baseline pre-measurement (current generator, computed before any threshold below)

`reports/g0_baseline_premeasure.json` (same code, base candidate only): S1 median **0.978** (k = 3: 0.980; B1-mid
reported 0.97-0.98 in-sample); S2 median runner-up NMSE **1.65** (0 of 96 tasks below 0.05: every wrong program is about
as far as an unrelated one, because each primitive has an independent random rotation); S3 depth-1 ARI **0.957 / 0.407 /
0.538** against depth-3 0.003 / -0.007 / 0.001 (N1 reported 0.46-0.77); S4 output variance 1.02 (max/min 1.18), norm
ratio 1.02 (max 1.04); S5 about 5 s per 200 updates at K = 6. The S2 result is why `m` was added before freezing: no
knob that leaves the rotations independent can bring a wrong program near a right one.

# Rules (registered)

Per candidate:
- `s1_moved`: S1 median (k = 1) `<= 0.90` (baseline 0.978).
- `s2_in_band`: S2 median runner-up NMSE in `[0.01, 0.5]` (baseline 1.65). Above 0.5 no wrong program explains even
  half the variance (inference stays trivial); below 0.01 a wrong program fits as well as a formed vocabulary fits
  its own tasks (O6-era canonical medians ~0.01), so inference would be hopeless rather than hard.
- `s3_usable`: depth-1 ARI `>= 0.30` in EVERY world and above that world's depth-3 ARI (baseline minimum 0.41).
- `s4_usable`: all finite; output variance max/min `<= 4`; norm ratio max `<= 3`; median output variance in
  `[0.25, 4]`.
- `moves` = `s1_moved + s2_in_band`; a candidate PASSES iff `moves >= 1` and `s3_usable` and `s4_usable`.

**Gate:** the base candidate's records reproduce the baseline pre-measurement exactly (every field but seconds).

**Label (first match):** gate fails -> `UNINTERPRETABLE`; some candidate passes -> `SELECTED`, the first passing
candidate by (more `moves` first, smaller `K`, fewer knobs changed from the base, lower S1, key); otherwise
`NO_CANDIDATE`. Also reported: the best passing-condition S1 mover and S2 mover separately (by smaller K, fewer
knobs, lower S1), so G1 can be planned on either axis.

**What the labels mean (stated before data):**
- `SELECTED`: G1 runs the confirmed online pipeline on that generator. If it moves only one row, the rungs the other
  row closed stay closed on it.
- `NO_CANDIDATE`: within the rotated family and this grid, no variant both breaks a saturation and keeps depth-1
  clustering; recorded as a result about the family, and the next G0 must change the family itself.
- `UNINTERPRETABLE`: the census code no longer reproduces its own baseline.

**Reported, not registered:** every statistic per candidate and world; the k = 3 S1; S1 label balance; S2 fraction
below 0.05 and second-wrong NMSE; S5 timings.

# Necessity

The necessity gate does not apply to G0 itself, because no learner is trained and no behaviour is elicited: it
measures properties of teacher worlds and of an untrained library. G0 is the step that decides whether LATER rungs
can have necessity: S1 is the refusal cost proxy for mid-state decisions (B1-mid measured the refusing input gate tied
with the mid-state gate at 0.97-0.98 linear accuracy), and S2 is the impostor scale for inference (how close the
best wrong program comes). Each later rung re-runs its own necessity census on the selected generator (G3).

# Discriminating power

**The decision rules, as they will be applied:** `s1_moved` iff the median of 72 held-out accuracies `<= 0.90`;
`s2_in_band` iff the median of 96 runner-up NMSEs lies in `[0.01, 0.5]`; `s3_usable`, `s4_usable` and the selection
order as above.

Source `reports/g0_generator_census_design/rates.py`, output `rates_output.txt` (2,000 draws, seed 11; units
resampled within world from the baseline pre-measurement; null = the current generator's own units; effect = the same
units shifted to a target median):

| rule | null (current generator) false-fire | rate at true median just outside the rule | detection at true median just inside |
|---|---|---|---|
| `s1_moved` | **0.000** (median 0.978) | 0.000 at 0.91 | **1.000** at 0.89 |
| `s2_in_band` | **0.000** (median 1.65) | 0.000 at 0.55; 0.000 at 0.008 | **1.000** at 0.45 and at 0.02 |

Bootstrap SDs of the medians are 0.0006 (S1) and 0.007 (S2): the rules resolve a true median 0.01 from either S1 bar
and 0.05 from the S2 upper bar.

**What the checks do not cover:**
- S3 is one ARI per world from one untrained library; it is not resampled. The baseline sits well above the bar
  (minimum 0.41 against 0.30), but a candidate near 0.30 is decided by three single values.
- The bands are judgements about what makes later rungs possible, not measured necessity: S1 at 0.90 has not been
  shown to make a mid-state gate beat an input gate, and S2 in band has not been shown to make learned inference
  pay. G3's censuses decide those on the selected generator.
- Untrained-library clustering is N1's PREMISE, not formation; G1 decides formation.
- S1 thresholds the projection at 0. If a candidate's states have a large mean, labels become unbalanced and a
  linear separator scores high by predicting the majority; that biases S1 toward NOT moving (conservative). Label
  balance is reported per candidate.
- The scaling of `m` (polar factor of a Gaussian sum) and the grid are design choices; a negative is about this grid.
- Three development worlds.

# Operational

117 jobs (39 candidates x 3 worlds), pool 3, largest K first, expected under an hour. Each (candidate,
world) record is written atomically to `artifacts/g0_generator_census/full/cells/` stamped with the protocol hash
(plan, config, generator, runner, commit) and reused on relaunch only under the same stamp; `run.log` has one line per
cell; relaunching the same command resumes. Restart path tested on the baseline pre-measurement. Independent scorer
`row.experiments.score_g0_generator_census` recomputes every statistic from the per-world records, re-checks the base
candidate against the pre-measurement, and refuses a report whose plan, config, generator or runner changed. Tests:
`tests/test_general_world.py` (bitwise reduction) and `tests/test_g0_generator_census.py` (rule, selection order,
gate, refusal).
