# G3b: at what observation noise does route choice bind on the formed vocabularies? (Tier 0 census)

Status: FROZEN 2026-10-13. **Tier 0, census over saved artifacts; no verdict.** Program: `GENERALITY_PROGRAM.md`, Track
G, step G3; decision 21 option (i), recommended by Claude after G3a and taken up when the PI answered "Continue. Use
your best judgement". Worlds 70-72, development band 6.

# Why

G3a (`NO_HEADROOM`, `45f6e55`) found that on the formed `m = 4` vocabularies route choice from 2 support examples is
as reliable as from 128: the targets are noiseless, so any wrong route that is not a near tie is excluded by a few
inputs, however small its margin. Observation noise is the knob that can make choice data-limited. This census asks,
on the six vocabularies that already exist and without training anything, at what noise level route choice starts to
cost query error, and whether the `m = 4` vocabularies (runner-up margin 0.7-3.9x) bind at a lower noise level than the
current generator's (79-148x). A window where `m = 4` binds and `m = 0` does not, at a noise level formation might
tolerate, is what would justify a probe that FORMS under noise.

# Construction

G3a's rebuild of G1's six `RW_SLEEP` vocabularies (anchor A1: per-task terminal bitwise equal to G1's record) and
G3a's 64 held-out depth-3 programs per world. For each program, Gaussian noise is added to the SUPPORT targets only:
`y_s + sqrt(v * var(y_s)) * eps`, with `var` over all support target entries, `eps` standard normal from
`SeedSequence([3201, world, program])`, the SAME `eps` at every noise level and for both `m` (common random numbers;
the two `m` are paired programs and inputs). The query targets stay clean, so the regret is the true query cost of
choosing from noisy support. Noise levels `v` in {0, 0.01, 0.03, 0.1, 0.3, 1.0}, `v` being the noise-to-signal ratio in
NMSE units (a formed vocabulary fits its tasks at NMSE ~0.01, its `m = 4` runner-up sits at 0.02-0.05, its `m = 0`
runner-up at 0.74-1.53).

Estimands: G3a's, verbatim (SG0's argmin with lowest-index ties, split into query halves, regret, 200-draw bootstrap
floor on the noisy support, near ties at 1%, inference failure), seeded per program exactly as G3a. Supports 128, 8, 2.
**Anchor A3:** at `v = 0` every program's row equals G3a's committed row bitwise (the route predictions are computed
once and the squared error is the second half of `sg0.per_example_squared_error`, verbatim).

# Rules (registered)

A cell is one (`v`, support) for one `m`, over 192 programs (3 worlds x 64). Its bind fraction is the share of programs
whose regret exceeds their own bootstrap floor (G3a's rule B statistic). Plausible noise: `v <= 0.1` (a judgement:
noise variance at most a tenth of the signal; whether formation tolerates it is the question of a later probe).
- WINDOW cell: `v <= 0.1`, `m = 4` fraction `>= 0.10`, and it exceeds the `m = 0` fraction at the same cell by `>= 0.10`.
- BOTH cell: `v <= 0.1`, `m = 4` fraction `>= 0.10`, excess `< 0.10`.
- HIGH cell: `v > 0.1`, `m = 4` fraction `>= 0.10`.

**Label (first match):** any WINDOW cell -> `NOISE_WINDOW`; else any BOTH cell -> `NOISE_ONLY`; else any HIGH cell ->
`HIGH_NOISE_ONLY`; else `NO_BIND`. Also reported: the onset noise level (smallest `v` with fraction `>= 0.10`) per arm
and support, every fraction, inference failures and identifiability.

**What the labels mean (stated before data):**
- `NOISE_WINDOW`: at a plausible noise level, the near-similar `m = 4` vocabulary makes route choice data-limited while
  the current generator's does not. Successor: a Tier 1 G1 probe that FORMS under that noise level at `m = 4` (does
  formation survive it?), and if it does, a learned-chooser rung sized on the binding cells.
- `NOISE_ONLY`: noise alone makes choice bind at a plausible level, and `m` adds little. Successor: the same probe at
  `m = 0` (the current generator) with noise; `m` is not needed for the inference axis.
- `HIGH_NOISE_ONLY`: choice binds only when noise is at least 30% of the signal. Claude's reading would be to close the
  learned-inference line in this program (decision 21 (iii)), since a world that noisy is unlikely to form.
- `NO_BIND`: even noise as large as the signal leaves choice free; same consequence, more strongly.

# Necessity

The necessity gate does not apply to this census itself, because no learner is trained and no behaviour is elicited:
it rebuilds frozen artifacts deterministically (anchors A1, A3) and evaluates fixed routes against noisy support. It
is the necessity census for a later learned-chooser rung: exhaustive support argmin is the arm that refuses learned
inference, and regret above the bootstrap floor is the measured cost of that refusal, at the scale of query NMSE.

# Discriminating power

**The decision rule, as applied:** per (`v`, support) cell with 192 programs per arm, WINDOW iff `v <= 0.1`, the
`m = 4` bind fraction `a >= 0.10` and `a - b >= 0.10` with `b` the `m = 0` fraction; label by first match as above.

Source `reports/g3b_noise_census_design/rates.py`, output `rates_output.txt` (independent binomial arms at true bind
rates `(qa, qb)`, 100,000 draws, seed 12):

| true `(qa, qb)` | P(WINDOW) | P(BOTH) |
|---|---|---|
| (0, 0), the null | **0.0000** | 0.0000 |
| (0.15, 0.15), equal binding | 0.0029 | 0.9715 |
| (0.30, 0.30), equal binding | 0.0152 | 0.9848 |
| (0.15, 0), window effect | **0.9751** | 0 |
| (0.30, 0.10) | 0.9937 | 0.0063 |
| (0.30, 0.20) | 0.4865 | 0.5135 |

So the per-cell false-fire of WINDOW is 0.0000 under the null and at most 0.015 under equal binding, and its detection
is 0.975 when the `m = 4` vocabularies bind 15% of programs and the `m = 0` ones none (0.457 at 10%, the coin-flip
point).

The window margin was raised from 0.05 to 0.10 while writing this section, because at 0.05 equal binding at rate 0.30
false-fired WINDOW in 0.147 of draws per cell; at 0.10 it is 0.015. Recorded here so the change is visible.

**What the checks do not cover:**
- the label takes the first WINDOW among up to 12 plausible cells (4 levels x 3 supports); per-cell false-fire under
  equal binding is at most 0.015, but the family-wise rate is higher, by an amount the correlated cells (same programs,
  same `eps` scaled) make hard to state. The number of WINDOW cells is reported, and a single WINDOW cell is read as
  weak;
- the arms are not independent binomials (paired programs, common noise), which makes the margin test conservative if
  anything, but the rates above assume independence;
- a formed vocabulary is used for noiseless-trained libraries only: a vocabulary formed UNDER noise may differ, which
  is exactly what the successor probe would test;
- `PLAUSIBLE_V = 0.1` and the 0.10 thresholds are judgements sized by the rates, not measured tolerances;
- six vocabularies from a Tier 1 run, depth 3, supports 128, 8, 2.

# Operational

6 jobs (`m`, world), pool 3: rebuild (~150 s) plus 64 programs x 6 noise levels x 3 supports x 200 bootstrap reductions
over precomputed route predictions. Expected under 20 minutes. Records written atomically to
`artifacts/g3b_noise_census/cells/`, stamped with the protocol hash (this plan, config, G1 and G3a reports, G1's
terminal hashes, reused modules, commit), reused on relaunch only under that stamp; `run.log`, `status.json`,
`run.pid`, `exit.json`, `precondition.json` archived to `reports/g3b_noise_census_<date>/`. Dry run: two real
(`m`, world) pairs at 4 programs (anchors A1 and A3 and every path), with the restart test. Independent scorer
`row.experiments.score_g3b_noise_census` recomputes the fractions and label from the per-program rows.

**Dry run disclosure.** The dry run is two REAL pairs (`m = 4` and `m = 0`, world 70) at 4 programs each, run after
the rules above were written (the window margin change came before it) and before freezing. Anchors A1 and A3 held.
At support 2 one of the 4 `m = 4` programs bound from `v = 0.03` and two at `v = 1`; one `m = 0` program bound at
`v = 1` only; nothing bound at supports 8 or 128. These 8 program-arms are among the census's 384; nothing in the rules
changed after the dry run. Timing: rebuild ~260 s, ~2.6 s per program.
