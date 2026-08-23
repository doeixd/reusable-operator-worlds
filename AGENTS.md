# Project intent

Reusable Operator Worlds (ROW) tests whether reusable computational substrates
reduce lifetime/prequential learning cost on tasks generated from recurring
hidden neural operations.

This is a careful scientific research project, not a demo, benchmark chase, or
rapid-prototyping exercise. Correctness of the experimental construct,
traceability of every number, and honest treatment of negative or invalid
results take priority over throughput and narrative continuity. A plausible
number is not a result until its instrument, artifact, and protocol have passed
the checks below.

# Working conventions

- Treat `neural_library_learning_v1_experimental_spec.md` as the research source
  of truth.
- Preserve paired comparisons: models in a comparison must receive identical
  worlds, task orders, examples, replay budgets, and evaluation sets.
- Score online examples before updating on them.
- Never expose hidden programs or primitive identities to non-oracle models.
- Keep experiment outputs reproducible from explicit world and model seeds.
- Commit completed, verified milestones as work progresses; do not leave finished
  implementation checkpoints only in the working tree.
- Use H1 Markdown headings for section beginnings in this file and
  `PROGRESS.md`.

# Scientific integrity standard

Treat every result as provisional until its registered estimand, controls,
instrument, artifacts, and scorer validate: pre-register confirmatory hypotheses
and decision rules; preserve exact pairing and strict train/held-out/future/sealed
separation; score before update; test positive, negative, and non-vacuity cases;
compare functions only on common on-trajectory states and reconstruct all model
state; run from clean committed code with one local full lifetime and one writer
per cell; fingerprint the complete protocol and fail closed on resume mismatch;
accept results only after exit codes, expected cell counts, finite metrics,
artifact freshness, `check_prereg.py`, `check_invalid.py`, and paired outputs pass;
and preserve exploratory/confirmatory/invalid/withdrawn/unresolved status by
appending corrections to `PREDICTIONS.md`, learnings, progress, and the paper
rather than rewriting history. A launched job or plausible number is not a
scientific result.

# Commands

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m row.experiments.scratch_difficulty --config configs/v1.yaml
```

# Project structure and layout

## Repository layout

```
.
├── src/row/                  # Python package (editable install)
│   ├── __init__.py            #   public API: ExperimentConfig, World, etc.
│   ├── config.py              #   frozen dataclass config, YAML loading, fingerprints
│   ├── world.py               #   deterministic hidden-operator worlds (the teacher)
│   ├── mixed_world.py         #   Benchmark D: per-primitive rho profiles
│   ├── task_group_world.py    #   V3 promotion testbed: cross-cutting task families
│   ├── metrics.py             #   MSE, NMSE, Gaussian log loss, prequential cost
│   ├── provenance.py          #   resolved-config fingerprints, artifact validation
│   ├── models/                #   all learner implementations
│   ├── experiments/           #   lifetime runners, sweeps, audits, plots
│   └── sitecustomize.py       #   machine workaround (WMI fail-fast, see learnings)
├── configs/                  # YAML experiment configs
├── tests/                    # unittest suite (one test per experiment module)
├── tools/                    # sealed-block launchers, logs, preregistration checker
├── artifacts/                # experiment outputs (model.pt, summary.json, config.yaml)
├── reports/                  # analysis JSON and figures
├── notes/                    # learnings.txt, spec sketches, synthesis notes
├── reviews/                  # reviewer feedback files and review-index.md
├── paper/                    # draft, figures, figure-generation script
├── AGENTS.md                 # THIS FILE: intent, conventions, structure, learnings
├── CLAUDE.md                 # front-door safety summary and pointer to AGENTS.md
├── PROGRESS.md               # running lab record (append per completed step)
├── README.md                 # public-facing summary
├── pyproject.toml            # package metadata, deps, entry point
└── *.md / *_spec.md / *_PLAN.md / *_CONFIRMATION_PLAN.md  # specs and protocols
```

## Source package (`src/row/`)

### `world.py` — the teacher / data-generation source of truth

Defines `Primitive`, `Program`, `Task`, `World`, and `WorldConfig`. The world
generates training and evaluation examples once from independent deterministic
NumPy streams; model code consumes those fixed arrays. Primitives use
independently spectral-normalized `U` and `V` matrices. Opaque task IDs are
random tokens decoupled from program order and primitive IDs. `rho` is
implemented through task-specific correlated teacher parameters followed by
spectral renormalization.

### `mixed_world.py` — Benchmark D (mixed recurrence)

Per-primitive `rho` profiles replace the single `reuse_rho`. Provenance stays
OUTSIDE `WorldConfig` (adding a field there would invalidate all existing
resolved-config fingerprints); runners record the profile in their own artifact
files. A uniform profile reproduces the homogeneous world bit-exactly.

### `task_group_world.py` — V3 promotion testbed

Family components are assigned per task group rather than per primitive, so
structure cross-cuts tasks and a task-invariant shared basis cannot absorb it.
The structureless control is the same generator at `eta = 0` (reproduces the
canonical mixed world bit-exactly), not a separate generator.

### `config.py` — configuration and fingerprints

Frozen dataclasses (`WorldConfig`, `ScratchModelConfig`, `EvaluationConfig`,
`ModelConfig`, `ReplayConfig`, `ExperimentConfig`). `load_config()` resolves a
YAML file into a complete `ExperimentConfig`. Every resolved config is hashed
into a `fingerprint.json` that exposes seeds, rho, model family/architecture,
learning rates, replay, alpha, activation, program depth, and git commit. Resume
paths validate the full resolved config, not merely summary model/rho fields.

### `provenance.py` — artifact validation

Resolved-config fingerprinting and validation. Every new experiment artifact
writes both `config.yaml` and `fingerprint.json`. `model.pt` is tensor-only
(embedding NumPy-valued summaries in a torch checkpoint breaks PyTorch's
restricted `weights_only=True` loader).

### `metrics.py` — scoring

MSE, NMSE, and Gaussian log loss. The density-to-mass term (fixed target
precision 1/256) gives the quantized-target coding interpretation. Report total,
per-online-example, and per-target-scalar Gaussian log loss.

### `models/` — learner implementations

| File | Models | Role |
|------|--------|------|
| `numpy_mlp.py` | `ScratchResidualMLP` | NumPy-only scratch baseline (no shared substrate) |
| `torch_oracle.py` | `LearnedOperator`, `OracleCompositor` | PyTorch oracle with access to hidden program structure |
| `learned_models.py` | `ContinuousBasisLearner`, `DenseLearner`, `DiscreteLibraryLearner`, `HypernetworkLearner`, `PresenceGatedDiscreteLibraryLearner`, `SharedParentResidualLearner`, `VariationalSharedResidualLearner` | V1/V2/V3 learned models (Dense-C, Continuous, etc.) |
| `gated_models.py` | `GatedInnovationLearner` | presence-gated / MDL-gated models |
| `promoting_models.py` | `PromotingSharedResidualLearner` | V3 PROMOTE operator: creates new shared abstractions from recurring private residuals |
| `lifecycle_models.py` | `LifecycleLibraryLearner`, `AbstractionRecord` | V4 lifecycle: RETIRE, FACTORIZE, retention, dormancy |

### `experiments/` — runners, sweeps, audits, plots

The experiment modules follow naming conventions:

- **`*_lifetime.py`** — full prequential lifetime runners (`oracle_lifetime.py`,
  `learned_lifetime.py`, `compiler_lifetime.py`, `consolidating_lifetime.py`,
  `mixed_lifetime.py`). These run the 64-task online protocol: score before
  update, replay, checkpoint probes on deep-copied models.
- **`sweep_*.py`** — resumable sweep drivers over worlds/rho/checkpoints
  (`sweep_rho.py`, `sweep_checkpoints.py`, `sweep_robustness.py`,
  `sweep_forward_transfer.py`, etc.). All are resumable by design: they skip
  worlds with existing `summary.json` and can be interrupted and relaunched.
  New sweep drivers must add `--jobs N`, but full-lifetime runs on this local
  Windows host use `--jobs 1`; parallelism is for isolated remote workers.
- **`audit_*.py`** — offline structural oracles and gate auditors
  (`audit_substitutability.py`, `audit_lifecycle_oracle.py`,
  `audit_factorization.py`, `audit_promotion_oracle.py`,
  `audit_obsolescence.py`, `audit_h9_rate_distortion.py`, etc.). These run on
  frozen artifacts and never mutate lifetime training.
- **`summarize_*.py`** — aggregate artifacts into report JSON
  (`summarize_rho_replication.py`, `summarize_robustness.py`,
  `summarize_stage2.py`, etc.).
- **`tune_*.py`** — hyperparameter tuning on development worlds
  (`tune_development.py`, `tune_mdl.py`, `tune_shared_residual.py`).
- **`plot_*.py`** — figure generation (`plot_exact_reuse.py`,
  `plot_structural_controls.py`, `plot_forward_transfer.py`, etc.).
- **`score_*.py`** — sealed-block scoring against frozen confirmation plans
  (`score_v3_sealed.py`, `score_variational.py`, `score_group_clustering.py`).
- **`quantize_artifact.py`** — int8 per-tensor quantization of operator weights.
- **`exact_route_posterior.py`** — exhaustive route posterior enumeration.

### `sitecustomize.py`

Machine-specific workaround loaded automatically via the editable install.
Forces WMI `platform.machine()` to fail fast on this Windows host so `import
torch` does not hang. Harmless on machines without the stall.

## Configuration (`configs/`)

- `v1.yaml` — canonical frozen config. Dense-C width 32, Continuous global/task
  LR 0.003/0.05, Hypernetwork 0.003/0.05. These are the frozen stage-two
  winners; the file must not silently revert to pre-tuning values.
- `smoke.yaml` — fast smoke test.
- `v3_variational_beta*.yaml` — variational task-code configs at different
  description-beta values.

## Tests (`tests/`)

One test file per experiment module (`test_learned_models.py`,
`test_oracle.py`, `test_promoting_learner.py`, `test_lifecycle_learner.py`,
`test_variational_learner.py`, `test_world.py`, etc.). Run with
`python -m unittest discover -s tests -v`.

## Tools (`tools/`)

Sealed-block launchers and logs: `run_v3_sealed.py`, `run_v3_taskgroup.py`,
`run_v3_variational.py`, `run_v4_lifecycle.py`, `run_component_a.py`,
`run_component_b.py`. `check_prereg.py` validates frozen confirmation-plan
hashes before sealed worlds are opened. Logs (`.log`, `.out`) record sealed-run
output for provenance.

## Artifacts (`artifacts/`)

Experiment outputs grouped by experiment type. Each artifact directory contains
per-world subdirectories with `model.pt` (tensor-only), `summary.json`, resolved
`config.yaml`, and `fingerprint.json`. Naming convention: `{experiment}_{scope}`
(e.g., `rho_development/`, `v3_sealed/`, `v4_dev/`, `checkpoints_development/`,
`paired_seed0/`). Smoke runs use `_smoke` suffix.

## Reports (`reports/`)

Aggregated analysis JSON and figures, organized by experiment
(`rho_worlds_0_9/`, `structural_controls/`, `v2_confirmatory_mixed/`,
`v3_sealed.json`, `v4_factorization.json`, etc.). Plots live in `figures/`.

## Notes (`notes/`)

- `learnings.txt` — the primary implementation-learnings and results log.
- `crossover.txt`, `v2-synthesis.txt`, `v3-sketch.txt`, `v4-sketch.txt`,
  `v5-sketch.txt` — working notes and spec sketches.
- `v4-spec-plan.md` — V4 spec planning notes.

## Reviews (`reviews/`)

Reviewer feedback files (`reviewer-feedback-NN.txt`) and `review-index.md` with
a linked entry and summary for each review. `reviewer-assessment-initial.txt` is
the founding assessment. Reviews are living documents; the index is updated when
new feedback arrives.

## Paper (`paper/`)

`draft.md` — the paper draft. `make_figures.py` — figure generation from
report JSON. `figures/` — generated figures.

## Specification and protocol documents

| File | Role |
|------|------|
| `neural_library_learning_v1_experimental_spec.md` | V1 research source of truth |
| `row_v2_experimental_spec.md` | V2 spec (shared-residual discovery) |
| `row_v3_experimental_spec.md` | V3 spec (PROMOTE / abstraction birth) |
| `row_v4_experimental_spec.md` | V4 original spec (preserved unrevised with gate-outcome banner) |
| `row_v4r_experimental_spec.md` | V4 revised: "When Does a Library Need a Lifecycle?" |
| `row_v5_experimental_spec.md` | V5 spec (representation economics); sealed program closed |
| `notes/v5-sketch.txt` | V5 sketch, superseded by the V5 spec; kept for history |
| `V5_CONFIRMATION_PLAN.md` | V5 frozen confirmation protocol (seeds 600-629; completed) |
| `V5_CLOSURE.md` | V5 closure record, including review-55 withdrawals and corrected interpretation |
| `EXPERIMENT_PLAN.md` | separates development (0–9) from confirmatory (100–129+) |
| `CONFIRMATION_PLAN.md` | V1 frozen confirmation protocol (seeds 100–129) |
| `V2_CONFIRMATION_PLAN.md` | V2 frozen confirmation protocol (seeds 200–229) |
| `V3_CONFIRMATION_PLAN.md` | V3 frozen confirmation protocol (seeds 300–329) |
| `PREDICTIONS.md` | standing predictions ledger (committed before data exists) |
| `SPEC_AUDIT.md` | spec-to-implementation audit |
| `RELEASE_PLAN.md` | release sequencing (nothing public before confirmation) |
| `PROGRESS.md` | running lab record (append per completed, verified step) |

## World seed partitions

- **Development:** seeds 0–9 (used for architecture, tuning, testbed design).
- **V1 confirmatory:** seeds 100–129 (frozen, scored against
  `CONFIRMATION_PLAN.md`).
- **V2 confirmatory:** seeds 200–229 (frozen, scored against
  `V2_CONFIRMATION_PLAN.md`; tests parameter intervals, not just signs).
- **V3 confirmatory:** seeds 300–329 (frozen, scored against
  `V3_CONFIRMATION_PLAN.md`).
- **V4 original:** seeds 400–429 were reserved; no original-V4 rung reached
  sealing eligibility. **V4R confirmatory** used the same band (frozen
  against `V4R_CONFIRMATION_PLAN.md`; closed 7/7).
- **V5 development:** seeds 500–509 (H19 causal grid; contaminated;
  never confirmatory). Do not use 510–599.
- **V5 confirmatory:** seeds 600–629 opened only after
  `V5_CONFIRMATION_PLAN.md` was frozen and hashed; the V5 sealed program is
  complete. Later audit corrections do not turn these seeds back into
  development data.

Sealed worlds must not be generated, inspected, or summarized until the
relevant confirmation plan is frozen with its hash in `tools/check_prereg.py`.

# Implementation learnings

- Python's built-in `hash()` is process-dependent and must not derive experiment
  seeds. ROW uses NumPy `SeedSequence` with explicit integer components.
- Primitive `U` and `V` matrices are independently spectral-normalized so random
  teachers have comparable operation scale.
- Training and evaluation examples are generated once per world from independent
  deterministic random streams; model code must consume those fixed arrays.
- Opaque task IDs are random tokens and are deliberately decoupled from program
  order and primitive IDs.
- Generate opaque IDs from fixed-length random bytes with an explicit uniqueness
  check. NumPy's `choice(..., replace=False)` cannot index the full `uint64`
  range on platforms whose sampling path uses signed indices.
- PyTorch 2.13 provides a native Python 3.14 wheel in this environment and is the
  autograd/runtime dependency for oracle and learned reusable models. The NumPy
  teacher remains the deterministic data-generation source of truth.
- World seed 0 produced tightly matched task output variance (approximately
  0.213–0.228) and no output saturation. The initial scratch optimizer produced
  a flat final-error trend but did not reach NMSE 0.05 in 128 examples; tune the
  scratch optimization control before using that threshold as a validity gate.
- The tuned NumPy scratch control uses learning rate 0.001 and four minibatch
  updates per arriving example. Across world seeds 0–2, final-NMSE/task-index
  correlations were -0.060, 0.002, and -0.039; every task reached NMSE 0.1 at
  support 64. NMSE 0.05 remained fully censored, so scratch validity currently
  rests on the flat 0.1 curve and flat continuous final error, not that stricter
  threshold.
- The oracle's random initialization is already below NMSE 0.1, so that threshold
  cannot demonstrate oracle transfer. Use NMSE 0.05, NMSE 0.02, zero-shot NMSE,
  and prequential cost for the positive-control claim.
- Oracle seed 0 reduced first-quarter to last-quarter zero-shot NMSE from 0.0345
  to 0.00192. Mean examples to NMSE 0.05 fell from 18 to 0, and examples to 0.02
  fell from 74.5 to 0. The unseen-composition zero-shot NMSE was 0.00215.
- The oracle's learned slots recovered the teacher primitives with a one-to-one
  mean normalized functional distance of 0.000539. Because the true route fixes
  slot identity, exact index alignment is expected; functional distance is the
  substantive operator-quality check.
- Reverse task order and two additional world seeds reproduced zero-example late
  performance at NMSE 0.05 and 0.02, so the oracle effect is not specific to the
  seed-0 curriculum.
- Dense baselines must mirror the teacher's three state-update/tanh stages. A
  conventional hidden MLP with only one output tanh begins at a severe geometric
  disadvantage. The fair dense architecture uses three task-conditioned
  state-space residual blocks with zero-initialized deltas.
- For `d=16`, a three-block Dense-P with hidden width 11 has 2,193 shared scalars,
  closely matching the eight-slot continuous basis's 2,112. Dense-C width 32
  approximately matches its matrix-multiply count and has 6,288 shared scalars.
- Continuous task-code LR 0.005 left routes diffuse (mean maximum coefficient
  0.216) and underperformed Dense-P. Raising only task-code LR to 0.05 increased
  the mean maximum coefficient to 0.466, improved one-to-one primitive distance
  from 0.0151 to 0.00349, and reversed the seed-0 comparison. Route inference is
  therefore an optimization bottleneck, not a cosmetic hyperparameter.
- In the seed-0 pilot, tuned continuous prequential NLL was -165,119 versus
  -155,993 for Dense-P, -161,984 for Dense-C, and -163,775 for width-128 dense.
  Tuned continuous novel-composition NMSE improved from 0.0453 to 0.00803 using
  32 code-only examples. Treat this as exploratory until paired-world replication.
- Tuned continuous beat Dense-P on all three exploratory worlds. Paired
  dense-minus-continuous prequential NLL differences were 9,126, 4,940, and
  5,746 nats (mean 6,604; three-world bootstrap interval 4,940–9,126). This is a
  replicated pilot, not a confirmatory estimate.
- Continuous also improved 32-shot frozen-library novel NMSE over Dense-P by
  0.0302, 0.0151, and 0.0114 across the three worlds.
- The simple 8-bit retained-code proxy is 29,184 bits for continuous versus
  33,928 for Dense-P. Actual symmetric per-tensor quantized evaluation changed
  mean final NMSE by at most about 2.2e-5 in the pilot artifacts.
- Parameter matching does not imply compute matching: the continuous basis uses
  an estimated 6,528 matrix/mixture multiply-adds per prediction versus 2,112 for
  Dense-P. Dense-C replication is required before making a compute-controlled
  claim.
- Dense-C replication is complete. Tuned continuous beat Dense-C on all three
  worlds by 3,135, 675, and 1,176 prequential nats (mean 1,662) at approximately
  matched multiply-add counts (6,528 continuous versus 6,144 Dense-C).
- Dense-C retains 66,688 8-bit proxy bits versus continuous's 29,184 because
  compute matching requires width 32 and 6,288 shared dense scalars. Continuous
  also improved 32-shot novel NMSE over Dense-C by 0.0210, 0.00944, and 0.00558.
- The seed-0 hard discrete library used 11/12 slots without collapse, recovered
  92.2% of teacher routes exactly (96.4% per position), and matched primitives
  at mean normalized distance 0.00229. Its weak prequential NLL (-134,784) is
  therefore a route-inference cost, not a failure to learn the operator library.
- Hard discrete zero-shot NMSE worsened from 0.103 in the first quarter to 0.160
  in the last because an unseen task's tie-broken default route becomes less
  generic as operators specialize. Nevertheless, examples to NMSE 0.02 fell
  from 128.9 to 25.3 and 32-shot novel hard-route adaptation reached 0.00413.
- Hardened discrete retention stores exact categorical route indices, not
  quantized training logits. Quantize only operator weights and reconstruct
  routes losslessly when validating the 26,112-bit proxy. This yields mean NMSE
  degradation 1.10e-5; quantizing logits directly is the wrong artifact model.
- Hardened discrete inference is only about 768 multiply-adds, versus 6,528 for
  the continuous mixture, exposing a strong storage/inference versus online
  learning-cost tradeoff.
- `rho` is implemented through task-specific correlated teacher parameters,
  followed by spectral renormalization of `U` and `V`. At `rho=0`, measured
  pairwise residual-function correlation is approximately zero; at `rho=1`, it
  is exactly one. Novel tasks receive independently derived correlated libraries.
- On development world 0, the causal endpoint reverses: Continuous beats Dense-C
  by 3,135 Gaussian-log-loss units at `rho=1`, while Dense-C beats Continuous by
  6,714 at `rho=0`. This is the first direct evidence that the advantage depends
  on latent computational recurrence.
- Checkpoint probes must operate on deep-copied models so code-only novel-task
  adaptation cannot alter lifetime training. Use four fixed novel programs at
  8, 16, 32, and 64 tasks.
- On exact-reuse world 0, Continuous mean 32-shot novel NMSE improved from 0.0298
  after 8 tasks to 0.00467 after 64; Dense-C improved from 0.0367 to 0.0159.
  This directly supports progressive learning-to-learn, beyond total loss alone.
- The current learnable-alpha checkpoint protocol replicates on all development
  worlds 0–9 using four fixed novel compositions per checkpoint. Mean Continuous
  versus Dense-C 32-shot NMSE is 0.02282/0.02300 after 8 tasks,
  0.01401/0.01652 after 16, 0.00687/0.01098 after 32, and 0.00343/0.00645 after
  64. Continuous wins 4/10, 8/10, 10/10, and 10/10 paired worlds, respectively.
- Both models improve from task 8 to 64 in every development world, but
  Continuous's mean per-world gain is 6.70x (range 5.15–9.28) versus Dense-C's
  3.64x (range 2.97–4.90). The increasing paired advantage supports a stronger
  learning-to-learn interpretation rather than a static model-quality offset.
- Call the density-based metric cumulative prequential Gaussian log loss. A
  fixed target precision of 1/256 adds the common density-to-mass term for a
  defensible quantized-target coding interpretation without changing model
  differences.
- Development worlds are 0–9. Confirmatory worlds 100–129 must remain untouched
  until symmetric tuning and all analysis rules are frozen in `EXPERIMENT_PLAN.md`.
- Symmetric stage-one tuning materially improved both models. Dense-C's best
  development-0–2 setting is global/task LR 0.001/0.05 with mean Gaussian log
  loss -168,768; its prior 0.001/0.005 setting scored -164,842. Continuous's
  best is 0.003/0.05 at -171,866; its prior 0.001/0.05 setting scored -166,505.
- After equal tuning, Continuous leads Dense-C by 3,098 mean Gaussian-log-loss
  units on worlds 0–2 and has better mean novel 32-shot NMSE (0.00237 versus
  0.00486). This supersedes both the untuned Dense-C comparison and the temporary
  observation that tuned Dense-C beat the old Continuous configuration.
- Stage-two finalists are Continuous 0.003/0.05 and 0.001/0.05, and Dense-C
  0.001/0.05 and 0.003/0.05, evaluated on development worlds 3–9.
- Corrected-architecture stage-two revalidation freezes Continuous global/task
  LR at 0.003/0.05 and Dense-C at 0.001/0.05. On worlds 3–9, Continuous's two
  candidates average -171,899 and -166,937 Gaussian log loss; Dense-C's average
  -168,112 and -165,729, respectively. The winners match the stage-one choices.
- The canonical `configs/v1.yaml` must freeze Dense-C at width 32. Stage-two and
  earlier compute controls explicitly overrode the former width-128 YAML value,
  so their artifacts are correct, but leaving 128 in the config would make a
  nominally frozen clean rerun reproduce the wrong baseline.
- With those frozen settings at exact reuse, Continuous beats Dense-C on all
  seven stage-two worlds by 2,554–5,054 cumulative Gaussian-log-loss units
  (mean 3,787) and improves mean 32-shot novel NMSE from 0.00731 to 0.00368.
  These remain development results, not confirmation.
- Learner operators no longer read teacher `alpha`. They initialize an independent
  learnable scale at 0.2. On world 0, decoupled Continuous scores -170,967 versus
  tuned Dense-C -166,521, so the earlier win is not caused by leaking alpha 0.35.
- Teacher-rank mismatch (teacher 16, learner 8) preserves a 3,296 log-loss
  advantage and better novel adaptation for Continuous. GELU learner mismatch
  preserves only a 441 advantage, showing that family alignment materially helps
  but does not wholly explain the seed-0 result.
- Adding a fixed identity candidate does not help Continuous on seed 0
  (-170,404 versus -170,967 without identity). The inability to skip a step is
  therefore not the source of the primary gain in this control.
- Per-task temperature re-annealing improves hard discrete log loss from -137,321
  to -146,146 and novel zero-shot NMSE from 0.150 to 0.0436 under the leak-free
  setup. Global annealing exaggerated route-inference difficulty, but per-task
  discrete still substantially trails Continuous.
- The leak-free oracle re-gate passes: early/late zero-shot NMSE is 0.0288/0.00159,
  early/late examples to NMSE 0.02 is 72.8/0, unseen-composition zero-shot NMSE
  is 0.00175, and primitive matching distance is 0.000484.
- Sparse world-0 `rho` controls show the crossover is above 0.75. At rho 0.5
  (measured residual correlation 0.064), Dense-C beats Continuous by 2,159 total
  Gaussian-log-loss units; at rho 0.75 (correlation 0.317), it wins by 1,615.
  Continuous already has slightly better 32-shot novel adaptation at both points.
- The complete leak-free seed-0 curve reverses between `rho=0.75` and `rho=0.9`.
  Dense-C-minus-Continuous cumulative-loss differences are -2,390, -2,280,
  -2,159, -1,615, +2,371, and +4,446 at configured rho 0, 0.25, 0.5,
  0.75, 0.9, and 1.0. Linear interpolation gives a descriptive crossing near
  configured rho 0.811, corresponding to measured residual correlation 0.454.
  This is a one-world development diagnostic, not a population estimate.
- Measured functional recurrence is strongly nonlinear in configured `rho` after
  spectral renormalization: the six seed-0 correlations are approximately
  -0.001, 0.003, 0.064, 0.317, 0.654, and 1.0. Plot and model effects against
  both the intervention and this measured explanatory variable.
- The full curve replicates on development worlds 1 and 2. All three worlds
  favor Dense-C at every `rho<=0.75` and Continuous at both `rho=0.9` and
  `rho=1`. Interpolated per-world crossings are 0.811, 0.852, and 0.851; the
  three-world mean curve crosses near configured rho 0.833, corresponding to
  mean measured residual correlation 0.488. Treat interpolation as descriptive.
- Expansion through development world 5 preserves the reversal without an
  exception: Continuous lifetime-loss wins are 0/6 worlds at rho 0, 0.25, 0.5,
  and 0.75, then 6/6 at rho 0.9 and 1.0. The six-world mean effects are -2,197,
  -2,308, -2,332, -1,427, +1,389, and +3,357; the mean crossing is rho 0.826
  (measured correlation 0.479), with per-world crossings spanning 0.811–0.852.
- The completed development sweep on worlds 0–9 is unanimous at every tested
  rho. Continuous lifetime-loss wins are 0/10 at rho 0, 0.25, 0.5, and 0.75,
  then 10/10 at rho 0.9 and 1.0. Mean Dense-minus-Continuous effects are -2,135,
  -2,165, -2,227, -1,337, +1,288, and +3,698. The descriptive mean crossing is
  rho 0.8264 at measured recurrence 0.4836; per-world rho crossings span
  0.8108–0.8590.
- The seed-0 suggestion that Continuous improves 32-shot novel transfer before
  it wins lifetime loss does not replicate. At rho 0.25, 0.5, and 0.75,
  Continuous wins novel transfer on only one of three worlds and has worse mean
  novel NMSE differences at all three points. Preserve this as a failed secondary
  hypothesis rather than a three-regime conclusion.
- Across worlds 0–5, Continuous wins 32-shot novel transfer on only 1–2 of six
  worlds at rho 0–0.9 and has worse mean transfer even at rho 0.9, despite its
  6/6 lifetime-loss wins there. It wins novel transfer 6/6 only at exact reuse.
  The current evidence supports distinct lifetime-cost and transfer behavior,
  but not the proposed ordering in which transfer improves first.
- On the full development set, Continuous novel-transfer wins are 2/10, 4/10,
  5/10, 2/10, 4/10, and 10/10 across the six rho values. Mean transfer is
  essentially tied at rho 0.5, worse for Continuous at rho 0.75 and 0.9, and
  reliably better only at rho 1.0. Do not claim a separate earlier transfer
  boundary from V1 evidence.
- Report total, per-online-example, and per-target-scalar Gaussian log loss.
  Compute accounting distinguishes training-forward all-slot evaluation from
  hardened inference; it excludes backward and optimizer operations.
- Learned alpha scalars use a no-weight-decay optimizer group. `forward_tasks`
  groups samples by task ID before forwarding, preserving gradients and enabling
  useful batching when replay contains repeated tasks.
- Do not run multiple full-lifetime PyTorch/SciPy experiments concurrently on
  this Windows host. Four parallel stage-two tuning processes exhausted the
  paging file and crashed the harness; run one lifetime process at a time and
  use artifact-level resume to recover completed worlds.
- Every new experiment artifact writes both the complete resolved `config.yaml`
  and `fingerprint.json`. The fingerprint hashes the resolved config and exposes
  seeds, rho, model family/architecture, learning rates, replay, alpha,
  activation, program depth, and git commit. Resume paths must validate the full
  resolved config, not merely summary model/rho fields; legacy artifacts may be
  backfilled only after that validation passes.
- The generic hypernetwork control uses a 24-scalar task code arranged as three
  eight-dimensional step codes. A shared two-layer generator produces low-rank
  operator deltas around one learned base operator, preserving the three-stage
  residual prior without an explicit reusable slot library. It has 2,705 shared
  scalars and 7,296 counted forward multiply-adds at `d=16`, close to
  Continuous's 2,120 scalars and 6,528 multiply-adds.
- Zero-initialized hypernetwork task codes must generate the shared base exactly
  while still receiving nonzero gradients. Use a bias-free code projection and
  small nonzero output weights with a zero output bias; zeroing the output
  weights would strand every new task code at initialization.
- The first untuned world-0 hypernetwork smoke run at global/task LR
  `0.001/0.05` scored -165,031 cumulative Gaussian log loss, median final NMSE
  0.00472, and novel 32-shot NMSE 0.0051. Treat this only as an implementation
  check until the control receives symmetric development tuning.
- Hypernetwork stage-one tuning on development worlds 0–2 selected global/task
  LR `0.003/0.05` with mean Gaussian log loss -170,246; `0.001/0.05` was second
  at -167,821. The selected hypernetwork beat Dense-C in all three paired worlds
  by a mean 1,478 log-loss units but lost to Continuous in all three by a mean
  2,014. Its mean novel 32-shot NMSE was 0.00589, worse than current Continuous
  (0.00238) and Dense-C (0.00486) on the same worlds.
- Dense-C with its task embedding reduced from 32 to 24 dimensions has 5,520
  shared scalars and the same 1,536 retained task scalars as Continuous and the
  hypernetwork. On world 0 it scored -166,567 versus -166,521 for Dense-32, a
  negligible 46-unit difference; replicate before closing this sensitivity.
- Hypernetwork stage two on worlds 3–9 confirms global/task LR `0.003/0.05`;
  its mean log loss is -170,038 versus -168,088 for `0.001/0.05`. The selected
  LR is frozen in `configs/v1.yaml`.
- Across all ten development worlds, Continuous beats the tuned hypernetwork on
  both lifetime loss and novel 32-shot NMSE 10/10, by mean advantages 1,907
  log-loss units and 0.00272 NMSE. The hypernetwork beats Dense-C on lifetime
  loss 10/10 by mean 1,791, but on novel adaptation only 6/10 by mean 0.000568.
  A generic continuous operator manifold is useful, but does not explain away
  the explicit slot basis's exact-reuse advantage.
- Dense-24 versus Dense-32 is a null sensitivity across worlds 0–9: Dense-24
  wins lifetime loss 5/10 and is worse by 73 units on average, while winning
  novel 32-shot NMSE 7/10 by only 0.000119 on average. The primary comparison is
  not driven by Continuous having fewer retained task scalars.
- Keep `model.pt` tensor-only. Summaries already live in `summary.json`, and
  embedding NumPy-valued summaries in a torch checkpoint breaks PyTorch's
  restricted `weights_only=True` loader. The quantizer has a provenance-checked
  fallback solely for legacy local artifacts.
- Current hypernetwork retention is 33,928 proxy bits (21,640 shared plus 12,288
  task-state bits) at 7,296 inference multiply-adds. The world-0 int8 smoke
  changed mean final NMSE by only 2.19e-6; replicate across development worlds
  before treating this as the refreshed retention result.
- Refreshed int8 retention on all ten development worlds gives 29,248 bits for
  Continuous, 33,928 for Hypernetwork, 56,448 for Dense-24, and 66,688 for
  Dense-C. Mean quantized-minus-float final NMSE is 1.15e-6, 3.23e-6, 0.64e-6,
  and 2.32e-6 respectively; no model's maximum observed task increase exceeds
  1.38e-4.
- The current per-task-annealed Discrete world-0 artifact retains 26,208 bits
  after adding its twelve learnable alpha scalars to the older proxy. It still
  uses only 768 hardened inference multiply-adds, with mean int8 degradation
  4.77e-6.
- Exploratory reverse-order and replay robustness on worlds 0–2 preserves the
  Continuous lifetime-loss advantage 3/3 in every condition. Mean Dense-minus-
  Continuous effects are 3,175 reverse, 3,165 with no replay, 3,492 at canonical
  1:1 replay, and 3,082 at 1:4 replay. Strong replay does not erase the effect;
  replicate through world 9 before closing the gate item.
- Ten-world robustness is complete. Continuous wins lifetime loss 10/10 with no
  replay, canonical replay, 1:4 replay, and reverse order. Mean advantages and
  bootstrap intervals are 3,129 [2,511, 3,743], 3,698 [3,233, 4,177], 3,350
  [2,959, 3,766], and 3,456 [3,145, 3,791] respectively.
- Reverse order has no consistent absolute penalty for either architecture:
  mean reverse-minus-forward loss is -57 for Continuous and -299 for Dense-C,
  with both bootstrap intervals spanning zero. The paired model advantage is
  therefore robust without implying one task order is intrinsically harder.
- Novel 32-shot transfer is not robust without replay: Continuous wins 6/10 and
  the mean advantage interval spans zero. It wins 10/10 under canonical replay,
  strong replay, and reverse order. Distinguish the replay-sensitive retained
  transfer endpoint from the replay-robust cumulative lifetime-cost result.
- The second initialization (Continuous seed 4001, Dense-C seed 3001) reproduces
  exact-reuse lifetime loss and novel transfer 10/10. Its mean advantages are
  3,081 log-loss units and 0.00269 NMSE, versus 3,698 and 0.00328 for the
  canonical seeds.
- Average initialization effects within each world before bootstrapping. The
  two-initialization world-level mean advantage is 3,390 with interval
  [2,962, 3,803]; the novel advantage is 0.00299 [0.00200, 0.00426]. Do not
  count 20 world-initialization pairs as 20 independent worlds.
- Same-architecture fresh-task baselines establish explicit forward transfer.
  Mean per-task fresh-minus-lifetime Gaussian log loss is +366.8 for Continuous
  and +211.6 for Dense-C; both are positive in all ten world means, with 97.5%
  and 93.75% positive individual tasks.
- Forward transfer increases with task index in every world for both models.
  Mean per-task slopes are +4.35 Continuous and +3.77 Dense-C log-loss units per
  task index. Continuous's extra mean transfer is positive 10/10 worlds at
  +155.2 [145.5, 165.3].
- Route similarity is a strong post-hoc explanatory variable: Continuous mean
  transfer rises from 26 for no matching route positions to 268 at one of three
  and 402 at two of three; Dense-C rises from -34 to 115 and 242. Teacher routes
  are used only for analysis, never learner training or prediction.
- Exact-reuse checkpoint matching shows progressive operator recovery in every
  development world. From 8 to 64 tasks, mean one-to-one primitive functional
  distance falls from 0.00786 to 0.00211 for Continuous and from 0.00745 to
  0.00131 for per-task-annealed Discrete; both improve 10/10 worlds.
- Matched slots also generalize to programs whose task codes have not yet been
  trained. Mean future-program teacher-route NMSE falls from 0.0300 to 0.0126
  for Continuous and from 0.0280 to 0.0106 for Discrete between checkpoints 8
  and 32.
- A teacher route through Hungarian-matched individual slots is not an oracle
  upper bound. Continuous's learned mixture route is better on completed tasks
  in all ten worlds at every checkpoint; at 64 tasks its mean NMSE is 0.00191
  versus 0.00937 for the matched-slot teacher route. This indicates distributed
  or compensatory mixture representations even while individual-slot matching
  improves.
- Discrete's learned hard route and matched-slot teacher route converge: at 64
  tasks their mean NMSEs are 0.00456 and 0.00484, with the learned route better
  in only 6/10 worlds. Teacher primitive identities and programs are used only
  in this post-hoc diagnostic and never enter training.
- Opaque-ID reassignment is an exact invariant for selected Continuous and
  Dense-C world-0 runs. With disjoint token sets and all task contents fixed,
  every metric row and summary value is identical after removing the token, and
  every final tensor is bitwise equal after mapping task-code keys to task index.
- Scrambled-ID provenance must remain outside `WorldConfig`: adding a default
  field there would invalidate all existing resolved-config fingerprints. The
  optional scramble seed is recorded only in explicitly scrambled learned-run
  configurations.
- The explicit 1:1 batch ablation uses four current-task examples and four
  replay examples at target batch 8. Because early current/replay pools are
  smaller, observed mean sizes are 1.98 for target 2 and 7.89 for target 8.
  Buffer-construction and replay-sampling RNGs are separated, and the sampling
  policy is paired across models.
- On development worlds 0–2, target batch 8 improves lifetime log loss for both
  Continuous and Dense-C 3/3, by means 1,554 and 2,956 respectively. It helps
  Dense-C more, narrowing the mean Dense-minus-Continuous advantage from 3,463
  at batch 2 to 2,061 at batch 8, but Continuous still wins 3/3.
- Continuous retains better 32-shot novel NMSE 3/3 at both batch sizes. Batch 8
  improves Continuous novel NMSE 3/3 and Dense-C 2/3. Treat this as a
  development sensitivity: batch 8 materially increases sample reuse and
  training compute, so it is not merely a vectorization change.
- Truncating the existing rho lifetimes gives mean configured-rho crossovers of
  0.869 after 16 tasks, 0.822 after 32, and 0.826 after 64. The 16-to-64
  crossover declines in 8/10 worlds, but the mean is not monotone because it
  edges upward from 32 to 64 tasks. H5a is partially, not fully, supported.
- Re-coordinating the 64-task mean effect curve by measured residual-function
  correlation makes it much more nearly linear: linear-fit R-squared rises from
  0.646 in configured rho to 0.974 in measured recurrence, and residual RMSE
  falls from 1,329 to 361 log-loss units. Much of the apparent elbow is a
  coordinate artifact.
- H5b's tighter-alignment prediction fails even though the mean curve smooths.
  At 64 tasks, per-world crossover population SD is 0.0173 in configured rho
  versus 0.0381 in measured recurrence (ranges 0.0482 versus 0.1308). Report
  the coordinate result as mixed, not as a general collapse onto one frontier.

# Standing scientific doubts

- The generic low-rank hypernetwork closes the largest structural-prior gap and
  produces an intermediate lifetime-loss result, but it still shares the
  teacher's three-stage low-rank residual family. Claims should distinguish
  evidence for explicit reusable slots from the broader benefit of aligned
  operator-manifold structure.
- Most development evidence and optimizer selection use the legacy effective
  batch of one current plus one replay example. The batch-8 sensitivity preserves
  the main result but benefits Dense-C more; the confirmation protocol and any
  symmetric retuning must be frozen explicitly before unsealing worlds 100–129.
- The free rho*(N) result does not establish a monotone economic law: the large
  16-to-32 shift plateaus by 64 tasks. The measured-recurrence coordinate
  smooths the mean curve but worsens raw cross-world crossing dispersion.
- Dense-C matches inference multiply-adds only. Continuous training backpropagates
  through every basis operator and uses materially more training compute.
- The initial n=3 bootstrap intervals are not inferentially meaningful. Public
  reporting should show paired world deltas directly until world count is larger.
- Identity-slot results are one world only; the negative ablation is not a
  population claim.
- Confirmatory worlds 100–129 remain sealed. Do not inspect them until these
  doubts and the development `rho` curve are resolved.
- The development crossover is complete and highly consistent, but the 0.8264
  interpolation lies between coarse grid points and remains descriptive. Do not
  treat it as a precisely located universal phase boundary.
- Continuous's slot-to-primitive diagnostic is basis-dependent: its own mixtures
  systematically outperform forced one-slot teacher routes. Report progressive
  functional matching as evidence of operator-quality improvement, not as proof
  that Continuous has uniquely recovered the teacher's discrete representation.
- The shared-parent residual control must optimize task routes and rank-two task
  residuals in separate parameter groups. Coupling both at task LR 0.05 produces
  large residuals and poor lifetime loss; residual LR is a substantive control.
- Guarded world-0 tuning at configured `rho=0.75` selected residual LR 0.01 and
  L1 penalty 0.01. Its cumulative log loss is -157,103, novel 32-shot NMSE is
  0.01748, and mean/maximum functional residual-to-parent ratios are 0.246/0.419.
  This is a tuning result only; the selected setting must be evaluated across
  recurrence and additional development worlds before interpretation.
- The frozen shared-residual control beats the better fixed Continuous/Dense-C
  model on lifetime loss and novel 32-shot NMSE in all nine world/rho pairs at
  configured rho 0.5, 0.75, and 0.9. Mean lifetime gains over the fixed envelope
  are 9,168, 7,458, and 3,745 log-loss units, respectively.
- At exact reuse, Continuous beats shared residual by 238, 163, and 336 log-loss
  units. Mean functional residual-to-parent ratio falls monotonically from 0.284
  at rho 0.5 to 0.026 at rho 1.0, with the endpoint decline present in all three
  worlds. This supports adaptive share-versus-specialize behavior.
- Shared residual stores 14,208 task-specific scalars over 64 tasks versus 1,536
  route scalars for Continuous, and uses 6,720 forward multiply-adds versus
  6,528. Its lower intermediate-reuse loss is a copy-on-write capacity result,
  not a storage-matched architecture win.
- Model 4's “only after Model 2 works” prerequisite is satisfied structurally:
  hard Discrete recovers operators and routes despite weak online route
  inference. Presence-gate pruning tests basis compression, so the route-learning
  cost is a limitation to report rather than a reason to skip the conditional
  model.
- Keep MDL presence gates distinct from task route logits. Training uses smooth
  global sigmoid gates with an expected-active-slot penalty; hardened evaluation
  masks sub-threshold slots before route argmax so an inactive operator can never
  be selected.
- Presence LR 0.01 with library penalty 1e-4 collapses all twelve relaxed gates
  below threshold on world 0, forcing the one-slot safety fallback and degrading
  log loss to -86,937. Tune presence LR at 1e-4 and 1e-3 before interpreting
  pruning; the default gate optimizer was two orders of magnitude too fast.
- `CLAUDE.md` is a short front-door safety summary and pointer to this file.
  Keep the detailed, authoritative conventions and implementation learnings
  here; mirror only the non-negotiable safety rules there.
- Background shell jobs die silently with the session that launched them.
  After any interruption, verify expected artifacts exist before assuming
  a launched run completed; relaunch idempotently (all sweep drivers are
  resumable by design — keep new ones that way).
- Notes and review files are living documents. Before any spec revision,
  diff them against the version last incorporated; staleness is not
  completeness.
- The MDL presence gate is bimodal at this scale: penalties either prune
  nothing or collapse the library, and gate pressure degrades novel-
  composition transfer even before pruning bites (11 active slots already
  fail sufficiency). Compress after evidence accumulates instead of
  penalizing during acquisition.
- Raw prequential wins can be reversed by description-length accounting:
  the shared-residual envelope win (+3.7k-9.2k nats) loses in all cells
  under a literal two-part code because rank-2 per-task residuals retain
  ~130k bits. Always report both currencies before "chooses its own
  sharing" style claims.
- Learner slot indices and teacher primitive indices live in different
  spaces. Any route-agreement metric must map through functional matching
  or stay in slot space (compare against the artifact's own hard routes);
  comparing raw indices silently returns zero.
- The exact route posterior (1,728 routes, 8,192 examples) takes over ten
  minutes unbatched on this machine while lifetimes run concurrently;
  batch the enumeration over examples if it becomes a repeated tool.
- Sealed-block discipline extended in V2: seeds 200-229 test parameter
  intervals (slope, crossing, R^2), not just signs; interval misses are
  reported as failures even when signs pass.
- Although each lifetime pins itself to one thread, local throughput is bounded
  by memory, I/O, and safe artifact ownership rather than core count. Earlier
  advice to launch up to 12 local processes is **superseded**: run one local
  full lifetime at a time. Keep `--jobs N` in sweep interfaces for isolated
  remote workers, where every worker owns disjoint output cells.
- Kaggle is available for throughput beyond local cores (token provided
  by the PI as an environment variable). Its value is parallel sessions,
  not per-run speed — the models are too small to benefit from a GPU.
  Pattern: each kernel clones the public GitHub repo, installs, runs a
  world-slice, and saves `artifacts/` as kernel output. Reserve it for
  work exceeding an overnight 16-core budget (V2 sealed block plus
  Benchmark D together, or Phase III cross-world training).
- NEVER write the Kaggle token (or any credential) into a tracked file,
  script, notebook metadata, or kernel source — the repository is
  public. Pass credentials only as inline environment variables at
  invocation time.
- The repository is now public at
  https://github.com/doeixd/reusable-operator-worlds. Push after each
  commit; the public history is part of the project's verifiability
  claim, so never rewrite published history (no force pushes, no
  amends of pushed commits).
- H6 analysis note: at rho 0.9 the GELU family-mismatch penalty
  (tanh-effect minus GELU-effect, paired per world) is roughly constant
  across worlds — 3,513 / 2,445 / 3,215 nats — suggesting misalignment
  enters the economics as an approximately ADDITIVE penalty (a shift in
  the intercept b of Delta(r) = a*r + b) rather than a slope change.
  When the rho 1.0 GELU runs land, compare their penalty (world 0:
  4,005) against these; if similar, the H6 status and the paper's
  Delta = f(r, A, C) discussion can state the quantitative form: family
  mismatch costs ~3-4k nats roughly uniformly over the high-recurrence
  range, moving the zero crossing right without bending the line. Fit
  the GELU points to their own line and report both coefficients rather
  than only the crossing.
- pip hangs on this machine for new package installs (stalls in network/
  index resolution even for `pip download --no-deps`; cause not
  diagnosed). Do not block work on pip. For Kaggle specifically, the
  `kaggle` package is unnecessary: the REST API accepts the access token
  as `Authorization: Bearer <token>` (verified 200 against
  /api/v1/kernels/list). Kernel workflow via curl: push a kernel with
  POST /api/v1/kernels/push (JSON metadata plus source), poll
  /api/v1/kernels/status/{user}/{slug}, fetch results from
  /api/v1/kernels/output/{user}/{slug}. Token comes from the environment
  at invocation time only — never from a file.
- Windows WMI on this machine intermittently stalls, hanging
  `platform.machine()` and therefore `import torch` (minutes per
  process) and pip. `src/sitecustomize.py` forces the WMI query to fail
  fast so `platform` falls back to environment variables; it loads
  automatically via the editable install. If new processes ever hang
  before any output, check for this class of stall first
  (`faulthandler.dump_traceback_later` locates it in seconds).
- Memory, I/O, and writer ownership—not cores—are the binding constraints for
  local lifetimes. Historical limits of 4-6 concurrent processes still produced
  paging failures and overlapping launchers. They are **superseded** by the
  one-local-lifetime rule. Never co-schedule installs or another experiment batch
  with a running lifetime.
- During a heavy batch, a hung shell command is evidence of LOAD, not of
  batch failure: lifetimes only write summary.json at completion, so an
  apparently empty output directory plus an unresponsive filesystem most
  likely means the batch is mid-wave and the disk is churning. Check
  artifact counts from a prior known state before killing anything, avoid
  repo-wide find during batches (use targeted ls), and remember TaskStop
  ends the parent shell while detached children survive as orphans.
- Variational task codes have three implementation traps, all measured
  on the V3 wake learner and all fatal if missed. (1) UNITS: KL is a
  sum of nats over a task's whole code, MSE is a mean over batch and
  dimensions; charge `beta * 2*sigma^2/(N*d) * KL` so beta = 1 is the
  literal MDL point (raw KL against MSE is ~1e4 too strong). (2) A
  GRADIENT-learned shared prior runs away — the KL gradient always says
  "shrink" when posteriors are concentrated and Adam's normalized step
  ignores the tiny magnitude, so the prior collapses (1.0 -> 0.0034 in
  one lifetime) and then annihilates the task state. Use the
  closed-form empirical-Bayes M step instead. (3) That closed form has
  a STABLE DEGENERATE FIXED POINT at initialization (mu ~ 0 gives
  s ~ sigma_init, whose mu/s^2 gradient pins every code at zero
  forever): estimate the prior from COMPLETED tasks only, after a
  warmup, with the posterior starting precise and the prior wide.
- A variational learner cannot be smoke-tested on a reduced-example
  world. At 16 examples/task the MDL-correct answer is an empty task
  code (task state buys ~5 nats of fit against ~500 nats of code), so
  correct refusal-to-encode is indistinguishable from a collapse bug.
  Smoke-test with `description_beta: 0.0` instead, which must reproduce
  the frozen non-variational baseline exactly.
- When auditing online objectives, count relative gradient/objective pressure, not raw regularizer appearances.- A functional tolerance must be normalized against the quantity whose
  loss it licenses, never against total output scale. V4.1's substitution
  test divided by total output variance while an abstraction contributes
  ~0.2% of it, so every abstraction substituted for every other and the
  oracle "compacted" 4-6 abstractions to 1. Two symptoms identify this
  class of error immediately: the causal control goes DEGENERATE (random,
  usage, and functional retirement score identically), and the NULL EDIT
  passes (deleting an abstraction outright scores a smaller deviation
  than replacing it). Both are now hard guards in
  `audit_lifecycle_oracle.py`.
- On the frozen V4 testbed V3's 4-6 abstractions are mutually DISTINCT,
  not redundant: contribution-relative substitution costs 0.86-1.60 of
  what the abstraction buys, and 99-100% of ordered pairs exceed a 10%
  tolerance. Priced in nats, compacting to one abstraction is net -1,180,
  -2,393, and -1,377 on worlds 0-2. The representation-fragmentation
  premise for V4.1-as-compaction is unsupported; the open direction is
  V4.2 synthetic merge (refit a new abstraction covering several distinct
  contributions), not retiring copies that do not exist.
- CONSTITUTIONAL, alongside "structural claims require functional
  substitutability": every SHARING claim needs a matched-BUDGET
  non-sharing alternative. "Shared beats unshared at full precision" and
  "shared beats unshared at equal bits" are different claims and only
  the second is evidence of reuse. V4.2's factorization gate passed the
  first and failed the second 9/10 worlds -- coarser private
  quantization of each atom beat a shared parameterized family at the
  same bit budget. This orders the edit vocabulary by ambition,
  KEEP < COMPRESS < SHARE/FACTORIZE < CREATE/FORK, and makes COMPRESS a
  first-class operator that any sharing proposal must beat first.
- V4's premise -- successful abstraction birth implies a maintenance
  problem -- is FALSIFIED at this scale. All three original rungs failed
  their development gates. V4R then opened seeds 400-429 against a
  hashed plan and closed 7/7: COMPRESS dominates, FACTORIZE/FORK do not
  pay at reachable M, and retention obeys the amortization law. This is
  a positive statement about V3: its library has little exploitable
  lifecycle slack. `row_v4_experimental_spec.md` is preserved unrevised
  with a gate-outcome banner; `row_v4r_experimental_spec.md` is the
  successor that was actually sealed. V5 is now a spec,
  `row_v5_experimental_spec.md`; `notes/v5-sketch.txt` is superseded
  and kept only as history.
- Before building another world in which an operator MUST fire, run an
  offline opportunity census with the learner frozen: sweep regimes and
  ask which edits have any oracle advantage at all. Two dormancy designs
  were built and both failed to instantiate retention value -- a census
  would have said so before the lifetimes ran.
- The world caps `tasks` at the number of distinct programs, which is
  `primitives ** program_depth` = 6**3 = 216 in the canonical config.
  N=256 is unreachable; the largest admissible scale point is 200.
  Raising it means adding primitives, which breaks comparability with
  every existing artifact. V4R §1.1's registered N in {64,128,256} was
  therefore unsatisfiable as written and is run at {64,128,200}.
- Retention obeys the SAME amortization law as abstraction birth:
  `RETAIN iff H_R * s_bar > lambda * D(A)`. Measured on the V5 probe,
  the predicted crossing (1,098 / 64.1 = 17.1 returning tasks) matched
  the observed one (17.9) with no fitted threshold. Dormancy LENGTH is
  the wrong variable and cost three failed world designs: per-task
  saving is flat across gaps, and only expected remaining reuse matters.
- A dormancy gap must CLOSE strictly before the lifetime ends, with
  enough tasks after it to measure. This error produced false readings
  three separate times (V4.1, the retention sweep, the fine gap sweep).
  Count returning tasks before reading any dormancy config's numbers.
- A mid-lifetime deletion is NOT a paired comparison: the arms stop
  being matched the moment the intervention changes what gets promoted
  next. Library state is path-dependent (deleting one abstraction at
  task 32 gave final sizes 5/5, 6/5, 5/2, and once 6/7 -- the deleted
  arm ending LARGER). Use the return-curve window where the arms are
  still comparable, never an end-of-lifetime J difference.
- In an evolving library the marginal carry cost of an abstraction is
  ENDOGENOUS: when deletion triggers a replacement promotion it saves 0
  bits, not `lambda * D(A)`. Per-object retention rules are therefore
  ill-posed; the decision is sequential.
- Abstractions are ~4-8x overparameterized: `D_min ~ 1-2 bits/scalar`
  against 8 stored, with ZERO behavioral cost at 6 bits and 127 nats at
  4. This is why every structural edit loses to local compression, and
  it means absolute (not paired) two-part figures in V1-V4 are inflated
  by that factor.
- Whenever a structural property is manipulated, measure separately what
  it did to COST and what it did to UTILITY. V5.1's registered
  proportionality failed only because residual rank moved both: rank 4
  costs 4x rank 1 and also saves 1.86x per use. The law
  `H* = lambda D*(A) / s_bar` survived precisely because the two were
  measured independently at each rank. Any future intervention on a
  representation's economics must report both halves before its
  crossing is read.
- Padding is NOT a valid intervention on description length. Under the
  rate-distortion currency established in V5.0,
  `D*(A + dead bits) = D*(A)`; charging for dead bits tests an
  artificial storage tax, not abstraction economics. Manipulate what an
  abstraction can compute, or manipulate the code, but do not inflate
  the serialization.
- Task-space capacity is not realized library size. Raising composition
  depth (L=4) did not raise M. The direct knob is F explicit recurring
  innovation families with m tasks each, N = Fm, holding m fixed while
  sweeping F -- then PROMOTE yields M ~ F. Use this for any experiment
  whose independent variable is library scale.
- A generator knob that is supposed to vary RELATEDNESS must not vary
  individual value. Frozen balance gates (each within 10% across the
  sweep): per-abstraction `D*`, `s_bar`, behavioral contribution, and
  promotion rate. A sweep failing any gate is unscoreable rather than
  weak -- this is the V5.1 confound restated as a precondition.
- Never run `git checkout <file>` (or any resetting command) while
  uncommitted edits exist in that file. During the V5 spec work this
  discarded eight applied consistency fixes that had to be re-derived
  and re-applied. Commit first, then reset; the commit is cheap and
  the reconstruction is not.
- A specification that changes its generator must be re-read for
  everything DOWNSTREAM of the generator. V5's Revision 3 replaced the
  meta-recurrence generator and removed its fixed centre `C`, but
  `teacher_G1` and H21's entire five-way instrument were still written
  against `C`. Editing a definition is not the same as editing what
  depends on it, and an independent audit found this where three
  self-reviews had not.
- A historical lifecycle batch found that five concurrent `slots=12` runs
  exhausted RAM and failed 113/120 cells; even the later `--jobs 3` rule is now
  **superseded locally**. Lifecycle and prospective lifetimes run serially on
  this host. The clustering holds `(tasks, centres, features)` tensors, so slot
  count still predicts peak memory on remote workers.
- Check a registered threshold against its own BASELINE before freezing
  it. The S0 arm registered `p_reuse >= 0.5`, and the unmodified world
  scored 0.25 — a bound the control already violates cannot detect the
  effect it was written to catch. Compute the baseline first, then
  register the threshold relative to it.
- A learner's routing table is not a proxy for an abstraction's value.
  Deleting one abstraction made 0/32 returning tasks adopt ANY
  abstraction though a second one stayed live, and the saving was no
  larger on tasks that routed to the deleted object (52.8) than on those
  that did not (65.9). `C_reacquire` prices the loss of the reuse
  pathway, not the direct use of one object.
- `D_retain - D_delete = D(A)` is an accounting convention in this
  codebase, not a measured difference: both arms' checkpoints store
  identical scalar counts, retirement is logical, and referencing a live
  abstraction does not shrink a task's stored residual. Fine as a
  convention, but never describe it as measured.
- `operator_slots` is not a neutral protocol constant. Halving it
  (12 -> 6) cost 24% of the per-use saving by producing a smaller
  library that fewer tasks reference. Absolute retention crossings must
  always be quoted with their slot budget.
- Look for the coordinate in which a construction is LINEAR before
  reaching for optimization. V5's meta-recurrence knob had to mix
  operators in function space; sharing `V` and `b` across the family
  makes the hidden features common, the residual linear in `U`, and a
  mixture of `U` matrices exactly a mixture of functions.
- Make a generator's invariants exact rather than expected. Matching
  component norms in expectation left a random cross term that failed
  the balance gate at 31.6%; projecting the private component out of
  the shared subspace made the norm exactly constant (0.0% spread).
- Before believing a gate FAILURE, check the instrument can express the
  tolerance. Integer bit depths made a per-object rate jump by whole
  bits and failed a 10% balance gate at 16.4%; interpolating in
  (bits, log error) gave 2.9%. The generator was never at fault.
- Never fit and score a shared-structure measurement on the same
  objects. In-sample subspace capture read 0.730 where the truth was no
  structure at all; leave-one-out on the same data read 0.021.
- Check that a registered term is ARCHITECTURALLY POSSIBLE before
  registering it. H29 predicted `D*(P_2) < D*(P_1)` from continued
  learning, but promoted abstractions carry `requires_grad=False` and
  join no optimizer group, so `P_2 == P_1` identically and the term is
  structurally zero.
- State the DENOMINATOR with any fraction-of-cells threshold. "Within
  15% in >= 4 of 6 cells" is ambiguous once a cell can be
  unobservable: sealed C2 read 79%/73% counting only worlds that
  crossed and 37%/63% counting all of them, so the clause resolved to
  neither pass nor fail.
- A crossing experiment must verify its range CONTAINS the predicted
  crossing before running. At F=12 with M_0=4 only 8 unseen members
  exist, so a predicted M* ~ 20 is unobservable and half the worlds go
  silent rather than wrong. This is the horizon-grid bracketing rule
  from H19, applied one level up.
- Check which side the TARGET belongs to before reading a comparison. A
  whole-library economy test defined each task's target as the teacher
  operator's own effect, so the teacher arm had zero residual by
  construction and "won" 16x. Any ratio that large is a rigged
  comparison, not a result. This is the same error class as V4.1's
  total-variance tolerance and the raw-ablation counterfactual.
- The residual tensor is not "the task's innovation". A learner solves a
  task through shared basis + route + private residual, so a family
  shared by two tasks can be split differently between route and
  residual and look absent in the residuals alone. Measure the
  EFFECTIVE task-conditioned operator, F_tau(z) - F_0(z) with
  task-specific information nulled, before concluding anything about
  what the learner represents. Same error family as comparing learner
  slot indices to teacher primitive indices.
- An audit that writes to a fixed report path must stamp its run or
  refuse a report older than its inputs. A scorer that dies partway
  leaves the PREVIOUS answer on disk and nothing in the output says so;
  H29's first two attempts were read as results that way.
- Before implementing a proposed search, check whether the objective is
  INVARIANT to the thing being searched over. Review 49 asked for a
  global rotation Q maximizing subspace capture; capture is invariant to
  a global orthogonal map of the point set, so that search is empty by
  construction. The well-posed version was the span question — are the
  teacher's atoms mixtures of the learner's — which is what the
  motivating example (B1 = A1+A2, B2 = A1-A2) actually expresses.
- A full-span least squares gives an UPPER BOUND on what any linear
  refactor can achieve. Use it to kill or keep a whole class of remedies
  in one measurement, rather than sampling the class one adapter at a
  time.
- A NESTED objective has TWO non-vacuity requirements, and V6 checked
  only one. The inner learner must actually learn, AND the outer
  gradient must actually depend on that learning. The V6 prospective
  arm verified the second and shipped an inner loop that moved the
  support loss by 0.000%, so it charged a zero-shot loss and silently
  ran a different objective than the registered one. Constitutional for
  any meta-learning rung: inner loss materially decreases; inner state
  materially changes; query loss is evaluated after that change; the
  outer gradient changes when the inner adaptation is disabled; and
  k = 0 versus k = many objectives are measurably different. The last
  is the cheapest and catches this whole class.
- Invalidated artifacts get a MACHINE-CHECKABLE manifest, not a
  paragraph. `artifacts/INVALID_MANIFEST.md` plus
  `tools/check_invalid.py`, run alongside `check_prereg.py`. And retire
  the path: rebuilding corrected runs into an invalidated directory
  makes "invalid" and "rebuilt" indistinguishable to the checker.
- A guard that parses its own input must fail when it parses nothing.
  The first `check_invalid.py` omitted `re.MULTILINE`, matched zero
  paths, and printed a clean pass over a manifest listing six.
- Two functions can only be compared when evaluated at the SAME inputs.
  V5's effective-operator and population-span audits built each task's
  vector on that task's own probe, then compared them coordinate-by-
  coordinate; coordinate j meant a different state per task. Corrected,
  R_effective went 0.190 -> 0.762 and span-unexplained 0.707 -> 0.491,
  reversing a published conclusion. Same error family as slot-index
  versus primitive-index and parameter-mean versus function: whenever
  two things are compared, ask what coordinate system each lives in.
- Reconstructing a learner from an artifact means reconstructing ALL of
  its state. `load_learner` restored promoted references but not
  retirement, so retired tasks were rebuilt with both the abstraction
  and the residual retirement had removed — a model that never existed
  during training.
- NESTED-LEARNING AUDIT, run before any meta-learning result is read.
  V6 produced six failures each of which could have supported a
  coherent but false story: (1) does the intervention change parameters
  at all — a frozen basis made every arm secretly identical; (2) does
  the inner learner actually learn — SGD moved the support loss 0.000%;
  (3) does the probe have dynamic range — a saturated probe reports the
  same number for every arm; (4) is the future genuinely unseen — the
  "held out" sibling was later trained on; (5) does the outer objective
  depend on the inner adaptation — k=0 versus k=many must differ;
  (6) does the result replicate across worlds — a positive world 0
  vanished at n=3.
- A negative endpoint does not identify its own mechanism. Phi said the
  intervention hurt; only a direct measurement of discrimination and
  task-code sensitivity could say whether the representation LOST
  information or merely became harder to optimize, and those imply
  different successor designs. Measure the mechanism before designing
  the fix.
- CONCURRENCY, corrected. Memory is the binding constraint, not cores;
  this host has 16 and a lifetime pins one thread, so serial execution
  wastes ~94% of it. Use a bounded pool: 3-4 concurrent for `slots=12`
  promoting/lifecycle runs (the promotion clustering step allocates
  `(tasks, centres, features)` tensors and is what exhausted RAM at 5),
  up to 6 for lighter models. Two independent failures taught this and
  they have different fixes: 5 concurrent lifetimes exhausted memory and
  killed 113 of 120 cells, while two launcher INSTANCES over the same
  output paths raced and logged completions for cells that had no
  artifacts. The first is fixed by a cap, the second by one writer per
  cell — a `ProcessPoolExecutor` over a job list gives both, a shell
  loop relaunched twice gives neither.
- Before scaling a batch up, check free memory against one run's
  resident size rather than guessing. A `slots=12` lifetime sits around
  340 MB; the ceiling is RAM divided by that, minus headroom, not the
  core count.
- Scorer CLI arguments are part of the experimental protocol. Before extending
  a curve, rerun its anchor cells and require exact reproduction; do not assume
  today's source defaults equal the arguments used for the prior report. H35's
  first scorer silently used 60 adaptation steps against a 40-step anchor and
  changed the absolute acquisition costs. Reports must record the executed
  steps, supports, learning rate, seeds, and noise scale.
- Write a completed report atomically before printing its presentation summary.
  A Windows console-encoding failure after H35 scoring discarded a finished
  computation because the report write came last; console output is not the
  scientific artifact.
- A pool job must carry and validate its own complete intervention record before
  treating `summary.json` as resumable. The V6 allocation pool accidentally
  reused H35's 8 inner steps for a 16-step sweep and silently skipped nine
  mismatched summaries. Encode sweep-specific inner/outer steps per job; require
  model, summary, provenance, and fingerprint files; refuse a non-empty
  mismatched target; and exit nonzero if any cell fails.
- Treat "oracle" adaptation as a finite operational envelope, never proof of a
  global optimum. Freeze the representation, optimize support only, keep query
  labels out of selection, include abundant-support and sparse-support cells,
  and require agreement across materially different optimizers before assigning
  a capacity, identifiability, or findability mechanism.
- ZERO IS A STATIONARY POINT OF `u.tanh(vz+b)`. A zero-initialized rank-2
  residual never moves (d/du = tanh(0), d/dv is proportional to u), and any
  argument that reaches the output only through that residual
  (`dL/dalpha = W^T dL/dr`) is pinned with it. This froze training once
  (exact-null eps) and an adaptation fit once (eps = 0 in the alpha-only
  protocol); both were caught only because a non-vacuity check required
  `alpha_norm > 0`. A literal null state in this family needs a gate, never a
  zero vector. Arguments that enter a basis operator's `U` matrix do not have
  this problem: `dL/dalpha_k = <dL/dU, U_k>` is nonzero at alpha = 0.
- The residual vector is not where family computation lives. Zeroing the
  whole residual channel (schema + eps) of live family tasks moved NMSE by
  about 2%, and 56/64 family tasks had been retired into promoted references
  where the channel does not fire. Two residual-side schemas (post-hoc PCA
  over the learner's own residuals, and one formed jointly online) were both
  infertile for exactly this reason; an argument inside a routed basis slot
  moved every endpoint. Parameterize the objects the learner routes through.
- The affine span of what the ordinary learner stores does not contain nearby
  novelty. Fitting only a coordinate in the span of all live residuals and
  abstractions (any rank, up to the full span) leaves an unseen-family member
  at 2.2-3.1x the free-residual endpoint, and the pre-retirement population
  is no better (2.75x). Retirement did not erase the directions; ordinary
  wake never formed them.
- `novel_family_tasks` are members of two families the lifetime never sees,
  all family operators sharing one rank-2 functional subspace at `r_meta=1`;
  held-out members of SEEN families are nearly free for every arm and are
  deliberately not the future. Re-read the generator before naming a future
  "sibling"; the per-family schema arm this error produced was ill-posed.
- Check a learner-statistic threshold against the BASELINE ARTIFACT before
  freezing it, every time. Ordinary routes are diffuse (mean maximum
  coefficient 0.29-0.44; any one slot averages ~0.08-0.12 mass), so a "route
  mass on P >= 0.5" usage criterion was unattainable by construction and its
  2x-baseline successor still never fired while the functional ablation
  passed everywhere. Prefer the functional criterion (zero the channel,
  measure the loss) as the deciding usage test; route statistics are
  diagnostics.
- A frozen-argument variant that is BITWISE the baseline is the cheapest
  strong control: `P(alpha)` at `alpha = 0` reproduces the ordinary artifact
  tensor-for-tensor, so every later difference is the channel. Build new
  architectures so that a switch recovers the baseline exactly, and verify it
  on the real artifact, not only the smoke world.
- Fertility of an in-basis argument scales with argument dimension and needs
  LEARNED directions: on worlds 0-2 the alpha-only future ratio fell
  monotonically from 3.4 (K=2) to 1.56 (K=16) while a matched-budget control
  with `U_k` frozen at random init stayed at ~3.1 and was essentially unused
  (alpha-zeroed ratio ~1.05). A single linear-in-U slot then SATURATES at
  K~16 (K=32/64: 1.59/1.57) even though present-task loss keeps improving;
  a second parameterized slot at K=32 reached 1.46 and passed 1.5x in two of
  three worlds while hurting the third. Treat K as a capacity knob, not the
  generator's rank, and treat the two-slot A verdict as marginal until a
  sealed block decides it.
- Heredocs fed to the Bash tool break on an apostrophe inside a quoted
  Python or Markdown body and leave NOTHING applied (the whole chain aborts).
  Write multi-file patches as a scratch `.py` with explicit UTF-8 I/O (this
  file is not cp1252-clean) and run it; check `git status` before assuming
  any part landed.
- Lifetimes write artifacts only at completion, so an interrupted or
  allocation-killed cell leaves no directory; a transient 3.66 MiB Windows
  allocation failure at a sleep is a commit-limit spike, not sustained
  pressure (three `slots=12` cells sat at 345 MB each with 9 GB free). Let
  the survivors run and relaunch the cell; the launcher skips completed
  cells and refuses mismatched records.

- THE H39 LINE'S CENTRAL LESSON (reviews 62-65; `H39_DEVELOPMENT_CLOSURE.md`,
  `reports/h39_confirmation.json`, `reports/h39_confirmation_followups.json`):
  the useful abstraction is not sitting in the finished collection of task
  solutions waiting to be extracted; the learner has to create and maintain
  the coordinate system while it learns. Online alpha-only acquisition of an
  unseen family costs 1.64x ordinary; post-hoc extraction from the ordinary
  learner's realized population costs 2.59x and from the argument learner's
  own residual population 6.33x, because an explicit argument channel drains
  family structure out of the residuals. Removing the arguments and
  re-optimizing the other task-local channels at matched budget still costs
  2.0x. Working definition of an abstraction from here on: it compresses
  what is invariant while exposing useful coordinates for what may vary —
  `A(alpha) + eps`, not `A`. The V6 prospective LOSS was the wrong
  mechanism; giving recurrence, variation, and innovation different
  representational ROLES was the right one. Thesis, reformulated:
  intelligence learns parameterized computational representations that
  compress what recurs while preserving cheap coordinates for novelty; the
  eventual language is made of parameterized primitives P_i(alpha), and
  adaptation is identify-primitive + infer-argument (+ a small patch).
  Existence and use are sealed; discovery (whether, which, how many, which
  dimensions) is the open rung. Do not describe the confirmed result in
  cluster language: its two slots form one distributed argument channel,
  and the confirmed world has a single family subspace.

- OPPORTUNITY GATE, constitutional (review 68, after the B2 gate): before
  any structural-recovery or discovery experiment, measure the
  counterfactual cost of REFUSING to represent the proposed structure —
  an oracle that is told the structure versus the learner that ignores it.
  If the gap is ~0, failure to recover the structure is not a learning
  failure; there is nothing economically worth recovering. On a world with
  two ORTHOGONAL family subspaces and teacher-level group classification
  of 1.000, the two-slot K = 32 learner ignored the groups (entropy ~0.93,
  ARI ~0), paid +0.27 / +0.10 / -0.03 log units on alpha-only acquisition
  for it, and was ~500 nats BETTER than the told-membership oracle on
  present cost: one 64-direction channel absorbs the union of two rank-2
  subspaces. Truth of a latent decomposition is not utility of
  representing it; abstraction tracks computational necessity, not
  ground-truth labels. Schema count and within-schema width trade off
  directly (2 x 32 versus 1 x 64), so "how many abstractions" is a
  WIDEN-versus-SPLIT decision under one D*, and the capacity knob (K per
  slot) is the independent variable that can make discrete identity
  necessary.

- THREE ECONOMIC QUANTITIES, KEPT SEPARATE (review 69, after H48b): current
  utility (does representing a structure make seen tasks cheaper), future
  fertility (does it make an unseen relative cheaper), and representational
  cost (does it shorten the description). At K = 4 on the two-subspace
  world the true grouping had future value (+0.1-0.5 log units) and no
  present value (-26 to -121 nats), so the label-free learner's refusal to
  separate the groups was RATIONAL under its objective. Before any
  discovery experiment, verify that a quantity computable from the
  learner's own past (leave-one-out reacquisition, a two-part description
  proxy, substitutability) actually prefers the fertile structure; if none
  does, non-discovery is not a learner failure. Never anneal a learner
  toward a discrete decision its objective never rewarded and call the
  result a discovery test. H49 is the census that asks this on the K = 4
  artifacts; its outcome decides whether a sleep operator has an objective.

- SCORING ECONOMY AND THE SHAM RULE (review 71, H50): when compute forces
  cuts, drop curve-refinement cells (fewer scored budgets), never the
  sample of the causal statistic (all LOO tasks); run corroborating
  instruments only on leading/best-wrong/sham arms. Any propose-
  reorganize-score loop needs a SHAM arm (same start, optimizer, data,
  parameter count, steps, no structure) — beating sham proves
  optimization, beating max(WRONG/RANDOM) proves abstraction, and both
  are required. Report the recovery fraction against the from-scratch
  reference so a yes/no separation becomes a mechanism claim (cheap
  counterfactual evaluation vs disguised retraining). Reused baseline
  rows inherit the original seeds (H50 Amendment 2).
