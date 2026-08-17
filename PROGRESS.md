# Current milestone

Milestone 007 diagnostics and figures, followed by Milestone 008 reuse sweep.

# Completed

- Initialized the Git repository and Python package.
- Implemented deterministic random residual primitives.
- Implemented programs, opaque task IDs, unique task generation, and fixed
  train/evaluation datasets.
- Added world validity diagnostics for output scale and saturation.
- Added a NumPy scratch residual MLP and examples-to-criterion experiment.
- Added unit tests for determinism, uniqueness, opacity, shapes, and metrics.
- Passed the initial test suite and smoke experiment.
- Ran the 64-task seed-0 scratch control: median final NMSE 0.0742, final-NMSE
  task-index correlation -0.0963, output variance range 0.2129–0.2284, and zero
  saturation.
- Prepared the verified foundation as the initial Git checkpoint.
- Installed PyTorch 2.13 for the oracle and subsequent learned models.
- Tuned the scratch control to learning rate 0.001 with four updates per example.
- Completed scratch controls on world seeds 0–2. Final-NMSE/task-index
  correlations were -0.060, 0.002, and -0.039; all tasks reached NMSE 0.1 at
  support 64, confirming no task-order difficulty trend at that resolution.
- Implemented the true-route oracle with six shared learned operator slots,
  predict-before-update scoring, completed-task replay, fixed support evaluation,
  model serialization, and unseen-composition testing.
- Confirmed oracle transfer on world seeds 0–2 and reversed seed 0. Across these
  runs, first-quarter mean examples to NMSE 0.05 ranged from 8.75 to 28.06 and
  fell to zero in the last quarter; examples to NMSE 0.02 ranged from 62.25 to
  96.38 and also fell to zero.
- On seed 0, zero-shot NMSE fell from 0.0345 in the first quarter to 0.00192 in
  the last quarter; unseen-composition zero-shot NMSE was 0.00215.
- Added functional primitive recovery. Seed-0 learned slots matched all six
  teacher primitives with mean normalized distance 0.000539.
- Implemented a shared non-oracle lifetime protocol, task-state-only novel
  composition adaptation, Dense-P, Dense-C, width-128 dense, and an eight-slot
  continuous reusable operation basis.
- Corrected a dense initialization confound by using three task-conditioned
  state-space residual/tanh blocks, matching the teacher's compositional cadence.
- Identified task-code inference as the continuous model's initial bottleneck.
  Raising task-code LR from 0.005 to 0.05 reduced seed-0 prequential NLL from
  -152,304 to -165,119 and primitive matching distance from 0.0151 to 0.00349.
- Seed-0 tuned continuous beat Dense-P, Dense-C, and width-128 dense on
  prequential NLL. Its frozen-library novel task improved from NMSE 0.0453 to
  0.00803 after 32 code-only examples.
- Replicated tuned continuous versus Dense-P on world seeds 0–2. Continuous won
  prequential NLL by 9,126, 4,940, and 5,746 nats; mean paired advantage was
  6,604 with a three-world bootstrap interval of 4,940–9,126.
- Continuous improved 32-shot frozen-library novel-composition NMSE over Dense-P
  by 0.0302, 0.0151, and 0.0114 on the same worlds.
- Added 8-bit retained-description evaluation. Continuous retained 29,184 proxy
  bits versus Dense-P's 33,928, and measured quantization degradation was
  negligible on these artifacts.
- Added analytic inference accounting: continuous uses about 6,528 multiply-adds
  per prediction versus Dense-P's 2,112.
- Replicated the compute-matched Dense-C control on all three worlds. Continuous
  won paired prequential NLL by 3,135, 675, and 1,176 nats and improved 32-shot
  novel NMSE on every world.
- Dense-C uses about 6,144 multiply-adds but retains 66,688 proxy bits, compared
  with continuous's 6,528 multiply-adds and 29,184 bits.
- Implemented the 12-slot hard discrete library with annealed relaxed training,
  hard argmax evaluation, route recovery, usage, collapse, fragmentation,
  duplication, and operator matching diagnostics.
- Seed-0 discrete recovered 92.2% of explained routes exactly and 96.4% of route
  positions, used 11/12 slots, and reached primitive matching distance 0.00229.
- Discrete prequential NLL was only -134,784 despite good final operators/routes;
  this isolates task-route inference as the main failure mode. Late-task examples
  to NMSE 0.02 still fell from 128.9 to 25.3.
- Hardened discrete retains 26,112 proxy bits and uses about 768 inference
  multiply-adds. Correct route-lossless 8-bit evaluation changes mean NMSE by
  only 1.10e-5.
- Implemented the full `rho` teacher continuum with task-specific correlated
  primitive parameters and measured functional-reuse diagnostics.
- Validated the causal endpoints: residual-function correlation is approximately
  zero at `rho=0` and one at `rho=1`.
- On development world 0, Continuous beats Dense-C by 3,135 prequential Gaussian
  log-loss units at `rho=1`, but loses by 6,714 at `rho=0`.
- Added four-program frozen-library checkpoints at 8, 16, 32, and 64 lifetime
  tasks. Continuous 32-shot novel NMSE improves from 0.0298 to 0.00467; Dense-C
  improves from 0.0367 to 0.0159.
- Added fixed target precision `1/256` and a quantized-target prequential log-loss
  total for a defensible coding interpretation.
- Generated and visually inspected six required exact-reuse pilot figures.
- Added `EXPERIMENT_PLAN.md` defining development seeds 0–9, sealed confirmatory
  seeds 100–129, symmetric tuning, causal metrics, and robustness work.
- Completed the symmetric six-setting LR grid for both Continuous and Dense-C on
  development worlds 0–2 (36 full lifetimes).
- Dense-C improved substantially when its task LR increased to 0.05, correcting
  an unfair pilot comparison. Its best stage-one mean log loss is -168,768.
- Continuous's best stage-one setting is global/task LR 0.003/0.05 at -171,866,
  with mean novel 32-shot NMSE 0.00237 versus Dense-C's 0.00486.
- Removed the teacher-alpha leak: learner operator scales now initialize at 0.2
  and are learned independently.
- Seed-0 family controls passed provisionally. Continuous beats Dense-C by 3,296
  with teacher rank 16 / learner rank 8; GELU learner mismatch retains a smaller
  441 advantage.
- A fixed identity basis slot did not improve Continuous, arguing against the
  fixed-depth convex constraint as the explanation for the primary result.
- Per-task discrete annealing improves log loss by 8,825 and greatly improves
  default-route zero-shot behavior, but remains far behind Continuous.
- Re-ran the oracle under independent learnable alpha and no alpha weight decay;
  the positive-control gate still passes with primitive distance 0.000484.
- Added grouped task batching, normalized prequential reporting, and explicit
  training-forward versus hardened-inference multiply-add accounting.
- Sparse seed-0 `rho` points show Dense-C wins total online loss by 2,159 at
  `rho=0.5` and 1,615 at `rho=0.75`; measured residual correlations are 0.064
  and 0.317. Continuous has slightly better novel 32-shot adaptation at both.
- Completed the six-point leak-free seed-0 recurrence curve. Dense-C wins at rho
  0, 0.25, 0.5, and 0.75; Continuous wins at 0.9 and 1.0. The paired cumulative
  loss effect crosses zero at a descriptively interpolated rho 0.811, or measured
  residual-function correlation 0.454.
- Added a validated machine-readable recurrence comparison and a visually checked
  two-panel crossover plot under `reports/rho_seed0`.
- Revalidated both optimizer finalists per architecture under learnable alpha on
  development worlds 3–9. Continuous 0.003/0.05 and Dense-C 0.001/0.05 remain
  the winners and are frozen in `configs/v1.yaml`.
- Selected Continuous beats selected Dense-C on all seven stage-two exact-reuse
  worlds by a mean 3,787 cumulative log-loss units; mean novel 32-shot NMSE is
  0.00368 versus 0.00731. The validated selection report is in
  `reports/stage2_current/selection.json`.
- Replicated all six rho points on development worlds 1–2 and combined them with
  seed 0. Each world selects Dense-C through rho 0.75 and Continuous from rho
  0.9; interpolated crossings are 0.811, 0.852, and 0.851.
- The three-world mean crossing is rho 0.833, or measured residual correlation
  0.488. The machine report and visually checked paired-world plot are under
  `reports/rho_worlds_0_2`.
- Falsified the provisional intermediate-transfer story: Continuous's better
  32-shot novel adaptation at rho 0.5/0.75 was seed-0-specific and does not hold
  in the three-world mean.

# In progress

- Expand the frozen six-point rho curve from worlds 0–2 to development worlds
  3–9, reusing their completed exact-reuse stage-two artifacts where possible.

# Next

- Complete and plot the development-world `rho` sweep with paired world deltas.
- Replicate checkpoint learning-to-learn curves across development worlds.
- Implement the generic low-rank hypernetwork and dense 24-dimensional task-code
  controls after the causal sweep.

# Decisions

- V1 starts with exact reuse (`rho = 1.0`). The reuse continuum follows only
  after the oracle and at least one learned reusable substrate work.
- Fixed arrays are part of a generated `World`, preventing accidental data-stream
  differences between paired models.
- The scratch validity gate uses NMSE 0.1 and continuous final error for now;
  NMSE 0.05 is fully censored by the current shallow scratch architecture and
  remains a sensitivity target rather than hidden or discarded evidence.
- The oracle gate passed. NMSE 0.1 is too loose for oracle initialization, so
  oracle transfer claims use the stricter thresholds and zero-shot behavior.
- The tuned continuous seed-0 win has now replicated on world seeds 1 and 2, but
  the three-world estimate remains exploratory rather than confirmatory.
- The continuous advantage over Dense-P replicated across three exploratory
  worlds, and it also survived the three-world Dense-C compute control. The
  sample remains exploratory (`n=3` worlds), not confirmatory.
- Hard discrete learned a compact and correct library but paid substantially
  higher online route-inference cost. Do not interpret its prequential loss as
  evidence against reusable computation; the continuous alternative succeeds.
