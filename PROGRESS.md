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
- Added resolved-config SHA-256 fingerprints to learned, oracle, and scratch
  artifacts. Sweep resume now rejects any architecture/protocol mismatch and
  validated/backfilled all 24 worlds 1–2 rho artifacts.
- Expanded the rho sweep through world 5. All six worlds favor Dense-C through
  rho 0.75 and Continuous from rho 0.9; the mean crossing is rho 0.826 or
  measured recurrence 0.479. A checked report and plot are under
  `reports/rho_worlds_0_5`.
- Across those six worlds, Continuous wins novel 32-shot adaptation 6/6 only at
  exact reuse; it has worse mean transfer at rho 0.9 despite winning lifetime
  loss 6/6. This further rejects the provisional transfer-first interpretation.
- Completed the frozen six-point rho sweep on all development worlds 0–9. Every
  world favors Dense-C through rho 0.75 and Continuous at rho 0.9/1.0. The mean
  crossing is rho 0.8264 or measured recurrence 0.4836; individual crossings
  span 0.8108–0.8590.
- The full report and visually checked plot are under `reports/rho_worlds_0_9`.
  Continuous's novel-transfer advantage is reliable only at exact reuse, so the
  transfer-first secondary hypothesis is closed as unsupported.
- Replicated current learnable-alpha checkpoint curves on all ten development
  worlds. Continuous moves from 4/10 paired wins after 8 lifetime tasks to 8/10
  after 16 and 10/10 after 32 and 64.
- Continuous's per-world 8-to-64 fresh-composition gain averages 6.70x versus
  Dense-C's 3.64x; both improve in every world. The machine report and checked
  figure are under `reports/checkpoints_worlds_0_9`.

# In progress

- Completed a line-by-line V1 obligation audit in `SPEC_AUDIT.md`. The
  confirmatory gate remains closed while batch-protocol freezing, shared-parent
  residuals, statistical freezing, and clean workflow rehearsal remain open.

# Two-initialization pilot complete

- Added Continuous seed 4001 and Dense-C seed 3001 on all ten development
  worlds, holding selected architectures and protocols fixed.
- The second initialization reproduces lifetime loss and novel 32-shot wins
  10/10, with mean advantages 3,081 and 0.00269 respectively.
- Averaging initialization effects within world gives a 3,390 lifetime-loss
  advantage with bootstrap interval [2,962, 3,803], and a 0.00299 novel NMSE
  advantage with interval [0.00200, 0.00426].
- The validated report is
  `reports/model_initializations/model-initializations.json`.

# Explicit forward transfer complete

- Trained selected Continuous and Dense-C freshly and independently on every
  task in all ten development worlds, using the same score-before-update
  protocol and architecture but no prior tasks or replay.
- Mean per-task fresh-minus-lifetime log loss is +366.8 Continuous and +211.6
  Dense-C; world means are positive 10/10 for both.
- Continuous has 155.2 more forward-transfer units per task than Dense-C on
  average, with all ten paired worlds positive and interval [145.5, 165.3].
- Forward transfer rises with task index in every world and is larger for tasks
  whose hidden routes share more positions with prior tasks.
- The validated report and visually inspected figure are under
  `reports/forward_transfer`.

# Checkpoint operator diagnostic complete

- Added analysis-only Hungarian matching from learned slots to hidden teacher
  primitives at 8, 16, 32, and 64 tasks, then evaluated programs through those
  matched slots under hidden teacher routes. No hidden identity or program enters
  learner training.
- Across worlds 0–9, one-to-one primitive distance improves from 0.00786 to
  0.00211 for Continuous and from 0.00745 to 0.00131 for per-task-annealed
  Discrete; every world improves for both learners.
- Matched-slot evaluation on future programs improves through task 32, directly
  showing operator-library improvement independently of future task-code
  inference.
- Continuous's own mixture routes outperform forced matched-slot routes in all
  ten worlds at every checkpoint. Discrete's hard learned routes converge with
  matched-slot routes by the end. The diagnostic therefore supports progressive
  operator quality but also exposes Continuous's distributed representation.
- The validated report and visually inspected figure are under
  `reports/operator_checkpoints`.

# Scrambled-ID validity check complete

- Added deterministic reassignment to an independent, disjoint opaque task-token
  namespace without changing programs, task order, examples, evaluation arrays,
  replay, model seeds, or optimizer settings.
- Continuous and Dense-C world-0 canonical/scrambled pairs have exactly equal
  metric rows and summaries after removing token labels. Their final shared and
  task-specific tensors are bitwise equal after task-index key normalization.
- The machine-checked result is `reports/scrambled_ids/scrambled-ids.json`.

# Batch-size sensitivity complete

- Implemented the V1 suggested target batch 8 as a paired 1:1 minibatch: the new
  current example plus up to three prior current-task examples, and four replay
  examples. Target batch 2 uses one current plus one replay example.
- Across development worlds 0–2, batch 8 improves lifetime log loss 3/3 for
  both models. Dense-C improves more, reducing Continuous's mean lifetime
  advantage from 3,463 to 2,061, but Continuous remains better 3/3.
- Continuous retains lower 32-shot novel NMSE 3/3 at both sizes. The result
  closes the missing sensitivity but requires an explicit protocol/retuning
  decision before confirmation because batch 8 increases sample reuse and
  training compute.
- The machine report is `reports/batch_sizes/batch-sizes.json`.

# Rho bridge analyses complete

- Reconstructed cumulative Dense-C-minus-Continuous effects at 2,048, 4,096,
  and 8,192 online examples from all 120 existing development rho logs, with
  every final sum validated against the tracked ten-world rho report.
- Mean configured-rho crossovers are 0.869, 0.822, and 0.826 after 16, 32, and
  64 tasks. Eight of ten worlds decline from 16 to 64, but the mean is not
  monotone, so H5a is only partially supported.
- Measured recurrence makes the six-point mean effect curve much smoother
  (`R²=0.974` versus `0.646`), but it does not align per-world crossings more
  tightly. H5b is mixed rather than confirmed.
- The validated report and visually inspected figure are under
  `reports/rho_bridge`.

# Order and replay robustness complete

- Completed reverse order and replay ratios 0, 1, and 4 on all ten development
  worlds for selected Continuous and Dense-C.
- Continuous wins lifetime loss 10/10 in every condition. Mean paired advantages
  are 3,129 no replay, 3,698 canonical replay, 3,350 strong replay, and 3,456
  reverse order; every ten-world bootstrap interval excludes zero.
- Absolute reverse-minus-forward effects are inconsistent for both models, so
  the result is not tied to one favorable curriculum.
- Novel 32-shot transfer weakens without replay (6/10 wins and interval spanning
  zero) but is 10/10 under canonical replay, strong replay, and reverse order.
- The validated report and visually inspected figure are under
  `reports/robustness`.

# Current retention complete

- Re-evaluated symmetric per-tensor int8 artifacts across development worlds
  0–9. Retained proxies are 29,248 bits Continuous, 33,928 Hypernetwork, 56,448
  Dense-24, and 66,688 Dense-C.
- Mean quantization degradation is at most 3.23e-6 NMSE for these models; the
  largest individual task increase across the sweep is 1.38e-4.
- The current per-task Discrete artifact retains 26,208 bits, uses 768 hardened
  inference multiply-adds, and changes mean NMSE by 4.77e-6.
- The validated report is `reports/retention/current-retention.json`.

# Structural controls complete

- Implemented a generic low-rank hypernetwork with the same 24 task-state
  scalars and three residual stages as Continuous but no explicit slots.
- Symmetric tuning selected global/task LR `0.003/0.05` in both stage one
  (worlds 0–2) and stage two (worlds 3–9); the setting is frozen in
  `configs/v1.yaml`.
- Across all ten development worlds, Continuous beats Hypernetwork on lifetime
  loss and novel 32-shot NMSE 10/10, while Hypernetwork beats Dense-C on
  lifetime loss 10/10. Mean log-loss advantages are 1,907 and 1,791,
  respectively.
- Dense-24 versus Dense-32 is effectively tied: 5/10 lifetime-loss wins and a
  mean -73-unit advantage for Dense-24. Matching retained task-state size does
  not remove the primary hierarchy.
- The validated report and visually inspected paired-world figure are under
  `reports/structural_controls`.

# Next

- Freeze batch 2 versus batch 8 and any required symmetric retuning for
  confirmation.
- Resolve the conditional MDL presence-gate decision and add the frozen
  world-level/task-level statistical summaries.
- Rehearse artifact generation and plotting from a clean checkout before opening
  the confirmatory seed gate.

# Shared-parent residual tuning complete

- Implemented a fixed shared eight-slot parent plus rank-two task-specific
  residual factors at each of three stages. Routes and residual factors have
  separate optimizer groups, and every run is rejected if any task residual's
  functional magnitude reaches the parent update magnitude.
- A nine-point nonzero-penalty grid on development world 0 at configured
  `rho=0.75` selected residual LR 0.01 and L1 penalty 0.01 by guarded cumulative
  Gaussian log loss.
- The selected tuning run scores -157,103 cumulative log loss, 0.01748 novel
  32-shot NMSE, and mean/maximum functional residual ratios 0.246/0.419. The
  result is a development tuning choice, not replication evidence.
- Added a serialized, resumable recurrence sweep and a fixed-envelope analyzer.
  The sweep reuses only the validated selected tuning artifact and otherwise
  runs one full lifetime at a time to avoid the prior paging-file failure.

# Shared-parent residual recurrence complete

- Evaluated the frozen control on development worlds 0–2 at configured rho
  0.5, 0.75, 0.9, and 1.0. The world-0 rho-0.75 point is explicitly retained as
  tuning; the corresponding worlds 1–2 results are untuned replication.
- The hybrid beats the better fixed Continuous/Dense-C model on lifetime loss
  in all 9 intermediate-reuse comparisons. Mean gains are 9,168 at rho 0.5,
  7,458 at rho 0.75, and 3,745 at rho 0.9. It also wins novel 32-shot NMSE in
  all nine.
- At rho 1.0 the specialized residual becomes nearly dormant: mean functional
  residual-to-parent ratio is 0.026 versus 0.284 at rho 0.5, declining in all
  three worlds. Continuous is then slightly better by a mean 246 log-loss units,
  as predicted for exact reuse.
- This closes the fixed-topology copy-on-write question, but the hybrid is not a
  storage-matched winner: it retains 14,208 task scalars over 64 tasks versus
  Continuous's 1,536 route scalars. The validated report and visually inspected
  figure are under `reports/shared_residual`.

# MDL presence-gate control started

- Resolved the conditional Model 4 gate in favor of implementation. Hard
  Discrete's final operator/route recovery satisfies the structural prerequisite;
  poor online route inference remains a separate optimization limitation.
- Added a distinct 12-slot presence-gated Discrete learner with relaxed global
  gates, expected-active-slot and route-entropy penalties, per-task temperature
  annealing, and hardened routing that cannot select inactive slots.
- A smoke lifetime and gate-gradient/masking tests pass. Symmetric penalty
  tuning and a full exact-reuse evaluation remain open.
- The initial presence LR 0.01 pilot over-pruned to the one-slot fallback and
  failed sufficiency. Added a resumable six-point lower-rate grid. Selection is
  frozen to the fewest sub-12 active slots with final median and novel 32-shot
  NMSE both at most 0.02, with log loss breaking active-count ties.

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
- Completed the shared-residual retained-description accounting
  (`reports/shared_residual/j-weighted.json`). Under the two-part MDL code
  (`lambda = ln 2` nats/bit) the shared-residual model loses to both fixed
  architectures in all twelve world/`rho` cells: its 130,624-bit retained
  footprint (~101k bits above Continuous) outweighs the 3.7k-9.2k-nat
  prequential gains; break-even `lambda*` is 0.037-0.143 nats/bit. The
  envelope win is therefore a prediction-cost result with the predicted
  allocation signature, not a code-length result at the current rank-2
  residual budget. Follow-ups recorded: rank-1/stronger-penalty residuals
  and sparsity-aware retention coding for the near-zero high-`rho`
  residuals.
- Completed the two-stage MDL presence-gate study on world 0
  (`reports/mdl_gating/mdl-gating.json`). Stage one was bimodal: five cells
  pruned nothing (12/12 active, novel 32-shot NMSE ~0.005) and the strongest
  penalty collapsed to one slot with broken accuracy. The stage-two bisection
  found a 7-active-slot library (teacher has 6) at penalty 5e-5 with
  negligible lifetime-accuracy cost, but both bisection cells failed the
  frozen novel-composition sufficiency limit (~0.033 versus 0.02), including
  the 11-slot cell, so the degradation tracks gate pressure rather than slot
  count.
- Explicit Model 4 decision (required by the spec audit): presence gating is
  recorded as not working as a compact-sufficient-library discoverer at this
  scale. Configurations achieve either a full library with good novel
  transfer or a smaller library with degraded novel transfer, never both.
  The result stands as a characterized negative; no further gate tuning.
- Passed the clean-checkout rehearsal at ed90ee2: fresh clone, 73/73 tests,
  full smoke lifetime with fingerprint-validated artifacts, and the plotting
  entry point loads. Caveat recorded: the rehearsal reused the installed
  Python environment; fresh-venv dependency installation remains untested.
- Froze the confirmatory protocol and pre-registered outcomes in
  `CONFIRMATION_PLAN.md` (three Holm-corrected primary outcomes, world-level
  exclusion rule, reporting rules) before opening any sealed world.
- Completed post-hoc operator recovery across the development recurrence sweep
  (`reports/rho_operator_recovery/operator-recovery.json`). Against an
  untrained-basis baseline of 0.0087, trained continuous operators sit
  farther from the shared primitives than untrained ones at `rho <= 0.5`
  (0.0115-0.0121), match baseline at 0.75 (0.0083), first drop below it at
  0.9 (0.0048), and crystallize at 1.0 (0.0017). Primitive recovery onset
  coincides with the performance crossover, and partial recovery at 0.9
  quantifies the statistical-versus-structural dissociation: lifetime wins
  appear before identifiable primitives do, and frozen-library transfer
  requires the crystallized regime.
- Opened sealed worlds 100-129 and ran the frozen confirmatory protocol
  (360 paired lifetimes; zero failures; no exclusions). All three
  pre-registered primary outcomes pass unanimously
  (`reports/confirmatory/confirmatory.json`): exact-reuse advantage 30/30
  worlds (mean +3,204 nats, median +3,177; sign test p = 1.9e-9);
  positive per-world recurrence slope 30/30 (mean +5,715 nats per unit
  measured recurrence; p = 1.9e-9); within-world sign reversal 30/30
  (p = 1.9e-9). All survive Holm correction (adjusted p <= 5.6e-9).
- Confirmatory secondaries: unanimous per-rho sign pattern (0/30 wins at
  rho <= 0.75, 30/30 at rho >= 0.9); the effect is linear in measured
  functional recurrence (R^2 = 0.935 vs 0.642 in configured rho; slope
  +5,716 nats per unit recurrence, zero crossing at r* ~ 0.46); crossover
  location rho* = 0.835 +/- 0.023 configured, r* = 0.499 +/- 0.050
  measured, across all 30 worlds. Per-example effect sizes range from
  -0.29 nats/example at rho = 0 to +0.39 at rho = 1.
- V1's central claim is now confirmatory: a reusable continuous substrate
  reduces lifetime prequential cost when and only when latent functional
  recurrence is sufficiently strong, with a reproducible crossover.
- Completed release-gate editorial items: public-facing README (question,
  metric, confirmed status, repository map, reproducibility statement), MIT
  LICENSE, and resolution of the robustness-means citation (the ten-world
  `reports/robustness/robustness.json` supersedes the unverifiable
  worlds 0-2 figures).
- Drafted the paper (`paper/draft.md`) under the RELEASE_PLAN write-up
  requirements: dedicated circularity section, toy-as-instrument framing,
  verbatim claims-not-made list, DreamCoder positioning, provenance
  paragraph, and the linearity result leading per the amended headline rule.
- Revised the paper to draft v0.2 per the paper review: new title and
  abstract centered on the statistical-reuse versus structural-abstraction
  distinction; economic formalization (Delta(r) = a r + b) moved into the
  introduction; "pre-specified sealed confirmation" wording with the git
  history as the verifiable timestamp; the adaptive-substrate section
  reframed as proof of concept with both currencies; the MDL negative
  condensed to a paragraph with grids deferred to the appendix; the
  provenance section toned to a disclosure; related work expanded around
  the intervention-on-recurrence novelty axis; restructured to
  phenomenon -> mechanism -> controls -> implications.
- Revised the paper to draft v0.3 per the second review: intercept
  reworded as an empirical sharing penalty; "ground-truth functional
  recurrence" phrasing; the pooled 180-cell regression explicitly labeled
  descriptive with per-world tests primary and a mixed-effects sensitivity
  noted for the appendix; forward transfer given an explicit FT_tau
  definition; the equal-start claim bounded to the 8-task checkpoint;
  the structural-abstraction claim scoped to the present learner and
  benchmark; Delta = f(r, A, C) added to the family-alignment section;
  the hypernetwork ordering interpretation spelled out; the envelope
  defined as L_env = min(L_Dense, L_Continuous) with fairness notes; the
  two-part code explained as literal description-length accounting; the
  truncated-lifetime result given its numbers and an appendix home; the
  provenance sentence retoned. Replaced figure 4 with the two-response-
  curves figure (lifetime advantage, recomposition advantage, operator
  recovery on one measured-recurrence axis).
- Revised the paper to draft v0.4 per the third review: the alignment
  dependence promoted into the abstract and contributions ("when the
  vocabulary fits, the value of using it is linear in recurrence"; the
  tenfold family-mismatch collapse stated up front); a caution added
  against reading meaning into the round r* = 0.50 crossing (a property
  of this experimental economy, already drifting with lifetime length);
  the recovery-onset/crossover coincidence explicitly hedged as a
  coincidence of locations pre-specified for V2 testing, not a causal
  claim; and a discussion paragraph connecting the scale-free criterion
  to scaled analogues (MoE routing, adapter libraries, model merging,
  fine-tuning transfer).
- Completed the related-work literature pass for the paper (draft v0.5).
  Two close methodological relatives found and positioned: Gerace et al.
  2022 (synthetic correlated datasets with parametric control of task
  correlation; single-transfer generalization rather than lifetime
  prequential economics) and Mittal et al. 2022 ("Is a Modular
  Architecture Enough?"; fixed modular world asking whether modules
  specialize, versus ROW varying how modular the world is). Related work
  rewritten in five threads (controlled relatedness models, controlled
  modularity studies, compositional continual learning including Mendez &
  Eaton, library learning, prequential evaluation, compositional
  generalization) with a defended novelty statement; 27-entry reference
  list added. Prose pass removed meta-commentary and reduced em-dash and
  changelog constructions.
- Revised the V2 spec per external review before resuming V2 runs: the
  research statement and internal-ceiling section rewritten around the
  linear return-on-recurrence law (amortization demoted to an early-
  lifetime mechanism finding, with the superseded expectation kept on the
  record); H9a given a frozen measurement procedure (primary attribution
  of task-step residuals by true-route position, avoiding the slot-
  identifiability failure V1's recovery result predicts; Hungarian slot
  matching demoted to descriptive with an ambiguity-counts-against rule;
  inference by sign test over per-world Spearman correlations); the Model
  8 compilation gate given a pre-registered monotone firing-rate shape
  prediction across the full rho grid; sealed-block point predictions
  added (slope in [4,000, 7,500] nats/unit, pooled root in [0.40, 0.60],
  per-world crossing mean in [0.42, 0.58], R^2 >= 0.85 and >= +0.15 over
  configured coordinates) so seeds 200-229 test parameter replication,
  not just signs; the hypernetwork-gap corollary promoted to step 002b;
  the 7b-dream falsifier protected from descoping; and the batch-size
  item updated to reflect the completed V1 ablation.
- Ran V2 step 001, the exact route posterior over the frozen world-0
  per-task-annealed discrete library (reports/v2_route_posterior/). H7 is
  strongly supported at the advantaged bound: the posterior scores
  -174,844 prequential (BMA-point), beating online Continuous by ~3,900
  nats and the online discrete learner by ~28,700, so the discrete
  deficit is entirely inference cost at this bound. Median 28.5 examples
  to posterior concentration below 0.1 nat. The posterior's MAP route
  matches the online hard route on only 25/64 tasks while outperforming,
  indicating behavioral near-equivalence among routes; averaging over
  that ambiguity is what hard commitment discards. One script defect was
  found and corrected during analysis: the original true-route agreement
  field compared learner slot indices with teacher primitive indices
  (different spaces) and is superseded by slot-space agreement fields;
  the report documents this.
- Published the repository publicly at
  https://github.com/doeixd/reusable-operator-worlds (all 47 commits of
  history, code, specs, plans, notes, reviews, reports, and paper draft;
  artifacts/ remains untracked by design — runs are deterministically
  regenerable from committed seeds and configs).
- Parked the V3 program sketch in notes/v3-sketch.txt (four workstreams:
  integrated wake/sleep learner, vocabulary-revision worlds, cross-world
  scale, and the LLM adapter-library external-validity bridge), with the
  explicit gate that the real V3 spec waits for V2's H8/H9 verdicts and
  the one-surrendered-control-per-rung rule.
- Ran V2 step 002b (hypernetwork at rho 0.9, worlds 0-2): the corollary is
  confirmed and stronger than predicted. The Continuous-over-Hypernetwork
  gap shrinks at rho 0.9 versus 1.0 in 3/3 worlds, and the hypernetwork
  outright beats Continuous at 0.9 in 2/3 worlds (-1,496 and -943; +917)
  while losing in all three at 1.0 (+2,820/+1,120/+2,102). A second
  representation-level crossover: the slotless manifold wins at partial
  recurrence; explicit slots pay only where identifiable primitives
  exist. Independent confirmation of the manifold-versus-crystallization
  account. H6 GELU runs: rho 0.75 complete (0/3, mean -3,632 versus tanh's
  -1,597); rho 0.9 worlds 0 and 2 negative (-1,142, -1,662) with world 1
  rerunning after a memory-pressure failure; rho 1.0 worlds 1-2 in
  flight to complete the shifted-not-vanished test.
- Completed V2 step 002, the GELU crossover-shift test (H6), on worlds
  0-2 at rho 0.75/0.9/1.0. Supported in direction with the crossing
  pushed to the boundary: the mismatched learner loses 0/3 at 0.75 and
  0.9 (where tanh wins 3/3 at 0.9) and reaches parity at exact reuse
  (mean +134, 1/3 wins). The return-on-recurrence slope survives at
  roughly 74% of the aligned slope while an approximately additive
  penalty of 2,035/2,881/3,359 mean nats shifts the line down; the
  gate-not-cost alternative is rejected. Spec section 10.2 updated in
  the same commit per the standing instruction: the circularity
  objection now has a quantitative answer.
- Updated the paper to draft v0.6 with the V2-era development results
  that bear on V1 claims: section 6.1 now carries the full three-rho GELU
  table and the quantitative circularity answer (approximately additive
  penalty, ~74% slope survival, parity at exact reuse), with the abstract
  and contributions reworded from "tenfold collapse" to the parity
  statement; section 6.2 adds the ordering inversion at rho 0.9 (the
  manifold beats explicit slots at partial recurrence, 2/3 worlds) as an
  independent confirmation of the statistical/structural account; section
  6.4 adds the exact-posterior result (the advantaged bound beats
  Continuous; MAP-route disagreement while outperforming; premature
  commitment, not discreteness, is the cost).
- Froze Model 8 gate thresholds before any consolidation run, per the
  spec's amendable-only-before-inspection rule: H_threshold = 0.1 nat
  (the posterior-concentration criterion whose empirical anchor is the
  28.5-example median from step 001) and kappa = 1.5 (MAP-route
  evaluation NMSE may exceed the soft mixture's by at most 50% for a
  task to compile). Plan: run Model 8 at all six rho values on worlds
  0-2 (18 lifetimes, parallel) so the endpoints decide H8a/H8b and the
  full grid evaluates the pre-registered monotone firing-rate shape,
  plus one forced-compile run at rho 0 world 0 for the
  compilation-should-hurt check. Baselines are the existing
  fingerprinted continuous artifacts at identical configurations.
- Ran the Model 8 gate-v1 grid (18 lifetimes plus a forced-compile
  control, artifacts/v2_consolidation/). The pre-registered firing-rate
  shape FAILED with inversion: rates 43/48/48/30/7/0 percent across rho
  0-1 — decreasing where non-decreasing was predicted. H8b fails as
  frozen; H8a is untestable (zero firing at exact reuse). Mechanism: the
  relative kappa test passes most easily where the soft baseline is
  worst, and at exact reuse single hard routes cannot match compensatory
  mixtures (consistent with V1's mixture-beats-forced-route finding).
  The forced-compile control confirms wrong compilation hurts (+1,208
  nats at rho 0); gated damage was small everywhere (mean deltas within
  ~650 nats, excluding one stale-baseline cell). Invoked the
  pre-registered single re-derivation: gate v2 replaces the relative
  test with the absolute sufficiency criterion (MAP-route evaluation
  NMSE <= 0.02, the constant frozen in the MDL-gating study); entropy
  threshold unchanged. Gate v1 artifacts retained; v2 grid runs under
  artifacts/v2_consolidation_gate2/.
- Ran the Model 8 gate-v2 grid (artifacts/v2_consolidation_gate2/). The
  pre-registered monotone shape now PASSES: firing rates 0/0/0/0/1/7
  percent across rho 0-1. H8b passes (the gate refuses all compilation
  below rho 0.9). H8a is weak-to-negative: only 2-8 of 64 tasks compile
  at exact reuse, mostly at the final sleep; storage gains are trivial
  and firing cells cost 0-404 nats. Mechanism, connecting to H7's
  non-identifiability finding: behavioral near-equivalence among routes
  in the overcomplete basis keeps posterior entropy above threshold even
  where compilable structure exists — the entropy criterion measures
  route identifiability, not compilability. With the single pre-
  registered re-derivation spent, Model 8 is recorded as a characterized
  negative for this gate family: relative criteria fire for the wrong
  reason (v1), entropy criteria barely fire at all (v2), and the two
  failures bracket the design space. A predictive-equivalence gate
  (BMA-versus-MAP behavioral gap) is the pre-registered candidate for
  any future iteration. Logging gap noted: per-task gate diagnostics
  (which criterion failed) were not recorded; a one-cell diagnostic
  rerun can decompose this if needed for the write-up.
- Implemented Benchmark D (V2 step 004): per-primitive rho profiles via
  src/row/mixed_world.py, with provenance kept outside WorldConfig (the
  fingerprint invariant) and a documented seed-scheme improvement — the
  epsilon draws are rho-independent, so uniform profiles reproduce
  homogeneous worlds bit-exactly (unit-tested), making the anchor
  conditions the existing homogeneous artifacts. All four validity gates
  pass on worlds 0-2 for the canonical profile (per-primitive measured
  recurrence 1.0/0.82/0.37-0.41/0.04-0.07/~0/~0 in profile order with a
  0.02 noise tolerance for the two ~zero primitives, decided at gate
  time; output scale, usage-versus-index, and scratch flatness all
  within limits). Runner mixed_lifetime.py reuses the full
  learned_lifetime protocol via world-factory injection.
- Froze the V2 confirmation plan (V2_CONFIRMATION_PLAN.md) before any
  seed in 200-229 exists, with decision rules pre-registered where
  development verdicts are pending: Component A (the six-point sweep,
  360 lifetimes) always runs and carries the four parameter-replication
  outcomes; Component B (canonical mixed-profile Benchmark D, 90
  lifetimes) runs iff development H9a rejects at p < 0.05 with majority
  negative correlations, and its envelope outcome is always reported
  with the two-part-code accounting, whose reversal would confirm the
  development finding. H7, H8, H6, 002b, and H10 are explicitly left as
  development-stage results.
- Completed V2 step 005 (Benchmark D, H9a). Allocation signature holds in
  10/10 development worlds (Spearman between per-primitive recurrence and
  residual allocation negative everywhere, -0.43 to -1.00, sign test
  p = 1.95e-3, frozen true-route-position attribution). Shared-residual
  beats the fixed-architecture envelope in 10/10 mixed worlds (mean
  +7,153 nats) and loses 0/10 under the two-part code, reproducing the
  homogeneous prediction. H9a is supported; Component B of the sealed V2
  confirmation triggers per the pre-registered decision rule.
- Completed V2 step 006 (H10, within-lifetime amortized inference). NOT
  supported at 64-task scale: the warm-started compiler loses to plain
  gradient descent 0/3 worlds at both rho endpoints (means -2,981 at
  exact reuse, -1,052 at rho 0), and dream augmentation halves but never
  erases the deficit. The protected dream falsifier passes: dream
  benefit +1,554 at rho 1 versus +512 at rho 0, three times larger where
  the library matches the world, so the compiler learns the program
  distribution but within-lifetime data is insufficient for one-shot
  amortization to beat direct optimization. Pre-registered fallback
  applied: deferred to cross-world scale.
- Recorded the V2 development synthesis (notes/v2-synthesis.txt; V2 spec
  section 9.5) before the sealed block, so the observations cannot be
  reshaped by it: inference is cheap to do well and expensive to do
  cheaply; allocation is easy while compression is hard because gradient
  descent optimizes nats and never sees bits; non-identifiability
  mandates functional instruments only; amortization is sample-starved
  rather than wrong (dream-benefit doubles as a ground-truth-free
  library-quality probe); and nothing yet wins both currencies, singling
  out cross-task promotion as V3's central mechanism. Added step 006b, a
  zero-compute mixture-versus-posterior correlation test of the
  approximate-Bayesian-inference mechanism hypothesis.
- Updated the V2 spec per the synthesis suggestions: 006b runs before the
  sealed block with a calibration check added (mixture entropy versus
  posterior entropy, distinguishing calibrated from rank-aligned
  Bayesian behavior); Benchmark E gains its pre-registered second job as
  the promotion testbed (residual clustering by family, both outcomes
  reported, scoping the draft V3 hypothesis "promotion wins both
  currencies in hierarchical worlds"); the sealed block gains a
  sequencing rule (Component A analyzed before B runs) and a durability
  rule (summaries committed per component; off-machine artifact archive
  before closure); and the synthesis is recorded as superseding
  notes/v3-sketch.txt as the V3 design constraints.
- Completed V2 step 007 with genuine 32- and 128-task lifetimes at the
  bracketing rho values (reports/v2_lifetime_length.json). The crossover
  is stationary across a 4x range of lifetime lengths: signs unanimous
  at every N (dense 3/3 at rho 0.75, continuous 3/3 at rho 0.9, for N in
  32/64/128) with magnitudes scaling roughly linearly in N. H5a's
  lifetime-length question is closed with real runs: the boundary is
  N-invariant beyond very early lifetime.
- Completed V2 step 008 (Benchmark E). Shared-residual beats the
  fixed-architecture envelope by +4,932/+5,074/+5,261 nats in 3/3
  hierarchical worlds (the fixed architectures approximately tie there).
  The pre-registered promotion post-hoc returns its early-warning
  branch: residuals do NOT cluster by family (within-family similarity
  ~0, cross-family ~0, same-primitive 0.008-0.015;
  reports/v2_hierarchical/residual-clustering.json). Diagnosis recorded:
  E's family components are primitive-indexed and task-invariant, so the
  shared basis absorbs them and residuals retain only task-idiosyncratic
  corrections — both a vessel warning (rank-2 residuals carry no shared
  structure here) and a design lesson (a true promotion testbed needs
  family structure that varies across task groups). Feeds the V3 spec.
- Ran V2 step 006b before the sealed block, as required. The
  mixture-as-approximate-posterior hypothesis is falsified: trained
  mixture weights are uncorrelated with exact route-posterior marginals
  (mean Spearman -0.03, chance-level sign rate, worlds 0-2) and
  miscalibrated by three orders of magnitude (1.4 versus 0.001 nats).
  Corrected mechanism, timestamped pre-sealed: soft mixing wins by
  exploiting the basis as a continuous function space, not by
  approximating route inference; route-committed solutions are better
  when found but gradient descent never finds them. Spec section 9.5 and
  the synthesis note updated in the same commit.
- Implemented the four ideas from the sealed-block reflection, with
  timestamps ahead of the sealed verdict: (1) the V3 constraint set in
  the spec now leads with differentiable description length (variational
  MDL / bits-back — task state as noisy codes with learned precision, so
  bits enter the gradient), including the draft hypothesis that a
  variational-coded shared-residual learner wins the two-part cell where
  every current model loses; (2) slots are demoted to an output of
  compression (manifold-first wake, per 006b/002b/recovery); (3) created
  PREDICTIONS.md, an append-only ledger of quantitative predictions with
  confidences (four founding entries); (4) implemented
  tools/check_prereg.py, which mechanically verifies frozen-file
  immutability and STATUS-cited artifact existence — and whose first run
  immediately caught a real subtlety (EXPERIMENT_PLAN.md was a living
  document during development; its invariant is stillness from d655ce0,
  before the confirmation freeze), now encoded in the manifest with
  justification.
- Incorporated reviewer-feedback-11/12/13 into the V2 spec before the
  sealed verdict: the three-regime claim named with its four independent
  instruments and the allocation/inference/compression/form research
  map; the V3 constraint set revised (bits in the gradient during wake
  with only irreversible discrete edits sleep-only; PROMOTE as the sole
  initial structural operation; discrete primitives as one endpoint of a
  basis-directions -> parameterized -> discrete -> macros ladder;
  task-group families for the promotion testbed, convergent with
  prediction P-2026-08-18-D; the LLM bridge staged measure -> factor ->
  learn); draft V3 hypotheses H11 (abstraction promotion, with the
  bits-flow smoking-gun figure) and H12 (hierarchical vocabulary)
  recorded, with hysteresis and acquisition-versus-retention staged for
  the nonstationary era; and pre-registered analyses 006c
  (functional-equivalence entropy, the candidate correct consolidation
  signal) and 006d (function-family intrinsic dimensionality across rho)
  added as non-gating development analyses with predictions stated.
- Prepended a supersession notice to notes/v3-sketch.txt enumerating the
  seven decisive changes from reviews 11-13 and the V2 verdicts
  (manifold-first wake, bits during wake, promote-only staging, discrete
  primitives as compression endpoint, task-group families, the staged
  LLM bridge, and the recorded H11/H12 drafts), pointing readers to spec
  section 9.5 as authoritative while preserving the original sketch as a
  thinking record.
- Read reviewer-feedback-14 in full (the grand-vision synthesis) and
  incorporated its deltas into the V3 constraint set: H11 gains the
  prospective third prediction (promoted abstractions must reduce future
  sample cost — compression without forward benefit is storage
  optimization, not abstraction); program synthesis and language
  revision separated as a stability principle; structural operations
  staged V3.1-V3.6 with branching tested as compression; learning
  progress formalized as decreasing conditional description length
  D(F | L_t); parsimony endogenous rather than hardcoded; and the
  five-milestone ladder recorded as the program's progress metric, with
  M1 marked achieved and V3 scoped to M2 plus M3.
- Rewrote notes/v3-sketch.txt as Revision 2 (Revision 1 preserved in git
  history): V3 scoped to milestones M2 plus M3 — abstraction birth with
  prospective value and refusal in structureless controls — with the
  manifold-first variational-coded wake, promote-only sleep under full
  accounting (rate-distortion form where practical), H11's three
  predictions, the V3.1-V3.6 operation staging (nonstationarity entering
  at FORK with hysteresis), binding design principles (functional
  instruments, synthesis/revision separation, endogenous parsimony,
  D(F | L_t) as the progress quantity), deferred workstreams with
  stricter gates, and the pre-V3 checklist keyed to the sealed block and
  the 006c/006d analyses.
- V2 sealed Component A complete (360 lifetimes on seeds 200-229, zero
  exclusions after a provenance path-normalization fix validated all
  cells). ALL FOUR pre-registered Family A outcomes pass: slope 6,194
  nats per unit recurrence (interval 4,000-7,500); pooled root 0.450
  (0.40-0.60) and per-world crossing mean 0.483 (0.42-0.58) with 30/30
  worlds crossing; R-squared 0.926 in measured coordinates with a +0.262
  margin over configured (thresholds 0.85 and +0.15); sign reversal
  30/30, p = 1.9e-9. Parameter replication achieved: the dose-response
  law's coefficients replicate across two independent sealed blocks.
  Reports committed per the durability rule.
- V2 CLOSED. Sealed Component B complete (90/90, zero failures): Family
  B passes both outcomes — allocation negative 30/30 (p = 1.9e-9) and
  envelope 30/30 (mean +7,192 nats) with the pre-registered
  two-part-code reversal 0/30 confirming the development finding. With
  Family A's four parameter-replication passes, all six pre-registered
  V2 confirmatory outcomes hold. Sealed artifacts (V1 and V2, 343 MB)
  published off-machine as GitHub release v2.0-confirmation. Spec
  section 12 records the closure, the full hypothesis ledger, and the
  Phase III gate opening with the pivot NOT taken (H9 passed, so V3
  proceeds promotion-first). SPEC_AUDIT re-audited at closure.
- Updated the paper to draft v0.7 with the V2 sealed results: abstract
  and contributions carry the parameter replication (second sealed
  block inside every frozen interval) and the read-versus-write
  asymmetry; section 4 gains the replication subsection including
  lifetime-length stationarity; new sections 5.3 (three regimes of
  representational form) and 5.4 (the pre-sealed falsification of
  mixtures-as-posterior); section 7 gains the sealed selective-sharing
  confirmation (allocation 30/30, envelope 30/30, two-part reversal
  0/30 as pre-registered) and the consolidation/compiler negatives with
  their design lesson; the discussion closes on the four-line research
  map; figure list extended with the interval-overlay, sealed
  allocation, and gate-shape figures.
- Ran the two non-gating pre-registered analyses. 006c FAILED its
  prediction informatively: route posteriors are deterministic at both
  rho endpoints given the full record and 510/512 routes are
  functionally distinct, so gate v2's under-firing is re-attributed to
  the absolute-NMSE bar (hard routes cannot behaviorally substitute for
  mixtures in a mixture-trained basis); the spec's H8 mechanism
  sentence, the paper, and the synthesis note were corrected in the
  same commit, and the correct V3 gate question becomes behavioral
  substitutability rather than posterior identification. 006d is weakly
  supported: participation ratio declines monotonically (6.5 to 5.1)
  but without regime structure — the three regimes do not manifest as a
  linear-dimensionality transition.
- Closed completion-audit open item 2: the GELU shared-residual mismatch
  control preserves the H9 allocation signature 3/3 and the envelope win
  3/3 on canonical mixed worlds (margins within ~15% of aligned; the
  mismatch penalty on shared-residual is roughly a third of what
  mismatch costs the continuous basis). Selective sharing is not
  strongly alignment-conditioned; spec, paper (v0.7), and audit updated.

# 2026-08-19 — reviewer-feedback-15 incorporated into V3 planning

- Read reviewer-feedback-15 in full (the post-closure review). Its five
  new requirements were folded into three artifacts: notes/v3-sketch.txt
  Revision 3, the V2 spec's section 9.5 constraint set (new item 7),
  and PREDICTIONS.md (entries E and F).
- Adopted: information migration (D_task down, D_shared up, D_total
  down, held-out behavior flat) as V3's pre-registered primary M2
  endpoint; promotion formalized as a change of prior (A earns its
  existence iff it reduces the KL cost of task residuals); a second
  mandatory refusal control with accidental non-predictive similarity,
  plus V(A) = V_retro + V_future - D(A) - mu*C(A) logging for every
  candidate; multi-code robustness (two-part, variational, prequential
  currencies all reported, divergence itself a result); and candidate
  H13 — whether the learner's internal promotion value crosses zero
  near the externally measured ~0.48 crossover.
- Longer-horizon review-15 items (equivalence-class routes, hidden-basis
  coordinate discovery, causal recurrence in units of saved learning,
  granularity discovery, search tax, memory hierarchy, task-boundary
  removal, compositional-closure depth gate, functional IBP) recorded as
  staged deferred workstreams in the sketch, not V3 scope.
- Citation hygiene flag recorded: feedback-15's reference list mixes
  verified arXiv IDs with unresolvable/garbled ones; every citation is
  verified independently before entering any spec or paper.
- Prereg check green (4 frozen files unchanged; all STATUS paths exist).

# 2026-08-19 — V3 spec written; sketch retired

- Converted notes/v3-sketch.txt Revision 3 into
  row_v3_experimental_spec.md (provisional draft), with V1/V2
  discipline: H11 with three mandatory predictions (migration curves as
  primary M2 endpoint, both-currencies win, prospective M3) plus the
  dual refusal requirement; H12 hierarchical vocabulary; H13
  internal-economics conditional on freezing V_hat(A) pre-run; explicit
  falsifiers for each; the task-group promotion testbed gated on
  P-2026-08-18-D clustering before any promoter runs; the
  accidental-similarity control as a build-and-validate deliverable;
  the unpromoted variational wake learner as the critical paired
  control; multi-code robustness as a success criterion; a sealed
  protocol for seeds 300-329 requiring a frozen V3_CONFIRMATION_PLAN
  (hash added to check_prereg) with parameter intervals and a one
  re-derivation rule; staged operations V3.1-V3.6 with the
  compositional-closure gate on LOOP.
- Retired the sketch with a pointer notice (Revisions 1-3 preserved in
  git history); updated CLAUDE.md quick pointers (V2 closed, V3
  active).
- Pre-run order fixed in the spec's checklist: variational wake learner
  and P-2026-08-18-A first, testbed validity gate second, control
  validation third, estimator freezes fourth, PROMOTE last.

# 2026-08-19 — V3 spec executability audit

- Audited the V3 spec for autonomous executability and closed the gaps
  that would have stalled an agent: concrete world defaults (2 hidden
  task groups of 32, rank-2 family perturbation with a single strength
  eta tunable only until the clustering gate passes, structureless =
  eta 0, accidental control = future-block directions resampled,
  8-task future block, checkpoints 8/16/32/64); variational learner
  defaults (shared-residual base, mu + log-sigma with learned
  per-tensor prior scale in a no-weight-decay group, KL-zero
  initialization, one reparameterized sample for updates and posterior
  mean for all scoring, beta grid {0.1, 0.3, 1.0} under the V2
  two-stage protocol with inherited LRs and a bounded budget); sleep
  defaults (SLEEPS 8/16/32/64, agglomerative functional clustering,
  cluster size >= 3, rank-2 candidate fit on deep copies, epsilon 0.02
  NMSE, lambda ln 2, mu 0 for V3.1, behavior-preserving re-anchoring
  with rollback-as-refusal); per-currency operational definitions of
  D_task/D_shared/D_total with references charged to D_task and the
  sign pattern required in both bit currencies; the V_future estimator
  selection rule (rank correlation against realized future-block
  savings on dev 0-2); and a new section 11 of execution notes
  (machine constraints, provenance extensions, promotion_ledger.json
  schema, tuning ledger, sealed-seed incident rule).

# 2026-08-19 — reviewer-feedback-16 pre-execution fixes applied to V3 spec

- Applied all ten of review 16's pre-execution edits. The two
  substantive repairs: (1) the accidental-similarity refusal control
  was information-theoretically impossible as designed (identical
  history distributions cannot support differential refusal); replaced
  with a drifting-family control (blocks of 16 with a fresh family
  direction each, instability observable from sequential evidence),
  while the original construction is kept as a secondary regime-change
  world whose registered prediction is SAME decision plus measured
  cost of unanticipated change; (2) fixed-width two-part bits could
  never fall without parameter removal, making H11.1 unwinnable by
  construction — promotion now includes a rank-reduction step (refit
  affected residuals at rank 1/0 within epsilon on a disjoint
  validation set) so migration is visible in literal bits.
- Also: wake substrate fixed to the frozen H9 shared-residual
  architecture (hypernetwork wake deferred to V3.1b) resolving the 3.1
  contradiction; the V3.1 abstraction family fixed explicitly at
  rank-2 U tanh(Vz+b) rather than claiming form discovery; the
  unpromoted-variational learner elevated into H11.2 as the PRIMARY
  causal comparison (frontier comparison secondary, with the
  "variational suffices, promotion inert" outcome named); disjoint
  Z_proposal/Z_validation/Z_audit probe sets made mandatory; L_mean
  and L_var = E_q[L]+KL separated (the L_mean+KL hybrid banned as a
  codelength); prior scales made per-tensor-TYPE, shared across tasks,
  charged to D_shared; the H11.1 matched-loss margin operationalized
  (delta_L default 1e-4, frozen in the confirmation plan, with
  rate-distortion reporting); H13 demoted to exploratory now, before
  promoter development, because the V_future estimator selection
  creates researcher degrees of freedom.
- Added P-2026-08-19-G (horizon experiment: promotion rate at least
  doubles in the long-horizon condition; conditional on H11) to the
  predictions ledger and staged it after H11 in the spec.
- P-2026-08-19-E is scored on the drifting-family control; the ledger
  entry itself is append-only and unedited.

# 2026-08-19 — reviewer-feedback-17 incorporated: registered diagnostics

- Read reviewer-feedback-17 in full (the post-spec prediction tree).
  Its concrete instruments were adopted into the V3 spec BEFORE any
  promotion run, as a new section 4.3 "Registered diagnostics and
  failure branches": the mandatory oracle factorization bound
  separating existence / discovery / use (with the gauge-freedom risk
  named — shared structure may rotate out of the residuals where
  PROMOTE looks); the M3 failure ladder (oracle reuse -> retrieval-only
  -> gradient initialization) naming the Phase III bottleneck; the
  coding ladder (KL -> entropy-coded -> sparse -> serialized bytes)
  separating a systems verdict from a theory failure; the
  posterior-collapse rule (report the full D(beta)-vs-L(beta)
  rate-distortion curves, promoted-curve dominance as supplementary
  evidence); the parameterized-family contingency as the single
  registered V3.1 deviation; and interpretation guards (behaviorally
  equivalent factorization expected, not weight recovery; H12 scores
  shared degrees of freedom before symbolic strata).
- The promotion criterion now includes a log2(M) candidate-selection
  charge so sleep cannot perform hidden search for free; the stale
  "form is never imposed" sentence was reconciled with the review-16
  fixed-form decision.
- Added P-2026-08-19-H (lifetime-length sweep N 16-256; monotone
  correct-promotion rate and an interpolable amortization threshold
  N*; conditional on H11) and staged the (r, N) phase diagram next to
  the horizon experiment in section 7.
- No architecture was added; feedback-17's V4-era material (search
  debt, copy-on-write forking, hierarchy, macro/loop/branch economics)
  is already staged and stays staged.

# 2026-08-19 — V4 sketch written

- Wrote notes/v4-sketch.txt Revision 1. V4 = vocabulary MAINTENANCE:
  can the language reorganize rationally under nonstationarity, shared
  mutable state, and search debt. Draft hypotheses H14 (rational
  MERGE/FORK/DELETE with matched refusal controls), H15 (hysteresis
  flagship: r_create > r_delete traces a loop whose area is the
  acquisition-vs-retention cost difference; falsifier = no loop), H16
  (copy-on-write beats update-in-place and never-update — FORK as a
  safety mechanism), H17 (priced retrieval gives an interior optimal
  library size). Introduces the minimal prospective model of future
  reuse as a measured component, per review 17.
- Because V3 has not run, the sketch's centerpiece is a branch table
  keyed to V3's possible verdicts (full pass; M3 access failure pulls
  retrieval work forward; KL-only currency verdict; parameterized-
  family promotion redefines lifecycle over families; existence
  failure blocks V4 in favor of coordinate discovery; refusal failure
  blocks V4 outright). Sealed seeds 400-429 reserved; sketch retired
  when the V4 spec is written after the V3 sealed block.

# 2026-08-19 — V3 checklist item 1: variational wake learner implemented

- Implemented `VariationalSharedResidualLearner` (V3 spec 3.1): the frozen
  H9 shared-residual architecture with every task-specific scalar (routes
  and rank-2 residuals alike) coded as a Gaussian posterior against a
  factorized prior whose scale is shared per task-state tensor TYPE.
  Sampling uses a dedicated generator (reproducible, never touches global
  RNG) and fires only inside gradient-enabled training forwards, so every
  scoring path keeps the posterior mean and score-before-update is intact.
  Retained task state is the mean alone (scales are training state), so the
  two-part comparison against shared-residual stays scalar-for-scalar
  matched. Wired as model kind "variational" through config, harness,
  mixed_lifetime, and configs/v1.yaml; 13 unit tests; full suite 91 green.
- Three implementation findings, each a real design correction:
  1. KL SCALE. Adding a KL summed in nats over a task's ~222 coordinates to
     an MSE averaged over batch and dimensions is dimensionally wrong (the
     first version was ~4 orders of magnitude too strong and collapsed every
     posterior). The wake loss now charges beta * 2*sigma^2/(N*d) * KL,
     which is exactly L_preq + beta*KL expressed in the optimizer's units
     and makes beta = 1 the literal MDL point.
  2. PRIOR LEARNING. A gradient-learned shared prior runs away: whenever
     posteriors are concentrated the gradient always says "shrink", Adam's
     normalized step makes the move size independent of the tiny gradient,
     and the collapsing prior then annihilates the task state it describes
     (measured 1.0 -> 0.0034 in one lifetime). Replaced with the closed-form
     empirical-Bayes M step, s^2 = mean(mu^2 + sigma^2) per tensor type.
  3. PRIOR WARMUP. The closed form has a stable degenerate fixed point: at
     initialization mu ~ 0 and sigma = 1e-3 make s ~ 1e-3 on the first
     update, whose mu/s^2 gradient pins every code at zero for the whole
     lifetime. The prior is therefore estimated from COMPLETED tasks only
     and not at all before a population exists (warmup 8 tasks); the
     posterior starts precise and the prior wide, so the code starts at high
     precision and relaxes where precision proves unnecessary.
  Also separated `coordinate_mean_information` (mu^2/2s^2 — what a sparse
  code recovers by dropping a coordinate, hence the pruning criterion) from
  `coordinate_kl` (the full code length including the precision term).
- Reduced-world smoke (16 examples/task) shows the learner declining to
  encode task state at all. This is CORRECT MDL behavior, not a bug: at that
  budget task state buys ~5 nats/task of fit against ~500 nats of code. With
  beta = 0 the learner reproduces the frozen baseline (max route coefficient
  0.213 vs 0.215, residual/parent 0.144 vs 0.116), confirming the plumbing.
  The consequence is methodological and is recorded as such: the variational
  learner can only be evaluated at the full 128-example budget, because a
  data-poor world cannot distinguish correct refusal-to-encode from
  collapse. It also puts the program's amortization economics INSIDE the
  learner, one level below where V3 expected to find them.
- Launched P-2026-08-18-A at full scale: variational on canonical mixed
  development worlds 0-2 against the existing frozen baselines
  (tools/run_v3_variational.py, detached, resumable, 3 jobs).

# 2026-08-19 — three pre-PROMOTE audits (PI-requested), one self-correction

- TOY AUDIT (row.experiments.variational_toy): does a Gaussian task code
  make "no information" mean "no perturbation"? Under the variational
  currency, no — it charges an unused task 22.0 KL bits against 27.8 for
  a task that genuinely needs adaptation, because q = p is the
  zero-information state but not the zero-perturbation state, so the
  optimizer buys quiet with precision. But the LITERAL-code control
  partly reverses the story: charged under a common sparse code (presence
  bitmap plus 8 bits per active scalar) both codes reach 4.00 bits on
  unused tasks, since a post-hoc pruner supplies the null state the
  parameterization lacks, and on used tasks the Gaussian is cheaper at
  worse distortion. The defect is therefore in the TRAINING SIGNAL, not
  the achievable storage: KL is a divergence, not a code length, and the
  Gaussian learner is charged for precision on coordinates a sparse code
  stores for free. Recorded that way rather than as "gating wins".
- KL-CHARGE AUDIT, and a change I made and then reverted. I first read
  the replay-induced tilt in integrated KL coefficient (mean x1.00 of
  intended, range 0.50-3.38, early tasks 3.03x late) as re-billing a
  one-time description cost, and changed the objective to charge only the
  current task. That was wrong and is reverted. The decisive quantity is
  KL pressure RELATIVE to likelihood pressure, and under the existing
  batch-mean policy it is exactly 1.000 for every task (range
  1.000-1.000, tilt 1.000x) because MSE is a mean over batch elements
  while KL is a mean over unique tasks present. Replay buys early tasks
  more optimization steps toward the same objective, not a higher price.
  The rejected policy would have made the ratio position-dependent
  (0.296-2.000, tilt 0.35x) and let replayed tasks accrete information
  free of charge. The batch running under it was killed and relaunched.
  Lesson recorded: audit the ratio before changing an objective.
- H9 RATE-DISTORTION BOUND (row.experiments.audit_h9_rate_distortion):
  is V2's ~130k-bit retention information content or storage format? On
  world 0, against 132,400 dense bits the best post-hoc code inside the
  1e-4 margin retains 117,966 — an 11% saving. Larger savings need real
  distortion (int4: 74,680 bits at +0.00024 NMSE; discarding all
  residuals and keeping routes: 29,440 bits, essentially Continuous's
  29,248, at +0.0089). The residuals carry genuine information, so no
  serialization choice rescues the two-part cell. This is the
  quantitative case for PROMOTE over better coding.
- Pre-registered P-2026-08-19-I (gated/rank innovation code, with the
  reuse-tracks-recurrence secondary) and P-2026-08-19-J
  (acquisition-then-freeze vs mutable task state), both BEFORE any gated
  learner exists and before any PROMOTE run.
- Built the V3 promotion testbed generator (src/row/task_group_world.py):
  hidden task-group families, held-out future block, drifting-family and
  regime-change variants. At eta = 0 it reproduces the canonical mixed
  world BIT-EXACTLY, so the structureless control is the same generator
  rather than one whose equivalence must be argued. Group separation is
  monotone in eta (0.002 / 0.019 / 0.047 / 0.166 / 0.442 / 1.137 at eta
  0 / 0.2 / 0.3 / 0.5 / 0.7 / 0.9) once the instrument centers out the
  task-invariant component that spectral renormalization leaves behind
  (without centering, a spurious 0.33 correlation floor).

# 2026-08-19 — P-2026-08-18-A scored: FALSIFIED (one clause passes)

- Stage-one tuning over the registered beta grid {0.1, 0.3, 1.0} plus an
  exploratory 3.0, selected on mean lifetime loss per the V2 protocol,
  chose beta = 0.3. The grid is monotone and never crosses: KL bits per
  task 1054 / 832 / 577 / 114 against mean losses -159,436 / -159,559 /
  -156,623 / -148,138 and mean maximum route coefficients 0.348 / 0.300 /
  0.152 / 0.125. Every bit saved costs route structure and predictive
  loss together.
- At the selected setting the prediction splits: the two-part win over
  both fixed architectures FAILS 0/3 (Continuous takes the cell by
  52-55k nats, retaining 29,248 bits against 117,534-118,278), while the
  envelope-retention clause PASSES at 0.80 / 0.87 / 0.83. The conjunction
  fails, so P-A is falsified; outcome appended to the ledger with its
  mechanism, and a STATUS block added to the V3 spec.
- The mechanism is measured, not inferred, and rests on three
  independent results: the bits are structural rather than precision
  (the untouched H9 baseline needs 89% of dense bits at the same margin
  in 3/3 worlds, so variational coding's 11% recovery bought nothing
  structural); a Gaussian code mischarges the identity state (22 bits to
  say "this task needs nothing"); and the per-tensor-type prior collapses
  winner-take-all, taking the ROUTE mechanism to uniform in 2/3 worlds at
  beta = 1 and 3/3 at beta = 3.
- Reading recorded in both spec and ledger: continuous information
  penalties alone cannot create structural compression, because
  shrinking a KL does not produce a shared object plus a reference to
  it. Only PROMOTE changes the representation class. This sharpens H11
  rather than weakening it, and it reassigns the wake phase's job to
  allocating information and exposing a null state that promotion can
  act on.
- Launched the remaining PI-requested audit: beta = 0 equivalence on
  full 128-example worlds (not just the smoke), which must reproduce the
  frozen shared-residual baseline up to the two documented differences
  (no L1 surrogate, and 1e-3 sampling noise).

# 2026-08-19 — beta = 0 equivalence audit passes at full scale

- The PI-requested audit that the variational learner reduces to the
  frozen H9 baseline when its information charge is switched off, run on
  full 128-example canonical mixed worlds rather than the reduced smoke.
  PASS 3/3: lifetime loss within +0.75% to +1.09% (1,204-1,734 nats of
  ~160,000), route specialization matching (0.369/0.374, 0.349/0.367,
  0.387/0.385), and comparable 32-shot novel transfer. The residual/parent
  ratio runs higher (0.283 against 0.217) exactly as the two documented
  differences predict: the variational learner disables the L1 residual
  surrogate and injects 1e-3 sampling noise. reports/v3_beta0_equivalence.json

# 2026-08-19 — V3.1's testbed fails its gate, for an instructive reason

- P-2026-08-18-D FALSIFIED as written, and the falsification redirects the
  V3 core experiment. On trained shared-residual learners over eta in
  {0, 0.5, 0.7, 0.9} and worlds 0-2, task-step RESIDUAL separation by
  hidden family is +0.0007 (structureless control), +0.0055, +0.0127,
  +0.0173 — monotone and 3/3 positive, but two orders short of the
  registered factor of 3. The operational check kills it: two-means
  partition recovery is 0.672 at eta 0.9 against 0.641 in the
  STRUCTURELESS CONTROL, so that figure is selection bias from
  best-of-restarts plus label-permutation freedom, not signal. A promoter
  could not find these families.
- Where the structure went: the ROUTES. Same instrument on route codes
  gives -0.0152 / +0.0043 / +0.2006 at eta 0 / 0.5 / 0.9, twelve times
  the residual figure. Per-group mean route distributions at eta 0.9
  differ by total variation 0.226 against 0.026 in the control, with
  group 0 placing 0.032 on slot 2 and 0.154 on slot 3 while group 1
  places 0.190 and 0.048. The basis has 8 slots for 6 teacher primitives,
  so it had exactly two spare, and it absorbed the two hidden families as
  dedicated slots.
- Reading: gradient descent performed the promotion operation during
  ordinary wake learning, for the description-length reason V3 predicts —
  a route reference costs 192 bits per task against 17,712 for a
  residual — with no sleep phase. H11's premise that task-specific
  adaptations CONTAIN the recurrent structure is therefore false in this
  testbed, which cannot support the V3.1 core experiment as specified.
  The sharpened question: when is recurring structure NOT addressable by
  the existing reference vocabulary, since that is the only regime where
  an explicit promotion operator can earn its place?
- Capacity is the lever, and it works directionally. With 8 hidden
  families against 2 spare slots, residual separation rises to +0.0374
  (2.2x, consistent 3/3 at 0.0340 / 0.0391 / 0.0391) while route
  separation falls to +0.1147 and becomes erratic across worlds. Testbed
  search is being logged in full per the spec's tuning rule; the next
  configuration adds deviation room (uniform rho 0.5, where the canonical
  profile's high-recurrence primitives leave the family component almost
  no room to act).

# 2026-08-19 — saturated-library testbed: the gate clears

- Interpretive correction applied first, per PI review: the earlier claim
  that the learner absorbed families into spare slots "for the
  description-length reason V3 predicts" overstates the evidence. Those
  slots were PREALLOCATED and already paid for; nothing charged the
  optimizer 192 bits against 17,712. The supported claim is that unused
  shared capacity gets used for recurrent structure. Whether a learner
  CREATES capacity when creation costs something is untouched and is
  exactly what V3 must still test. Corrected in the spec and appended as
  a dated correction to the ledger outcome (never a rewrite).
- The implicit-promotion finding is nonetheless CAUSAL, not merely
  correlational (row.experiments.audit_implicit_promotion, three
  interventions): substituting a task's route with the wrong family's
  mean costs +0.00344 more NMSE than its own family's mean, positive
  3/3 worlds; single-slot ablation is sharply family-differential
  (world 0 slot 3 damages group 0 by +0.00419 against group 1's
  +0.00003; world 1 slot 5 reverses the asymmetry). Shared functional
  objects addressed by family-specific references, built during wake.
- Capacity sweep (world 0, eta 0.9) confirms the PI's prediction that
  capacity pressure alone does not relocate the structure. Routes and
  residuals at (K, F): (6,2) +0.2285/+0.0196, (8,2) +0.1407/+0.0196,
  (6,8) +0.0381/+0.0338, (8,8) +0.0638/+0.0340. Even with no nominal
  spare slot the basis reorganizes and serves families through routes.
  Raising the family signal made it worse: uniform rho 0.5 drove routes
  to +0.4333 while residuals reached only +0.0459.
- SATURATE-THEN-INTRODUCE is what works. With K = 6 saturated on the base
  primitives for the first block, families appearing only afterward, and
  the basis frozen at onset, the channel inverts: residual separation
  +0.0523 against routes +0.0121, and the matched structureless control
  sits at -0.0025. Partition recovery is 0.823 against the control's
  0.677, a 0.146 gap where the failed original testbed managed 0.031. An
  earlier onset with more deviation room does marginally better (+0.0544,
  recovery 0.854). Every configuration tried is logged here and in the
  artifact tree per the spec's tuning rule.
- Implemented partial basis freezing so the library can GROW at onset,
  which is the oracle for promotion: freeze the saturated slots and let
  (K+2) - K new ones learn. Running now to establish the PI's gate,
  J*_{K+1} < J*_K, before any promoter is built.
