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

# 2026-08-19 — PROMOTE implemented; the testbed fails a THIRD time, deeper

- Froze a candidate testbed (K=6 saturated, canonical profile, 2 families
  at eta 0.9, onset and basis-freeze at task 16) on three gates: residual
  separation +0.0459 against the control's -0.0025, partition recovery
  0.861 against 0.677, and a rank-2 oracle worth +50,439 nats 3/3.
- The first oracle formulation FAILED (0/3) because it grew the library by
  whole rank-8 operators (265 scalars each) rather than by the rank-2
  abstraction V3.1 actually specifies (66 scalars), and because it added
  capacity without retiring the residuals the capacity replaces. Both
  halves of promotion have to be priced together.
- Implemented PROMOTE as literal capacity birth: functional clustering of
  task innovations, a rank-2 abstraction fitted per cluster, acceptance by
  behavioral substitutability on a probe set DISJOINT from the one used to
  propose and fit, retirement of the members' private residuals, and a
  ledger entry for every candidate whether it fires or refuses. Six unit
  tests including refusal and the three-sign migration signature.
- First full run: 26 abstractions for 2 true families, and refusal failed.
  Not a bug — at lambda = ln 2 a private rank-2 residual costs ~1,098 nats
  and buys ~215, so any behavior-preserving grouping pays and a
  retrospective-only promoter is a vector quantizer. This is exactly what
  P-2026-08-19-E predicted.
- Added the prospective term, estimated from observed history alone (fit
  the candidate on all but the most recent members, require it to fit
  those held-out members within epsilon). It did NOT discriminate either:
  structured promotes 7.0 abstractions against the control's 6.7.
- The reason is deeper than the gate could see, and it invalidates the
  frozen testbed. Substituting family means scores 0.02618 NMSE against
  0.02642 for ONE GLOBAL mean and 0.02661 for ZEROING every residual —
  differences of ~0.0004. The oracle's +50,439 nats was almost entirely
  "residuals are not worth their bits", not "families are shared". The
  gate measured that family structure is PRESENT in residuals; what
  promotion needs is for it to be BEHAVIORALLY LOAD-BEARING, which is a
  different and stricter property. Recorded as a gate-design lesson: a
  validity gate for promotion must include a discard control (zeroing the
  channel) and a no-structure control (one global abstraction), or it
  will pass worlds where promotion degenerates into deletion.
- Root cause across all three failures: the family was a PERTURBATION of
  existing primitives, which a 6-slot basis approximates well, leaving the
  residual holding a small correction. Implemented the redesign the
  diagnosis implies: post-onset tasks call a genuinely NEW primitive drawn
  independently of the base library, so a frozen library cannot express it,
  the residual becomes load-bearing, and family members need the same new
  computation. Running the gate on it now.

# 2026-08-19 — the promotion testbed finally works, via two corrections

- NEW-PRIMITIVE FAMILIES. Post-onset tasks now call a genuinely new
  primitive drawn independently of the base library, rather than a
  perturbation of an existing one. Immediate effect: partition recovery
  from residuals goes to 1.000 in 3/3 worlds and residuals become
  load-bearing (keeping them scores 0.0127 NMSE against 0.0224 with them
  zeroed, so they halve the error).
- GAUGE FREEDOM, exactly as reviewer-feedback-17 warned. Even with
  perfectly identified families, averaging members' rank-2 PARAMETERS
  captured only 11.9% of the residuals' behavioral value, because two
  tasks can compute the same innovation under different rotations and
  scalings of (U, V) and the mean of gauge-inequivalent parameters is not
  the mean of the functions. Fitting the abstraction FUNCTIONALLY by
  gradient descent — which the spec specified and the first implementation
  quietly replaced with a parameter mean — recovers 53.4%.
- The discrimination the earlier testbeds never had: two family
  abstractions capture 53.4% of residual value against 24.0% for one
  global abstraction, a 29.4-point advantage consistent 3/3 (48.7/21.7,
  60.2/24.7, 51.5/25.5). Promotion in this world is about family
  structure, not about discarding a worthless channel.
- An intermediate misstep worth keeping: the new primitive's position was
  first varied per task "to avoid a positional artifact", which left
  family members' residuals at different steps so their shared abstraction
  averaged incompatible functions (9% capture). Fixing the position costs
  nothing, since position carries no family information.
- Functional fitting is now wired into PROMOTE and the six unit tests
  still pass. Re-running the operator on the structured testbed and the
  structureless control, which together test P-2026-08-19-E.

# 2026-08-19 — leave-one-out substitutability: the gate the earlier proxies were reaching for

- Ran the PI-insisted leave-one-out test: the abstraction substituted into
  task tau is fitted WITHOUT tau's own residual, so what is measured is
  cross-task reuse rather than joint compression of a fixed collection.
  World 0: private 0.01274, family 0.01789, global 0.02047, zero 0.02237 —
  family capture 46.5% against global 19.8%, a +26.8 point advantage. The
  joint fit gave 53.4%, so leave-one-out costs only ~7 points: the shared
  function genuinely transfers to a member it was never fitted on.
- Two probe distributions agree closely (common-domain 46.5%, on-trajectory
  46.9%), so the abstraction is equivalent both as an operator and as a
  substitution on the states the task actually visits.
- The gate is now stated operationally and replaces the original
  "within-family similarity exceeds cross-family by a factor of 3", which
  was an indirect proxy that certified two unusable worlds:
    load-bearing?    removing the private computation must cost something
    compressible?    one fitted shared function recovers much of it
    family-specific? it must beat a single global abstraction
  The four-way audit (private / family / global / zero) becomes the
  standard promotion diagnostic, because only the spread between those
  four separates abstraction from generic compression from deletion.
- PROMOTE with functional fitting separated structured from control
  (8.7 abstractions against 6.0, refusals 0-1 against 2-3) but the control
  still promoted. Replaced the prospective test with the leave-one-out
  estimator the PI suggested: a candidate must beat the single GLOBAL
  abstraction on members it was not fitted on. A candidate that only
  compresses its own members is a quantization artifact; one that predicts
  a task it never saw is reusable structure.

# 2026-08-19 — H11.1 observed: bits migrate, 3/3 worlds

- Leave-one-out replicates 3/3: family capture 46.5 / 58.0 / 49.3 per cent
  against a single global abstraction's 19.8 / 22.6 / 23.7, mean advantage
  +29.3 points, with the on-trajectory probe agreeing with the
  common-domain probe throughout. The abstraction transfers to members it
  was never fitted on, so this is cross-task reuse rather than joint
  compression, and the testbed is validated.
- The V_transfer gate (a candidate must beat the single global abstraction
  on members it was not fitted on) inverts the promote/refuse balance:
  structured promotes 6.7 and refuses 3.0, the control promotes 3.3 and
  refuses 6.0. A 2:1 discrimination where the retrospective-only criterion
  had none, though the control still promotes rather than refusing
  outright.
- H11.1 MIGRATION OBSERVED, 3/3 worlds, with shared capacity that was NOT
  preallocated: D_task 110,592 -> 15,732 / 9,472 / 14,151 (falls ~87%),
  D_shared 12,720 -> 20,640 / 26,976 / 22,224 (rises), D_total 123,312 ->
  36,372 / 36,448 / 36,375 (falls 70%). Mean two-part gain +60,244 nats,
  positive 3/3.
- One gap the run exposed immediately: prequential loss was IDENTICAL
  between promoted and unpromoted (-158,347 both), because a new task
  began with a fresh private residual and never consulted the library.
  Promotion was buying description length with no forward benefit, which
  is storage optimization rather than abstraction and would have failed
  H11.3 by construction. Implemented library reuse: at a fixed point in a
  task's own examples it may select any existing abstraction or none, from
  its own observed data only, with "none" always available so reuse is
  never forced. Rerunning both conditions with reuse enabled.

# 2026-08-19 — H11.1 AND H11.2: the first model to win both currencies

- With library reuse enabled, promotion improves BOTH the prediction and
  the code, 3/3 worlds:
    world 0: L -158,347 -> -159,240 (+893 nats), J gain +57,869, 30/40 tasks reuse
    world 1: L -159,830 -> -161,122 (+1,292),    J gain +64,840, 33/40
    world 2: L -158,648 -> -159,409 (+761),      J gain +48,969, 31/40
  Mean loss gain +982 nats and mean two-part gain +57,226, both positive
  3/3. Every V2 model lost the two-part cell; this is the first learner in
  the program to win prediction and description length at the same time,
  and it does so by creating shared capacity that was not preallocated.
- The forward half is what turned a storage result into an abstraction
  result. Before reuse, promoted and unpromoted had IDENTICAL prequential
  loss: bits fell, nothing learned faster. Allowing a new task to select an
  existing abstraction from its own early examples (or decline) is what
  makes the library pay in prediction.
- Refusal now discriminates on three measures at once: structured promotes
  5.0 and refuses 4.3 with 31.3 of 40 tasks reusing the library, while the
  structureless control promotes 3.3, refuses 6.3, and has only 15.0 tasks
  reuse. The control still promotes rather than refusing outright, which is
  reported as a limitation rather than tuned away.

# 2026-08-19 — H11.3 passes: promoted abstractions cut future learning cost

- Held-out future block (8 tasks per world drawn from the same hidden
  families, never in any lifetime; deep-copied models, library frozen,
  matched budget and data): 32-shot NMSE 0.01678 -> 0.01425, 0.02261 ->
  0.01856, 0.02556 -> 0.02384, mean improvement +0.00277 positive 3/3;
  adaptation nats improve by +11 / +21 / +10, positive 3/3. Every future
  task in every world chose to reuse the library (8/8, 8/8, 8/8) with
  "none" always available.
- ALL THREE MANDATORY H11 PREDICTIONS NOW HOLD on development worlds 0-2:
  H11.1 migration (task bits -87%, shared bits up, total -70%, 3/3);
  H11.2 both currencies (loss +982 nats and two-part +57,226, 3/3);
  H11.3 prospective value (3/3 on both measures). Refusal discriminates
  but does not fully refuse: structured promotes 5.0 with 31.3 of 40 tasks
  reusing against the control's 3.3 and 15.0.
- Extending to development worlds 3-9 before setting any sealed interval,
  so the confirmation plan is frozen on ten worlds rather than three.

# 2026-08-19 — V3 development complete: H11 holds 10/10 on every prediction

- Ten-world development set on the frozen testbed, promoted against the
  identical unpromoted learner:
    loss gain          mean +1,350 nats,  range [+761, +2,513],  10/10
    two-part gain      mean +55,697 nats, range [+46,127, +64,840], 10/10
    D_total reduction  mean 63.6%,        range [52.6%, 74.3%],  10/10
    library size       mean 5.3,          range [3, 7]
    tasks reusing      mean 31.8 of 40,   range [22, 39]
- H11.3 future block, ten worlds: 32-shot NMSE improvement mean +0.00310,
  range [+0.00172, +0.00438], 10/10 positive; adaptation nats improve by a
  mean +16, 10/10 positive; 8 of 8 future tasks reuse the library in nine
  worlds and 7 of 8 in the tenth.
- So all three mandatory H11 predictions hold unanimously on development:
  migration (H11.1), both currencies (H11.2), prospective value (H11.3).
  The promoted learner is the first in the program to win prediction and
  description length at the same time, and it does so with shared capacity
  it created rather than capacity handed to it.

# 2026-08-19 — V3 CLOSED: sealed block passes 5/5

- Sealed block on seeds 300-329 executed the configuration frozen at
  bcc8319 before those seeds existed. 90 lifetimes, zero failures.
    O1 two-part gain   PASS  +55,292 nats, positive 30/30 (dev +55,697)
    O2 loss gain       PASS  +1,174 nats,  positive 30/30 (dev +1,350)
    O3 migration       PASS  three-sign 30/30, -63.3%     (dev -63.6%)
    O4 prospective     PASS  32-shot +0.00311, 30/30      (dev +0.00310)
    O5 refusal         PASS  reuse ratio 1.80x, floor 1.4 (dev 2.48x)
  The parameters replicate, not merely the signs: two-part gain within
  0.7% of development and total-bits reduction within 0.3 points, on
  thirty unseen worlds against intervals registered in advance.
- H11 is confirmed in all three mandatory parts. An abstraction is born
  (M2) and it makes related future tasks cheaper to learn (M3), which is
  what V3 was defined to be. It is also the first learner in the program
  to win prediction and description length at the same time, in the
  literal two-part cell where every V2 model lost, using capacity it
  created rather than capacity it was given.
- Limits recorded in spec section 12 rather than left implicit: the world
  is deliberately constructed and four earlier testbeds failed; refusal
  is graded and its absolute form is falsified; the learned abstraction
  is not the teacher's primitive but the learner's own compensation; and
  promotion's compute is logged, not charged, leaving the search-cost
  economics to V4.
- The methodological result is the substitutability ladder: parameter
  similarity < clustering < latent identity < functional similarity <
  cross-task substitutability < prospective value, with every rung below
  the top certifying a world where promotion was really routing,
  quantization, or deletion. The four-way leave-one-out gate (private /
  family / global / zero) is the instrument that separated them.
- V4 sketch's branch table resolves to its first case, so V4 proceeds as
  written; sealed seeds 400-429 remain untouched.

# 2026-08-19 — learnings, paper, and lab log updated for the V3 closure

- notes/learnings.txt gains the sealed V3 section: the five outcomes with
  their development counterparts, the four failed testbeds and what each
  one eliminated, the substitutability ladder, the two-halves point (bits
  migrated but nothing learned faster until library reuse existed), the
  graded-refusal falsification, and the scope limits.
- paper/draft.md to v0.8: new section 7.5 "Promotion: creating shared
  computation (third sealed block)" placed after the resource-economics
  section and before related work, covering the constructed world and why
  three earlier designs failed, the four-way leave-one-out validity gate,
  the operator, the 30/30 results, and an explicit "what promotion does
  not do" paragraph carrying the falsified absolute-refusal criterion.
  Abstract extended with the third block. Section 9 gains a promotion
  limitation covering world construction and uncharged search cost.
- The header now distinguishes three sealed blocks by seed range so a
  reader cannot conflate them.

# 2026-08-19 — plan for the V4 spec

- Wrote notes/v4-spec-plan.md: a blueprint for the V4 specification,
  incorporating notes/v4-sketch.txt Revision 1, reviewer-feedback-26 (the
  V4 review) and -27 (the overall assessment), the V3 spec's closure
  section, the V3 confirmation plan as template, the two standing
  post-H11 predictions, and the sealed-V3 learnings, with each planned
  section mapped to its source.
- Seven decisions fixed so the spec does not re-litigate them. The
  substantive one reverses the sketch: the sketch bundled merge, fork and
  delete into a single H14, and both reviews argue for RETAIN/DELETE
  alone as V4.1 because bundled operations are undiagnosable and DELETE
  attacks V3's measured weakness directly (5.3 abstractions for 2
  families on development, 6.2 on sealed worlds). Also fixed: library
  size is a diagnostic and never an outcome; edits are justified by
  substitution and never identity; migration costs are logged from V4.1
  and charged from V4.2; V4.1 stays in the validated V3 world.
- Six prerequisites ordered before the spec is frozen, including the V3
  matched-compute audit that reviewer-feedback-27 flagged as load-bearing
  and that was not run before sealing, and an offline clairvoyant oracle
  as the V4 validity gate (lifecycle regret J_online - J_oracle), which
  is the DELETE analogue of V3's four-way substitutability gate. If that
  gate fails the world is redesigned BEFORE the spec is written, which is
  the lesson four V3 testbed failures paid for.
- The most important single addition from the reviews is the hysteresis
  lag control: any slowly adapting system shows an apparent create/delete
  loop from lag alone, so the sweep-rate series (does the gap survive as
  v -> 0) is preregistered as the falsifier rather than left as a design
  note.

# 2026-08-19 — V4 spec drafted; P1 audits answered, one narrows a V3 claim

- P1 audits, ten development worlds (reports/v3_compute_audits.json):
    frozen comparator        loss +1,350 (10/10), two-part +55,697 (10/10)
    2x-updates comparator    loss +3,015 (10/10), two-part +57,362 (10/10)
    UNFROZEN-basis comparator loss -5,766 ( 0/10), two-part +48,581 (10/10)
  Compute is not the explanation: doubling the comparator's gradient steps
  makes it worse, so promotion's advantage widens. But an unfrozen basis
  predicts better in aggregate while losing the two-part objective badly.
- Re-evaluated against the FINAL model, the unfrozen basis is worse on the
  sixteen pre-onset tasks (+0.00415 NMSE, 10/10) and better on the
  forty-eight post-onset ones (-0.00398, 0/10). It buys prediction by
  forgetting. Promotion needs no such trade because abstractions are
  additive. V3 spec gains section 13 recording that its prediction claim
  holds against a frozen substrate specifically.
- Instrument lesson: final_nmse is written at task completion and cannot
  see later drift in shared parameters. The first interference pass used
  it and reported +0.00000 in 0/10 worlds, a pure artifact. All V4
  interference measurement re-evaluates against the final model.
- Wrote row_v4_experimental_spec.md against notes/v4-spec-plan.md: H14
  survival (primary), H15 deduplication, H16 safe shared update, H17
  hysteresis with the sweep-rate falsifier, H18 priced retrieval
  (exploratory); the dormancy pair and per-operation refusal controls; the
  real-options retention rule with its evidence window frozen in advance;
  a deterministic greedy edit policy; survival table, lifecycle regret,
  migration ledger, semantic regression suite, decision dataset.
- The audits are folded into the spec as adopted consequences rather than
  commentary: the unfrozen basis joins the standing comparator set,
  old-task interference becomes a standing endpoint at every rung, and
  H16's falsifier is pre-answered in the affirmative so V4.3 now tests
  whether copy-on-write recovers the unfrozen basis's prediction advantage
  WITHOUT its forgetting.

# 2026-08-19 — V4.1 validity gate FAILS, and finds a flaw in the charter

- Built the timed lifecycle oracle (B1) and ran the V4.1 world-validity
  gate. FAILED: 0 of 5 abstractions are worth deleting, oracle gain 0
  nats, at every kappa in {0, 1e-4, 1e-3, 1e-2}.
- The reason is structural. Every abstraction carries 3-17 dependents,
  and deleting one restores n x 198 scalars of private residual against
  saving 198. Promotion fires only when it saves bits, so DELETE is
  exactly its inverse and can only pay when the promotion should not have
  happened — which V3's promotion criterion already prevents.
- That contradicts V4's own charter. "Invent hypotheses cheaply, then make
  persistence expensive" requires cheap birth, but V3's PROMOTE is
  conservative: three or more members, behavioral substitutability within
  epsilon, a bits improvement, AND prospective generalization. V4.1 as
  specified paired a conservative birth rule with a deletion rule that had
  nothing to delete. The gate caught the inconsistency before any operator
  was tuned, which is exactly what it is for.
- FIRST RUN OF THE GATE WAS A FALSE POSITIVE, recorded because the failure
  mode looks like success: it reported 3.7 of 5 deletions paying and 4,026
  nats of gain, with every abstraction showing ZERO dependents. The cause
  is that task_reference and retired are plain Python containers, absent
  from state_dict, so a reloaded V3 model reports an empty reference table
  and the audit silently analysed an unused library. The oracle now
  REFUSES to run without a persisted reference table, and
  LifecycleLibraryLearner writes one (sync_lineage, task_reference and
  retired_task_ids in the artifact). This is the gap V4's lineage existed
  to fill, arriving earlier than expected.

# V4.1 gate: FAILS under a corrected substitution tolerance (2026-08-19)

Built the exact behavioral-cover oracle (subset search over the whole
library, at most 2^7 = 128 evaluations, so a solution rather than a
bound) plus the three-policy causal control (functional / usage /
random retirement at matched counts).

The control immediately went degenerate: all three policies stranded 0
dependents at identical cost. That is the signature of a substitution
relation carrying no information, and it was. The tolerance normalized
deviation against TOTAL OUTPUT VARIANCE while an abstraction contributes
about 0.2% of it, so removing an abstraction outright scored a SMALLER
deviation (0.0016-0.0024) than replacing it with another
(0.0022-0.0039) — the null edit passed.

Re-normalized against the abstraction's own contribution, the result
inverts: substitution costs 0.86 / 1.25 / 1.60 of what the abstraction
buys in worlds 0-2, and 99-100% of ordered pairs are distinguishable at
a 10% tolerance. Priced in held-out Gaussian nats, compacting to one
abstraction costs 4,474 / 7,883 / 5,769 to save 3,294 / 5,490 / 4,392
bits: net -1,180 / -2,393 / -1,377, mean -1,650.

Consequences. The earlier +2,928-nat H14 structured-versus-control
result and the "4-6 abstractions compact to a cover of size 1" oracle
result are both retracted in `PREDICTIONS.md` (appended, not rewritten).
The representation-fragmentation premise for V4.1-as-compaction is
unsupported on this testbed: V3's abstractions are individually weak but
mutually distinct, which is a different problem. Both the operator
(`lifecycle_models.consolidate`) and the oracle now use the
contribution-relative denominator, the oracle carries a hard null-edit
guard and an honest pass/fail line, and the spec's H14 section records
the gate failure. All six lifecycle tests still pass — the redundancy
test uses a genuinely duplicated abstraction and survives the tighter
tolerance, which is the discrimination we want.

Next: V4.2 synthetic merge (refit one abstraction covering several
distinct contributions), which the gate does not block.

# V4.1 blocked on BOTH routes; V4.2 factorization gate passes (2026-08-19)

Route 1, REDUNDANCY, already failed (retraction above). Route 2 is
OBSOLESCENCE, which the §2.2 dormancy pair exists to test, and which the
plain structured world cannot contain because its families stay live to
the end. Built `audit_obsolescence.py` and ran the pair.

First configuration was wrong and the control caught it. With
`dormancy = (32, 64)` the returning regime resumes exactly at the last
task, so the RETURNS arm never returns and both arms were byte-identical
for the whole lifetime; they scored identically to the last nat. Fixed
to `(32, 48)`, leaving 16 resumed tasks.

With the corrected gap the arms differ but the REFUSAL CONTROL FAILS.
The permanent arm retires at t_d in {24: 4, 32: 1, 48: 3, 64: 2} and the
returning arm at {24: 4, 32: 1, 48: 3, 64: 1} — indistinguishable. The
returning arm is supposed to retain through the gap and does not want
to. Diagnosis: after the gap the learner PROMOTES FRESH abstractions
rather than reusing the dormant one, so there is no retention obligation
for the control to refuse. The world does not yet test real options.

Consequence: V4.1 is blocked on redundancy AND on obsolescence, and no
RETIRE operator may be tuned against either. The obsolescence auditor now
refuses to report a single-arm verdict at all, because deleting at the
final consolidation point removes bits and charges nothing for
already-scored tasks, so end-of-life cleanup always "pays".

V4.2 FACTORIZATION GATE — PASSES. Ran the behavioral function-space
diagnostic (never parameter space, per V3's gauge-freedom result): is
A_i(z) ~ C(z) + sum_k a_ik B_k(z)? A shared mean plus a rank-2 functional
family captures 83.4 / 67.7 / 78.7 percent of the abstraction-to-
abstraction spread in worlds 0-2. With m points the spread has at most
m-1 dimensions, so these are scored against an isotropic null of
67.7 / 41.5 / 51.4 percent: excesses of +15.7, +26.2, and +27.3 points,
positive 3/3. Proxy bits fall from 6,336 / 9,504 / 7,920 to about 4,830
under a shared C plus rank-2 basis plus 8-bit coefficients.

This is a DIAGNOSTIC, not yet a result: 17-32% of the spread is
unexplained at rank 2 and that residual distortion has not been priced
in nats. Pricing it is exactly the mistake that produced the retracted
H14 number, so the V4.2 build must compare refitted C + Delta_i against
the original A_i in held-out Gaussian loss before any claim is made.

# V4R census, first result: COMPRESS dominates 12/12 (2026-08-19)

Ran V4R §1's opportunity census with the V3 learner frozen and no
operator implemented. Each library is scored on the ambition ladder
KEEP < COMPRESS < FACTORIZE, in both currencies, with FACTORIZE required
to beat MATCHED-BUDGET private compression rather than merely full
precision atoms.

| regime | library size | COMPRESS net | FACTORIZE net | winner |
| --- | --- | --- | --- | --- |
| N=64  F=2 | 4-6  | (V4.2 result) | loses 9/10 | COMPRESS |
| N=64  F=4 | 6-9  | 3,084 / 5,420 / 3,779 | 1,948 / 3,094 / 2,541 | COMPRESS 3/3 |
| N=64  F=8 | 3-4  | 2,371 / 2,302 / 2,388 | 1,521 / 1,249 / 991 | COMPRESS 3/3 |
| N=128 F=4 | 9-12 | 4,872 / 5,269 / 7,385 | 4,312 / 3,725 / 3,313 | COMPRESS 3/3 |

Two findings.

Library size is NOT monotone in family count. F=8 produces SMALLER
libraries (3-4) than F=4 (6-9) because 64 tasks split eight ways leaves
too few tasks per family to reach `minimum_cluster = 3`. Family count is
capped by lifetime length, so the scale axis has to be driven by N.

Doubling the lifetime does drive it -- N=128 F=4 reaches libraries of
9-12 -- and COMPRESS still wins 3/3, with the margin WIDENING rather
than closing (7,385 versus 3,313 in world 2). So there is no
`M*_factorize` below a library of about twelve abstractions: the
prediction registered in V4R §1.1 that "many related atoms" favors
FACTORIZE is not supported anywhere in the sampled regimes.

Method note. The first N=128 run was invalid and is discarded: the
census resolves `root/F{families}`, so passing the N=64 root with the
N=128 config scored 64-task artifacts against a 128-task world
generation. It reported a spurious FACTORIZE win in world 1. Artifacts
were moved to a separate root and the cell rerun.

# V4R RETAIN: crossover located, refusal control passes (2026-08-19)

Replicated the frozen-library retention oracle at the scale its own
registered prerequisite demands: family onset moved from 16 to 8 so
pre-gap abstractions form reliably, all ten development worlds, and a
gap sweep. 40 lifetimes, single-family worlds, library frozen after the
gap so online PROMOTE cannot manufacture a replacement.

| arm | gap | scored | mean V_retain | positive | median |
| --- | --- | --- | --- | --- | --- |
| returns | 8 | 10 | +1,594 | 7/10 | +2,113 |
| returns | 16 | 10 | -240 | 6/10 | +22 |
| returns | 32 | 9 | -2,068 | 0/9 | -1,962 |
| permanent | 16 | 9 | -2,068 | 0/9 | -1,962 |

The crossover `g*` lies between gap 8 and gap 16: short gaps favor
RETAIN (+1,594, 7/10 worlds), gap 16 is marginal (mean -240 but median
+22 and 6/10 positive), and long gaps favor DELETE decisively.

The refusal control PASSES for the first time in this arc: in the
permanent arm the dormant abstraction is worth nothing to post-gap tasks
in 0/9 worlds. Retention value appears where the regime returns and
nowhere else, which is the discrimination three online dormancy designs
could not produce.

Consistency check, not a bug: `returns` at gap 32 and `permanent` at gap
16 report identical figures because a returning regime that resumes at
task 64 resumes at the lifetime's last task and therefore never returns.
Both conditions are "family gone permanently from task 32", so they
generate the same worlds and must score alike. Earlier in V4 the same
identity appeared as a defect precisely because it was NOT expected
there; here it is the arithmetic working.

Status. V4R §2.1's registered prerequisite for implementing an online
retention policy — an oracle crossover in gap length — is now satisfied.
This is still an oracle over frozen libraries, not a learner result: no
retention policy has been implemented, and the confound that blocks the
online version (PROMOTE firing on noise) is unchanged.

# V4R scale axis complete: COMPRESS wins 15/15 (2026-08-19)

Finished V4R §1.1's registered lifetime-length sweep. N=256 proved
unreachable: the world caps `tasks` at the number of distinct programs,
`primitives ** depth` = 6**3 = 216, and raising it means adding
primitives, which would break comparability with every existing
artifact. The sweep therefore runs at N in {64, 128, 200}, with 200 the
largest admissible point, and that substitution is recorded in AGENTS.md
rather than left implicit.

| N | library size | COMPRESS net | FACTORIZE net | winner |
| --- | --- | --- | --- | --- |
| 64  | 6-9   | 3,084 / 5,420 / 3,779 | 1,948 / 3,094 / 2,541 | COMPRESS 3/3 |
| 128 | 9-12  | 4,872 / 5,269 / 7,385 | 4,312 / 3,725 / 3,313 | COMPRESS 3/3 |
| 200 | 11-16 | 8,346 / 6,897 / 6,000 | 5,553 / 4,540 / 2,700 | COMPRESS 3/3 |

Tripling the lifetime roughly doubles the library, from 6-9 to 11-16
abstractions, and COMPRESS still wins every cell. Across the whole
census — F in {2,4,8} and N in {64,128,200} — COMPRESS beats FACTORIZE
in 15/15 regime-worlds, and the margin does not close with scale.

So no `M*_factorize` exists below sixteen abstractions, and §1.2's
registered possibility is the realized outcome at every scale this
testbed can reach: lifecycle machinery costs more than the slack it
recovers. Locating a sharing threshold would require a library an order
of magnitude larger than the 216-program world can generate, which is a
statement about the testbed's ceiling, not only about the learner.

## V4 synthesis: economy over representational transformations

Synthesized the full V4 development program into the paper draft
(section 10.1), review feedback (reviewer-feedback-34.txt), and
learnings. The V4 development program falsified its original premise
(lifecycle management) but produced a stronger result: the economic
rules that govern when a neural library should restructure.

Key development findings:
- V3's abstractions were behaviorally distinct, not redundant
- COMPRESS (local precision/rank reduction) beats FACTORIZE,
  RETIRE, and FORK across the entire reachable census (15/15
  regime-worlds, including tripled lifetimes)
- The retention law H_R * s_bar > lambda*D(A) was not fitted:
  predicted crossing 17.1, empirical 17.9 (within ~5%)
- Promotion and retention are the same amortization decision in
  opposite temporal directions
- Dormancy length is not the relevant variable; only expected
  remaining reuse matters
- In a self-modifying library, abstraction value is conditional on
  library state V(A | L_t, H_t, pi), making the problem sequential
  structural decision-making
- The recursive economic law N*s > C appears at every level (tasks,
  retention, schema, macro)
- Missing pressures identified: scale, meta-recurrence,
  nonstationarity, reacquisition cost, retrieval cost

Added section 10.1 to paper/draft.md as development-stage outlook
(not confirmatory). Updated review-index.md and learnings.txt with
the full synthesis. No sealed worlds were run; V4 remains
development-only.

# Retention: the amortization law, measured and confirmed (2026-08-19)

Three dormancy designs had failed to instantiate retention value. The
cause was found and it was not the world: V3's `select_reference` scans
every abstraction ever created, so a "deleted" abstraction remained
adoptable and deletion was a no-op. `LifecycleLibraryLearner` now
overrides it to reuse only from the live library (an override, never an
edit to the frozen V3 class).

With deletion made real, reacquisition cost is positive 9/9 and the
return curve is the correct instrument: a mid-lifetime deletion stops
the arms being paired the moment it changes what gets promoted next, so
end-of-lifetime J is uninformative while the return window is not.

The boundary turned out NOT to be a dormancy length. Per-task saving is
flat across gaps 4-16 (63.5 / 64.6 / 63.9 / 62.2 / 66.2), so an earlier
interpolated `g* = 15.2` was pure horizon truncation and is withdrawn.
What governs the decision is expected remaining reuse.

Registered before the sweep, from independently measured quantities:
`H_R* = lambda * D(A) / s_bar = 1,098 / 64.1 = 17.1` returning tasks.

Controlled sweep (gap fixed, library frozen at the gap so
`D_retain - D_delete = D(A)` exactly, verified by zero post-gap births):
V_retain = -621, -407, -131, +140, +404, +782 at H_R = 8, 12, 16, 20,
24, 32. Monotone, 3/3 worlds per cell, crossing at 17.9 against 17.1
predicted, with no fitted threshold. This is the V1/V3 amortization
criterion applied one level up.

Boundary condition: with re-promotion restored, C_reacquire is unchanged
(within 5%) but the marginal carry cost is endogenous -- 0 nats when
deletion merely triggers a replacement -- and V_retain becomes
non-monotone. A per-object retention rule is not well-posed in an
evolving library; the decision is sequential.

# V3 coding-frontier audit: the description claim survives (2026-08-19)

The question that had to be settled before any successor benchmark: does
promotion's description saving survive when BOTH representations are put
on their own behavioral rate-distortion frontiers, rather than compared
at a fixed 8-bit serialization?

Answer: yes, and it grows. Across all 30 V3-sealed worlds, at the
scorer's own accounting scope, the reduction rises from 62.6% at fixed
precision to 68.7% at the frontier, larger in 30/30 worlds.

Getting there required a bit-exact re-run. V3 artifacts do not persist
`task_reference`/`retired`, so the promoted-versus-private split could
not be reconstructed and the first attempt reported reductions of -9.6%,
-5.5%, -5.5% -- a loading bug, not a measurement, now recorded as void.
`LifecycleLibraryLearner` is behaviourally identical to the promoting
learner with its flags off, so seeds 300-329 were re-run to persist the
table; reproduction verified bit-exact (delta 0.0 nats) before use.
These are already-scored worlds; 400-429 were not touched.

Reconciling 67.6% against the paper's 63.3% took two attempts. The first
explanation (reference bits) was wrong and worth 0.1 points. The actual
cause is route/code state, 1,170 scalars retained in BOTH arms, which my
audit omitted; adding the same constant to numerator and denominator
pulls the ratio toward 1, so omitting it inflated the reduction. At the
scorer's scope the fixed-precision figure reproduces to within 0.7
points.

Provenance fixed at the source: every promoting artifact now writes a
`reference_table` summary field. Verified non-vacuously (56 promoted
tasks, 4 abstractions on a real run); 115 tests pass. Summary field
only, so no frozen model behaviour and no fingerprint change.

# V4R SEALED BLOCK OPENED AND SCORED (seeds 400-429, 2026-08-19)

`V4R_CONFIRMATION_PLAN.md` frozen at commit 2aec65c and hashed into
`tools/check_prereg.py`; `python tools/check_prereg.py` verifies the
trail (6 frozen files unchanged). Thresholds were transcribed into
`score_v4r_sealed.py` before any sealed world existed. Seeds 400-429
were untouched until the hash was registered.

This is the project's first PREREGISTERED NEGATIVE: the registered
prediction was that no structural edit pays.

    O1 count     COMPRESS wins 30/30           (need >= 27)      PASS
    O1 interval  mean margin 1,362 nats        (need 1k-4k)      PASS
    O2           FACTORIZE wins at M <= 16: 0  (need 0)          PASS
    O3           FORK pays in 2/30 worlds      (need <= 2)       PASS

    Library sizes: min 3, max 9, mean 5.5.
    FORK: 126 abstraction-cells evaluated, 2 paying.

4/4 registered census criteria met. The V4R development negative
REPLICATES out of sample: in the canonical ROW regime, at every scale
this benchmark can reach, the best available refactoring is local
private compression, not lifecycle restructuring.

O3 deserves a note rather than a clean tick. FORK pays in exactly 2/30
worlds against an allowance of <= 2 -- it passes at the boundary, not
comfortably, where development had 0/33. The registered threshold held,
but a replication would be prudent before the claim is stated as "FORK
never pays"; the defensible statement is "FORK pays rarely and the rate
is within the preregistered bound".

O4 (the retention amortization law) is NOT scored here. It needs the
horizon sweep protocol at 6 horizons x 2 arms x 30 worlds, which is a
separate run; the census cells stand on their own.

# V4R O4: the retention law confirmed on sealed worlds (2026-08-20)

240 lifetimes, 4 horizons x 2 arms x 30 sealed worlds, controlled
protocol (gap (32,40), last sleep at the gap so
`D_retain - D_delete = D(A)` exactly). Scored by
`score_v4r_retention.py`, thresholds transcribed from the plan frozen at
2aec65c before any of these runs existed.

| H_R | C_reacquire | s_bar | V_retain | verdict |
| --- | --- | --- | --- | --- |
|  8 |   486 | 60.8 | -612 | DELETE |
| 16 |   976 | 61.0 | -122 | DELETE |
| 24 | 1,461 | 60.9 | +363 | RETAIN |
| 32 | 1,961 | 61.3 | +864 | RETAIN |

    monotone in H_R                 PASS
    s_bar 61.0 (need 50-75)         PASS
    crossing 18.0 (need 14-22)      PASS
    O4: PASS (3/3).  0 worlds excluded for post-gap births.

WHAT IS AND IS NOT NON-TRIVIAL HERE. The scorer reports a derived
prediction of 18.0 against an observed crossing of 18.0, and that exact
agreement is partly TAUTOLOGICAL: the crossing is by definition where
`H_R * s_bar = lambda * D(A)`, so with s_bar constant the crossing must
equal `carry / s_bar`. The two substantive results are:

  1. s_bar IS constant -- 60.8 / 61.0 / 60.9 / 61.3 across horizons,
     cv 0.3% over 30 worlds. The per-use saving does not depend on the
     horizon, which is what makes the law linear rather than a fit.
  2. The DEVELOPMENT-derived prediction of 17.1 (from dev s_bar 64.1)
     lands at 18.0 out of sample -- 5% error, inside the preregistered
     [14, 22]. That is the real confirmation.

# V4R SEALED BLOCK CLOSED: 7/7 (2026-08-20)

    O1 count     COMPRESS wins 30/30              PASS
    O1 interval  mean margin 1,362 nats           PASS
    O2           0 FACTORIZE wins at M <= 16      PASS
    O3           FORK pays 2/30 (allowance 2)     PASS (at the boundary)
    O4 monotone / s_bar / crossing                PASS x3

The project's first preregistered NEGATIVE replicates out of sample,
with one positive mechanism inside it. In the canonical ROW regime the
best available refactoring is local private compression; the single
structural edit that pays is retention, and it pays exactly when
expected remaining reuse repays the abstraction's code cost.

O3 remains the weak tick: 2/30 against an allowance of 2, where
development had 0/33. The threshold held and is not being reinterpreted,
but the defensible phrasing is "FORK pays rarely, within the
preregistered bound", not "FORK never pays".

# V5.0: the component rate-distortion frontier D*(R) (2026-08-20)

The currency for everything downstream, replacing the arbitrary 8-bit
retention proxy. Minimum bits/scalar per component, each scored ONLY on
the computations that depend on it, 10 sealed worlds, zero vacuous
cells:

| component | participants | eps=2 | eps=10 | eps=30 |
| --- | --- | --- | --- | --- |
| private (per-task residuals) | 8.2 | 5.8 | 5.0 | 4.2 |
| shared (abstractions)        | 55.8 | 4.8 | 3.9 | 3.0 |
| basis (operator basis)       | 64.0 | 5.7 | 4.4 | 3.6 |
| routes (per-task codes)      | 64.0 | 3.0 | 2.0 | 2.0 |

Every component sits well below 8 bits, so the proxy overstates
description length by 1.6x (private) to 4x (routes).

THE STRUCTURAL FACT. Shared abstractions are cheaper per scalar (3.9
bits) than the private residuals they replace (5.0). Promotion does not
merely move description from many places to one; the destination is
intrinsically more compressible. That is the mechanism behind the V4R
audit result that promotion's description reduction GROWS at the
frontier (62.6% -> 68.7%), and it was not visible under a uniform proxy
that charged both sides 8 bits.

A NON-VACUITY GUARD is built in, because a component scored over tasks
it does not participate in returns 0.0 at every depth -- "infinitely
compressible" when it means "never measured". That exact false result
occurred once during the V4R audit. All 40 component-cells here are
non-vacuous.

CONSEQUENCE FOR THE RETENTION LAW, and it is a live prediction rather
than a re-analysis. The law `RETAIN iff H_R * s_bar > lambda * D(A)`
was confirmed with D(A) priced at the 8-bit proxy, giving H_R* = 18.0.
At the measured frontier, D*(A) = 198 scalars x 3.9 bits = 772 bits =
535 nats, so the same law predicts

    H_R* = 535 / 61.0 = 8.8 returning tasks

Re-scoring the sealed O4 measurements at that carry gives V_retain of
-49, +440, +926, +1,426 at H_R = 8, 16, 24, 32 -- a crossing just above
8, consistent with 8.8.

This re-scoring is NOT an independent test: C_reacquire and s_bar are
unchanged, so moving the carry term necessarily moves the crossing to
carry/s_bar. The genuine causal test is V5.1 -- manipulate D(A) at the
generator (e.g. residual rank), predict H_R* proportionally, and check
that the observed crossing follows. Only that distinguishes a
quantitative law from an accounting identity.

# 2026-08-20 — V5 sketch written

- Wrote `notes/v5-sketch.txt` Revision 1. V5 is representation
  economics, not MACRO/LOOP/BRANCH (those moved to V6, per reviews
  35/40/41). Question: under what conditions the cheapest useful
  representation stops being a static library of independently
  compressed abstractions, and whether one prospective score can track
  that boundary.
- Nested claims, gated in order: LAW (H19) → PHASE DIAGRAM (H20/H21)
  → SELECTION (H22–H24). A later rung is not a consolation prize; H20
  blocked by the 216-program ceiling is a registered respectable
  outcome, not a failure to retune.
- Records V5.0 as a starting fact, H19–H24 as already frozen in
  `PREDICTIONS.md` before any V5 world, and the in-flight H19 protocol
  deviation (rank at the generator vs bits/scalar at the coder) rather
  than silently adopting it. Worlds 500–509 are V5 development;
  600–629 reserved for confirmation. Compositional-closure probe is on
  the pre-spec checklist, not skipped again.
- Sketch is not a spec. It retires the day `row_v5_experimental_spec.md`
  is frozen. Next: score the H19 grid and append the outcome before
  writing the spec.

# 2026-08-20 — V5 sketch Revision 2

- Expanded `notes/v5-sketch.txt`. Revision 1 named the question;
  Revision 2 makes it executable: worked H19 arithmetic (rank 4's
  8-bit crossing is off the current horizon grid), the
  `residual_rank > 2` config cap, slots=12 as a deviation from V3/V4R,
  an s-arm recipe with an uninformative-arm rule, H20 cheap-first
  slice, numeric G3, a written H22 estimator, structural-regret
  definitions, a threats table, and an operational composition probe.
- In-flight `artifacts/v5_causal/` is recorded as rank-1 only and is
  not H19. D1–D12 freeze the decisions a spec should not re-litigate.
  Checklist reordered: score the rank-1 diagnostic and run the
  composition probe before another lifetime sweep.

# 2026-08-20 — V5 preflight: diagnostics, rank cap, paper hygiene

- Scored the in-flight rank-1 causal grid (worlds 500-509, slots=12)
  and the rank-2 horizon grid (worlds 0-2, slots=12) as diagnostics,
  not H19. Rank 1: s_bar 36.5, crossing 15.2. Rank 2: s_bar 60.1,
  crossing 17.9 (reproduces V4R O4 development). Unpaired H* ratio
  1.18 against D ratio 2.0; s_bar may move with rank. Reports:
  `reports/v5_causal_rank1_dev.json`,
  `reports/v5_horizon_rank2_dev.json`, `reports/v5_preflight.json`.
- Compositional-closure probe on Continuous checkpoints 0-2:
  depth-8/depth-3 median NMSE 4.39 / 4.26 / 4.09, all under the
  draft 5x gate, no saturation. V6 is not blocked. Module:
  `src/row/experiments/probe_compositional_closure.py`.
- Lifted `residual_rank` cap from 2 to 4 in shared-residual,
  variational, and gated configs; default remains 2. Tests in
  `tests/test_config.py`. Did not launch rank-4 lifetimes.
- Paper draft to v0.9: header names the fourth sealed block;
  section 10.1 no longer says no V4 sealed block was run.

# 2026-08-20 — README brought up to four sealed blocks

- Public README no longer stops at V1. Status table covers V1–V4R
  sealed verdicts, names V5 as a sketch with 600–629 reserved, and
  corrects the seed partition (200–229 and 400–429 are closed, not
  sealed-and-waiting). Repository map lists V2–V4R specs, confirmation
  plans, and `notes/v5-sketch.txt`.

# 2026-08-20 — reviews 44 and 45 incorporated

- `reviews/review-index.md`: entry for
  `reviewer-feedback-44.txt` (response to the V5.1 causal result;
  V5.2 design as recursive amortization). Entry for
  `reviewer-feedback-45.txt` recording that it carries no new content
  — its two halves restate feedback 42 and re-paste feedback 43
  verbatim. Kept for provenance.
- `PREDICTIONS.md`: registered H25 (schema crossing
  `M* = D*(S)/s_bar_schema` predicted from unfitted quantities),
  H26 (`dM*/drho < 0`, with generator balance gates frozen at 10% as a
  precondition on scoring), and H27 (shared scalars are individually
  cheaper because promoted abstractions have lower effective functional
  rank). All three registered before any V5.2 world exists.
- Withdrew the padding follow-up proposed in the V5.1 note. Review 44's
  objection is accepted and recorded in both `PREDICTIONS.md` and
  `AGENTS.md`: under the rate-distortion currency,
  `D*(A + dead bits) = D*(A)`.
- `AGENTS.md`: four learnings — separate cost from utility under any
  structural manipulation; padding is not a valid `D` intervention;
  realized library size needs an F-families knob, not depth; a
  relatedness knob needs frozen balance gates.

- `notes/v5-sketch.txt` to Revision 3. H19 closed with the scored V5.1
  numbers (reading (a) falsified, reading (b) confirmed, partials P1
  and P3). H20's generator replaced with the norm-preserving
  functional-space mixture and given balance gates as a scoring
  precondition, plus the parameter-free point prediction
  `M* = D*(S)/s_bar_schema` (H25/H26). Padding excluded by D13; D14
  and D15 added. Two new rungs: H27 (why shared scalars are
  individually cheaper; artifact-only, runs in parallel) and H28 (the
  same morphism under different coordinates, with a mandatory adapter
  complexity budget; registered, unscheduled). Open question 8 marked
  settled in the uncomfortable direction — no pure D manipulation is
  known in this substrate.

# 2026-08-20 — V5 promoted from sketch to spec

- `row_v5_experimental_spec.md` created from `notes/v5-sketch.txt`
  Revision 3: 28 numbered sections in V2/V3/V4R house style, plus a
  new §0 recording V5.0 and V5.1 as closed rungs and stating what
  V5.1's split verdict forces on the rest of the document (measure
  cost and utility separately; H20's balance gates are scoring
  preconditions).
- Frozen at promotion: §6 decisions D1–D15, §9 operational constants,
  §14 statistical plan. Sketch hedging ("draft", "the spec may")
  removed from those sections; §22 keeps the open questions that the
  freezes genuinely do not answer, marked non-blocking.
- §25 pre-run checklist rewritten against reality: items 1, 2, 4 and 6
  discharged (rank cap lifted, causal grid scored, closure probe run,
  H19 branch resolved). Still owed before seeds 600–629 may be
  opened: the paired `operator_slots = 6` check, one informative
  s-arm, the `r_meta` teacher-validity AND balance gates on worlds
  0–2, and a frozen hashed `V5_CONFIRMATION_PLAN.md`.
- `notes/v5-sketch.txt` banner-marked SUPERSEDED and left unrevised,
  as the V3 and V4 sketches were. Pointers updated in `CLAUDE.md`,
  `AGENTS.md`, and `README.md`; `row_v3_experimental_spec.md` is no
  longer described as the active spec.

# 2026-08-20 — reviews 46 and 47 (of the spec) incorporated

- Both reviews read `row_v5_experimental_spec.md` itself. Spec taken to
  Revision 4 before freezing; neither review asked for a rewrite and
  none was made.
- H19: primary s-arm replaced by S0, a post-gap return-value gain
  `y = f_base(x) + g A(x)`, which moves s_bar while leaving the
  abstraction byte-identical (registered check: D*(A) constant to 2%
  across g). S1/S2 demoted to secondary cuts with their confounds
  stated.
- H20: validity instrument replaced — pairwise correlation reports
  r_meta = 0 for a perfectly shared subspace (A_1 = B[1,0],
  A_2 = B[0,1]) and would have voided a valid generator. Now
  functional shared-subspace capture with leave-one-family-out R_LOO
  as the reported number. Split into H20a (frozen schema after an
  M_0 = 4 calibration; where H25 is scored) and H20b (re-fit; tests
  the learner, three registered outcomes including library collapse).
  Promotion rate removed from the balance gates and made a reported
  outcome (D17).
- H21 gained a novel-combination holdout; H22 scores PROMOTE in the
  common Delta J currency after a legality gate and decomposes online
  failures into four error terms; H23's paired-causal and policy rules
  separated; H27 gained the P_0/P_1/P_2 causal decomposition; H28
  ungated from H20 and raised to a V6 entry question; the closure
  probe extended into an error-vs-depth law with p90, worst decile and
  per-call error (interface stability).
- New decisions D16–D18; open question 11 settled; staging reordered
  so V5 may close after H19/H20/H21/H27 with H22–H24 as a side branch.
- `PREDICTIONS.md`: H25 protocol amendment (frozen schema, or the
  crossing does not score H25), H26 amendment (three gates, not four),
  and H29 (the causal decomposition, registered before the audit).
- Reviews saved as `reviewer-feedback-46.txt` / `-47.txt`, indexed, and
  the method lessons appended to `notes/learnings.txt`.

# 2026-08-20 — V5 spec Revision 5 (internal-consistency audit)

- Ran an independent audit of `row_v5_experimental_spec.md` against
  reviews 46 and 47 rather than self-certifying Revision 4. Coverage
  came back complete except for H28's ungating, which had not
  propagated to the branch table or §5.
- Most serious finding, now fixed: Revision 3 replaced the H20
  generator and removed its fixed centre `C`, but `teacher_G1` and
  H21's five-way instrument were still written in terms of `C`. The
  fitted schema is now defined explicitly as
  `S_hat = C_hat + B_hat alpha` with `C_hat` the fitted family mean,
  and the generator's lack of a centre stated as the reason the
  structureless control behaves as it does.
- Also fixed: the isotropic null is now defined once (95th percentile
  of 100 matched isotropic refits) instead of being invoked by three
  gates and defined by none; H25 stated in one form, with the
  undefined "savings already accrued" term removed; the 8-bit horizon
  rows marked as reference arithmetic, since they violate the spec's
  own >= 4-task bracketing rule; H19 described as confirmed on the
  D-arm with the rung explicitly still partial; H20's falsifier moved
  onto the H20a exogenous slice, which is what D16 says decides it;
  H21's gate stated once (G1); H28's staging status made consistent;
  inheritance attributed to V4 rather than V4R; a wrong D11 reference,
  the stale `measured_r` endpoint, the retired rank-4 branch, the
  superseded B1 world, §22's numbering, and D12's position all
  corrected.
- Eight earlier fixes were lost to a `git checkout` on uncommitted
  work and re-applied. Learning recorded in `AGENTS.md`.

# 2026-08-20 — S0 built and Stage 1 run (H19 s-arm)

- Built B1, the return-value gain: `TaskGroupSpec.return_gain` scales
  the family primitive's residual `alpha` on RETURNING tasks only,
  exposed as `mixed_lifetime --return-gain`, recorded in
  `rho_profile.json`. Eight tests in `tests/test_task_group_world.py`,
  the load-bearing one being that `g = 1.0` is bit-exact against the
  pre-B1 world, so every existing artifact and fingerprint stays valid.
- Chose `alpha` scaling over the additive blend
  `y = f_base + g*(f_full - f_base)`. The blend is the literal
  registered formula but its target is not realizable by ANY composition
  of primitives, so at `g != 1` even a perfect learner would carry
  irreducible misspecification and `s_bar` would be confounded with it.
  Rationale recorded in the spec, since a reviewer could otherwise read
  the choice as a deviation.
- `src/row/experiments/preflight_v5_s0.py`: eight invariants, each
  printing its DENOMINATOR (`routed_to_A 14 / 32`, `0 late / 2 born`),
  because this project has twice shipped a guard that printed PASS over
  zero rows.
- `src/row/experiments/score_v5_s0.py`: scores the arm, hashes the
  carried abstraction tensors to assert carry invariance across `g`,
  voids any cell whose pre-intervention delta is nonzero, and refuses
  crossings the grid does not bracket rather than interpolating.

## Results

STAGE 1 (fixed window N=72, H_R=32, 10 worlds, both arms, 60 cells,
0 failures, 0 exclusions, 0 leaks):

| g | s_bar | C_reacquire | p_reuse | s_conditional |
| --- | --- | --- | --- | --- |
| 0.5 | 28.9 |   925 | 0.41 | 27.0 |
| 1.0 | 58.1 | 1,858 | 0.44 | 52.8 |
| 1.5 | 85.2 | 2,725 | 0.44 | 80.2 |

- s_bar ratio 2.95 against a gain ratio of 3.00 — very nearly linear,
  better than the sub-linear behaviour expected from the tanh and the
  mismatch channel.
- Carry invariance: abstraction checksum identical in 10/10 cells. The
  cost side is the SAME NUMBER across gains while the utility side moves
  threefold.
- Mechanism: `p_reuse` flat, `s_conditional` triples — the registered
  reading A (the abstraction became more important), not B (routing
  abandoned it). The relative refusal threshold never fired.
- `g = 1.0` reproduces the V4R sealed operating point: 58.1 vs 61.0.
- Stage-2 grid computed by the frozen rule and committed BEFORE any
  Stage-2 cell exists: g=0.5 -> N {72,76,78,82}; g=1.0 -> {54,58,60,64};
  g=1.5 -> {48,52,54,58}. Registered at +/- 15%.

SLOTS=6 PROTOCOL ROBUSTNESS (g=1.0, N=72, 10 worlds, 20 cells): s_bar
44.3 against 58.1 at slots=12, via a smaller library (2.2 vs 3.3
abstractions) that fewer tasks reference (46.7 vs 59.0). `D(A)` is
unchanged, so the law predicts the crossing moves as 1/s_bar, 18.9 ->
24.8. The D-arm's ABSOLUTE crossings are therefore protocol-dependent
and do not transfer between slot budgets; its internal rank comparison
is unaffected because ranks 1/2/4 all ran at slots=12. Whether the law
itself survives at slots=6 is NOT tested and needs its own grid.

## Not yet run

Stage 2 (240 cells, ~3h, grids frozen); the slots=6 crossing grid around
24.8; D* currency grids (predicted crossings 18.5 / 9.2 / 6.3, which the
8-bit grids do not bracket).

# 2026-08-20 — H19 CLOSED: PASS. The amortization law predicts out of sample

Stage 2 ran the three frozen grids: 240 cells, zero failures, zero
post-gap births, zero pre-intervention leaks. Every predictor came from
Stage 1's separate fixed-window runs, so no cell served as both
predictor and outcome.

| operating point | D(A) | s_bar | H* predicted | H* observed | err | chi |
| --- | --- | --- | --- | --- | --- | --- |
| D-arm rank 1 |  99 | 36.8 | 14.9 | 15.2 | 2.0% | 1.019 |
| D-arm rank 2 | 198 | 61.0 | 18.0 | 18.0 | 0.0% | 1.000 |
| D-arm rank 4 | 396 | 68.3 | 32.2 | 32.4 | 0.6% | 1.008 |
| S-arm g=0.5  | 198 | 28.9 | 38.0 | 38.7 | 1.8% | 1.019 |
| S-arm g=1.0  | 198 | 58.1 | 18.9 | 19.2 | 1.6% | 1.016 |
| S-arm g=1.5  | 198 | 85.2 | 12.9 | 13.0 | 0.8% | 1.009 |

mean |chi - 1| = 0.012 against a registered 0.15; range [1.000, 1.019]
against registered bounds [0.7, 1.3]. Report:
`reports/v5_h19_collapse.json`.

All seven conditions of the criterion frozen before g=1.5 was read are
met. Routing validity passed in its strong form: p_reuse is IDENTICAL
between g=1.0 and g=1.5 in every world inspected, so the gain moved
payoff without touching selection.

- The claim, stated to admit what each arm did: the D-arm manipulates
  abstraction CAPACITY (cost and utility together), the S-arm holds the
  stored abstraction bit-identical and moves only its post-return
  utility regime; across both, the threshold is predicted by
  C_carry / s_bar. Neither `H* ~ D` (falsified at 46.8% in V5.1) nor
  `H* ~ 1/g` is the law.
- Limits recorded alongside the result rather than after it: six points
  on a line and not a crossed 3x3 design (every S-arm point shares
  D(A) = 198); one currency (8-bit — the D* crossings 18.5 / 9.2 / 6.3
  are unbracketed by these grids); one protocol point (slots=12, one
  dormancy geometry); development seeds 500-509, nothing sealed.

## Spec and rung status

V5.1 moves from partial P1 + P3 to PASS. §25 checklist item 3 is
discharged for the s-arm; the slots=6 crossing grid and the D* grids
remain as named debts, and 600-629 stay untouched.

# 2026-08-20 — B2 built: the r_meta generator, and its gates PASS

Checklist item 5's teacher-side half is discharged. `src/row/meta_world.py`
and `src/row/experiments/audit_meta_recurrence.py`; 9 tests in
`tests/test_meta_world.py`; report
`reports/v5_meta_recurrence_gates.json`.

THE MIXTURE IS FUNCTIONAL, EXACTLY, NOT BY ARGUMENT. A primitive
computes `tanh(z + alpha * h(z) U^T)` with `h(z) = tanh(z V^T + b)`.
The generator SHARES `V` and `b` across the family operators, so `h` is
one fixed function of the input and the residual contribution is LINEAR
in `U`. A mixture of `U` matrices is then exactly a mixture of the
functions they compute, at every input at once — which is how the spec's
"defined in functional space because of gauge freedom" is satisfied
rather than approximated.

    theta_f(r) = sqrt(r) * B alpha_f + sqrt(1 - r) * C_f

with the shared basis orthonormalized in matrix space and `C_f`
PROJECTED OUT of that subspace. The projection is not decoration: with
the two components merely independent, their random inner product made
`||theta_f||` wobble and the balance gate failed at 31.6% spread on
contribution. Orthogonality makes `||theta_f||^2 = r + (1 - r) = 1`
exactly, for every family at every r.

GATES, worlds 0-2, F=4, m=16, K=2, probe 256:

| r_meta | R_LOO | null | R_in-sample | contribution | D* bits | norm |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.021 | 0.049 | 0.730 | 0.00771 | 3.92 | 1.945 |
| 0.5 | 0.165 | 0.049 | 0.845 | 0.00781 | 3.99 | 1.945 |
| 0.9 | 0.547 | 0.049 | 0.970 | 0.00776 | 3.88 | 1.945 |
| 1.0 | 1.000 | 0.049 | 1.000 | 0.00770 | 3.88 | 1.945 |

- VALIDITY: r=0 inside the isotropic null, R_LOO monotone, r=1 far
  above the null. PASS.
- BALANCE: contribution 1.4%, rate 2.9%, norm 0.0% spread, all inside
  the registered 10%. PASS. Promotion rate is deliberately not gated.
- GATES: PASS — the sweep is scoreable.

Two instrument notes worth keeping. The in-sample capture reads 0.730
at r_meta = 0, where the truth is "no shared structure at all"; only the
leave-one-family-out version reports 0.021. Fitting and scoring on the
same families would have manufactured the phenomenon H20 is looking for.
And the D* reading is INTERPOLATED in (bits, log error): integer depths
made the per-abstraction rate jump by whole bits, which alone produced a
16.4% spread and a false balance-gate failure.

# 2026-08-20 — H20a run: the amortization law holds one level up

`src/row/experiments/audit_schema_economics.py`, report
`reports/v5_schema_economics.json`. Exogenous atoms, schema fitted on
M_0 = 4 and frozen, matched distortion budget on every piece. F=12,
K=2, worlds 0-2. Offline.

| r_meta | s_bar_schema | M* pred | M* obs | matched-budget winner |
| --- | --- | --- | --- | --- |
| 0.00 |  -33 |  inf | none | COMPRESS |
| 0.50 |   -9 |  inf | none | COMPRESS |
| 0.70 |   24 | 39.7 | none | FACTORIZE |
| 0.90 |  103 |  8.0 |  7.5 | FACTORIZE |
| 0.95 |  157 |  5.1 |  5.7 | FACTORIZE |
| 1.00 |  495 |  1.6 |  2.0 | FACTORIZE |

- H26 (dM*/dr_meta < 0): SUPPORTED, monotone across the sweep. The
  COMPRESS -> FACTORIZE boundary is between r_meta 0.5 and 0.7.
- H25: PASS at r_meta 0.90 (6.4%) and 0.95 (10.1%); MISS at 1.00
  (25.7%), reported as a miss. Observed M is an integer, so at a
  predicted 1.6 the finest attainable error is 25% and ceil(1.6) = 2 is
  what was observed — the law is consistent there, the registered
  metric is not resolvable there. A better small-M criterion is
  registered for next time and NOT applied retroactively.
- The r_meta = 0 cell reproduces V4R's negative as the low-relatedness
  limit of a knob rather than as a separate finding: the schema has
  negative per-member saving and never pays.

Scope: exogenous atoms by design (D16), so this says nothing about
whether PROMOTE reaches the region. H20b is the separate experiment and
its outcome 3 — the learner collapsing meta-structure into fewer atoms
instead of forming a schema — is still live.

# 2026-08-20 — H20b first reading: the library does not track F

Meta world wired into `mixed_lifetime` (`--r-meta`, `--meta-families`,
`--meta-tasks-per-family`, `--meta-subspace-rank`), provenance recorded
in `rho_profile.json`. Cheap-first slice, F=4, N=72, worlds 0-2:

    r_meta = 0.0   realized M = 5, 6, 7   mean 6.0
    r_meta = 1.0   realized M = 7, 9, 5   mean 7.0

PROMOTE produces 1.5-1.75x more abstractions than there are families,
and the count does not fall as relatedness rises. Outcome 3 (collapse)
does not occur; outcomes 1 and 2 assumed M ~ F and that premise fails.
The F knob controls task structure, not realized library size, at this
operating point. n = 3, spread 5-9: a direction, not an estimate.

H20b's G2 is deliberately NOT run. A learned abstraction is a flat
198-vector of (u, v, b), so schema-fitting it is parameter-space
fitting — the gauge-sensitive move the spec forbids. H20a escaped this
because the teacher's operators share hidden features by construction.
A correct G2 needs functional-effect accounting at matched budget;
building it in a hurry is how V4.2 produced a sharing claim that passed
at full precision and failed at equal bits.

# 2026-08-20 — H20b G2: the schema economy exists, the learner misses it

`src/row/experiments/audit_learned_schema.py`, report
`reports/v5_learned_schema.json`. Schema fitted in effect space
(gauge-free, since two learned atoms computing the same function can
have unrelated 198-vectors) and charged in parameter space; uncovered
members pay full private price, so FACTORIZE is never flattered by an
invented leftover code.

    r_meta = 0.0   FACTORIZE 0/3 worlds   3,157 vs 1,630 bits
    r_meta = 1.0   FACTORIZE 0/3 worlds   3,814 vs 2,258 bits

G2 fails 0/6. Checked against the vacuous-guard failure mode before
reporting: the schema leaves 90.9% (r=0) and 87.3% (r=1) of a held-out
atom's effect variance unexplained, so 0% coverage is real and not the
budget being unreachable.

The finding is the CONTRAST with H20a. Over teacher family operators a
schema pays and its crossing is predicted (H25 passes at r_meta 0.90 and
0.95). Over the learned library, PROMOTE makes 1.5-1.75x more atoms than
families and they share almost none of the family's functional
structure. The representation class is fine; the promoter is what fails
to reach it. Run only on the learned library — the design before reviews
46 and 47 — this would have read as "higher-order factorization does not
pay", and D16's exogenous-atom arm is what makes the two separable.

n = 3 worlds, one operating point. A direction, not an estimate.

# 2026-08-20 — H21: prospective reuse passes at exact meta-recurrence

`src/row/experiments/audit_schema_transfer.py`, report
`reports/v5_schema_transfer.json`. Leave-one-family-out, F=8, K=2,
worlds 0-2, target held-out MSE 1e-3.

    r_meta 1.0   schema reaches target from 1 example, independent 8
                 retains 2 arguments vs 128 operator scalars      PASS
    r_meta 0.9   schema plateaus at 1.21e-3, never reaching target FAIL
    r_meta 0.0   schema plateaus at 8.4e-3                         FAIL

The failures are the registered falsifier working: at partial
relatedness a rank-2 schema cannot EXPRESS a member that is only partly
in the subspace, so its error plateaus regardless of support. Capacity,
not sample efficiency.

Caveat recorded with the result: the independent baseline is
unrealistically strong here, because shared hidden features make
acquisition an exactly-determined linear solve that hits machine
precision at support 8. The 1-vs-8 win is a lower bound on the schema's
advantage; the sub-1.0 failures are measured against a baseline no real
learner attains.

# 2026-08-20 — H27: gap confirmed, mechanism falsified; H29 blocked on provenance

`src/row/experiments/audit_coding_geometry.py`, report
`reports/v5_coding_geometry.json`. 16 artifacts, matched participant
count.

    mean D* gap (private - shared)        +0.253 bits/scalar
    mean spectral gap (private - shared)  -0.157
    rank correlation of the gaps          -0.009  (registered |r| >= 0.5)

H27 FAILS both halves. Shared abstractions have HIGHER sigma_2/sigma_1
(~0.70 vs ~0.54), so they use more of their functional dimension, not
less — the opposite of the registered direction — and the two gaps are
uncorrelated. Effective-dimensionality reduction is eliminated as the
mechanism. The D* gap's SIGN survives, reproducing V5.0's direction, so
noise purification, the selection effect, and restructuring remain live.

H29 is blocked, and not by compute: a finished artifact holds P_2 and
the surviving private residuals but not P_0, the residuals as they stood
at the promoting sleep. Nothing checkpoints them. Same class of defect
as the missing `task_reference` table that once voided a coding audit;
the fix is to record the consumed member residuals in the sleep path.
No approximation substituted.

# 2026-08-20 — sealed C1: stage 1 replicates, first crossing lands at 3.0%

Sealed Stage 1 (seeds 600-609, fixed window N=72, 60 cells, 0 failures):
s_bar 26.6 / 57.3 / 84.3 at g = 0.5 / 1.0 / 1.5, against development's
28.9 / 58.1 / 85.2 — within 8% at every gain, with the abstraction
checksum identical in 10/10 cells. The S0 intervention itself
replicates on untouched seeds.

Stage-2 grids derived by the frozen deterministic rule and committed
before any Stage-2 cell ran:

    g=0.5  H_hat* 41.3  N 76, 80, 82, 86
    g=1.0  H_hat* 19.1  N 54, 58, 60, 64
    g=1.5  H_hat* 13.0  N 48, 52, 54, 58

First batch complete (80/80, 0 failures). g=0.5 observed crossing 42.5
against predicted 41.3 — 3.0% error, chi 1.031, V_retain monotone with a
single sign change (-157, -56, -17, +106). Inside the registered
[0.7, 1.3] and the tighter 15%.

g=1.0 and g=1.5 are running; the C1 verdict and the full chi collapse
wait on them. Configs v5_h80 and v5_h86 added for the g=0.5 grid.

# 2026-08-20 — SEALED C1: PASS. The law predicts on untouched seeds

300 cells (60 Stage 1 + 240 Stage 2) on seeds 600-609. Zero failures,
zero exclusions, every crossing in-grid.

| g | s_bar | H* pred | H* obs | err | chi |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 26.6 | 41.3 | 42.5 | 3.0% | 1.031 |
| 1.0 | 57.3 | 19.1 | 18.9 | 0.8% | 0.989 |
| 1.5 | 84.3 | 13.0 | 12.9 | 0.5% | 0.993 |

mean |chi - 1| = 0.016 against a registered 0.15; range
[0.989, 1.031] against [0.7, 1.3]. The crossing moves 3.3x across the
sweep while the carried abstraction stays bit-identical.

Sealed status: C1 PASS. C2 passes its three unambiguous clauses
(monotone M*, r_meta=0 never pays, ceil-exact 30/30) with its 15% clause
unresolved by a denominator the plan failed to specify. C3 and C4 are
registered negatives and not yet run on sealed worlds.

# 2026-08-20 — V5 SEALED BLOCK COMPLETE: 2 pass, 1 split, 1 mixed

Seeds 600-629 opened against `V5_CONFIRMATION_PLAN.md`, frozen at
1ed227d and hashed into `tools/check_prereg.py` before any sealed world
was generated. `python tools/check_prereg.py` verifies.

| rung | verdict | headline |
| --- | --- | --- |
| C1 amortization law | **PASS** | mean \|chi-1\| = 0.016, range [0.989, 1.031] |
| C2 schema crossing | **SPLIT** | 3 clauses pass; the 15% clause unresolved |
| C3 learned library | **PASS** | 3/3 clauses; FACTORIZE 0/6 at both r_meta |
| C4 coding geometry | **MIXED** | 1/3 clauses; the D* gap does not hold per-world |

C1. 300 cells, zero failures, zero exclusions. Crossings 42.5 / 18.9 /
12.9 against predictions 41.3 / 19.1 / 13.0 made before the runs
existed. The crossing moves 3.3x while the carried abstraction is
bit-identical in every arm.

C2. Monotone M*, r_meta = 0 never pays, and the ceil criterion exact in
30/30 worlds. The 15% clause is unresolved because the plan never said
whether a world with no observable crossing counts as a failure: 79% and
73% on worlds that crossed, 37% and 63% on all worlds. Cause is
reachability — at F=12 with M_0=4 only 8 unseen members exist and sealed
r_meta 0.90 predicted M* ~ 20.

C3. Replicates harder than development: unexplained fraction 0.921 at
r_meta = 1.0 against 0.873, and M > F in 12/12 cells (mean 7.2 and 7.7
against F = 4).

C4. Two clauses fail in opposite directions. The D* gap holds in only
6/12 cells — a coin flip, mean +0.087 bits/scalar against development's
+0.253 — so the per-world regularity I registered from V5.0's aggregate
does not exist at this operating point. And the correlation registered
as absent (-0.009 in development) is +0.888 on sealed worlds. What
survives cleanly: shared abstractions never show faster spectral decay,
0/12 sealed and 0/16 development.

## What V5 established

The amortization law, at two levels and out of sample:

    H* = lambda D*(A) / s_bar        atom retention, sealed, chi ~ 1
    M* = D*(S) / s_bar_schema        schema birth, monotone in
                                     relatedness, ceil-exact 30/30

and one clean negative with a located cause: the economics exist over
teacher operators, and PROMOTE does not reach them, because it
fragments families into 1.5-1.9x more atoms that share almost none of
the family's functional structure.

## Still open

H29's measurement (instrument built this session — the P_0 snapshot —
but existing artifacts predate it, so it needs fresh runs); the slots=6
crossing grid; D* currency grids; H22-H24, a side branch under D18.

# 2026-08-20 — reviews 48-49 acted on: three hypotheses eliminated

Review 48 argued V5's bottleneck had moved from "find a better economic
rule" to "can the learner build representations in which the available
economics are exploitable", and named four hypotheses. Review 49
corrected an over-claim in my first answer. Net result: three of the
four are eliminated and the remaining one is not fixable after the fact.

    A  fragmentation      REJECTED, best k-subset leaves 0.92 at k=3
                          against 0.94 at k=1 (`audit_fragmentation.py`)
    C  local coordinates  not reachable — nothing for an adapter to align
    global rotation       DEAD, teacher operator 0.707 unexplained by the
                          FULL span of the learner's innovations
                          (`audit_population_span.py`)
    D  wake objective     what remains

Along the way, H29 corrected: capture over the learner's private
residuals is 0.095, but over the EFFECTIVE task-conditioned operator
(`audit_effective_operator.py`, I_tau = F_tau - F_0) it is 0.190 —
double. The residual tensor was the wrong computational unit. Still 5x
below the teacher's 1.000, so the honest reading is "a fifth of the
family geometry, half of it invisible in residuals alone".

V5's closing negative: POST-HOC REFACTORING CANNOT BE THE REMEDY. No
local alignment, global reparameterization, or sleep oracle over the
promoted library can recover structure outside the span of what wake
produced. Prospective pressure has to exist while the representation is
formed — a V6 question, and explicitly not established here.

Reviews saved as `reviewer-feedback-48.txt` (indexed) and 49 pending
index. Instrument lessons in `notes/learnings.txt` and `AGENTS.md`:
stale-report hazard, on-trajectory probing, invariance-before-search,
and the third instance of a target belonging to one arm of a comparison.

# 2026-08-21 — CODE REVIEW 55 DOCUMENTATION AND INTERPRETATION CORRECTION

Updated `notes/learnings.txt`, `PREDICTIONS.md`, and `paper/draft.md` to
record the conclusion-level consequences of review 55. The original
effective-operator value (`0.190`), population-span value (`0.707`
unexplained), replay `Phi = +0.072`, all old V6 arm comparisons, and the
old H32 reading are explicitly withdrawn.

The corrected H29 audits restore retirement state, reproduce the actual
model rollout, and evaluate all functions in common coordinates. They
reverse the localization result: `R_effective = 0.762`, full-population
unexplained variance `0.491`, best-single unexplained `0.695`, while the
promoted library remains at `0.921` unexplained and FACTORIZE remains
0/6. The paper now states the resulting interpretation: ordinary wake
encodes distributed family structure, but PROMOTE extracts the wrong
unit by promoting one private residual at a time.

V1-V4 and V5's primary causal economics remain unchanged. H30-H32 remain
UNRESOLVED pending completed, clean scoring from the repaired V6
pipeline. The repaired runner has been launched elsewhere in the shared
workspace, but completion or a new verdict is not claimed by this
documentation milestone.

# 2026-08-21 — SCIENTIFIC OPERATING STANDARD HARDENED

Expanded `AGENTS.md` and `CLAUDE.md` so future work treats ROW explicitly
as careful scientific research rather than rapid experimentation. Added
claim-status discipline; construct, instrument, comparison, holdout, and
coordinate-system validity rules; complete model-state reconstruction;
full-protocol fingerprint/resume requirements; and artifact acceptance and
quarantine rules. The top-level scientific standard is intentionally one
compact paragraph; detailed incidents remain in implementation learnings.

Resolved contradictory local concurrency guidance. The authoritative rule
is now one full-lifetime PyTorch/SciPy process at a time on this Windows
host, one writer per artifact cell, and no experiment launches from dirty
or changing code. `--jobs N` remains an interface requirement for isolated
remote workers, while local full lifetimes use `--jobs 1`.

Updated stale V5 pointers in both files: V5 and its confirmation plan are
closed, `V5_CLOSURE.md` records the review-55 corrections, and seeds
600-629 are completed confirmatory worlds rather than unopened reserves.

# 2026-08-21 — review 55: three conclusions retracted, ten defects fixed

An independent code review found conclusion-impacting bugs in audits
that had already produced reported results. Retractions were published
before fixes, and two headline V5 numbers changed.

## Retracted and corrected

    R_effective            0.190  ->  0.762
    full-span unexplained  0.707  ->  0.491
    all V6 arm comparisons        ->  void (future-label leak)

Three defects produced the first two: each task's innovation was
evaluated on ITS OWN inputs and then compared coordinate-by-coordinate,
so the subspace fit ran across incomparable axes; the loader restored
promoted references but not retirement state, reconstructing a model
that never existed; and the "on-trajectory" rollout ran only the routed
basis. The third came from `_sibling_of` drawing prospective siblings
from `world.tasks`, so the hook trained on a later task's labels before
that task legitimately arrived.

V5-D is withdrawn. What V5 actually found: the wake learner encodes ~76%
of the family structure, distributed across route and residual and
across the population rather than concentrated in any single object, and
PROMOTE captures almost none of it. The gap is between what the learner
COMPUTES and what the promoter EXTRACTS — the extraction unit, not the
wake objective. "Post-hoc refactoring cannot be the remedy" is
withdrawn; population-level refactoring is now what the evidence points
at, and reviews 48/49's hypotheses B and global-rotation are live again.

## Scope of damage, checked rather than assumed

Only V6 needs re-running: its bugs were in the LIFETIME. The audit bugs
read frozen artifacts, so V5 was re-scored without a new lifetime. V1-V4
untouched. Sealed C1 involves no audit. Sealed C2 verified bit-identical
after the `family_operators` API change (first F operators compared
across four seeds). C3/C4 already used one shared probe per world and
never had the coordinate bug.

## Fixes

Leak closed (siblings from `held_out_family_tasks`, zero overlap with
the lifetime verified); coordinates aligned in both audits; retirement
state restored; rollout matches `forward`; replay matched to ordinary
acquisition learning rates (route 0.05 / residual 0.01 / shared 0.003
rather than one group at 0.003); H31 given symmetric UNSEEN unrelated
futures; fertility scorer returns prequential acquisition cost, endpoint
and steps-to-target, defaulting to the registered gate; protocol knobs
fingerprinted and resume refuses a mismatched arm; compute accounting
covers `lifecycle` and `prospective`.

Two guards added that would have caught this class: a save/load
functional-equivalence test with a companion that verifies the guard can
FAIL, and the nested-objective non-vacuity tests. `check_invalid.py` was
itself vacuous on first write (missing `re.MULTILINE`) and now refuses
to pass when it parses nothing.

## State

`artifacts/v6*` deleted rather than resumed. The first valid V6 run is
executing: 12 cells, serialized, every fix applied. H30-H36 remain
untested — every previous prospective number came from an arm that was
not adapting, or was training on future labels, or both.

# 2026-08-21 — H35 LOW-PRESSURE CURVE COMPLETE: OPTIMUM NOT SUPPORTED

Completed the six registered low-pressure lifetimes (outer steps 1 and 2,
worlds 0--2) and scored them with the existing ordinary and eight-step anchors.
All H35 artifacts passed protocol, world-seed, completeness, and unfrozen-basis
validation before any checkpoint was loaded.

The corrected 40-step acquisition instrument gives Phi_related of -0.432,
-0.833, and -8.575 at pressures 1, 2, and 8. Sign counts are 1/3, 2/3, and
0/3 positive. Current-lifetime deltas are -46, +64, and +1,281 nats; the first
two are mixed across worlds, while pressure 8 is harmful in all three. Thus no
low-pressure cell is a robust benefit and H35's registered non-monotonic
optimum is not supported. The narrower high-pressure-harm result stands.

Added `score_v6_pressure.py` with fail-closed full-grid preflight, complete
protocol metadata, paired related/unrelated/within-family effects, explicit
Phi and lifetime-cost decisions, and atomic report output. During validation,
a 60-versus-40 adaptation-horizon mismatch was caught because the new scorer's
first result failed to reproduce the old anchor costs. The final report uses
40 steps and reproduces both ordinary and pressure-8 anchors exactly.

One separate partial-freeze allocation cell failed from a memory error during
the shared bounded-pool run. It is unrelated to H35, is not counted as an H35
failure, and remains resumable; no allocation conclusion is recorded here.

# 2026-08-21 — V6 ALLOCATION COMPLETE AND V6 CLOSED IN DEVELOPMENT

Preflight of the nominally 44/45 allocation grid found nine cells produced with
8 prospective inner steps rather than the allocation protocol's 16. Preserved
those artifacts under `artifacts/v6_alloc_superseded_inner8`, hardened the
bounded-pool launcher, and reran the nine mismatches plus the missing
`free6/ordinary/world_1` cell. The repaired run completed 10/10 with zero
failures; all 45 active cells pass full protocol validation. Ordinary/replay
reruns were bit-exact across the inert inner-step difference.

Added `score_v6_allocation.py`, which fail-closes on an incomplete or mismatched
grid and combines the corrected 40-step fertility scorer, aligned
effective-operator audit, matched-budget learned-schema audit, and lifetime
allocation statistics. The report is `reports/v6_allocation.json`.

The structural endpoint is a noisy frontier rather than a sharp boundary. From
free0 to free6, explicit library size decreases in all nine paired arm-world
comparisons while R_effective increases in all nine. Intermediate means are not
strictly monotone. Thus where plasticity is allowed influences whether recurring
information is absorbed by the continuous shared basis or emitted as explicit
promoted objects.

The frontier contains no fertile/schema regime. Prospective Phi is negative at
every plastic setting and negative in 3/3 worlds at free2, free3, and free6.
FACTORIZE wins 0/12 scoreable prospective cells and 0/34 scoreable cells overall.
Prospective current-loss disadvantage grows from +107 nats at free1 to +3,362
at free6. H33 is not supported, and the allocation result does not rescue H30.

V6 is closed in development without opening confirmatory seeds. H30, H31, H33,
and H35 are unsupported; H32's original threshold is non-diagnostic; H34 and
H36 were correctly gated off. The remaining question is localization:
representational opportunity versus adaptation findability.

# 2026-08-21 — H37 V6R LOCALIZES REPRESENTATIONAL OPPORTUNITY LOSS

Froze `V6R_ADAPTATION_GEOMETRY_PLAN.md` and its preregistration hash before
writing or running the audit. Added `audit_v6r_adaptation_geometry.py` with
exact historical-anchor reproduction, fixed high-budget Adam and LBFGS fits,
support-only restart selection, full frozen-representation reconstruction, and
machine-tested decision gates. The first invocation stopped before new fitting
because the anchor reader expected a richer report schema; it wrote no report.
After fixing only that reader, all 12 anchor values reproduced exactly.

The completed report is `reports/v6r_adaptation_geometry.json`. At k=128,
primary Adam 0.01 gives ordinary/prospective scaled query endpoints
0.01814/0.02976 (gap +0.01162, +64%). Every task and world gap is positive.
Adam 0.05 gives +0.01145 and LBFGS +0.01207; all three optimizers pass the
registered representational-opportunity-loss gate, and every fit is finite.
The primary k=1 gap is +0.03426 and support-selected restarts preserve it.

H37 therefore resolves the registered fork toward lost representational
opportunity under the allowed task-local interface, not optimizer-only
findability. This remains an operational finite-budget result, not proof of a
global optimum. The earned successor is an existence/use test of an explicit
shared-schema, fast-argument, private-innovation factorization.

# 2026-08-21 — REVIEW 58 INTEGRATED: H38-H46 REGISTERED AS THE FERTILITY DECISION TREE

Saved review 58 as `reviews/reviewer-feedback-58.txt` and indexed it along
with the previously unindexed reviews 56 and 57 (plus a numbering note
recording that 54/55 have no preserved feedback files — review 55's content
lives in PREDICTIONS.md and the learnings). Updated the review-index header,
which still described the arc as ending "into V3 design."

Registered review 58's hypothesis tree H38-H46 in `PREDICTIONS.md` with the
reviewer's per-hypothesis predictions attached, the binding discipline rule
(earn each component separately; H37 chooses the branch), and two honest
status annotations where the review's writing predates data this project
already holds: H43's directional claim is supported by the completed
allocation sweep while its intermediate-fertile-regime hope is not, and H37
itself has already resolved to the representational-opportunity-loss branch
(`reports/v6r_adaptation_geometry.json`). This falsifies the reviewer's core
prediction that high-budget ordinary/prospective endpoints would become equal,
but not the narrower expectation of partial shrinkage: primary Adam reduced
the k=1 endpoint gap by 88%, while a replicated residual and the k=128 gap
remained. The active branch is therefore H39 -> H40 -> H41 -> H44;
H38/H42/H46 remain registered as the findability research line but are not
licensed as the next mechanism.

Recorded the durable lessons in `notes/learnings.txt`: acquisition cost
decomposes as C_express + C_find; a good abstraction compresses what is
common while providing cheap coordinates for what can vary; and each
component of SHARED SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION + LEARNED
UPDATER must be earned by its own hypothesis rather than built as a unit.

# 2026-08-21 — REVIEW 59 INTEGRATED: REPRESENTATION BRANCH CONFIRMED, H39 CHECKLIST REGISTERED

Saved review 59 as `reviews/reviewer-feedback-59.txt` and indexed it. The
review reads V6R as closing the optimizer-only rescue and deprioritizes learned
optimizers for the main ROW line; the earned next mechanism is the oracle-form
SHARED SCHEMA + FAST ARGUMENT + PRIVATE INNOVATION architecture on the V6
worlds.

Its H39-H42 labels collide with the review-58 numbering already in
`PREDICTIONS.md`. Kept the registered numbers and wrote the mapping into a new
REVIEW 59 ledger entry (review-59 H40 is H39's matched-capacity control;
review-59 H41 is H41; review-59 H42 is registered H40). Registered the
reviewer's eight-item ordered checklist, with items 1-5 fixed as the H39 gate
and 6-8 assigned to H41/H40/H44, and recorded the proposed three-term
abstraction criterion as a candidate without an instrument. Learnings appended.

No code, artifacts, or runs. Next step: freeze an H39 plan (architecture,
matched-budget generic-code control, V6R k=128 instrument reuse, gate
thresholds) before writing learner code.

# 2026-08-21 — H39 PLAN FROZEN; CENSUS C0 NEGATIVE 3/3; H39 NOT RUN

Froze `H39_EXISTENCE_PLAN.md` (`b6fc27d`) and registered it in
`check_prereg.py`. Before writing any code, re-reading found that the census
gate C0 fitted its schema leave-one-family-out, a different question from the
one the factorized arm faces; appended Amendment 1 (`16906ff`) and re-froze.

Wrote `census_h39_schema.py` with tests (`63a8e9e`): within one ordinary
artifact, PCA the live family residual vectors plus promoted abstractions into
a rank-8 linear schema, then fit only a fresh route code and an 8-dim alpha on
each held-out sibling at k=128 under protocol B1 with the representation
frozen and the private residual zero. Report written atomically;
`reports/h39_census.json`.

Result: alpha-only endpoints 0.07860 / 0.03452 / 0.05537 against ordinary
V6R anchors 0.02226 / 0.01539 / 0.01675, ratios 3.53 / 2.24 / 3.31 versus the
registered 1.5 threshold. 0/3 worlds pass; per the frozen verdict table H39 is
NOT RUN. No factorized or control lifetime was launched. Fits are non-vacuous
(3-10x reduction from k=0, converged by 1,000 updates).

Exploratory calibration (separate, non-registered reports): rank 16 and the
maximum available rank (the population's full affine span, 26/12/34 dims) give
ratios 2.2-3.4, so the shortfall is span, not rank. The ordinary lifetimes
retired 39/64, 56/64, 32/64 family residuals, leaving 13-35 vectors with a
flat spectrum in two worlds.

Recorded the verdict and exploratory finding in `PREDICTIONS.md` and the
lessons in `notes/learnings.txt`. Next: a successor plan must specify how a
schema is obtained other than post hoc from the final residual population;
nothing is licensed until it is frozen.

# 2026-08-21 — REVIEW 60 FILED; H39 WORLD-0 JOINT-FORMATION PILOT FROZEN AND LAUNCHED

Filed review 60 (`reviews/reviewer-feedback-60.txt`, indexed). Froze
`H39_PILOT_PLAN.md` (`effaf6b`) and registered it in `check_prereg.py`. Two
amendments were appended and re-frozen before any pilot artifact was read:

- Amendment 1 (`189b4cb`): "exact-null eps at birth" is unrealizable — a
  scratch smoke showed alpha and eps norms exactly 0.0 and identical loss for
  a=2 and a=8, because `u.tanh(vz+b)` is stationary at zero and
  `dL/dalpha = W^T dL/dr` vanishes there. eps keeps the ordinary 1e-3 init.
- Amendment 2 (`99e3c73`): `novel_family_tasks` are members of two families
  the lifetime NEVER sees, all family operators sharing one rank-2 subspace
  at r_meta=1; the per-family "grouped" arm is ill-posed and withdrawn. The
  primary arm is pooled a=2 (the oracle FORM); pooled a=8 is secondary. A
  correction about the word "sibling" was appended to `PREDICTIONS.md`; no
  census number changes.

Implemented `FactorizedLifecycleLearner` (`residual = W alpha + eps`, all
residual readers routed through a new base-class `effective_residual`
accessor), a `factorized` model kind with complete provenance records and
resume refusal, a read-only history snapshot hook, unit tests including a
save/load functional-equivalence guard with a failing companion, a
bounded-pool launcher, and `score_h39_pilot.py`. Full suite: 169 tests OK.

The ordinary history rerun reproduced the V6 world-0 loss exactly
(-191,906.786). The first grouped cell died at its first sleep with a
transient 3.66 MiB Windows allocation failure (no partial output). The a=8
pooled cell completed (-191,446.8). The primary a=2 cell is running; no
pilot endpoint has been read.

# 2026-08-21 — H39 PILOT SCORED: BRANCH C; LINEAR SCHEMA NOT FERTILE

The primary a=2 pooled cell completed (-191,604.5). The first scorer run was
refused by the registered non-vacuity check: alpha norm 0.0 in every
alpha-only fit because `eps = 0` is a stationary point; Amendment 3 freezes
eps at the shared 1e-3 init instead, re-frozen before re-scoring; the refused
report is preserved as `reports/h39_pilot_discarded_stationary_protocol.json`.

`reports/h39_pilot.json`: ordinary rerun bit-exact; anchor reproduced to
1e-12; parity +302 nats; alpha-only k=128 4.18x ordinary (three optimizers
agree), full fit 1.06x; channel ablation shows the residual channel carries
~2% of live family-task fit with 56/64 tasks retired; historical span 2.75x
at full rank ("ordinary wake never formed those directions"). Verdict per the
frozen table: **branch C**. Nothing licensed. Recorded in `PREDICTIONS.md`
with reviewer-prediction scorekeeping and in `notes/learnings.txt`.

# 2026-08-21 — H39b P(alpha)-IN-BASIS PILOT: BRANCH U (THRESHOLD MISCALIBRATED); CHANNEL IS USED

Froze `H39B_PSLOT_PILOT_PLAN.md` (`83ac418`), implemented
`ParameterizedSlotLearner` (basis slot 12 becomes P(alpha) with U_0 + sum
alpha_k U_k; frozen-alpha control bitwise equal to ordinary, verified on the
smoke world and on the real artifact), wired `--model pslot`, tests (full
suite OK), launched P2 / P8 / P2-frozen in a pool of three, and scored with
`score_h39b_pslot.py`.

Results: P2 -507 nats vs ordinary, alpha-only k=128 3.94x, full fit 0.94x,
alpha-zeroed ratio 1.38, route mass on P 0.093; P8 -1,028 nats, alpha-only
1.89x (three optimizers agree), full fit 0.79x, alpha-zeroed 2.68, route mass
0.169. Frozen verdict U via the route-mass criterion, which the ledger entry
annotates as miscalibrated (ordinary routes are diffuse; any one slot ~0.08-
0.10). Nothing licensed. Recorded in `PREDICTIONS.md` and `notes/learnings.txt`.

# 2026-08-21 — H39c K-SWEEP FROZEN AND LAUNCHED (WORLDS 0-2)

Froze `H39C_KSWEEP_PLAN.md` (`1e99904`) with channel-use thresholds
registered RELATIVE to the measured ordinary slot-12 route mass per world
(0.0971 / 0.1147 / 0.1168; threshold 2x), K in {2, 4, 8, 16}, and a matched-
budget control G_8 (U_k frozen at init, alpha learns). Added the
`--freeze-matrices` control with a unit test, sweep cells to the bounded-pool
launcher (15 cells, 3 workers), and `score_h39c_ksweep.py` (fail-closed
non-vacuity per cell, verdicts A / A-capacity / P / B / NOT COMPARABLE). The
sweep is running; no cell has been read.

# 2026-08-21 — H39c K-SWEEP SCORED: VERDICT P

All 15 cells completed (zero failures) and scored with `score_h39c_ksweep.py`
(`reports/h39c_ksweep.json`). Alpha-only k=128 ratios fall monotonically with
K in all three worlds (K=16: 1.75 / 1.58 / 1.35); the frozen-direction control
G_8 sits at 3.3 / 2.8 / 3.2 with alpha-zeroed ratios ~1.05; present-task loss
improves with K everywhere (K=16: -1,119 / -708 / -844 nats); the full
interface beats ordinary in 13/15 cells. No K reaches 1.5x in two worlds, so
the frozen verdict is P: real, learned, capacity-limited at K <= 16. Licenses
a larger-K / multi-slot development plan only. Recorded in `PREDICTIONS.md`
and `notes/learnings.txt`.

# 2026-08-21 — H39d CAPACITY SWEEP FROZEN AND LAUNCHED

Froze `H39D_CAPACITY_PLAN.md` (`c6b1b7b` plan commit; hash registered):
arms P32, P64 (one slot) and M2K16, M2K32 (two parameterized slots, matched
total arguments) on worlds 0-2; usage criterion is the functional alpha-
zeroed ablation with route mass reported only; verdicts A / P+ / S against
H39c's K=16 points. Generalized `ParameterizedSlotLearner` to `pslot_count`
slots with the single-slot state-dict layout unchanged (tests: two-slot at
alpha=0 equals ordinary bitwise; second slot gets gradient), added
`--pslot-count`, launcher cells, and `score_h39d_capacity.py`. Full suite OK.
12 cells running; none read.

# 2026-08-21 — H39d SCORED: VERDICT A (MARGINAL) VIA TWO-SLOT K=32

12/12 cells complete, zero failures. `score_h39d_capacity.py` first crashed
serializing a two-slot alpha (no report written; flatten fix, no numerics
changed) and then completed: `reports/h39d_capacity.json`. Single-slot K =
32 / 64 are flat against K = 16 (means 1.59 / 1.57 vs 1.56); two slots at
K = 32 reach 1.27 / 1.74 / 1.36 and pass the fertile rule in worlds 0 and
2. Frozen verdict A, recorded with explicit marginality caveats in
`PREDICTIONS.md`; it licenses only the writing of a confirmation plan on
seeds 700-729, which are not opened. Learnings appended.

# 2026-08-21 — H39 CONFIRMATION BLOCK OPENED; REVIEW 61 AMENDMENT; DEVELOPMENT CLOSURE

Froze `H39_CONFIRMATION_PLAN.md` after PI approval (`1c98017`), wrote
`tools/run_h39_confirmation.py` (prereg check enforced at launch, pool of
3, one writer per cell, world-major so triples complete together) and
launched the 90-lifetime block on seeds 700-729. Review 61 arrived after
launch asking for one decision-rule change (E2 as the geometric mean of R_M
in [1.2, 1.8]; the 1.5x fraction as a C0 continuity statistic only; scope
sentence "existence and use, not discovery"); appended as Amendment 1 and
re-frozen (`f78f209`) before any sealed cell was read, and implemented in
`score_h39_confirmation.py` before any sealed number existed. Filed and
indexed review 61; registered the block opening, the final rule, and our
own prediction (CONFIRMED-RELATIVE more likely than CONFIRMED) in
`PREDICTIONS.md`. Appended the H39 re-audit to `SPEC_AUDIT.md` and wrote
`H39_DEVELOPMENT_CLOSURE.md` with the ladder, what was and was not
established, the methodological record, and reviewer scorekeeping. The
block is running; no sealed cell has been opened.

# 2026-08-21 — H39 SEALED BLOCK SCORED: CONFIRMED

The 90-lifetime block on seeds 700-729 completed with zero failures;
`score_h39_confirmation.py` validated every record, passed every non-
vacuity check, and read CONFIRMED from the amended table: E1 0.886 [0.717,
1.054]; E2 geomean 1.636 [1.495, 1.7997] vs ceiling 1.8; E3 -1,545 [-1,649,
-1,435]; E4 0.815 [0.764, 0.866]; E5 30/30. Recorded in `PREDICTIONS.md`
with the 0.0003 E2 margin and the fact that the pre-amendment rule
(fraction 0.433) would have read CONFIRMED-RELATIVE; appended the sealed
verdict to `H39_DEVELOPMENT_CLOSURE.md`; learnings appended. Nothing further
is licensed by this entry; the next rung (discovery) needs its own plan.

# 2026-08-22 — REVIEW 62 FILED; DISCOVERY REGISTERED (H47-H49); PRE-DESIGN AUDIT

Filed and indexed review 62. Registered membership / cardinality /
representation-type discovery as H47-H49 with the reviewer's predictions
and the binding "economics before ontology" safeguard. Answered three of
the six attack points on the sealed report from the frozen-direction
control already in it: E3 is not capacity (G +53 nats vs M -1,545); E4
survives a matched budget only in part (G 0.917 vs M 0.815; learned share
~0.10); no oracle grouping was used (architecture supplied only). Noted
that M already soft-discovers membership, so H47 must be framed against a
labelled-routing oracle. Two cheap audits identified for the remaining
points (census on M's matrices; zero-alpha-then-refit). No plan drafted,
no run launched.

# 2026-08-22 — REVIEW 63 FILED; NEXT-STEPS PLAN AMENDED TWICE; STAGE A RUNNING

Filed and indexed review 63. Appended Amendment 1 to
`H39_NEXT_STEPS_PLAN.md` (A1 claim narrowed to the extensional population;
A2 robustness band with a second optimizer; L redefined as an exact route-
mask oracle verified at correct-slot fraction 1.000; H_late control;
relative tolerances; Stage C rebuilt census-first around the economic K*
with K_max = 8 and the full marginal D* charge) and re-froze. Wrote
`audit_h39_confirmation_followups.py`. A one-world sampled smoke exposed a
refit-budget confound in A2 (ratio 0.58 against the intact model);
Amendment 2 registered a matched-budget denominator (alpha-zeroed refit /
alpha-free refit) before the registered run and re-froze. Re-smoke: A2
2.08 on two tasks. The registered 30-world run is in progress; no Stage A
result has been read.

# 2026-08-22 — STAGE A COMPLETE: A1 PASS, A2 PASS

The registered 30-world audit finished (`reports/h39_confirmation_followups.json`,
`tools/h39_followups.log`). A1: census on O 2.59x and on M 6.33x against
online M 1.64x — both exceed the +0.5 margin; the narrowed "extensional
population" sentence stands. A2: matched-budget compensated ratio 2.01
(1.69-2.54), no robustness band entered. Recorded in `PREDICTIONS.md`
(including one missed diagnostic prediction), appended to the closure,
learnings added. Stage A gate satisfied; Stage B may be frozen next.

# 2026-08-22 — REVIEW 64 FILED; H47 BASELINES MEASURED; H47 REDRAFTED (NOT FROZEN)

Filed and indexed review 64. Wrote `audit_h47_baselines.py` and measured
M's routing on worlds 0-2 (`reports/h47_baselines.json`): conditional
entropy over the two parameterized slots 0.92-0.95 bits of 1.0, ARI ~ 0 —
M uses both slots as one distributed argument channel and never commits.
Re-reading `meta_world.family_operators` confirmed that at r_meta = 1 all
families share one rank-2 subspace, so the planned slot "membership" was
an arbitrary partition. Registered the finding in `PREDICTIONS.md`,
redrafted `H47_MEMBERSHIP_PLAN.md` as B1 (discreteness and premature
commitment on the current world: M, L_arbitrary, H_early, H_late, with
relative tolerances from the measured spreads) and B2 (true membership on
a two-subspace world via a `schema_groups` generator extension with a
bitwise G = 1 check). Not frozen; awaiting review.

# 2026-08-22 — REVIEW 65 (SYNTHESIS) FILED AND PROPAGATED

Filed the post-Stage-A synthesis verbatim as review 65, indexed it,
annotated its one superseded line (M does not discover membership; the
confirmed world has a single family subspace), and propagated the durable
content to `AGENTS.md` (the H39 central lesson and the reformulated
thesis) and `notes/learnings.txt`.

# 2026-08-22 — REVIEW 66 FILED; H47 B1 FROZEN, BUILT, AND LAUNCHED

Filed and indexed review 66 (H39 reread as one continuous parameterized
manifold; B1 reframed as the cost of imposing discrete commitment on a
continuous family; B2 generator tests; H48 split into schema count and
within-schema dimensionality). Appended Amendment 1 to
`H47_MEMBERSHIP_PLAN.md` freezing B1 (arms M / L_arb / H_early / H_late,
baseline-relative tolerances, the four-way result matrix, registered
predictions) and registered its hash. Implemented route policies on the
two-slot learner (per-task hard mask over the parameterized slots; global
temperature on their conditional; policies off are bitwise M; tests), the
runner hooks and provenance records, loader tolerance for pre-policy
artifacts, launcher cells, and `score_h47_b1.py`. Full suite OK. The nine
B1 lifetimes are running; none read.

# 2026-08-22 — H47 B1 SCORED: MIXED; ORDERING HOLDS IN SIGN BELOW TOLERANCE

9/9 B1 cells completed; `score_h47_b1.py` validated records and
non-vacuity (mask 64/64; H arms at 0.004-0.011 bits, pair comparable) and
read MIXED: L_arb a COST on J (36-69% of the M-O gain) but NEUTRAL on
R_alpha (+0.135 mean, tolerance 0.15); H_early and H_late NEUTRAL on all
three with the predicted ordering M < H_late < H_early in every world.
Recorded in `PREDICTIONS.md` with the tolerance-calibration note and
scorekeeping; learnings added. Next: B2 generator extension
(`schema_groups`) with its bitwise G = 1 test and functional-separation
audit, then B2's own plan.

# 2026-08-23 — REVIEW 67 FILED; B2 GENERATOR BUILT AND VERIFIED; GATES FROZEN; OPPORTUNITY GATE RUNNING

Filed review 67. Implemented `MetaFamilySpec(schema_groups=G)` with group 0
on the original seeds and later groups projected out of earlier spans;
G = 1 reproduces the previous generator bitwise on seeds 0, 1, 2, 700
(sha256 over all task arrays, futures, and operators). Unit tests added.
Wired `--schema-groups`, the `mask_group` route policy, and provenance.
Froze Amendment 2 (teacher gates G2-G5; the opportunity gate stated as the
membership tax itself; staged arms). The first teacher-gate run was
refused: G4's 10% per-family bound fails on the baseline G = 1 world, and
G3's mixture search covered only an affine line; Amendment 3 fixed both
(group-level balance with the baseline spread as covariate; least-squares
span fits normalized by the family step's contribution) and re-froze.
Gates then passed 3/3 (`reports/h47_b2_world_gates.json`), with the note
that G2/G3/G5 are near-tautological at r_meta = 1 and the economic gate
is the decisive one. M_G2 and L_true are running on worlds 0-2; the gate
scorer `score_h47_b2_gate.py` is committed. No B2 endpoint has been read.

# 2026-08-23 — B2 OPPORTUNITY GATE: NO-OPPORTUNITY; H ARMS NOT LAUNCHED

6/6 gate cells completed; `score_h47_b2_gate.py` validated the G = 2
records and non-vacuity and read the membership tax at +0.271 / +0.099 /
-0.027 log units (gate >= 0.15 in >= 2/3 worlds: 1/3). Verdict
NO-OPPORTUNITY; per the plan H_early / H_late are not launched. M_G2
routes the two orthogonal groups as one distributed channel (entropy
~0.93, ARI ~0), beats the oracle mask on present cost by ~500 nats in
every world, and trails it slightly on the full interface. Recorded in
`PREDICTIONS.md` with scorekeeping (both prediction sets wrong in the
same direction) and the registered follow-up (restrict K so that discrete
identity becomes economically necessary); learnings appended.

# 2026-08-23 — REVIEW 68 FILED; OPPORTUNITY GATE MADE CONSTITUTIONAL; H48b WIDTH SWEEP FROZEN AND LAUNCHED

Filed review 68 and elevated its opportunity-gate rule into AGENTS.md.
Froze `H48B_WIDTH_SWEEP_PLAN.md` (K in {2,4,8,16,32} on the G = 2 world,
M_K versus L_true,K only; rules alpha/present/full pay; outcomes
CROSSOVER / CAPACITY NOT BINDING / INNOVATION BUFFERS / FULLY LICENSED;
our prediction: outcome 3 at K = 4, possibly 4 at K = 2). Added launcher
cells and `score_h48b_width.py`. 24 lifetimes running; none read.

# 2026-08-23 — H48b SWEEP INTERRUPTED BY SESSION EXIT; RELAUNCHED IDEMPOTENTLY

The Claude Code session that launched the width sweep exited; its
background pool died with it (the learnings' silent-death case). State on
inspection: 6/24 cells complete (K = 2, 4, 8, both arms, world 0), three
cells mid-flight with no partial output, no Python processes running.
Relaunched `python tools/run_h39_pilot.py width`, which skips completed
cells and refuses mismatched records; the killed cells restart from
scratch. No cell has been read.

# 2026-08-23 — H48b SCORED: MIXED; FUTURE CHANNELS PAY FOR IDENTITY AT K <= 8

30/30 cells validated and scored (`reports/h48b_width.json`). Alpha pays
at K = 2, 4, 8 and full pays at K = 2, 4, 8, 32; present never pays
(oracle cost falls from ~-510 to ~-40 nats but crosses zero in one world
only); K* undefined; outcome MIXED per the frozen table. M_K sharpens its
routing at small K without aligning to the groups (ARI ~0). Recorded in
`PREDICTIONS.md` with scorekeeping and the registered next plan
(discovery arms at K = 4 with a present-tolerance band, frozen before
any run); learnings appended.

# 2026-08-23 — REVIEW 69 FILED; H49 DISCOVERABILITY CENSUS FROZEN AND RUNNING

Filed review 69 (do not anneal M at K = 4: the grouping has future value
but no present value, so non-discovery is rational; first find a
retrospective signal). Froze `H49_DISCOVERABILITY_PLAN.md`: on the
existing K = 4 artifacts (M_4 label-free, L_4 told), six candidate
partitions applied as re-fit policies; C_LOO, a two-part D* proxy, and
substitutability from experienced tasks only; outcomes A DISCOVERABLE /
B UNDERDETERMINED / C SIGNAL NEEDS ORGANIZATION; our prediction C. Wrote
`audit_h49_discoverability.py` (mechanics smoke-tested); the ~2,300-refit
census is running. Nothing read.
