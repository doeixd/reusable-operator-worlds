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
