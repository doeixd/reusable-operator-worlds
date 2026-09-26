# Audit scope

This audit maps `neural_library_learning_v1_experimental_spec.md` to the current
implementation and evidence as of commit `7e66639`. Confirmatory worlds 100–129
remain sealed. “Complete” means implemented and checked on the stated
development scope; it does not turn a development result into a confirmatory
claim.

# Complete core benchmarks

- Deterministic random residual teacher primitives, 64 unique length-three
  programs, opaque randomized task IDs, and fixed train/evaluation arrays are
  implemented.
- Exact reuse and the six-point reuse continuum are implemented with measured
  functional recurrence. The continuum is complete on development worlds 0–9.
- Online examples are scored before update. Paired models receive identical
  worlds, task orders, examples, replay policy, and evaluation arrays.
- Scratch difficulty, uniform output scale, and the leak-free oracle positive
  control pass.
- Dense-P, Dense-C, Continuous, hard Discrete, and a generic low-rank
  hypernetwork control are implemented. Dense-24 closes the task-state-size
  sensitivity.
- Cumulative prequential Gaussian log loss, examples-to-criterion, frozen-library
  novel composition, functional matching, route recovery, specialization, and
  forward multiply-add accounting are implemented.
- The required exact-reuse and reuse-sweep figures exist and have been visually
  inspected.

# Complete development conclusions

- Continuous beats Dense-C on exact-reuse lifetime loss across all ten
  development worlds and becomes progressively better at fresh-composition
  adaptation over the lifetime.
- The causal effect reverses consistently: Dense-C wins through configured
  `rho=0.75`; Continuous wins at `rho=0.9` and `rho=1.0` in all ten worlds.
- The generic hypernetwork beats Dense-C but loses to Continuous on lifetime loss
  in every development world. This separates the benefit of a continuous
  operator manifold from the additional benefit of an explicit reusable slot
  basis.
- Hard Discrete learns good operators and routes but pays high online route
  inference cost. Per-task annealing reduces, but does not close, that gap.
- Alpha leakage, teacher-rank mismatch, activation mismatch, fixed identity, and
  dense task-code dimension have been investigated.

# Protocol validity checks complete

- Opaque task-ID reassignment is exactly invariant for selected Continuous and
  Dense-C runs.
- The effective batch-size discrepancy is explicitly ablated on worlds 0–2.
  Batch 8 improves both models and preserves Continuous's advantage, while
  narrowing it because Dense-C benefits more. The confirmation batch and any
  retuning remain a freeze decision rather than an untested obligation.

# Open diagnostic obligations

- Add world-level median and bootstrap intervals for the ten-world reports and a
  task-level model-by-lifetime analysis for examples-to-criterion. Existing
  three-world intervals are only exploratory.

# Open model investigation

- The conditional MDL decision is resolved: hard Discrete works well enough in
  final operator/route recovery to justify the basis-pruning question. Model 4
  is implemented; its penalty tuning and exact-reuse evaluation remain open.

# Explicitly outside V1

- Prospective macro regimes are reserved for Benchmark C after V1.
- Dynamic fork/merge/delete, self-refactoring, prospective abstraction, and the
  full train/inference/peak-capacity objective are Phase II.
- The strong self-refactoring hypothesis is explicitly not required for V1.

# Confirmation gate status

The gate remains closed. Oracle transfer, high-reuse advantage, reuse dependence,
novel-composition improvement, broad capacity controls, learned-model task-order
robustness, replay/forgetting separation, and the two-initialization pilot pass.
A clean checkout artifact-to-report rehearsal is still required by
`EXPERIMENT_PLAN.md` before opening worlds 100–129, and the remaining diagnostic
obligations below should be resolved or explicitly deferred.

# Execution order

1. Freeze batch protocol, tuning, and statistical summaries.
2. Tune and evaluate the MDL presence-gated library.
3. Run a clean artifact/plotting rehearsal.
4. Re-audit the gate before touching confirmatory worlds.

# Closed during this audit

- Current learnable-alpha int8 evaluation is complete. Continuous,
  Hypernetwork, Dense-24, and Dense-C were behaviorally evaluated on all ten
  development worlds, and current per-task Discrete was checked on world 0. The
  tracked result is `reports/retention/current-retention.json`.
- Reverse-order and replay-ratio robustness are complete on worlds 0–9.
  Continuous wins lifetime loss in all paired worlds at replay ratios 0, 1, and
  4 and under reverse order. The tracked result is
  `reports/robustness/robustness.json`.
- The two-initialization pilot is complete. Both model-seed pairs reproduce
  lifetime loss and novel-transfer wins in all ten development worlds. The
  tracked result is
  `reports/model_initializations/model-initializations.json`.
- Same-architecture fresh-task forward transfer is complete on worlds 0–9.
  Both models benefit from lifetime experience, but Continuous acquires more
  transfer in every world and the gain rises with task index. The tracked result
  is `reports/forward_transfer/forward-transfer.json`.
- Checkpoint true-route operator analysis is complete on worlds 0–9 for
  Continuous and per-task-annealed Discrete. Primitive matching and program
  performance through matched slots improve from 8 to 64 tasks in every world;
  Continuous's learned mixtures also reveal that a forced one-slot teacher route
  is not an oracle upper bound. The tracked result is
  `reports/operator_checkpoints/operator-checkpoints.json`.
- Explicit scrambled-ID invariance passes exactly for selected Continuous and
  Dense-C world-0 runs. Reassigned IDs are disjoint, while normalized metric
  rows, summaries, and final tensors are identical. The tracked result is
  `reports/scrambled_ids/scrambled-ids.json`.
- The batch-size sensitivity is complete on development worlds 0–2. A paired
  target batch 8 improves both models and preserves Continuous lifetime and
  novel-composition wins 3/3, although Dense-C's larger improvement narrows the
  architecture effect. The tracked result is
  `reports/batch_sizes/batch-sizes.json`.
- The free measured-recurrence and truncated-lifetime bridge analyses are
  complete across all ten development worlds. The reuse threshold drops from
  16 to 32 tasks and then plateaus; measured recurrence smooths the mean effect
  curve but does not tighten per-world crossover alignment. The tracked result
  is `reports/rho_bridge/rho-bridge.json`.
- The shared-parent plus rank-two task residual control is complete on worlds
  0–2 across rho 0.5, 0.75, 0.9, and 1.0. It beats the fixed-model envelope at
  every intermediate-reuse point, then yields slightly to Continuous at exact
  reuse as its functional residual ratio approaches zero. The tracked result is
  `reports/shared_residual/shared-residual.json`.

# Re-audit at cf804aa (2026-08-18)

Everything listed as open in the original audit (at `7e66639`) is now
closed; this section supersedes the gate status above.

## Closed since the original audit

- Reverse order and replay 0/1/4: complete on all ten development worlds,
  Continuous 10/10 under every condition (`reports/robustness/`).
- Second model initialization: complete, 10/10 on both metrics
  (`reports/model_initializations/`).
- Explicit forward transfer and checkpoint true-route diagnostics:
  complete (`reports/forward_transfer/`, `reports/operator_checkpoints/`).
- Scrambled-ID invariance: complete, bit-exact (`reports/scrambled_ids/`).
- Batch-size deviation: ablated; advantage survives at batch 8 with ~40%
  shrinkage (`reports/batch_sizes/`).
- Shared-parent + residual: complete, with two-part-code accounting
  (`reports/shared_residual/`, including `j-weighted.json`).
- MDL presence gating (Model 4): explicit decision recorded — does not
  work as a compact-sufficient-library discoverer at this scale;
  characterized negative (`reports/mdl_gating/`).
- Clean-checkout rehearsal: passed at `ed90ee2` (fresh-venv dependency
  install still unexercised; noted in `CONFIRMATION_PLAN.md`).
- Statistical freeze and confirmation: `CONFIRMATION_PLAN.md` frozen at
  `e0b0552`; seeds 100-129 run (360 lifetimes, zero failures, zero
  exclusions); all three pre-specified primaries passed 30/30
  (`reports/confirmatory/`). **The V1 gate is closed and confirmed.**

## Governing documents now

- V1 spec, `EXPERIMENT_PLAN.md`, `CONFIRMATION_PLAN.md`: frozen history.
- `row_v2_experimental_spec.md`: the active spec (provisional header
  retired in effect by V1 confirmation; STATUS annotations are the live
  state). `RELEASE_PLAN.md` governs publication.

## V2 position at this re-audit

- Bridge analyses B1/B2/B4 done (`reports/rho_bridge/`,
  checkpoint sweep); B3 done via step 001.
- Step 001 (Model 7a exact posterior): done; H7 strongly supported at the
  advantaged bound on world 0 (`reports/v2_route_posterior/`).
- Step 002 (GELU crossover shift, H6) and 002b (hypernetwork at rho 0.9):
  runs in flight (`artifacts/v2_gelu_crossover/`,
  `artifacts/v2_hyper_rho09/`).
- Next after those: 003 Model 8 consolidation with the pre-registered
  gate shape prediction; then Benchmark D.
- Paper: draft v0.5 with verified references (`paper/draft.md`), seven
  figures regenerable from `paper/make_figures.py`.

# Re-audit at V2 closure (2026-08-19)

- V2 implementation order: steps 001-009 all executed with written
  outcomes (001 H7; 002 H6; 002b manifold corollary; 003 Model 8 gates
  v1/v2; 004 Benchmark D with passing gates; 005 H9a; 006 H10 with
  dream falsifier; 006b mechanism falsification; 007 lifetime-length
  stationarity; 008 Benchmark E with the promotion post-hoc negative;
  009 sealed block, both components, all six outcomes).
- Frozen documents unchanged (tools/check_prereg.py green throughout).
- Sealed artifacts archived off-machine (GitHub release
  v2.0-confirmation); summary reports committed per the durability rule.
- Non-gating pre-registered analyses outstanding by design: 006c
  (functional-equivalence entropy) and 006d (function-family
  dimensionality) — inputs to the V3 spec, not V2 obligations.
- Governing documents now: V2 spec closed (section 12); V3 proceeds
  from section 9.5 plus notes/v3-sketch.txt revision 2, with its own
  spec to be written before any V3 run; sealed seeds 300-329 reserved.

# V2 completion audit (item-by-item, 2026-08-19)

Every closed item's artifact verified present on disk; prereg checker
green (4 frozen files unchanged, all STATUS paths exist).

## Hypotheses — all ten have written verdicts with artifacts

| Item | Verdict | Evidence |
|---|---|---|
| H5a | Partial: early movement then stationary; settled with genuine 32/64/128-task lifetimes | reports/rho_bridge, reports/v2_lifetime_length.json |
| H5b | Half: linearizes (R^2 0.97 dev, 0.93/0.926 sealed x2), dispersion not tightened | reports/rho_bridge |
| H6 | Supported-at-boundary; ~additive penalty 2.0-3.4k nats; parity at rho=1 | artifacts/v2_gelu_crossover |
| H7 | Strongly supported at the advantaged bound (world 0 only — extension to worlds 1-2 optional, noted in STATUS) | reports/v2_route_posterior |
| H8 | Characterized negative for the gate family; H8b passes under gate v2; re-derivation budget spent and respected | artifacts/v2_consolidation, _gate2 |
| H9a | Supported 10/10 development AND 30/30 sealed | reports/v2_mixed, reports/v2_confirmatory_mixed |
| H9b | Supported prequentially both settings; reversed under two-part code both settings (as pre-registered for sealed) | reports/shared_residual (incl. j-weighted), sealed Family B |
| H10 | Not supported at 64 tasks; dream falsifier passes (3x ratio) | artifacts/v2_compiler |
| 002b | Confirmed, stronger than predicted | artifacts/v2_hyper_rho09 |
| 006b | Falsified, pre-sealed timestamp | reports/v2_mechanism |

## Implementation order — all eleven steps executed

B1-B4 bridges done (B3 entropy curves live inside the route-posterior
report; B4 is figure 3). Steps 001-009 all closed as recorded in section
12; step 009's sequencing rule (Component A analyzed before B ran) and
durability rule (summaries committed; 343 MB archive at release
v2.0-confirmation) both complied with.

## Deviations and open items, honestly enumerated

1. **7b/7c naming:** the implemented compiler variant warm-starts the
   code and continues online gradient descent — i.e., the 7c HYBRID. A
   pure one-shot 7b (no refinement) was never run separately; since the
   hybrid already loses to plain gradient descent and the pure variant
   has strictly less optimization, it is dominated a fortiori and the
   H10 verdict is unaffected. Recorded for precision.
2. **Constraint 6.3 (family-mismatch variants for every V2 model
   family) was not applied to Models 5, 7, or 8.** Moot for 7/8 (both
   negative), but Model 5's H9 results carry no mismatch control. OPEN
   ITEM: run a GELU shared-residual variant before the V2 paper claims
   H9 unconditionally, or state H9 under the same alignment condition
   as V1's claims (recommended default).
3. **Constraint 6.5 (batch forward_tasks before any Benchmark D sweep)
   was never implemented**; all sweeps ran the per-sample loop. Impact
   is wall-clock only — no correctness effect — but the constraint as
   written was violated. Recorded as a deviation.
4. **H10 accounting gap:** the compiler network's own capacity was
   never charged into D. Moot given the negative verdict; must be fixed
   in any V3 reuse of the machinery.
5. **006c/006d remain outstanding by design** (pre-registered,
   non-gating; inputs to the V3 spec).
6. **H9a's spec STATUS block predates the sealed replication**; the
   sealed result lives in section 12. Cross-reference noted here rather
   than editing a dated status block.
7. **PREDICTIONS.md outcome log is still empty** — correct, since none
   of its four entries' experiments (variational two-part, finer GELU
   grid, 256-task lifetimes, task-grouped families) have run; they are
   V3-era.

## Verdict

V2 is complete as specified: every hypothesis has a written,
artifact-backed outcome; every implementation-order step executed; both
sealed components passed all six pre-registered outcomes; the frozen
documents are untouched; and the deviations above are enumerated rather
than silent. The two items that should not slip: the Model 5 mismatch
control (open item 2) before any unconditional H9 claim, and the
006c/006d analyses before the V3 spec freezes its gate design.

## Completion-audit follow-ups (2026-08-19, later)

- Open item 2 CLOSED: the GELU shared-residual mismatch control ran on
  canonical mixed worlds 0-2; allocation and envelope both survive 3/3
  (reports/v2_mixed/sr-gelu-control.json). H9 is not strongly
  alignment-conditioned; spec and paper updated.
- 006c ran: prediction failed informatively; gate v2's mechanism
  re-attributed from entropy to the absolute-NMSE bar, with corrections
  applied to the spec, paper, and synthesis in the same commit
  (reports/v2_func_entropy/).
- 006d ran: weakly supported — monotone ~20% dimensionality decline, no
  regime structure (reports/v2_dimensionality/).
- Remaining watch-list is now only V3-era items (H10 compiler-capacity
  accounting on reuse; forward_tasks batching before any large V3
  sweep; PREDICTIONS.md outcomes as their experiments run).

# Re-audit at H39 development closure (2026-08-21)

Scope: the H39 line (reviews 58-61), from census C0 through the H39d
capacity sweep, and the code it added. The V1-V6 audits above are untouched
by this work; the items checked here are the ones the new code could have
broken.

## Plans versus implementation, item by item

- `H39_EXISTENCE_PLAN.md` (frozen `b6fc27d`, Amendment 1 `16906ff`):
  census C0 implemented in `census_h39_schema.py` exactly as amended
  (PCA over live residuals + abstractions within one artifact; alpha-only
  B1 fit, eps frozen at zero, k=128; 1.5x in >= 2/3 worlds). Verdict NOT RUN
  is the plan's own. The rank-16 and max-rank runs are labeled exploratory
  in their reports and required a non-registered output path.
- `H39_PILOT_PLAN.md` (frozen `effaf6b`; Amendments 1-3): every amendment
  was appended before the data it concerned was read (eps stationarity found
  on a scratch world; the unseen-family correction before any pilot artifact
  was opened; the alpha-only fit's stationarity after the registered
  non-vacuity check refused the first scorer run, whose report is preserved
  as discarded). Branch C read from the frozen table.
- `H39B_PSLOT_PILOT_PLAN.md` (frozen `83ac418`, no amendments): branch U
  read from the table; the route-mass threshold is annotated as
  miscalibrated in `PREDICTIONS.md` and NOT re-judged.
- `H39C_KSWEEP_PLAN.md` (frozen `1e99904`): 15/15 cells; verdict P from the
  table. The baseline-relative route-mass term never fired; the plan's
  verdict did not depend on it.
- `H39D_CAPACITY_PLAN.md` (frozen `c6b1b7b`): 12/12 cells; verdict A from
  the table, recorded with its marginality. The functional usage criterion
  replaced route mass as deciding, as the plan states.
- `H39_CONFIRMATION_PLAN.md` (frozen `1c98017`; Amendment 1 `f78f209`):
  lifetimes launched under the original freeze; the amendment changed only
  the scorer's E2 rule and was registered before any sealed cell was read
  or scored. The scorer implements the amended rule.

## Code-level checks

- `PromotingSharedResidualLearner.effective_residual` is an identity for
  every pre-existing learner; the ordinary world-0 rerun with the history
  hook reproduced the V6 artifact BITWISE (loss and every tensor), so the
  base-class change and the snapshot hook are read-only for existing
  models.
- `ParameterizedSlotLearner` at `alpha = 0` with arguments frozen
  reproduced the ordinary artifact bitwise on all ordinary tensors (real
  world-0 artifact, not only the smoke world). Multi-slot at `alpha = 0`
  equals ordinary in unit tests; the single-slot state-dict layout is
  unchanged, so H39b/H39c artifacts load under the multi-slot code.
- Loaders reconstruct abstractions, references, retirement, schema/slot
  assignment, and both slots' argument matrices; save/load functional-
  equivalence tests with failing companions exist for both new learners.
- Scorers: anchors reproduced to 1e-12 before any new value is read; fits
  are support-only with fixed budgets; reports are written atomically;
  every cell is reported; non-vacuity fails closed. The two scorer crashes
  this line produced (serializing a 2xK alpha; a stationary alpha-only
  protocol) both occurred BEFORE the atomic write and left nothing on disk.
- Launchers: bounded pool of three, one writer per cell, complete
  intervention records, resume refusal on mismatch, nonzero exit on any
  failure; the confirmation launcher additionally runs `check_prereg.py`
  before opening any seed >= 700.

## Deviations and open items

1. Two thresholds were registered without a baseline check (route mass on
   P in H39b; its 2x successor in H39c). Neither changed a verdict; both are
   annotated. The functional ablation is now the deciding usage criterion
   and is written into AGENTS.md.
2. `schema_share` in the H39 pilot is a ratio of ~2% NMSE differences and is
   reported, not relied on.
3. `D*` for the argument matrices is an 8-bit scalar-count proxy throughout
   the H39 line; the rate-distortion instrument was not run on these
   artifacts. Any description-length claim about `P(alpha)` needs it.
4. The ordinary per-task anchors were mis-transcribed once in the ledger
   and corrected by appended note on the same day; no decision used them.
5. Slot structure at matched capacity (M2K16 vs P32) is NOT supported;
   only the K=32 two-slot arm crossed, in two worlds.

## Verdict

The H39 development line is complete as registered: six frozen plans, six
verdicts read from their own tables, every amendment dated and prior to the
data it governs, every new learner bitwise-equivalent to the baseline under
its null switch, and every scorer fail-closed. The sealed confirmation
block is open; its verdict will be read from the amended table and nothing
else.


# H49-H51 re-audit (2026-08-25)

Scope: the code added or changed for the discoverability census (H49), the
reorganization loop (H50), and the reorganizability testbed (H51).

New or changed implementation, and its equivalence control:

1. `src/row/models/pslot_factorized_models.py` (H51 arm R_2) composes the
   parameterized slots with a pooled, separately addressable innovation basis
   (`residual = W a + eps`). Controls verified on the real classes, not only a
   smoke world: `freeze_schema=True` reproduces the `pslot` forward output
   bitwise, the free learner is bitwise equal at `a = 0` (the component is
   exactly zero at initialization), and `dL/da` is nonzero there, so the
   zero-stationary-point trap does not apply. `schema_dim = 4` was registered
   before any lifetime ran, chosen so both G3 denominators pass.
2. `learned_lifetime.py` / `mixed_lifetime.py` gained the
   `pslot_factorized` kind. A learner may now return several task-local fast
   arguments; every caller routes them through `_as_params`, and
   single-argument learners are unaffected.
3. `audit_h49_discoverability.refit` was extended to zero AND re-fit an
   arm's extra task-local fast state (`schema_alphas`, `trace_coefficients`).
   Models carrying neither attribute — every H49/H50 artifact — take the
   identical path, so H50's numbers remain reproducible by the same function.
   Independently confirmed by H51: R_1a's three measured `m = 0` rows
   reproduce H49's recorded M_4 values exactly (0.00560 / 0.00549 / 0.00447).
4. `score_h39b_pslot.load_pslot` takes the artifact's own world seed and
   builds the composed learner from its record; `factorized_fit` accepts a
   multi-argument fast interface. Both were reached by defects found in the
   standing pre-run audit (plan Amendments 4-5), fixed before any row was
   written, and disclosed with the result.
5. `audit_h51_reorganizability.py` caches each SCORED cell under a protocol
   fingerprint (budgets, model records, refit steps and LR, migration rates
   and seeds) and refuses a cell computed under a different fingerprint.
   Migration is re-run every launch, so no cached quantity can outlive the
   state it was computed from.

Registered-versus-implemented check: H51's migration operator, candidate set,
seeds, learning rates, batch policy, LOO sample, margins and decision table
are H50's, unchanged and imported rather than re-coded — the arms differ only
in the wake representation, which is what the plan registers as the sole
independent variable. Balance gates are computed from the artifacts before
any causal reading and now fail loudly when an input cannot be read.

Outstanding, unchanged from the H39 audit: `schema_share` remains a ratio of
~2% NMSE differences and is reported as a diagnostic only; the
rate-distortion instrument has still not been run on these artifacts, so `D*`
figures in this line are the H49 proxy and are comparable within an arm only.


# Export-branch re-audit (2026-08-26): E0 through E2

Scope: the modules added for the export/composition/synthesis branch, audited
after its first licensed claim (composition).

New implementation, and what makes each trustworthy:

1. `audit_e2_feasibility.py` — pure combinatorics, no model. Establishes that a
   held-out stratum EXISTS at usable size before a world is designed around it.
   Two defects were caught and disclosed before any verdict: the training set
   was bounded by rather than equal to the lifetime, and the constructibility
   verdict read one fill objective while the other satisfied it.
2. `audit_e0_export.py` — E0.1 contextual substitutability and the E1.0 gate.
   Distances are CONTRIBUTION-normalised (Amendment 2 corrected a denominator
   that repeated the V4.1 total-scale error) with a null-edit guard that reads
   exactly 1.000 by construction and a degenerate-context guard. The artifact's
   alpha convention is read from the checkpoint rather than assumed, and legacy
   pickle checkpoints load through the same provenance-checked fallback the
   quantizer uses.
3. `audit_e0_residual.py` — the four-condition residual audit, with a live-task
   disaggregation added before recording because most tasks are retired and
   contribute a guaranteed zero. Discloses that the refit condition lifts
   retirement, a configuration the lifetime never used.
4. `audit_e1_export.py` — frozen-library export. Mode-consistent support
   diagnostics (Amendment 2; the first pass was VOIDED for mixing train-mode and
   eval-mode endpoints and is preserved at
   `reports/e1_export_void_firstpass.json`). The wrong-library control is a
   library from an incompatible world, because a slot permutation is vacuous
   under free route inference.
5. `audit_e1r_recurrence.py` — the recurrence control. Uses a held-out TASK
   protocol because programs are not comparable objects across `rho`, and reads
   MEASURED RECURRENCE from the project's own registered diagnostic
   (`world_functional_reuse.json`) after two hand-rolled versions were wrong in
   opposite directions. The `rho = 1` cells reproduce E1 within 0.13 log units
   under an independently built protocol.
6. `support_split_world.py` — E2's generator. Follows the `mixed_world` /
   `task_group_world` pattern and adds NO field to `WorldConfig`, so no existing
   resolved-config fingerprint is invalidated. The split is searched over seeded
   attempts against STRUCTURAL constraints only (stratum sizes, balance, context
   counts) and never against a model or loss; the accepted attempt index and the
   full program list are written to the artifact before training.
7. `audit_e2_composition.py` — three strata with the E1 arm set, so E1 and E2
   are directly comparable. Verifies in code that no training program touches a
   withheld placement, that no held-out program appears in training, and that
   every H3 program contains a withheld placement.

Registered-versus-implemented check: E1 and E2 share arms, budget (2,000 Adam
steps at lr 0.01), interface (E1-P), margins (+0.15) and scorer, so the rungs
differ only in which programs the lifetime saw. Every long scorer caches each
completed cell under a protocol fingerprint and refuses a cell computed under a
different one.

Outstanding: the object-to-primitive assignment margin is 0.001-0.019 throughout,
so functional identity is weakly determined even where export and composition
succeed; every claim in this branch is phrased as substitutability and use, never
as recovery. `D*` accounting for the program variable has not been run (that is
E3) and no depth beyond 3 has been tested (E8).


# Sealed export-confirmation audit (2026-08-27)

Scope: `score_export_confirmation.py` and `tools/run_export_confirmation.py`, the
only new code between E3 and the sealed block.

- The scorer was dry-run against DEVELOPMENT artifacts before being pointed at the
  band, and caught two defects that would each have corrupted it. (1) WORLD
  MISMATCH: `World.generate` produces the same 64 opaque task IDs from the same
  seed but different programs (63/64 differ), so reconstructing the world that way
  paired a support-split model's routes with another world's targets; the symptom
  looked like a scientific failure of C1b/C1c rather than a bug. Fixed to build
  through `generate_support_split_world`, with a FATAL assertion that the
  reconstructed programs equal the recorded split. (2) PROCESS-DEPENDENT SEED: the
  held-out sampler derived a `SeedSequence` component from Python's built-in
  `hash()`, which is randomized per process; replaced with a sha256-derived
  integer and verified identical across processes.
- Structural assertions are fatal, not advisory: held-out programs absent from
  training, H2's unseen adjacent pair, H3's withheld placement, split coverage and
  balance, `D_train = 3` with `D_test in {2, 4}`, the variable-depth executor
  bitwise-identical at depth 3, non-saturated behavioural rates, and every
  claim-bearing arm adapting by more than 1%.
- Scoring reuses the E1/E2/E3/E8 instruments unchanged (`adapt_cell`,
  `oracle_cell`, `behavioural_rate`, `quantize_operator`, `predict_with_route`),
  so the sealed block and the development rungs are measured by the same code.
- Cells are cached under a protocol fingerprint; the block survived a harness
  crash mid-scoring without losing work, because the scorer was launched detached
  and every completed cell was already on disk.

Outstanding, unchanged: the object-to-primitive assignment margin remains small
(0.001-0.019), so every claim is phrased as substitutability and use rather than
recovery; SYNTHESIS is unclaimed pending E5.

# Track B partial re-audit (2026-09-15): rotated substrate through SO3 - RECORD-CONSISTENCY LEVEL ONLY

Status: PARTIAL. This is not the full spec-to-implementation re-audit that
CLAUDE.md requires after major milestones; that audit is still owed and is
tracked in `RESEARCH_STATUS.md`.
- **How it was produced:** a read-only background subagent drafted a first pass
  over the fourteen frozen plans from the rotated substrate through SO3. Claude
  reviewed that draft against the program document, the reports and the code.
- **What changed in review:** the draft's blanket "verified consistent" labels
  were downgraded, two of its statements were corrected, and one finding it
  missed was added.
- **What this section does not do:** it reruns no scorer and recomputes no
  artifact.

## Verified directly in code

- **SO2 terminal estimand.** `src/row/experiments/audit_so2_online_gate.py`
  computes `M_terminal` with `score` (G5R Stage D) on the terminal model and
  reports `end_of_task_median` beside it, with a last-task anchor. This matches
  the plan wording and the disclosed pre-relaunch fix.
- **SO2-P replay confound.** `src/row/experiments/learned_lifetime.py` builds
  `TaskReplayBuffer(seed + 1 if replay_seed is None else replay_seed)`, one
  generator for storage and sampling when `update_batch_size` is `None`. This
  confirms correction b250973: the replay arms were gradient-matched but
  stream-confounded.
- **J2A route gap `g`, a NEW FINDING in this review.**
  - `audit_j2a_staged_library.py` defines `g` as the median over trained tasks
    of `log(opt query error / enum query error)`, identically to
    `audit_j0_library_quality.py`. It is not clipped and can be nonzero (J0
    measured nonzero values on poor libraries).
  - In J2A, `g` = 0.0 on all twelve libraries, including the six failure
    controls (NON-STAGED and RESET, trained median NMSE 0.92-1.06). On those
    controls, exhaustive search picks the as-trained route for only 59-81% of
    tasks (95-98% on staged libraries).
  - The J2A PROGRESS entry's sentence "as J0's threshold predicted for libraries
    this far below 0.47" is therefore true only of the six staged libraries. On
    the controls, gradient routing matched exhaustive search on a majority of
    tasks despite library quality beyond J0's 0.62 failure point. That is not
    what J0's threshold predicts, though these controls are a different library
    kind from J0's oracle-acquired SO1 libraries.
  - The median also hides any minority of tasks on which the routes differ.
  - The EXPORTS verdict is unaffected: it uses exhaustive-search routes.
    Correction appended to PROGRESS.md and PREDICTIONS.md.

## Record-consistent, not code-verified

For G5/G5R, the G5R diagnosis and its LBFGS correction, G5R Stage D, SO0, SO1
(with its restart and anchor-diagnostic amendments), SO1R, J0 (with Amendment
1), J1, J1c, J1c-R, the SO2 interference census, and SO3, this pass checked
only that:
- each plan's registered arms, gates, thresholds and labels match what its
  PROGRESS/PREDICTIONS entries report;
- each disclosed deviation is recorded with its fix.

It did not read each runner and scorer line by line against its plan, and did
not recompute report cells. No inconsistency was found at that level.

Disclosed deviations confirmed present in the record: SO2's end-of-task
construct error (fixed); SO2's strict-reload probe code (fixed); SO2-P's replay
description (corrected); the G5R diagnosis's unregistered H-LBFGS penalty
(quarantined and rerun); SO0's masked unit-test exit code (fixed); SO1's
original launch defects (fixed by amendment before relaunch); G5R's
comparability label wording (fixed, regression-tested).

## Corrections to the first-pass draft

- **Stop rule.** The draft named the binding rule "stop rule 2 (substrate
  exists; the online writer does not reliably acquire it)". In
  `POST_E6_RESEARCH_PROGRAM.md`, Track-B stop rule 2 is "oracle routes pass but
  learned routes do not" (fired after SO1). The rule binding the online result
  is the third: "Online SO2 fails: do not run branching or iteration". Other
  documents written the same day, including this project's status index and
  plans, say "stop rule 2" for that online rule. The intended rule is the
  online-failure rule, and the program document's numbering is authoritative.
- **SO3.** The draft's verdict grouped SO3 with SO2 as programs that "FAIL". SO3's
  registered label is `SO3_FAILS` for its two candidate changes, but its
  unchanged protocol passed the terminal criterion in 3/3 fresh worlds, with
  stream spread 0.009-0.066. It does not license B2 (no BASE gate, no margin),
  and it does show SO2's negative is not a stable property of the protocol.

## Deviations and open items

- **Open:** the full code-level re-audit of the fourteen milestones listed above
  (runner and scorer against plan, at least one recomputed cell per milestone).
- **Open:** the stop-rule numbering in documents written 2026-09-15 refers to the
  online-failure rule as "stop rule 2". A clarifying appendix, not a rewrite, is
  owed where it matters for decisions. Correction recorded here and in
  `RESEARCH_STATUS.md`.
- **Recorded:** the J2A `g` attribution (above).

## Verdict

PARTIAL. No undisclosed plan-to-report inconsistency was found at record level.
One finding was verified in code: J2A's zero route gap on its failure controls
is not the J0-threshold prediction the record attributes it to. The full
spec-to-implementation audit remains owed. No verdict, threshold or artifact
changes.


# Re-audit (2026-09-25): SO4, N1, N1b, N1c, O1 - record consistency and estimand-to-code

Status: PARTIAL, but deeper than the 2026-09-15 pass. For each of the five
milestones, every runner and scorer was read line by line against its frozen
plan and amendments. Headline numbers were then recomputed from the committed
report JSON with stdlib and numpy only:
- per-cell medians from the stored per-task values;
- world medians, clause counts and labels;
- SO4 margins from the 48 recorded program pairs;
- the N1c registered split, recomputed independently from per-task values plus
  the canonical programs in `artifacts/so2_online_gate/cells/STAGED_w{0,1,2}/stage3/world_programs.json`;
- the O1 STAGED/PLAIN last-task anchors, from `artifacts/o1_online_anchor/work/*/stage3/metrics.jsonl`.

**Method limits.**
- Written read-only while O2 runs, so it did not import `torch` or `row.*`,
  rerun any scorer, or reload any model. Scorer results are taken from the
  archived validation records and PROGRESS.
- `PREDICTIONS.md`, `PROGRESS.md`, `RESEARCH_STATUS.md`, `README.md` and
  `paper/draft.md` have uncommitted edits in the working tree (git status
  2026-09-25). This audit read the working copies.

## SO4 (`SO4_B2_RETEST_PLAN.md` eba13b0, `SO4_AMENDMENT_1.md` 4968698; run 69b0692)

**(1) Estimands vs code.**
- `M` is `stage_record`'s terminal median over `world.tasks` whose ids are in
  `model.task_codes`, which is the 64 canonical stage-3 tasks, using G5R Stage D
  `score`. It is not the lifetime's `final_nmse`: `end_of_task_median` is
  reported beside it, with a last-task anchor at `ANCHOR_TOLERANCE = 1e-6`, as
  the plan requires. Recomputed from `terminal_per_task`: all 12 cells and 4
  PLAIN medians match, each with 64 tasks.
- `W` is `np.median` over the 3 streams (`audit_so4_b2_retest.py:248`).
- The margin matches G5R verbatim (`margin_task`/`margin_pair`,
  `audit_so4_b2_retest.py:173-192`, against `audit_so2_online_gate.export_margin`,
  `audit_so2_online_gate.py:111-134`):
  - seed `SeedSequence([1500, world])`;
  - 12 programs;
  - `index_offset = 95000 + i`;
  - scratch `scratch_model(cfg, "rotated_discrete", 7717)`;
  - `ADAPT_STEPS = 2000`;
  - the natural log of the scratch geometric mean minus the natural log of the
    trained geometric mean;
  - computed on the stream-0 terminal library.

  All 48 recorded pairs have `steps = 2000`. Recomputed margins from the pairs:
  -0.075939 / +3.780567 / +5.167760 / +5.096878, exactly equal to the report.

**(2) Record list vs output.**
- Present:
  - per-cell end-of-task median, lost and gained counts, recency Spearman,
    stage-3 drift and the J2A 64-program export;
  - per-stage prequential cost (prefix stages and cells);
  - per-cell wall seconds;
  - PLAIN terminal and export.
- "example-gradients" is not recorded. `online_examples` is tasks x
  examples_per_task, a count of online examples and not of example-gradients
  (`audit_so4_b2_retest.py:133`). PLAIN records neither this count nor seconds
  (`:161-170`). Cosmetic.

**(3) Decision rule vs scorer.**
- Runner `classify` (`:242-266`) and scorer (`score_so4_b2_retest.py:119-129`):
  - both use `W <= 0.05` in 3 or more of 4 worlds;
  - both implement Amendment 1's cap: no stream above 0.10 in a
    terminal-passing world (`<=` 0.10 passes);
  - both require margin `>= 0.75` in 3 or more of 4 worlds.
- `SO4_ACQUIRES_ONLY`'s "the margin misses in 2 or more worlds" is equivalent
  to "not 3 of 4" when there are four worlds.
- The labels partition every case, and `HARNESS_FAILED` takes precedence.
- Recomputed:
  - `W` = 2.1705 / 0.0125 / 0.0908 / 0.0162, so the terminal worlds are {7, 9},
    2 of 4;
  - the maximum stream is 0.0639 in world 7 and 0.0197 in world 9, so the cap
    holds;
  - margins pass in 3 of 4;
  - label `SO4_FAILS`, matching the report and run.log.
- G0 omitted and explicit both give `worst_per_task_abs_diff` 0.0 and pass.
  All last-task anchors are 0.0.

**(4) Record consistency.**
- These match the JSON: the PREDICTIONS / PROGRESS / RESEARCH_STATUS / paper
  tables (stream terminals, W, margins, PLAIN 1.894 / 1.993 / 1.880 / 2.045, the
  median lost count of 9 in world 8, and 0/64 PLAIN export).
- These descriptive claims were checked and hold:
  - world-6 stage-1 prefixes 0.12-0.25 and stage-2 0.042-0.111;
  - worlds 7-9 stage 2 at 0.0096-0.067;
  - gained counts of 6-30 in passing cells;
  - export 64/64 in the best cells and 0-1 in the worst;
  - world 8 stream 1: end-of-task 0.037, terminal 0.173, 33 lost.
- Two defects are listed below: D1 and D2.
- Imprecise but not wrong: world 6's prequential cost is 9.6M-19.9M, which the
  record calls "2-6x the other worlds'". Cell-to-cell it is 1.6x-6.9x
  (others 2.89M-6.06M).

**(5) Arms as constructions.**
- Model seed 7000.
- Stream 0 is `None` (seed+1). Streams 1-2 use `SeedSequence([7400, w, s])`.
- STAGED carries the library through `carry_library` with no task codes.
- PLAIN is a fresh model on the stage-3 world at stream 0.
- G1 recomputes the world-6 stream-0 prefix in memory.

All of this matches the plan. Running the per-pair margin jobs as a pool (a
scheduling change) was verified bitwise against `export_margin` in the dry run
(PROGRESS).

## N1 (`N1_ANCHOR_SUPPLY_PLAN.md` with Amendments 1-3; run 3d2f5c6)

**(1) Estimands vs code.**
- Every arm is scored by `score` on the canonical 64 length-3 tasks only
  (`n1_anchor_supply.py:230`). STAGED is scored through J1c's `run_arm`
  stage-3 `final_per_task`.
- Recomputed: every cell has 64 per-task values, and its median equals
  `terminal_median`.

**(2) Record list.**
- Recorded: pool size and examples, the depth histogram and the first-N draw
  indices, as Amendment 3 requires.
- `first_64_draws` actually holds 128 indices (64 updates x batch 2). The name
  is cosmetic.

**(3) Decision rule vs scorer.**
- Both runner `triage` (`:266-287`) and scorer (`score_n1_anchor_supply.py:87-99`)
  implement:
  - `m < 0.05` (strict) in 2 or more of 3 worlds;
  - otherwise the guard;
  - otherwise `median(s)/median(m) >= 5.0`.
- Latent divergence from the plan's wording, which could not affect this run:
  - Amendment 1 makes `PARTIAL` uninterpretable "in that world". The code
    instead emits a global label, `PARTIAL_UNINTERPRETABLE`, which is not in the
    registered label set, whenever any world's SHAM is below 0.45.
  - The ratio is a cross-world median ratio, not a per-world one.

  Not triggered: SUFFICE fired and SHAM is at least 1.037 everywhere.
- Recomputed:
  - INTERLEAVED is below 0.05 in 3 of 3 (0.00925 / 0.01011 / 0.00658);
  - the SHAM ratio is 114.80;
  - verdict `ANCHORS_SUFFICE`, matching the report.
- INTERLEAVED and SHAM draw identical indices in all 3 worlds (recomputed).
- All three STAGED cells equal the J1c `terminal_median` exactly.

**(4) Record consistency.**
- The PROGRESS / PREDICTIONS / RESEARCH_STATUS / README / paper numbers all
  match the JSON.
- Defects D3 and D4 are listed below.

**(5) Arms as constructions.** Defects D3, D5 and D6 are listed below.
- SHAM fillers: 124 distinct, unused length-3 programs, seeded
  `[1912, w, ...]`. They use `world.tasks[0].teacher_library`. This is correct
  only because `reuse_rho = 1.0` in `configs/v1.yaml`, so `_task_library`
  returns the shared library (`world.py:251-252`). They are executed in the
  same primitive order as `Program.execute`.
- INTERLEAVED and SHAM share pool size, seed `[1911, w]` and draws. This
  matches Amendment 3.

## N1b (`N1B_ANCHOR_DOSE_PLAN.md`; run fc8a742)

**(1)-(3) Estimands and rules vs code.**
- The pool is `anchors(k) + canonical + fillers[:124-k]`.
- DOSE subsets: permutation `[1913, w]`, take the first k, re-sorted to N1
  order.
- LENGTH triage: four labels, strict `< 0.05`, 2 of 3.
- `k*` uses the persistence rule over {0, 8, 32, 124}, with endpoints read from
  the committed N1 cells, plus a NON_MONOTONE flag.
- The NEAR_THRESHOLD band [0.03, 0.08] is inclusive.

Runner and scorer agree, and both match the plan. The scorer also fails if the
N1 endpoints do not bracket the curve.

**Recomputed.**
- Every per-task median matches `terminal_median`.
- LENGTH: L1_ONLY passes 3 of 3 (0.0175 / 0.0112 / 0.0067). L2_ONLY fails 3 of
  3 (1.163 / 1.116 / 1.144). Label `L1_SUFFICES_ONLY`.
- DOSE: 0 F, 8 F, 32 T, 124 T, so `k* = 32` and the curve is monotone. No
  near-threshold cell.

All of this matches the report.

**(4) Records.**
- The PROGRESS / PREDICTIONS / RESEARCH_STATUS / README / paper / AGENTS numbers
  match.
- The post-hoc coverage table matches `coverage.json` (passing cells 6 of 6 and
  failing cells 0 of 6, with DOSE_8_w2 at 5 of 6).
- DOSE_32 carries 16-19 length-1 anchors.

**(5) Arms.** They match the plan. The endpoint-identity and 32-update prefix
tests were checked in the record, not rerun. There is no finding beyond D5's
inherited N1 items.

## N1c (`N1C_ANCHOR_COVERAGE_PLAN.md`; run 6251a17)

**(1)-(3) Estimands and rules vs code.**
- The excluded operation `x` is the first draw of `[1914, w]`, shared by all
  arms.
- The arms match the plan:
  - COVER6_K6 has one anchor per operation;
  - COVER5_K6 has five firsts plus one extra from a seeded doubled operation;
  - COVER5_K18 has 18 anchors from the five-operation pool.
- The PRIMARY 2x2 map and the SECONDARY rule match the plan.
- The descriptive LOCAL/GLOBAL split is applied to failing (`>= 0.05`) COVER5
  cells.

**Recomputed.**
- The excluded operations are 4 / 3 / 4, matching the plan's pre-launch
  statement.
- PRIMARY `K6_INSUFFICIENT`: all six K6 cells are between 1.04 and 1.09.
- SECONDARY `COUNT_COMPENSATES`: 0.199 / 0.0129 / 0.0259.
- The split was recomputed independently from per-task values and canonical
  programs, and all 9 cells match exactly. For COVER5_K18_w0: using-x 35 tasks
  at median 1.1346, not-using 29 at 0.0252, so LOCAL. The other failing COVER5
  cells are GLOBAL.

**(4) Records.** The PROGRESS / PREDICTIONS / RESEARCH_STATUS / README / paper /
AGENTS numbers all match.

**Scorer independence (cosmetic).** `score_n1c_anchor_coverage.py:62-65` is
commented "Recompute the registered split from per-task values rather than
trusting it". It checks only that the two counts sum to 64, and classifies
LOCAL/GLOBAL from the runner's stored medians. This audit's independent
recomputation found them correct.

## O1 Tier 1 (`O1_ONLINE_ANCHOR_TIER1_PLAN.md`; run 2e99765)

**(1) Estimands vs code.**
- Terminal medians are over the canonical 64 only:
  - the single-lifetime arms score `canonical ∩ model.task_codes`, which
    excludes anchors and the novel probe;
  - STAGED/PLAIN use SO2's stage-3 `terminal_per_task`.
- Recomputed: all 12 medians match per-task values, and every cell has 64 tasks.
- KNOWN, confirmed: in the single-lifetime arms `end_of_task_median` was the
  median over all 188/124 stream tasks (`o1_online_anchor.py:106`). STAGED's
  covers the stage-3 64. This was corrected in PROGRESS 2026-09-24 using
  `stream_audit.json`. The canonical-64 values (0.633 / 0.244 / 0.713 and
  1.319 / 1.243 / 0.627) and the "17-39x" claim recompute exactly (17.3x, 38.8x,
  17.5x in the three passing cells).
- The stale values remain in the stamped report. That is correct under the
  append-only rule.

**(2) Record list.**
- KNOWN, confirmed:
  - STAGED/PLAIN hard-code `route_lengths_match_plan: True` (`:130`);
  - `so2.run_arm` computes `export_margin` and `export_diagnostic` at scale 1
    (`audit_so2_online_gate.py:192-194`), and O1 discards both
    (`o1_online_anchor.py:124-132`).
- Additional, a minor record gap: O1's record also drops SO2's
  `anchor_abs_error` for STAGED/PLAIN. The stream audit covered only the six
  single-lifetime cells.
  - This audit closed the gap from the work-directory `metrics.jsonl`. All six
    STAGED/PLAIN last-task anchors are 0.0, the last task ids match the record,
    and the stage-3 end-of-task medians equal the recorded ones.
  - The O2 draft already records the anchor for every arm.

**(3) Decision rule vs scorer.** Runner `summarize` and the independent scorer
agree:
- LIVE if SHUFFLED or MIXED_L1 has `< 0.05` in 2 or more of 3 worlds;
- denominator 3;
- PLAIN floor check;
- interleaving check on the first 20 depths.

Recomputed: STAGED 2, SHUFFLED 2, MIXED_L1 1, PLAIN 0, so LIVE, matching the
report.

**(4) Records.**
- The PROGRESS / PREDICTIONS / RESEARCH_STATUS / README / paper tables match the
  JSON: 0.272 / 0.0269 / 0.0188; 0.0366 / 0.473 / 0.0184; 0.191 / 0.149 /
  0.0358; 2.005 / 1.923 / 1.913.
- These also match:
  - cell wall times (STAGED about 85 min, PLAIN about 76 min, SHUFFLED about
    9 min, MIXED_L1 about 6 min);
  - the effect sampler U(0.0088, 0.0400), which equals SO3's seven passing BASE
    streams (min 0.008756, max 0.040012);
  - the null sampler, SO2 PLAIN 1.910-1.967.
- Defect D7 is listed below.

**(5) Arms.**
- Model seed 5000 (`stage_setup(..., 5000)`; SO2 `MODEL_SEED`).
- Replay stream 0 (no `replay_seed`).
- Shuffle seeds `[1920, w]` and `[1921, w]`.
- SHUFFLED is the STAGED multiset. MIXED_L1 is 60 length-1 tasks plus the 64
  canonical ones.

The stream audit confirms the consumed order, 128 examples per task, and routes
at their planned length. Everything matches the plan. The once-over-the-stream
temperature anneal and single replay buffer are disclosed in the plan.

## DEFECTS

| id | milestone | severity | defect and evidence |
|---|---|---|---|
| D1 | SO4 | affects a reported number (descriptive; no verdict) | `PROGRESS.md:5725-5726` says the scratch comparator's "geometric-mean NMSE" is "1.1-4.5". The per-world scratch geometric means recomputed from the 48 pairs are 2.082 / 2.806 / 2.426 / 2.447, a range of **2.08-2.81**. 1.1-4.5 is roughly the range of individual program scratch NMSEs (actual 0.915-8.06), not of geometric means. |
| D2 | SO4 | affects a reported number (wording; no verdict) | `PREDICTIONS.md:8797-8798` says "the replay stream moves terminal error by up to two orders of magnitude (world 6: 0.203 to 2.239)". That is a factor of 11.0; the largest within-world spread is 11.0x (world 6), and world 8 is 10.75x. That is one order of magnitude, which is how PROGRESS, RESEARCH_STATUS, README and the paper state it. A correction appendix is owed, since PREDICTIONS is append-only. |
| D3 | N1 | affects a reported number's provenance (no verdict; NONE is not in the triage) | Amendment 1 section 2 registers `NONE` as "the published 64-slot length-3 baseline... unchanged", with non-vacuity referent 0.92-0.97 (`N1_ANCHOR_SUPPLY_PLAN.md:156-158, 315-317`). Commit 3d2f5c6 says "NONE is the published 64-task floor". The code instead re-trains NONE with `train_pooled` on N1's pool stream `SeedSequence([1911, w])` (`n1_anchor_supply.py:136-137, 227-229`), not J1c's baseline stream. It is a fresh realization, not the published cell: published J1c baseline 0.9608 / 0.9542 / 0.9187 against N1 NONE 0.9575 / 0.9037 / 0.9037. Two of three worlds fall outside the registered 0.92-0.97 referent. The scorer checks only `NONE > 0.05` (`score_n1_anchor_supply.py:81-85`). The records say "matching the published 0.92-0.97 floor to within its spread" (`PROGRESS.md:7116`, `PREDICTIONS.md:9303`) without saying that NONE was re-run under a different stream or that the registered range check was implemented as `> 0.05`. |
| D4 | N1 | cosmetic (construction wording) | Amendment 3 establishes that the offline trainer has no stream positions: it samples minibatches uniformly from a pooled set. Yet the records describe INTERLEAVED as "anchors at random positions": `README.md:52`, `paper/draft.md:1968`, `PROGRESS.md:7125`, `AGENTS.md:2074` and `RESEARCH_STATUS.md:184`. The construction is pooled uniform sampling over 188 tasks. |
| D5 | N1 | cosmetic / process (registered but not implemented) | (a) The plan registers "Every arm is described with `row.arm_provenance.describe_arm` and checked with `assert_arm`" (`N1_ANCHOR_SUPPLY_PLAN.md:61-62`). None of n1/n1b/n1c/o1 imports `arm_provenance`, and N1's disclosed deviations do not mention this. Arm constructions are instead enforced by `validate_cell` and the scorer. The A1 audit's general no-retrofit decision does not cover a plan that registered it. (b) `n1_anchor_supply.py:355-356` declares `--dry-run`, and `main()` never reads it, so the flag would launch the full 12-cell run. The dry run in commit 3d2f5c6 must have been done by other means; this cannot be verified from the repo. (c) The archived `precondition.json` records only the relaunch (14.7 GiB). The original launch at 6.1 GiB, against the registered 8 GiB and the code's 6.0 GiB (disclosed), is visible only in `run.log`. |
| D6 | N1 | latent, no effect | `triage` emits a global `PARTIAL_UNINTERPRETABLE` label, which is not among the registered labels, and uses a cross-world median ratio. Amendment 1 says the guard applies "in that world" (`N1_ANCHOR_SUPPLY_PLAN.md:374-378`, `n1_anchor_supply.py:274-281`). It was not reachable in this run. |
| D7 | O1 / status | cosmetic (index staleness, working copy) | The `RESEARCH_STATUS.md` header (line 9, "Nothing is running", "O2... is DRAFTED, awaiting PI decision 11") and lines 80-86 ("O2 DRAFTED ... not frozen ... Awaiting PI decision 11") contradict the same file's decision 11, "ANSWERED 2026-09-25 ... Frozen `b3c1c85`", and the fact that O2 is now running. Lines 724-727 still say SPEC_AUDIT does not cover SO4. That becomes stale once this section is appended. |
| D8 | SO4 | cosmetic (record list) | The plan's "example-gradients" is not recorded. `online_examples` counts online examples. PLAIN records no seconds (`audit_so4_b2_retest.py:133, 161-170`). |
| D9 | N1c | cosmetic (scorer independence) | The scorer's split "recomputation" does not recompute (`score_n1c_anchor_coverage.py:62-65`). The values are correct by this audit's independent recomputation. |

The known items were confirmed, not re-reported:
- O1's single-lifetime end-of-task population (corrected 2026-09-24/25);
- O1's hard-coded STAGED `route_lengths_match_plan`;
- O1's unused `export_margin`, which also covers the discarded `export_diagnostic`;
- the O2 baseline unit error. The fixed value, 14/24 = 0.583 over SO2 3 + SO3
  BASE 9 + SO4 12 unchanged-protocol cells, recounts correctly from the
  committed reports.

## Checked with no finding

- All five decision rules. Runner and scorer implementations are equivalent to
  the frozen rules for these properties:
  - thresholds, and strict `<` for the N1-line and O1 rules (as registered);
  - `<=` for SO4 W and the cap (as registered);
  - denominators (3 worlds; 4 worlds for SO4);
  - label partitions.

  The labels recompute from the JSON for every milestone.
- Every cell median in all five reports equals the median of its stored
  per-task values, and every scored set has exactly 64 canonical tasks.
- SO4:
  - G0 is 0.0 for both variants;
  - all anchors are 0.0;
  - G1 to G4 are true;
  - the margin construction is identical to G5R/SO2 and all 48 pairs are at
    2000 steps.
- N1: the STAGED bitwise anchor and the identical INTERLEAVED/SHAM draws.
- The N1b/N1c pools are 188 tasks and 24,064 examples, anchor counts and depths
  are as registered, and N1b's endpoints are the committed N1 cells.
- The sampler values cited in the O1 and SO4 plans were verified against the
  committed reports:
  - SO3 BASE and STORE_8, including 0.122;
  - SO2 PLAIN;
  - SO2 STAGED 0.126 and 0.085.
- The disclosed deviations are present:
  - N1's 6 GiB precondition, its late `status.json` and its reboot/resume;
  - O1's 8 GiB fail-closed first launch and its performance-pass miss.

## Not checked

- No scorer was rerun and no model was reloaded (torch was off-limits while O2
  runs).
- The unit-test counts and the dry-run and restart-test outputs cited in
  PROGRESS were not verified.
- `check_prereg`, `check_invalid` and `check_adequacy` were not rerun.
- The N1 Amendment 2 ARI gate and the N1b endpoint-identity and prefix tests
  were taken from the record.

## Verdict

PARTIAL at code level. No registered verdict or label changes: SO4_FAILS,
ANCHORS_SUFFICE, L1_SUFFICES_ONLY with `k* = 32`, K6_INSUFFICIENT with
COUNT_COMPENSATES, and O1 LIVE all recompute from the committed JSON.

Two reported numbers are wrong in the record: D1 (a PROGRESS range) and D2 (a
PREDICTIONS magnitude). One arm's construction is misdescribed against its
registration: D3, N1 `NONE` is a re-run, not the published cell, and its
registered non-vacuity range was implemented as `> 0.05`. The rest are
cosmetic or process gaps.

**Parent verification (2026-09-25).** The site-level reading was done read-only
by a subagent. The parent independently confirmed three defects:
- D1: SO4 scratch geometric means recomputed from the 48 margin pairs are
  2.08 / 2.81 / 2.43 / 2.45 for worlds 6-9.
- D2: 2.239 / 0.203 = 11x.
- D3: N1 `NONE` = 0.9575 / 0.9037 / 0.9037, against J1c 0.9608 / 0.9542 / ...

Corrections are appended in `PREDICTIONS.md` (D2, D3) and `PROGRESS.md` (D1,
D3). The D4 wording was fixed in `README.md`, `paper/draft.md` and
`RESEARCH_STATUS.md`. D5(b), the unused `--dry-run` flag in
`n1_anchor_supply.py`, is recorded as a hazard and NOT fixed in code. The N1
scorer digests that runner, so editing it would break re-scoring N1. Never run
that module with `--dry-run` expecting a dry run.
