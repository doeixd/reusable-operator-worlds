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
