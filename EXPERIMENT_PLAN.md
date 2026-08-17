# Purpose

This plan separates architecture development from confirmatory evaluation for
Reusable Operator Worlds. It prevents iterative tuning on the worlds used for
the final claim.

# World partitions

- Development worlds: seeds 0–9.
- Confirmatory worlds: seeds 100–129.
- Confirmatory worlds remain uninspected until the model families, tuning rule,
  metrics, and exclusions are frozen from development results.

# Primary causal comparison

Compare the continuous reusable basis with Dense-C on identical worlds at:

`rho = [0, 0.25, 0.5, 0.75, 0.9, 1.0]`.

The primary world-level effect is:

`Dense-C cumulative prequential Gaussian log loss - Continuous cumulative
prequential Gaussian log loss`.

Positive values favor the reusable basis. The central causal prediction is that
this effect increases with measured functional recurrence and configured `rho`.

# Lifetime learning-speed comparison

At 8, 16, 32, and 64 completed tasks, freeze shared parameters and adapt fresh
task state on four fixed unseen compositions. Report NMSE after 0, 1, 2, 4, 8,
16, and 32 examples. The primary checkpoint statistic is mean 32-shot NMSE.

The frozen checkpoint sweep is complete on development worlds 0–9. Continuous
improves from mean 32-shot NMSE 0.02282 after 8 tasks to 0.00343 after 64 and
wins all paired worlds from checkpoint 32 onward. Dense-C improves from 0.02300
to 0.00645. The tracked report is
`reports/checkpoints_worlds_0_9/checkpoint-replication.json`.

# Development rho gate result

The frozen six-point causal sweep is complete on development worlds 0–9. All ten
worlds favor Dense-C through rho 0.75 and Continuous at rho 0.9 and 1.0. The
descriptive mean zero crossing is configured rho 0.8264, corresponding to mean
measured residual-function correlation 0.4836. The tracked machine report is
`reports/rho_worlds_0_9/rho-replication.json`.

The novel 32-shot transfer outcome does not show an earlier reusable-representation
boundary; Continuous is reliably better on that metric only at exact reuse.
Confirmatory worlds remain sealed pending replicated checkpoint stability and a
clean artifact/plotting rehearsal.

# Symmetric tuning

Tune Continuous and Dense-C under the same staged rule on development worlds:

- global learning rate: `[0.0003, 0.001, 0.003]`;
- task-state learning rate: `[0.005, 0.05]`;
- replay ratio fixed at `1.0` for the primary tuning stage;
- architecture sizes fixed at Continuous rank 8 / eight slots and Dense-C width
  32 to preserve the planned compute comparison.

Stage one evaluates the full grid on development worlds 0–2. Stage two evaluates
the best two configurations per architecture on worlds 3–9. Select by mean
cumulative prequential Gaussian log loss, breaking near ties with mean 32-shot
checkpoint novel-composition NMSE. Freeze the resulting configuration before
confirmation.

The stage-one grid predates the learnable-alpha architecture correction and is
therefore provisional. Revalidate finalists under the corrected model before
freezing confirmation.

# Frozen development configuration

Corrected-architecture stage two on development worlds 3–9 selected and freezes:

- Continuous: global learning rate `0.003`, task-state learning rate `0.05`;
- Dense-C: width `32`, global learning rate `0.001`, task-state learning rate
  `0.05`.

The primary selection metric chose the same winners as provisional stage one.
The machine-readable validation and paired world effects are stored in
`reports/stage2_current/selection.json`. Confirmatory worlds remain sealed.

# Coding metric

The continuous-density score is called cumulative prequential Gaussian log loss.
For a coding interpretation, targets are assigned fixed precision `1/256` and
the common `-d log(delta)` term is added per example to produce cumulative
prequential quantized-target log loss. Model differences are identical under
both forms.

# Retained-description metric

Report evaluated symmetric 8-bit operator/global weights plus task-specific
state. Continuous and dense task coefficients are counted as 8-bit scalars.
Hardened discrete routes are counted as categorical indices. Quantization scale
overhead is reported as excluded from the V1 proxy.

# Compute scope

Dense-C is matched to Continuous on approximate inference-forward multiply-adds,
not total training FLOPs. Continuous training evaluates and backpropagates through
every basis slot. Report training-forward multiply-adds separately and state that
backward/optimizer costs are excluded.

# Online update batch

The current protocol performs each update on one current example plus one replay
example, an effective batch of two at replay ratio 1. This differs from the
suggested batch size eight but is symmetric across models. It must be disclosed
and later ablated.

# Robustness queue

After the causal reuse sweep:

- replay ratios `0`, `1`, and `4`;
- dense task embedding dimension 24;
- generic low-rank hypernetwork baseline;
- teacher rank 16 versus learner rank 8;
- teacher activation/family mismatch;
- reverse task order for key learned-model conditions.

# Structural control definitions

The generic low-rank hypernetwork uses one opaque eight-dimensional code for
each of three task steps. A shared two-layer generator maps each code to `U`,
`V`, and bias deltas around one learned rank-eight base operator. This preserves
the low-rank three-stage residual family but contains no explicit operator slots.
Its 24 task-specific scalars exactly match Continuous. Tune its shared learning
rate symmetrically on development worlds before comparing cumulative
prequential Gaussian log loss and frozen-library novel adaptation.

The Dense-C task-code sensitivity changes only `task_embedding_dim` from 32 to
24 while retaining width 32, three residual blocks, and the selected optimizer.
This is a task-state-size control, not a compute-matched replacement for the
primary Dense-C baseline.

The hypernetwork's symmetric development tuning selected global/task LR
`0.003/0.05` in both stages. Across worlds 0–9, Continuous beats it on lifetime
loss and novel 32-shot NMSE in all ten paired worlds; the hypernetwork beats
Dense-C on lifetime loss in all ten. Freeze this optimizer for any follow-up.

Dense-24 and Dense-32 are effectively tied across worlds 0–9, so the sensitivity
is closed: retained task-state dimension does not explain the primary result.

# Confirmation gate

Run seeds 100–129 only after:

- both architecture configurations are frozen;
- all six `rho` values pass endpoint validation on development worlds;
- checkpoint evaluation is stable;
- artifact and plotting commands pass from a clean checkout;
- no confirmatory-world outputs have been inspected.

# Robustness gate result

Selected Continuous and Dense-C have now been evaluated on all development
worlds under reverse task order and replay ratios 0, 1, and 4. Continuous wins
cumulative lifetime loss in all ten paired worlds under every condition; strong
replay does not remove the advantage. Novel 32-shot transfer is replay-sensitive
and is not reliable in the no-replay condition. The confirmation gate still
requires a second model initialization and clean workflow rehearsal.
