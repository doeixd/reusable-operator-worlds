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

# Robustness queue

After the causal reuse sweep:

- replay ratios `0`, `1`, and `4`;
- dense task embedding dimension 24;
- generic low-rank hypernetwork baseline;
- teacher rank 16 versus learner rank 8;
- teacher activation/family mismatch;
- reverse task order for key learned-model conditions.

# Confirmation gate

Run seeds 100–129 only after:

- both architecture configurations are frozen;
- all six `rho` values pass endpoint validation on development worlds;
- checkpoint evaluation is stable;
- artifact and plotting commands pass from a clean checkout;
- no confirmatory-world outputs have been inspected.
