# SO2 interference census (Tier 0, descriptive)

Status: written before any census number exists, committed with its code
before it runs. Not preregistered and not a verdict: it neither changes nor
reinterprets `SO2_FAILS` (6a4f707). Its purpose is to decide whether the
successor question ("what does the online protocol remove?") should target
INTERFERENCE and, if so, where.

# Motivation

In SO2, staged worlds 1 and 2 ended with terminal error above end-of-task error
(0.126 vs 0.077, 0.085 vs 0.035). World 0 improved (0.019 vs 0.065). World 1's
terminal error rose at every stage boundary. This is an unregistered
observation, and it has alternative explanations: stage-3 tasks may be learned
and then overwritten, arrive unsolved, or a few tasks may dominate the medians.

# Inputs (existing frozen artifacts only)

`artifacts/so2_online_gate/cells/{STAGED,PLAIN}_w{0,1,2}`: per-stage
`model.pt` / `model_state.json`, `metrics.jsonl` (per-task `task_summary`), and
`result.json` (terminal per-task NMSE). No training, no new data, no held-out
or sealed seeds. Every model is reconstructed with all of its state, including
the terminal novel-probe code, exactly as the independent scorer does.

# Questions and descriptive measures

Q1 **Where degradation lives.**
- For each stage-3 task: `log10(terminal / end_of_task)`.
- By task position: quartile means, Spearman correlation with position, the
  fraction of tasks degraded by more than 2x, and the number crossing 0.05 in
  each direction.
- The same measure for every earlier stage.

Q2 **Arrival readiness.** Zero-shot NMSE by task position, as quartile medians.
This separates "arrives solved" from "is learned and then lost".

Q3 **Backward interference across stage boundaries (STAGED only).**
- Take the task codes a stage-`s` model learned, run them through a later
  stage's library, and score on stage-`s` tasks.
- Report the median NMSE and the count at or below 0.05 for s1->s2, s1->s3 and
  s2->s3, beside that stage's own terminal.
- This moves a code into a library it did not co-form with, so it is an upper
  bound on forgetting, not a pure measure of it.

Q4 **Library drift.** Per-slot relative functional change,
`||f_after(x) - f_before(x)|| / ||f_before(x)||`, on a fixed Gaussian probe
(`SeedSequence([2026, world])`, 512 states) between consecutive stages, as the
median and maximum over slots.

# Non-vacuity and consistency guards (the census refuses to report otherwise)

- Recomputed terminal per-task NMSE equals `result.json` to within 1e-6.
- The transplant evaluated with its OWN stage library reproduces that stage's
  terminal median to within 1e-6, so the transplant mechanics are exact.
- End-of-task values read from `metrics.jsonl` reproduce the recorded
  end-of-task median.

# What it cannot establish

It cannot say when inside stage 3 degradation occurs, because only stage-end
models exist; it gives the ordering by task position only. It does not show
that interleaving or replay would fix anything. Any intervention is a separate
Tier 1 or Tier 2 plan.
