# SO2 Amendment 2 (2026-09-14): the novel-composition probe reports unavailability

Status: FROZEN at this commit, before any SO2 cell exists. Supplements,
without rewriting, `SO2_ONLINE_GATE_PLAN.md` (6347243) and
`SO2_AMENDMENT_1.md` (77412e8).

# Defect

The plan states that SO2 runs `learned_lifetime.run` unchanged apart from two
registered additive changes (`model=` injection and `return_model`). Building
the runner showed a third change is required, for a reason the plan should
have anticipated: a curriculum stage can hold more tasks than its program
space has programs. Stage 1 is 60 tasks over the 6 length-1 programs, so every
program is used and no unseen composition exists. The lifetime's
novel-composition probe assumed at least one unseen program and divided by
zero (`_novel_data`), and its checkpoint variant then indexed a curve that was
never computed (`_novel_checkpoint`). The stage-1 lifetime could not run at
all.

# Correction

Two guards in `learned_lifetime.py`, both reached only in the case that
previously raised, so no existing result can change:

- `_novel_composition_available(world, config)` returns False when the world's
  tasks already use every program of its length, and
  `_adapt_novel_composition` then returns
  `{"available": False, "reason": ...}` instead of crashing.
- `_novel_checkpoint` returns the same unavailability marker instead of
  averaging curves that do not exist.

The novel-composition probe is a DIAGNOSTIC. It is not part of SO2's
estimands, thresholds or classification, and it remains available and
unchanged in stage 3 and in the PLAIN arm, where unseen programs exist (the
canonical world uses 64 of 216). SO2 reports, per stage, whether the probe was
available.

Nothing else changes: the stages, budgets, arms, estimands, thresholds,
classification, predictions and consequences stand as frozen, and the export
margin remains G5R's construction per Amendment 1.
