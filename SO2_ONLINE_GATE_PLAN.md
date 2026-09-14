# SO2: does staged formation survive the online lifetime?

Status: FROZEN at this commit (hash recorded in `tools/check_prereg.py` in the
following commit), before any SO2 code or cell exists. Development worlds 0-2
only. This is B2 of `POST_E6_RESEARCH_PROGRAM.md`, opened by `J1C_ACQUIRES`
(d4e6808), its replication (3bf6a59) and the export audit `EXPORTS` (686e4d3),
with the curriculum's price recorded (aa79aa3: compute-free, +124 tasks).

# Question

Everything Track B has established about staged formation is OFFLINE: tasks
available together, sampled i.i.d. ROW's claims are about lifetimes - tasks
arrive in sequence, every online example is scored BEFORE it is trained on,
and cost accumulates. Does staged formation acquire the rotated substrate
under that protocol, and does the resulting library beat a from-scratch
learner on held-out programs by the margin G5R registered?

# Construction (and what it does not do)

The lifetime loop is `learned_lifetime.run` unchanged, with model kind
`rotated_discrete_fast`, its canonical online protocol (score before update,
replay, canonical train/evaluation separation) and its existing `world=`
injection. Two additive changes, both optional and default-off, are registered
here: an optional `model=` injection (mirroring `world=`, so a stage can
continue from the previous stage's library instead of a fresh one), and a
returned terminal model handle. No existing caller changes.

- **SO2-STAGED:** three consecutive lifetimes per world - 60 length-1 tasks,
  64 length-2 tasks, then the canonical 64 length-3 tasks - carrying ONLY the
  shared library between them (fresh task codes; the optimizer is rebuilt).
  Every example of every stage is scored before update and its prequential
  cost is counted. Stage boundaries are SUPPLIED, as is the stage task
  distribution: SO2 tests whether staged formation survives the online
  protocol, not whether a learner discovers the curriculum for itself, and
  the report says so.
- **SO2-PLAIN:** one lifetime per world on the canonical 64 length-3 tasks
  alone, same kind, same config, same seeds. The matched non-staged control.
- **SCRATCH (held-out margin):** for each held-out program, a from-scratch
  learner of the same architecture built as E1/E8 build it, trained on that
  program's 128 support examples only, scored on its 256 query examples.

Held-out programs are J2A's: 64 of the 152 length-3 compositions the canonical
world never trains on, `SeedSequence([1709, world])`, verified disjoint, built
from the world's own teacher library. Routes for the frozen library are chosen
by exhaustive support-only search, as in J2A and E1; teacher programs never
reach any learner.

# Estimands and registered classification

Per world, on the canonical length-3 tasks and the held-out programs:

- `M_terminal`: terminal median query NMSE over the 64 canonical tasks.
- `G_export`: median over held-out programs of
  `log10(scratch NMSE) - log10(frozen-library NMSE)`, the G5R margin in log
  units (positive favours the library).
- Reported beside them: cumulative prequential Gaussian log loss over the
  whole stream, total example-gradients, operator applications, wall-clock
  seconds, retained bits (int8 proxy), and the staged arm's extra tasks.

Evaluated in order:

0. HARNESS_FAILED: any arm scores an example before it is trained on out of
   order, a library fails to transfer between stages, task ids collide across
   stages, any non-finite value, a missing cell, or the SO2-PLAIN arm is not
   the same construction as SO2-STAGED's third stage apart from its
   initialization.
1. SO2_PASSES: `M_terminal <= 0.05` in at least 2 of 3 worlds AND
   `G_export >= 0.75` in at least 2 of 3 worlds, for SO2-STAGED.
2. SO2_ACQUIRES_ONLY: `M_terminal <= 0.05` in at least 2 of 3 worlds but the
   export margin misses in 2 or more.
3. SO2_FAILS: otherwise.

SO2-PLAIN does not enter the classification; it is the contrast that says
whether staging did the work, and G5R's 0/3 online failure is the historical
reference.

# Registered predictions

- SO2_PASSES: 0.55. Offline staged formation reached 0.005-0.007 and exported
  at 64/64, but the online protocol sees each example once, in order, with
  replay rather than i.i.d. resampling, and G5R failed 0/3 online.
- `M_terminal <= 0.05` in 3/3 worlds: 0.5.
- `G_export >= 0.75` in at least 2/3 worlds given the terminal threshold is
  met: 0.75 (J2A's libraries exported at their trained loss, and a scratch
  learner on 128 examples is weak).
- SO2-PLAIN fails the terminal threshold 3/3: 0.85 (its offline counterpart
  reached 0.92-0.97 and G5R failed online).
- Staged prequential cost over the whole stream EXCEEDS plain's: 0.7 (it
  scores 124 extra tasks online, each paid for at arrival).

# Registered consequences

- SO2_PASSES: "a strong substrate is learnable under this registered online
  protocol and cost", the statement B2 defines, with the curriculum recorded
  as a supplied intervention. The C2 control-flow rungs may then be opened by
  their own frozen plan, which must first re-run the branching and iteration
  opportunity gates on the SO2 artifact itself.
- SO2_ACQUIRES_ONLY: the substrate is acquired online but the vocabulary claim
  does not carry over from J2A; report the split and do not open C2.
- SO2_FAILS: staged formation is an offline result. Track B stop rule 2 stands
  and the successor question is what the online protocol removes - arrival
  order, single-pass examples, or replay.

# Cost and run discipline

Per world: three staged lifetimes (188 tasks total) plus one plain lifetime,
plus 64 scratch fits per world for the margin. Measured before launch in the
dry run; the plan is not amended if it proves slower. Clean committed code;
the independent scorer is committed WITH the runner before launch; dry run
and performance pass first; durable hashed record and tensor-only model per
cell and per stage; relaunch resumes; timestamped `run.log` and
`status.json`; detached launch; independent recomputation re-scoring saved
models; logs copied to `reports/` and committed with the result.

# What this cannot establish

It does not show a learner discovering its own curriculum, nor anything about
program lengths other than 1-3, control flow, the ordinary substrate's sealed
results, or any confirmatory band. Development worlds only. A pass licenses
online learnability AT THIS PROTOCOL AND COST with a supplied stage schedule.
