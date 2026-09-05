# Concurrency calibration plan (infrastructure, not an experiment)

Status: DRAFT, to be committed after G5R Stage D completes. This plan changes
how many processes a batch may run and how independent cells are dispatched. It
changes no model, no estimand, no threshold, and no artifact contents. It is
therefore NOT a preregistered experiment, but it does amend a constitutional
working rule in `AGENTS.md`, so it carries acceptance criteria and a bitwise
equivalence gate, and it is written before the code exists.

# The rule being amended, and why

`AGENTS.md` currently caps local execution at a bounded pool of 3-4 concurrent
`slots=12` promoting/lifecycle runs and up to 6 for lighter models. That cap was
derived from two real failures: five concurrent lifetimes exhausted memory and
killed 113 of 120 cells, and two launcher instances over the same output paths
raced and logged completions for cells with no artifacts.

Those failures had different causes and only one of them is about count. The
race was fixed by one-writer-per-cell, which is now standard. The memory failure
was diagnosed by outcome (a crash) rather than by measurement, and the resulting
constant has been carried unmeasured ever since.

Measured on this host, 2026-09-04, during the Stage D audit:

    one discrete lifetime, resident            364 MB
    physical memory free                       9.8 GB
    cores                                        16
    threads per lifetime                          1 (torch.set_num_threads(1))

At 364 MB the memory argument does not bind at 3-4. The cost of the unmeasured
constant is direct: Stage D's nine independent offline cells run serially on one
of sixteen cores.

# The amendment

Replace the constant with a measured rule, unchanged in intent (memory is the
binding constraint, not cores):

    cap = min(cores - 2, floor((free_memory - reserve) / measured_RSS))

with `reserve` = 4 GB of headroom and `measured_RSS` the resident size of ONE
run of the family being launched, measured immediately before the batch rather
than assumed. Families are measured separately: the promoting and lifecycle
learners allocate `(tasks, centres, features)` tensors during clustering and are
expected to be materially larger than the discrete family's 364 MB. If a
family's resident size has not been measured, the old constant stands for it.

FREE MEMORY IS NOT STABLE ON THIS HOST, so the cap is not a launch-time
constant either. Measured 2026-09-04: free memory fell from 9.8 GB to 4.7 GB
within hours with no research process changing, because non-research processes
(three `node` instances, ~6.3 GB) grew; a 3-worker gate batch launched under
the earlier figure was killed by the harness's low-memory guard under the later
one. The rule is therefore applied at DISPATCH: a worker starts only if live
free memory minus `reserve` still covers one more `measured_RSS`; otherwise the
pool waits. A cap computed once and trusted for the batch's lifetime would have
been wrong by 2x that afternoon.

The one-writer-per-cell requirement and the `ProcessPoolExecutor`-over-a-job-list
pattern are UNCHANGED and remain mandatory; nothing here licenses a shell loop or
a second launcher instance.

# Second change: pool the independent offline cells

Audit drivers that compute independent one-writer cells serially (Stage D's
C_lo/L_lo/L_hi grid is the immediate instance) dispatch them through the same
bounded pool their lifetimes already use. Requirements, all of which Stage D's
cells already satisfy and which any pooled driver must:

- every cell writes to a disjoint output path and its own report key;
- every cell's randomness comes from an explicit `SeedSequence` with integer
  components, never from arrival order, wall time, or a shared global generator;
- every worker sets `torch.set_num_threads(1)`;
- the report is written atomically after each cell, under a protocol
  fingerprint, and a cell computed under a different fingerprint is refused;
- a failed worker fails the batch with a nonzero exit code rather than leaving
  a silently missing cell.

# Acceptance gate (bitwise, and it can fail)

Parallelism is admissible only if it is invisible in the output. Before the rule
is used for any scientific batch:

1. EQUIVALENCE. Run a small multi-cell batch (three offline cells, one short
   world) serially and again through the pool at cap. Every artifact and every
   report cell must be BITWISE identical, including `metrics.jsonl` byte hashes.
   Any difference means a cell's randomness or state leaks across the schedule
   and the change is rejected, not tuned.
2. MEMORY. Peak system memory across the pooled batch must stay under
   `free - reserve`, sampled while it runs, with no swap growth.
3. NO SPEED CLAIM WITHOUT THE FAILURE MODE. Report wall clock for both arms,
   and confirm the pooled arm produced the same number of cells with the same
   exit code. A faster batch that dropped a cell is a failure.

If (1) fails, the finding is more valuable than the speedup: it means some
existing sweep's results depend on its schedule.

Gate 1 RESULT (2026-09-05, machine otherwise idle): nine real Stage D cells (C_lo/L_lo/L_hi shapes x rotated worlds 0-2, registered SeedSequence streams, 48 updates each) run serially and then through a 3-worker ProcessPoolExecutor: **9/9 cells bitwise identical** (every checkpoint, per-task NMSE, routing diagnostic, and parameter-change statistic), 663.0 s serial versus 251.5 s pooled (2.64x on 3 workers). A first attempt the previous afternoon was killed by the harness's low-memory guard when non-research processes consumed 5 GB; that is the dispatch-time rule's origin, not a gate failure. Result file: scratch `equivalence_gate_result.json`, reproduced by `equivalence_gate.py` (scratch; the pooled driver itself is not yet in the repository).

# What this plan does not do

It does not touch the learner numerics; the 2x batched-slot forward measured in
`notes/performance_audit.txt` is a separate, versioned proposal and is NOT
authorized here. It does not raise the cap for any family whose resident size
has not been measured. It does not apply to remote workers, which already own
disjoint cells. And it may not be applied to a batch that is already in flight:
a run's concurrency is fixed at its launch.
