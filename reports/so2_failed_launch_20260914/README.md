# SO2 first launch: failed before any cell, and the pre-relaunch audit

Not a result. Operational records of the first SO2 launch, preserved before the
relaunch.

# What happened

- 15:13 UTC, commit `84b1e6c`: launched. The STAGED_w1 cell finished its three
  lifetimes (15:14-15:22). The export margin then raised `KeyError('nmse')` on
  its first adaptation (15:27). `adapt_cell` reports `query_nmse`. No durable
  cell was written, and no classification exists.
- `a31a397` fixed the key and added a contract test.
- 15:29: a relaunch was refused by the clean-code guard because uncommitted H28
  edits were in the tree (`stderr2.log`). SO2 then sat idle while H28 work moved
  HEAD.

Files: `run.log`, `status.json`, `exit.json`, `stdout*.log`, `stderr*.log`, and
`so2_online_gate_partial_report.json` (the untracked report stamped to
`84b1e6c`, with empty `cells`). The failed launch's stage models were moved
out of the run root to `artifacts/so2_online_gate_failed_84b1e6c/`, so the
relaunch starts on an empty root.

# Pre-relaunch audit (2026-09-14), against the frozen plan and Amendments 1-2

1. **Construct error, fixed: M_terminal measured end-of-task error.**
   - The runner classified on the summary's `final_nmse` median. That is each
     task's `curve[max(curve)]`, measured right after that task trained, while
     the library was still changing.
   - The plan's `M_terminal` is "terminal median query NMSE", and B2 in
     `POST_E6_RESEARCH_PROGRAM.md` separates terminal from end-of-task error
     for online arms. J1c's `terminal_median` came from G5R Stage D's `score`.
   - The independent scorer already re-scored the TERMINAL model, so runner and
     scorer disagreed. On the failed launch's STAGED_w1 stage 3: terminal
     0.1264, end-of-task 0.0770.
   - Fix: the runner scores the terminal model with Stage D's `score` and
     reports the end-of-task median beside it. It adds Stage D's last-task
     anchor (terminal equals end-of-task, within 1e-6, because nothing trains
     after the last task) as a HARNESS_FAILED guard in both runner and scorer.
     On the saved models the anchor is 0.0 at all three stages.
   - This implements the frozen text; it is not a plan change.
   - DISCLOSURE: both numbers above were seen for one world and one arm before
     relaunch. The fix follows the plan's literal wording and the program's
     definition; it was not chosen by comparing them against the threshold.
2. **Scorer defect, fixed: strict reload failed.**
   - The lifetime's terminal novel-composition probe leaves
     `task_novel_composition_0` in the model wherever an unseen program
     exists: stage 2 and stage 3, not stage 1.
   - The scorer's `strict=True` load raised on it. It now registers that code
     when the stage record says the probe was available.
   - The probe trains only its own code with the shared library frozen, after
     terminal metrics. The library hash and the canonical tasks' predictions
     are unchanged, which the anchor confirms.
3. **Margin construction confirmed as G5R's.**
   - `adapt_cell` recasts either arm to `VariableDepthDiscrete`, whose forward
     applies each `RotatedLearnedOperator` sequentially. Both arms therefore
     adapt through G5R's own executor, whether the lifetime used the fast or
     sequential kind.
   - Measured cost on the failed launch's STAGED_w1 model, 2000 steps:
     146.5 s trained arm, 390.7 s scratch arm, and 3.0 s for the 64-program
     diagnostic.
   - DISCLOSURE: this timing exposed one program's NMSE (trained 0.152,
     scratch 1.591) and the diagnostic count (5/64) for a model the relaunch
     will probably rebuild exactly. No construction or decision depends on
     them.
4. **Performance pass: bitwise-neutral scheduling.**
   - Cells are independent, single-threaded, fixed-seed computations. The
     launcher now runs them in a 3-worker `ProcessPoolExecutor`, STAGED
     (longest) first. Each worker writes only its own cell; the parent alone
     writes report, status and log.
   - `status.json` now carries running cells and an ETA.
   - If a cell fails, the pool waits for the other running cells to finish
     (their cells are saved durably) before recording the failure.
   - Speeding up the sequential adaptation executor would change floats, so it
     is not applied.
