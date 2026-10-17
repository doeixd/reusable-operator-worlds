# G1R: are the `tanh` learner's collapsed depth-3 tasks stale routes? (Tier 0, re-route census)

Status: FROZEN 2026-10-17.

Runner `row.experiments.g1r_tanh_reroute`; independent scorer `row.experiments.score_g1r_tanh_reroute`; tests
`tests/test_g1r_tanh_reroute.py`. World 70, stream 0, the two saved G1S2 wake terminals. No new lifetime: every cell
starts from a saved model and applies existing constructions. Diagnostic, with a pre-stated reading that decides
whether the `tanh` line continues.

# Why

G1S2 (`reports/g1s2_tanh_budget.json`) labelled `IMPROVES` (0.092). Post hoc its depth-3 failure is bimodal: at wake
u = 1 after 65,536 sleep updates, 44 of 64 depth-3 tasks sit at 0.062-0.107 and 20 at 0.71-1.06 (u = 2: 30 and 34),
while depth 1 / 2 stay at 0.017 / 0.045. That is the signature the rotated line identified as STALE ROUTES (O6:
collapsed tasks sit near the no-anchor floor; sleep alone cannot repair them because the error is coherent, A1), and
neither G1S nor G1S2 re-routed. The confirmed pipeline re-routes before sleep. The PI agreed on 2026-10-17 to test this
first, under a stop rule: the `tanh` tuning line continues only if re-routing rescues the collapsed block and a later
formation probe forms on fresh worlds.

# Construction (arms are constructions)

For each saved G1S2 wake terminal (`artifacts/g1s2_tanh_budget/work/u{u}/lifetime/model.pt`, u in {1, 2}, sha256 in
the protocol), reloaded with `o3.load_shuffled_terminal` under G1S2's `apply_budget(u)` (lr 0.003, gain 1):

| arm | construction |
|---|---|
| `RR_SLEEP_u{u}` | `o5.reroute` verbatim (exhaustive search per stream task over all `12^d` routes on its 64-per-task reservoir, minimal logit swap), then `g1s2.sleep_curve` (`o2c.consolidate`'s loop, bitwise per G1S2 gate B3) for 65,536 updates, scored at 8,192 / 16,384 / 32,768 / 65,536. O6's confirmed order (wake -> re-route -> sleep) at G1S2's longest sleep |
| `SLEEP_RR_u{u}` | the same sleep WITHOUT re-routing first (must reproduce G1S2's `SLEEP_u{u}` at every checkpoint), then `o5.reroute` on the slept model, scored (`rr`), then `o2c.consolidate` for 8,192 more updates with the same seed, scored (`rr_sleep`) |

Four cells. Every scored point records the terminal on the canonical 64, per-task terminals, per-depth medians over the
188 stream tasks, depth-1 identity ARI and the library hash; each re-route records how many routes changed.

**Gates.** R1: with `o5.FrozenLibrary` rebound to `FrozenTanhLibrary` (o5 imports its own reference, which G1T's
rebinding did not cover), the search view reproduces the reloaded learner's hard prediction bitwise on one stream task
of each depth. R2: `restore()` returns the rotated view and the defaults. R3: both terminals exist. **Per cell:** pool
188 x 64, learner, lr and gain as set, sleep lr 0.003, library changes between checkpoints, all points finite with
denominators 60 / 64 / 64.

**Anchor A0.** Every reload reproduces G1S2's `WAKE_u{u}` per-task terminals exactly, and `SLEEP_RR_u{u}`'s sleep
reproduces G1S2's `SLEEP_u{u}` per-task terminals exactly at all four checkpoints.

# Reading (written before any cell)

- **Collapsed set** `C_u`: tasks above 0.5 in G1S2's `SLEEP_u{u}` at 65,536 (20 and 34; the two modes are separated by
  0.107 to 0.706, so the cut is not sensitive). **Stale fraction**: the share of `C_1 + C_2` (54 tasks) whose terminal
  is below 0.2 after re-routing the slept model (`SLEEP_RR` point `rr`).
- **Primary**: the best checkpoint of `RR_SLEEP_u1` (G1S2's chosen wake) among those with `d1_ari >= 0.8`.

Labels:
- A0 fails -> `ANCHOR_MISS`: nothing is read;
- primary `< 0.05` -> `FITS`. Next: a frozen formation probe on worlds 71-72 (paired with G1T) and fresh band-6
  worlds at lr 0.003, gain 1, wake u = 1, with re-routing and the sleep length chosen here; never world 70;
- else stale fraction `>= 0.5` -> `STALE_ONLY`: the collapse was stale routes, but even with routes repaired the
  working tasks do not reach the bar. By the stop rule the `tanh` tuning line stops here, recorded as "forms with the
  full pipeline to about X, not to 0.05";
- else -> `NOT_STALE`: the collapse is not stale routes; the line stops.

Secondary reads: `RR_SLEEP` against G1S2's sleep alone at every checkpoint (does re-routing BEFORE sleep also improve
the working mode, as O6 found on the rotated form?); routes changed; per-depth profiles.

# Necessity

The behaviour is re-routing's repair of collapsed tasks. The arm that refuses it is sleep alone, G1S2's own cells,
reproduced here as A0: its refusal cost is measured, 20 + 34 tasks at 0.71-1.31 against 0.06-0.11 for the rest (scale:
about ten times the working mode, the whole gap between `IMPROVES` and a pass). Impostors: (a) re-routing could lower
error by re-assigning tasks onto slots that overfit the 64 retained examples; the terminal is scored on held-out
evaluation examples, so this cannot pass; (b) a "rescue" that breaks identity; the identity filter excludes it from the
primary. Difficulty band: the target is the rotated pipeline's 0.009-0.03; a result below 0.005 would be checked
against the teacher before use.

# Discriminating power

**The rule:** `FITS` iff `RR_SLEEP_u1`'s best identity-keeping checkpoint `< 0.05`; else `STALE_ONLY` iff the stale
fraction over the 54 collapsed tasks `>= 0.5`; else `NOT_STALE`.

Simulated (200,000 draws, seed 20261017):
- stale fraction, tasks drawn binomially with a per-vocabulary rescue rate drawn from a Beta (correlated within a
  vocabulary): null (rate ~0.1, Beta(1, 9)) false-fire 0.0003; effect (rate ~0.8, Beta(8, 2)) detection 0.995; a
  rate of ~0.5 lands either way (0.53), the intended ambiguity;
- `FITS`, terminal log-normal: null (repair fixes the collapse but leaves the working mode at its 0.086 median, log-sd
  0.25) false-fire 0.015 for one checkpoint and at most 0.058 if the four checkpoints were independent (they are one
  trajectory, so the true rate is near the first figure); effect (centred on 0.02, log-sd 0.5) detection 0.967.

**What the checks do not cover:**
- one world, one stream, two vocabularies from one world; world 70 has chosen every setting so far, which is why any
  probe that follows excludes it;
- end-of-stream re-routing only; the in-stream variant (O8/O11) is not run;
- the 0.5 and 0.2 cuts are judgements, set from G1S2's observed gap (0.107 to 0.706);
- the reading's stop branches are a program decision agreed with the PI, not a scientific verdict on the form.

# Operational

4 cells, pool 4 (~340 MB each). Each cell is one 65,536-update sleep (~37 min on this host from G1S2) plus five or six
scorings and one or two re-routes (`12^3` enumeration per depth-3 task, under a second each); `SLEEP_RR` adds 8,192
updates. Expected about 45 minutes. Cells are written atomically to `artifacts/g1r_tanh_reroute/cells/`, stamped with
the protocol hash (this plan, config, the G1S2 report, both terminals' sha256, every reused module, commit); relaunch
resumes; at exit the runner archives logs and cells into `reports/g1r_tanh_reroute_<date>/`. Dry run: the same four
cells with sleep checkpoints 64 and 128 (no reading), restart test included. Launched detached from clean committed
code; the scorer recomputes the reading from the cells and G1S2's committed cells.

**Write-time check (2026-10-17), item by item against the implementation learnings.** Opportunity: re-routing can
rescue the collapsed block or not, and the working mode can move or not. Discrimination: rule simulated against null and
effect samplers (above). Thresholds against baselines: the collapsed cut 0.5 and rescue cut 0.2 sit inside G1S2's
observed gap (0.107 to 0.706); the `FITS` bar is the program's 0.05; the baseline (sleep alone, A0) reads `NOT_STALE`
with stale fraction 0 by construction, so the null branch is reachable. Denominators: the stale fraction is over the 54
collapsed tasks named by G1S2's committed cells (20 + 34, checked in a test against the real report); depth counts
60 / 64 / 64 per point. Estimands against code: every terminal is the final model on the canonical 64 at that point;
the collapsed set comes from G1S2, not from this run, so it is not fitted on the data it scores. Arms are stated as
constructions. Non-vacuity: A0 can fail; R1 compares the search view to the learner; the library must change between
checkpoints. Found one gap, fixed before freezing: `o5_reroute_sleep` holds its own `FrozenLibrary` reference, which
G1T's name rebinding did not cover, so `o5.reroute` would have searched the tanh library through the ROTATED candidate
function; it is now rebound and gated (R1). Same class as the E5 `S` arm: a reused function is only as reused as every
name it resolves.

**Dry run disclosure.** The four cells on the real world-70 terminals with sleep checkpoints 64 and 128 (no reading).
The first restart-test launch exited 1 before writing `run.log`, most likely the host precondition (free memory was
hovering at 2-2.7 GiB; the precondition raises before the log opens), and its record was overwritten by the next
launch, so the cause is inferred, not shown. Repeated with 4.5 GiB free: stopped after one cell, four orphan workers
killed by `parent_pid`, relaunched; the finished cell was reused byte-identical (sha256 `fde85774...`), the rest ran,
exit 0, archive written. Because the dry run uses the real terminals, its numbers are a PEEK at world 70: re-routing then
128 sleep updates gave 0.27-0.29, and `SLEEP_RR` (128 sleep, re-route, 8,192 sleep) gave 0.088 / 0.099 against G1S's
0.184 for 8,192 sleep alone; routes changed 35-36 (u1) and 61-63 (u2). The reading above was written before the dry
run and was not changed after it.

