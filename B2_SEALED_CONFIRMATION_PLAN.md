# C-seal: sealed confirmation that the learner discovers which tasks iterate, while forming the vocabulary (Tier 2)

Status: FROZEN 2026-10-13, before any sealed world was generated. **Tier 2, CONFIRMATORY.** Program:
`GENERALITY_PROGRAM.md`, Track C-seal (decision 19). Sealed band 1100-1129 (allocated with decision 19, verified
unused as world seeds): this plan uses worlds **1100-1114**; 1115-1129 stay sealed and unused. Anchor cell on
development world 57; dry run on development world 67 (both band 5, already used by B2-hard v2).

# Why

The control-flow line's positives are all Tier 1 on development worlds. The strongest is B2-hard v2
(`B2_HARD_V2_PLAN.md`, `reports/b2_hard_v2.json`, worlds 57-59): with every depth-1 candidate STARTING at the plain
one-step form, the learner formed the vocabulary 9/9 and discovered which of the 108 candidates iterate without being
told (wake state-weight AUC 1.000, recall 0.986, specificity 1.000, terminal loops below 0.38 in 9/9), and the
zero-start control rooted in 7/9 cells against 0/9 (`ROOTS_PREVENTED`). B2-depth2 reproduced the construction with
two-step loop bodies. This plan confirms the depth-1 construction on sealed worlds, unchanged.

**Scope stated in advance:** a confirmation on the rotated-residual generator. It does not address whether the result
holds in other worlds (Track G).

# Constructions (B2-hard v2, unchanged)

Runner `row.experiments.b2_sealed`. Every cell is `b2_hard_v2.run_cell` with that module's frozen constants:
`loop_stream` (O2's SHUFFLED depth-3 stream: 60 length-1, 64 length-2, 64 canonical length-3 tasks, plus 48 loop tasks;
236 tasks, one random order, 128 examples per task); candidates are the 108 depth-1 tasks (planned depth given, as
disclosed in B2-hard v2).

| arm | construction | cells |
|---|---|---|
| `ONESTEP` (confirmatory) | `LoopLearner(loop_plan = all 108 candidates, state_halt, step_bias, detached halting input, halt_decay 0.1, halt_init 4)`; wake; end of stream: `deep_reroute.reroute` of the depth-2/3 tasks, the ONE_STEP data test per candidate (count form L in {0, 1} against the loop form; keep the loop iff its support MSE `< 0.5 x` the count form's), then B2-hard's decayed sleep (8,192 updates) | 15 worlds x streams 0-1 = 30 |
| `ZERO` (mechanism control) | identical except `halt_init = None` (zero start) | 15 worlds x stream 0 = 15 |
| `GIVEN` (instrument gate, cost reference) | B2-online v2's `LOOP` (loop identity given; v2's re-route, re-fit and sleep) | 15 worlds x stream 0 = 15 |
| anchor | `ONESTEP` on development world 57 stream 0, rerun | 1 |

61 cells. The anchor and every arm reuse B2-hard v2's cell function, so the dry run checks all three arms against
B2-hard v2's committed dry records bitwise, and the anchor checks the full-scale path against its committed
development record.

# Rules (registered)

Per cell, as in B2-hard v2: terminal canonical median (query NMSE, hard routes and hard halting), terminal loop median,
wake state-weight AUC (L2 norm of each candidate's halting weights after wake; loop positive, straight negative; ties
half), end-of-stream decisions, rooted operations at the end of wake (a teacher operation all of whose straight
anchors are best fit at a fixed count L >= 2; teacher programs enter only this record).

**Gate.** `anchor` = the rerun `ONESTEP_w57_s0` reproduces its committed record in `reports/b2_hard_v2.json` exactly
in `library_sha256`, the per-task canonical, loop and anchor NMSE, the 108 decisions and the rooted operations.
`given` = `GIVEN` forms (canonical median `< 0.05`) in at least 12 of its 15 cells.

**Primary label (`ONESTEP`, 30 cells):** `k_formation` = cells with canonical median `< 0.05`; `AUC_med` = median of
the 30 wake AUCs; `recall` = loop tasks kept as loops / 1,440; `specificity` = straight candidates given the count
form / 1,800; `n_terminal` = cells with loop median `< 0.38`.

| condition (first matching row) | label |
|---|---|
| not (`anchor` and `given`) | `UNINTERPRETABLE` |
| `k_formation <= 26` | `FORMATION_BROKEN` |
| `AUC_med >= 0.9` and `recall >= 0.8` and `specificity >= 0.95` and `n_terminal >= 27` | `CONFIRMED` |
| otherwise | `NOT_DISCOVERED` |

**Secondary label (mechanism):** `r_one` = rooted `ONESTEP` cells of 30, `r_zero` = rooted `ZERO` cells of 15.

| condition (first matching row) | label |
|---|---|
| `r_zero <= 1` | `NO_ROOTS_IN_CONTROL` |
| `r_one <= 1` | `ROOTS_PREVENTED` |
| otherwise | `ROOTS_NOT_PREVENTED` |

Changed from B2-hard v2's `r_one == 0` of 9 to `r_one <= 1` of 30, so that one rooted cell in a block three times
larger does not by itself decide the label (detection at a 2% per-cell root rate: 0.87 here, against 0.55 for
`r_one == 0` of 30). Stated before any sealed data.

**What the labels mean (stated before data):**
- `CONFIRMED`: on sealed worlds, a learner whose every depth-1 candidate starts as a plain step forms the vocabulary
  reliably (at least 27 of 30) and discovers which tasks need data-dependent iteration without being told. With
  `ROOTS_PREVENTED`, the mechanism (the plain start prevents iterated-slot impostors) is confirmed too.
- `FORMATION_BROKEN`: the development 9/9 did not hold; the one-step start does not make formation reliable.
- `NOT_DISCOVERED`: formation holds but a discovery clause fails; read which.
- `UNINTERPRETABLE`: the anchor did not reproduce (a construction change, not data) or the identity-given reference
  failed to form on the sealed worlds.

**Registered early stops:** if the anchor cell does not reproduce, every pending cell is cancelled (the run would be
`UNINTERPRETABLE`). Once 4 `ONESTEP` cells on sealed worlds fail formation (`FORMATION_BROKEN` is then determined),
the pending `ONESTEP` cells are cancelled; `ZERO` and `GIVEN` still run, because they are needed to read the failure.
A cancelled cell counts as not formed, and `CONFIRMED` requires all 30 `ONESTEP` cells.

**Reported, not registered:** the cost of discovery (paired `ONESTEP` stream 0 against `GIVEN`: loop-median ratio,
cells at or below `GIVEN`); `ZERO`'s formation count; whether every failing `ZERO` cell is rooted (the B2-hard wake
census prediction, 4 of 4 in B2-hard v2); wake, re-route-only and terminal medians per arm; count accuracy; routes
changed; per-task decisions; seconds per cell.

**Non-vacuity (fails the cell, `b2_hard_v2.validate`):** 236-task stream, 64 canonical and 48 loop tasks scored, 108
candidates and 60 anchors in the root record; sleep changed the library; wake moved the halting parameters from their
initial values; each arm's `halt_init` is the registered one; 108 decisions in every candidate arm.

# Necessity

- **Target behaviour:** discovering which tasks need data-dependent iteration, without being told, while forming the
  vocabulary.
- **Refusal arm:** not re-run; measured twice in development (B2-online v2's `CONSTCOUNT`, B2-hard's `ALLCOUNT`: the
  same iteration machinery with no state access), terminal loop medians 1.07 and 1.074 against 0.255-0.305 with state
  halting.
- **Measured refusal cost and its scale:** loop median ~1.07 refusing against ~0.29 discovering; the scale is that
  gap and the 0.38 bar.
- **Impostors:** the ROOT impostor (a slot iterated L times standing in for a primitive; the `ZERO` arm reproduces
  it); identity leaking (identity enters only the AUC, recall, specificity and the root record); keeping loops
  everywhere or nowhere (specificity, recall); the planned depth (narrows to 108 candidates; disclosed).
- **Difficulty band, higher-is-harder:** loop median NMSE, band `(0.0005, 0.5)`; `GIVEN` near 0.28 in development.

# Discriminating power

**The decision rules, as they will be applied:** `CONFIRMED` iff the anchor reproduces, `GIVEN` forms in at least
12/15, `k_formation >= 27` of 30, `AUC_med >= 0.9`, `recall >= 0.8` of 1,440, `specificity >= 0.95` of 1,800 and
`n_terminal >= 27` of 30. `ROOTS_PREVENTED` iff `r_zero >= 2` of 15 and `r_one <= 1` of 30.

Source `reports/b2_sealed_design/rates.py`, output `rates_output.txt` (seed 43; 1,000 draws of 30 cells for the AUC
clause, exact binomials otherwise):

| clause | null | false-fire | effect | detection |
|---|---|---|---|---|
| `k_formation >= 27` of 30 | per-cell 0.75 (unreliable; B2-hard's broken rate was 0.67) | **0.037** | per-cell 0.95 / 0.98 | **0.939 / 0.997** |
| `n_terminal >= 27` of 30 | per-cell 0.75 | **0.037** | per-cell 0.98 | **0.997** |
| `AUC_med >= 0.9` | d' = 0 | **0.000** | d' = 2 (AUC ~0.92) | **0.999** |
| `recall >= 0.8` of 1,440 | per-task 0.75 | **0.000** | per-task 0.85 | **1.000** |
| `specificity >= 0.95` of 1,800 | per-task 0.93 | **0.0003** | per-task 0.97 | **1.000** |
| `GIVEN` gate `>= 12` of 15 | per-cell 0.5 | 0.018 | per-cell 0.98 | **0.9998** |
| `ROOTS_PREVENTED` | start has no effect (both arms at the zero-start rate 7/19 = 0.368) | **0.000** | one-step roots at 0.02 / 0 per cell | **0.871 / 0.990** |

Development baselines these are checked against: B2-hard v2 `ONESTEP` 9/9 formation, AUC 1.000, recall 0.986,
specificity 1.000, `n_terminal` 9/9; `GIVEN` 9/9; `ZERO` rooted 7/9.

**What the checks do not cover:**
- The formation null is "unreliable" at 0.75 per cell; at 0.80 the clause passes with probability 0.12 and at 0.85
  with 0.32. `CONFIRMED` therefore says formation is reliable at roughly the 0.9 level or better, not that it never
  fails.
- `ROOTS_PREVENTED` detection drops to 0.55 once the one-step start roots in 5% of cells.
- Pooled recall and specificity treat tasks as independent; per-cell values are reported. ONESTEP's two streams share
  each world, so the 30 cells are 15 worlds with two draws each, not 30 independent worlds.
- One generator family, planned depth given, one-operator loop bodies, linear halting, K = 6, 64 retained examples.

# Operational

61 cells on a pool of 4 (memory measured on the dry run; `LoopLearner` lifetimes are below the `slots=12` promoting
footprint), anchor first, then `ONESTEP`, `ZERO`, `GIVEN` (longest first). Development per-cell times: `ONESTEP`
~2,300-2,900 s, `ZERO` ~2,400 s, `GIVEN` ~1,400-1,600 s; about 9.5 hours. Performance pass: the cell function is
frozen development code that the anchor must reproduce bitwise, so no implementation change is admissible; the wins
taken are scheduling (longest first) and the pool size. Restartable, durable stamped cells, protocol fingerprint (this
plan, config, the B2-hard v2 report, this runner, B2-hard v2's full protocol including every module hash, commit);
`run.log`, `status.json` (with an early-stop note), `exit.json`, `precondition.json`; detached launch; reserve 2.0 GiB
(PI 2026-10-03). Dry run: all three arms on development world 67 at scale 16, each reproducing B2-hard v2's dry
record bitwise, with a restart test. No sealed world is generated before this plan is committed and its hash is in
`tools/check_prereg.py`. Independent scorer `row.experiments.score_b2_sealed` recomputes every clause from per-task
records and the B2-hard v2 report and must agree with the runner. Tests `tests/test_b2_sealed.py` (labels, cell list,
gates, early-stopped scoring, `CONFIRMED` requiring every cell, scorer agreement, tampered summary, changed plan).
