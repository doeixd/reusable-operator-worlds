# G1V: does the `tanh` learner form online with the full pipeline on worlds that chose nothing? (Tier 1, EXPLORATORY)

Status: FROZEN 2026-10-17.

Runner `row.experiments.g1v_tanh_probe`; independent scorer `row.experiments.score_g1v_tanh_probe`; tests
`tests/test_g1v_tanh_probe.py`. Development band 6: worlds 71, 72 (G1T's, paired) and 73, 74, 75 (fresh; verified
unused as world seeds, 73 occurs only as a `SeedSequence` component in `learned_lifetime`), stream 0.

# Why

Decision 22 (opened by G1R): G1R's frozen `STALE_ONLY` branch stops the `tanh` line, while the stop rule Claude stated
to the PI would continue to a formation probe because re-routing rescued 54/54 collapsed tasks. The PI answered "Use
your best judgement to continue"; Claude chose (a), this probe. It is not a further setting: every setting is fixed by
world 70 (G1S lr 0.003 / gain 1; G1S2 sleep length; G1R wake u = 2, the better vocabulary once routes are repaired, and
re-route-then-sleep order) and is evaluated on five worlds that chose none of them. It answers Track G's first question
for a second operator form: does online formation survive the change of form?

# Construction (arms are constructions)

| arm | construction |
|---|---|
| `WAKE_w{w}` | `o2.run_single('SHUFFLED')` verbatim under `g1s2.apply_budget(2)` (lr 0.003, gain 1, 2 updates per example in every stage) |
| `RR_SLEEP_w{w}` | on that wake terminal: `o5.reroute` verbatim with the tanh search view (G1R), then `g1s2.sleep_curve` (`o2c.consolidate`'s loop, bitwise per G1S2 gate B3) for 65,536 updates with seed `[1941, w, 0, 64]`, scored at 8,192 / 16,384 / 32,768 / 65,536; `g1t.formed_s1` at 65,536 |

10 cells. Each scored point records the terminal on the canonical 64, per-task terminals, per-depth medians over the 188
stream tasks, depth-1 identity ARI and the library hash.

**Gates.** V1: every stage config of every world carries lr 0.003 and 2 updates per example, the initial gain is 1, and
`o5`'s search view is the tanh view. V2: that view reproduces a fresh learner's hard prediction bitwise at depths 1-3.
V3: `restore()` returns the defaults. **Per cell:** G1T's `o2.validate_cell` on every wake and the lifetime's own
resolved lr and updates; pool 188 x 64, lr, sleep lr and gain; finite terminals; denominators 60 / 64 / 64; the library
changes between checkpoints; formed S1 over 24 hyperplanes.

# Reading (written before any cell; exploratory, no verdict)

Per world, at the REGISTERED point 65,536 (never the best checkpoint): FORMS iff terminal median `< 0.05` AND
`d1_ari >= 0.8`. Over all five worlds (no anchor filter; the denominator is 5):
- `>= 4` -> `FORMS`. Next: G3 on the formed `tanh` vocabularies (does a decision on a computed state become necessary
  on this form? G1T's partly fitted vocabulary read formed S1 0.92 against the teacher's 0.877, the wrong direction);
- `<= 1` -> `FAILS`: the `tanh` line stops, recorded as "on fresh worlds the full pipeline reaches X"; next is the
  synthesis-cycle census (`SYNTHESIS_CYCLE_PROPOSAL.md`);
- otherwise -> `MIXED`: possible, not reliable (the O1 -> O2 lesson); the line stops here too, with the rate recorded,
  since a reliability rung on a near-threshold form is not worth its cost against the synthesis question.

Secondary reads: worlds below 0.1; collapsed tasks (> 0.5) at 65,536; the sleep curve; routes changed; per-depth
profiles; formed S1 against the teacher's (G0F 0.875-0.878 on worlds 70-72); worlds 71-72 against G1T's `RW_SLEEP`
there (0.393 / 0.388).

# Necessity

The behaviour is online formation of the `tanh` vocabulary. The arm that refuses it is G1T's construction (rotated
hyperparameters, in-stream re-route, 8k sleep) on worlds 71-72: its refusal cost is measured, 0.393 / 0.388 against the
0.05 bar (scale: about eight times the bar), and on world 70 the full pipeline reached 0.054-0.066 (G1R). The impostor
is a vocabulary that fits without separating the operations, excluded by the `d1_ari >= 0.8` clause, which is part of
the rule. A second impostor is choosing a lucky checkpoint, excluded by reading only the registered 65,536 point.
Difficulty band: the target is the rotated pipeline's 0.009-0.03 (G1); world 70 sits at 0.054-0.066, just above the bar,
so this probe sits near the threshold by design and the near-threshold fragility is stated below.

# Discriminating power

**The rule:** `FORMS` iff at least 4 of 5 worlds have terminal `< 0.05` and `d1_ari >= 0.8` at 65,536; `FAILS` iff at
most 1; else `MIXED`. Exact binomial over five independent worlds, `p` the true per-world rate:

| true `p` | P(`FORMS`) | P(`FAILS`) | P(`MIXED`) |
|---|---|---|---|
| 0.05 (null) | 0.00003 | 0.977 | 0.023 |
| 0.30 | 0.031 | 0.528 | 0.441 |
| 0.50 | 0.188 | 0.188 | 0.625 |
| 0.70 | 0.528 | 0.031 | 0.441 |
| 0.90 (effect) | 0.919 | 0.0005 | 0.081 |

So `FORMS` false-fires at 0.00003 under the null and is detected at 0.919 under the effect; `FAILS` is detected at 0.977
under the null and false-fires at 0.0005 under the effect. World 70's 0.054-0.066 says the true per-world rate may be
intermediate, where `MIXED` is the likely and intended outcome.

**What the checks do not cover:**
- near-threshold fragility: world 70 sat at 0.054-0.066, so single-stream cells will scatter around 0.05 and one
  world's label can flip on stream noise; the pattern and the `below_loose` count are read with the label;
- one stream per world;
- end-of-stream re-routing only; in-stream re-routing (O11's pipeline) is not run on this form;
- the settings were chosen on world 70 and are not re-tuned; a different world might prefer others;
- the thresholds are the program's (0.05, ARI 0.8); `FORMS_K = 4` of 5 is a judgement.

# Operational

10 cells, pool 3 (~340 MB each): the five wakes first (~33 min each at 2 updates per example, from G1S2), each sleep
submitted when its wake exists (~41 min, from G1R). Expected about 2 hours. Cells are written atomically to
`artifacts/g1v_tanh_probe/cells/`, stamped with the protocol hash (this plan, config, the G1T and G1R reports, every
reused module, commit); relaunch resumes; at exit the runner archives logs and cells into
`reports/g1v_tanh_probe_<date>/`. Dry run: world 99, scale 1/16, sleep checkpoints 64 and 128, both arms, with a
restart test. Launched detached from clean committed code; the scorer recomputes the label from the cells.

**Write-time check (2026-10-17), item by item against the implementation learnings.** Opportunity: world 70 sat at
0.054-0.066, so a fresh world can land on either side of 0.05. Discrimination: exact binomial rates above. Threshold
against baseline: G1T's paired cells (0.39) read FAILS, world 70 sits just above, so every label is reachable.
Denominator: five worlds, stated, no anchor filter (fresh worlds have no rotated reference). Estimands against code:
the label reads ONLY the registered 65,536 point, never the best checkpoint, and a test shows an earlier checkpoint
below the bar does not count (the first-crossing rule's cousin: picking the best of four points would inflate the rate).
Fit and score: no setting is chosen on these worlds. Arms are constructions; the sizing lesson from G1R is applied (the
FULL pipeline, re-routing included). Non-vacuity: per-cell checks on lr, updates, gain and pool can fail; the library
must change between checkpoints. Found one operational gap from G1R's dry run and fixed it here: the host precondition
now writes a `run.log` line before it can refuse, so a refused launch is visible rather than inferred.

**Dry run disclosure.** World 99, scale 1/16, checkpoints 64 and 128: gates passed; stopped after one cell, one orphan
worker killed by `parent_pid`, relaunched; the finished cell was reused byte-identical (sha256 `df524abc...`), the
other ran, exit 0, archive written. Dry terminals (~1.05) mean nothing at that budget. No `tanh` learner has been run
on worlds 73-75; worlds 71-72 have only G1T's cells at the rotated hyperparameters.

