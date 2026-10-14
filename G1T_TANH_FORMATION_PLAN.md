# G1T: does online formation survive the `tanh` operator form? (Tier 1, EXPLORATORY)

Status: FROZEN 2026-10-14. **Tier 1, exploratory; no verdict.** Program: `GENERALITY_PROGRAM.md`, Track G, decision 20 (a).
G0F (`reports/g0f_form_census.json`, `SELECTED` `tanh2`) found that a teacher operator without an identity skip,
`P(z) = c_2 Q tanh(2 R z + b)`, breaks the near-linearity that closed both Track G axes in the rotated family (linear
predictability S1 0.877 against 0.978) while keeping untrained-library depth-1 clustering and single-operator
learnability. Claude asked whether to adopt it, which means changing the program rule that the learner does not change
(the learner must take the same form); the PI answered "sure" (2026-10-14). Development band 6, worlds **70-72** (G0's,
G0F's and G1's worlds; G1 trained rotated vocabularies on them, nothing has trained on their `tanh` teachers). Dry-run
world 99.

# Why

Every confirmed formation result (J1c through O11) was earned with the rotated form. Adopting the `tanh` form is only
worth it if the confirmed fully online pipeline still forms a vocabulary of that form; if it does not, the formation
story is specific to the rotated form, which is itself a headline. This probe runs that pipeline, unchanged except for
the operator form of teacher and learner, on the three worlds where its rotated version passed 3 of 3 (G1, `m = 0`).

# Construction (arms are constructions; all on stream 0)

**Teacher.** `row.form_world` f1, form `tanh`, gain 2: G0F's `tanh2` library for each world (stream 120). Task ids,
programs, inputs, stream order and stage task sets are those of the committed rotated streams; only the library, and so
every target, differs (gates E1, E2).

**Learner** (`row.models.tanh_form`, implementation `tanh_form_planned_depth_v1`). `PlannedDepthRotatedLearner` with
each of its 12 slots replaced by `P(z) = c Q tanh(A z + b)`: free `A` (`d x d`), free `b`, learnable scale `c`, and
the rotated learner's Householder `Q` (same seeds and reflection counts, so slots alternate between the two components
of O(d)). The teacher form is inside this family exactly. Initialization depends only on the model seed (5000): `A`
Haar-orthogonal at gain 1 (not the teacher's 2), `b = 0`, `c = 1/sqrt(E[tanh(x)^2])`. The scale is named `alpha`, so
the optimizer gives it no weight decay, as for the rotated `alpha`. Task codes, temperature, hard routes and per-task
depth are inherited unchanged. 6,252 slot parameters (rotated kind: ~6,500). Hyperparameters are `configs/v1.yaml`'s
`discrete_model` (global/task LR 0.001/0.05, weight decay 1e-4, temperatures 1.0 -> 0.1): tuned for the rotated form
and NOT retuned for this one.

The runner `row.experiments.g1t_tanh_formation` rebinds names only, as G1 did; no committed module is edited:
`o2.stage_setup` (the form world's stages), `o2.planned_model` and `so2.build_fast` (the tanh learner), `o2.lifetime`
(the tanh learner where the first stage would have built the default kind), and `FrozenLibrary` in `o8`, `census` and
`g1` (`FrozenTanhLibrary`, the same search interface; its candidates are the learner's own function). `learned_lifetime`
still receives the kind label `rotated_discrete_fast`, which selects the hyperparameters above and labels the lifetime's
own fingerprint and compute accounting; those two lifetime fields therefore describe the rotated kind, and every cell
records the learner actually trained.

| arm | construction | role |
|---|---|---|
| `REROUTE_WAKE` | `o8.run_arm` verbatim: O1's SHUFFLED stream (60 length-1 + 64 length-2 + 64 canonical length-3 tasks), wake with exhaustive in-stream re-routing | the confirmed pipeline's wake half |
| `RW_SLEEP` | `o3.run_sleep` verbatim on that cell's `REROUTE_WAKE` terminal | **the confirmed pipeline** (O11 sealed: 45/45 rotated) |
| `SHUFFLED` | `o2.run_single('SHUFFLED')` verbatim: the same stream, wake alone (no re-routing, no sleep) | O11's wake-only arm |
| `SLEEP` | `o3.run_sleep` verbatim on that cell's `SHUFFLED` terminal | O11's sleep-only arm |
| `STAGED` | `o2.run_staged('STAGED')` verbatim: J1c's length curriculum, library carried between stages | the most reliable formation path known (diagnostic) |
| `PLAIN` | `o2.run_staged('PLAIN')` verbatim: the 64 canonical tasks only | the refusing arm (no anchors) |

6 arms x 3 worlds = 18 cells. `SHUFFLED` and `SLEEP` were added before freezing, after reviewer 86
(`reviews/reviewer-feedback-86.txt`) asked whether the SAME interventions explain success on a new form: with
`REROUTE_WAKE` and `RW_SLEEP` they are O11's four mechanism arms (sealed rotated: wake alone 14/45, sleep alone 38/45,
re-routing alone 40/45, both 45/45). **Rotated reference, not re-run:** G1's `m = 0` cells are exactly this pipeline with the
rotated teacher and learner on the same worlds, stream, task ids, programs, inputs, order and seeds
(`reports/g1_formation_probe.json`, hashed into the protocol): `RW_SLEEP` 0.0091 / 0.0122 / 0.0094, `REROUTE_WAKE`
0.0128 / 0.0124 / 0.0102, `PLAIN` 1.95 / 1.92 / 1.92, formed `d1_ari` 1.00 in all three. G1 did not run `SHUFFLED`,
`SLEEP` or `STAGED`; their rotated references are O11's and J1c's rates on other worlds, not paired cells.

**Gates (run before any cell, recorded in `gates.json`):** E1, the form stream equals the committed stream in task
ids, programs, inputs, plan and every stage's tasks (worlds 70 and 99); E2, every task's targets differ; F1, the teacher
library reproduces G0F's `tanh2` S1 record of world 70 exactly; L1, the search view reproduces the learner's hard
prediction bitwise at depths 1, 2 and 3; L2, every rebound name builds or searches the tanh learner. All pass on the
working copy.

# Measurements

- Terminal median NMSE on the 64 canonical tasks (O11's estimand), every cell.
- `RW_SLEEP`, on the slept model rebuilt with `run_sleep`'s own calls (must reproduce its per-task terminal exactly):
  G1's `formed_diagnostics` verbatim (`d1_ari` between teacher primitives and the formed routes of the 60 length-1
  tasks; SG0's identifiability on 64 unseen length-3 programs, near ties, runner-up support NMSE, search winner's
  query NMSE), and the FORMED vocabulary's S1: G0's `s1_linear` with its own draws (24 hyperplanes per world, 72 over the three, paired with
  G0F's teacher S1 of the same world: 0.875 / 0.878 / 0.877) on the library in which teacher primitive `p` is replaced
  by the slot most of `p`'s length-1 tasks route to (mapping and its distinct-slot count recorded).
- `REROUTE_WAKE`: wake-terminal `d1_ari`, routes changed by re-routing, staleness (O8's fields). `SHUFFLED`: its
  wake-terminal `d1_ari`. Every cell records its wall seconds, which with the re-routing seconds give the compute each
  intervention costs (reviewer 86, priority 3; reported, not registered).

# Reading (written before any cell; exploratory, no verdict)

A slept cell FORMS iff its terminal median `< 0.05` (O11's threshold) AND its `d1_ari >= 0.8` (formed rotated
vocabularies 0.97-1.00; an untrained `tanh2` library already clusters at 0.42-0.47, G0F S3, so a lower bar would not
separate formation from initialization). A world is ANCHORED iff G1's rotated `RW_SLEEP` passed there (all three did).
Over anchored worlds, count forming `RW_SLEEP` cells:
- all form -> `FORMS`: the confirmed pipeline forms a vocabulary of the `tanh` form at the rotated hyperparameters;
  the next rung is the necessity census for a mid-state decision (G3) on these vocabularies;
- none -> `FAILS`; read with `STAGED`:
  - `STAGED` passes -> the form can be formed, but the online pipeline does not carry over to it;
  - `STAGED` fails too -> this learner at the rotated form's hyperparameters cannot form the `tanh` form; the next rung
    is a learner-side sizing (initial gain, learning rate) on one world, not abandonment of the form;
- otherwise -> `MIXED`: more streams or worlds are needed before anything is read;
- no anchored world -> `ANCHOR_MISS`.

A passing terminal with `d1_ari < 0.8` is reported as an IMPOSTOR (a vocabulary that fits the tasks without recovering
operation identity) and does not count as forming. Secondary reads, every label: the MECHANISM pattern - passes of `SHUFFLED`, `SLEEP`, `REROUTE_WAKE` and `RW_SLEEP` set
beside O11's rotated ordering (wake < sleep, re-route < both); with three worlds this is a pattern to read, not a rate
to compare; `PLAIN` (are anchors still needed?),
`REROUTE_WAKE` (does wake alone pass, as on the rotated form?), the formed identifiability against G1's rotated values
(margin 79-148, no near ties), and formed S1 against the teacher's 0.877 (does a formed vocabulary inherit the
non-linearity the form was chosen for?).

# Necessity

The behaviour is formation of a reusable vocabulary from an order-free stream. The arm that refuses it is `PLAIN` (no
anchors); its refusal cost on these worlds with the rotated form is the whole effect, terminal 1.92-1.95 against
0.009-0.012 (G1 `m = 0`; scale: two orders of magnitude), and it is measured here for the `tanh` form. The expected
impostor is a vocabulary that fits tasks without separating the operations, exposed by the `d1_ari >= 0.8` clause,
which is part of the rule rather than a diagnostic. Difficulty band: the teacher-side runner-up NMSE of `tanh2` is
1.76-1.81 (G0F S2, above G0's band `[0.01, 0.5]`), so route inference is not expected to be hard on this form; the
probe measures where the formed vocabulary sits.

# Discriminating power

**The rule, as applied:** among anchored worlds, `FORMS` iff every `RW_SLEEP` cell has terminal median `< 0.05` and
`d1_ari >= 0.8`, `FAILS` iff none does, else `MIXED`.

With three worlds, one stream each, the rule is a sizing instrument, not a test. Exact binomial rates for three anchored
worlds, where `p` is the true per-cell rate of forming:

| true `p` | P(`FORMS`) | P(`FAILS`) | P(`MIXED`) |
|---|---|---|---|
| 0.05 (null: formation broken on this form) | 0.0001 | 0.857 | 0.143 |
| 0.50 | 0.125 | 0.125 | 0.750 |
| 0.95 (effect: as on the rotated form) | 0.857 | 0.0001 | 0.143 |

So under the null the false-fire rate of `FORMS` is 0.0001 and its detection under the effect is 0.857; `FAILS` is
detected at 0.857 under the null with false-fire 0.0001 under the effect. Intermediate rates mostly land in `MIXED`,
the intended outcome: one stream per world can say formation is POSSIBLE or ABSENT, not that it is RELIABLE (O1 -> O2).
The identity clause was checked against its baselines before it was set: formed rotated 0.97-1.00, untrained `tanh2`
0.42-0.47.

**What the checks do not cover:**
- one stream per world, cells near 0.05 are fragile; the pattern across cells is read;
- the hyperparameters were tuned for the rotated form; a `FAILS` is about these settings, which is why `STAGED` and the
  pre-stated branches above are part of the reading;
- the reading was written after G0F and G1 and with knowledge of their numbers; worlds 70-72 have been measured by
  G0, G0F (teacher statistics) and G1 (rotated vocabularies), never with a `tanh` learner;
- the formed S1 and identifiability reads have no registered thresholds; they size the next rung;
- the lifetime's own fingerprint and compute accounting carry the rotated kind label (disclosed above);
- the learner initialization (gain 1, `b = 0`) is a design choice; a different one could form differently.

# Operational

18 cells, pool 3, longest first (`REROUTE_WAKE`, then `SHUFFLED`), each sleep arm submitted when its wake terminal
exists. Timing on this
host: one update 28 ms (rotated 31 ms); one `12^3` route enumeration on 64 examples 28 ms (rotated 21 ms), so the
re-routing wake should take ~1.2x G1's 1,440 s. Expected about 1.5 hours in all. Each cell is written atomically to
`artifacts/g1t_tanh_formation/cells/`, stamped with the protocol hash (this plan, config, G0F and G1 reports, every
reused module, the form and learner modules, commit) and reused on relaunch only under that stamp; the PARENT writes
every `run.log` line (G0F learning); `status.json`, `run.pid`, `exit.json`, `precondition.json` and `gates.json` are
kept and archived to `reports/g1t_tanh_formation_<date>/`. Restart path tested on the dry run (world 99, scale 1/16).
Launched detached from clean committed code. Independent scorer `row.experiments.score_g1t_tanh_formation` recomputes
the reading from the cell records and the G1 reference, and refuses a report whose plan or inputs changed. Tests:
`tests/test_g1t_tanh_formation.py`.

**Write-time check (2026-10-14).** Found one defect before freezing: the formed-S1 measurement said "72 hyperplanes"
for one world, where `s1_linear` draws 24 per world (72 over three) and the runner validates 24; corrected above. The
two mechanism arms were added after reviewer 86 and before freezing.

**Dry run disclosure.** World 99 at scale 1/16, all six arms through `validate`: every path executed; gates E1, E2, F1,
L1, L2 passed; stopped after two cells, orphan workers killed, relaunched: both completed cells were reused byte-identical
and the rest ran (`PLAIN` reproduced the earlier dry run's terminal exactly). The dry-run log prints headline values for
world 99 at 1/16 budget (terminals 0.99-1.97, which mean nothing at that budget); no probe world (70-72) was run with a
`tanh` learner before freezing.
