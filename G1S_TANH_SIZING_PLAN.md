# G1S: learning rate and initial gain for the `tanh` learner (Tier 1, EXPLORATORY, sizing)

Status: FROZEN 2026-10-14.

Runner `row.experiments.g1s_tanh_sizing`; independent scorer `row.experiments.score_g1s_tanh_sizing`; tests
`tests/test_g1s_tanh_sizing.py`. Development band 6, world 70, stream 0. Exploratory: this rung CHOOSES a setting for
the next probe and produces no verdict.

# Why

G1T (`G1T_TANH_FORMATION_PLAN.md`, `reports/g1t_tanh_formation.json`) returned `FAILS` with `STAGED` failing too, and
its plan pre-stated the branch: "this learner at the rotated form's hyperparameters cannot form the `tanh` form; the
next rung is a learner-side sizing (initial gain, learning rate) on one world, not abandonment of the form". The read-
only depth diagnostic archived with G1T located the failure: depth-1 tasks route to the right slot (`d1_ari` ~1) but sit
at 0.13-0.16 after sleep (0.19-0.27 at the wake terminal), and the error compounds with depth to 0.39-0.46. G0F's S6
had fitted one same-form operator to <= 0.0074, but with Adam at lr 0.01 for 2,000 full-batch steps, ten times the
lifetime's lr 0.001. The two settings that govern per-operation fit and that G1T inherited unchanged are the shared
learning rate and the slots' initial gain (1, against the teacher's 2). This rung varies exactly those.

# Construction (arms are constructions; world 70, stream 0)

Each SETTING is a pair `(lr, gain)`; six settings, `lr` in {0.001, 0.003, 0.01} x `gain` in {1, 2}. Applying a setting
is G1T's `use_form()` name rebinding, plus two values:
- `config.discrete_model.global_learning_rate = lr`, set in every stage config, so the wake lifetime AND sleep's
  `o2c.consolidate` (both read it through `learned_lifetime._training_values`) run at it; the task-code learning rate,
  weight decay, replay and everything else stay v1.yaml's;
- `tanh_form.INIT_GAIN = gain`: each slot's initial `A` is `gain` times the same Haar-orthogonal draw, and its initial
  output scale is `tanh_gain_scale(gain)`, so a fresh slot keeps unit output variance at either gain.

| arm | construction |
|---|---|
| `WAKE` | `o2.run_single('SHUFFLED')` verbatim: G1T's order-free stream, wake alone |
| `SLEEP` | on that setting's `WAKE` terminal, `o3.run_sleep`'s exact call sequence (`load_shuffled_terminal`, the 64-per-task reservoir, `o2c.consolidate` with 8,192 updates and seed `[1941, w, stream, 64]`, `score`), inlined so the slept model is measured without sleeping twice |

`SLEEP` after wake alone rather than G1T's confirmed pipeline (re-routing wake, then sleep): on the `tanh` form the two
slept results differed by +0.017 / -0.066 / -0.047, inside the world-to-world spread, and plain wake is ~22% cheaper
(787 s against 1,008-1,027 s). What is being sized is per-operation fit, which both pipelines share.

12 cells. The baseline setting `(0.001, 1)` is G1T's construction exactly.

**Gates (before any cell).** S1: under every setting, a fresh learner's `A` has every singular value equal to the set
gain (1e-5), its output scale is `tanh_gain_scale(gain)`, and all three stage configs carry the set `lr`; at the
baseline the fresh learner is bitwise G1T's. S2: `restore()` returns the defaults. A test shows S1 fails when the
gain is not applied. **Per cell:** `WAKE` records the `global_learning_rate` in the lifetime's own resolved
`config.yaml`, `SLEEP` the rate consolidation read, and each must equal the setting; G1T's `o2.validate_cell` runs on
every `WAKE`.

**Anchor A0.** The baseline's `WAKE` and `SLEEP` per-task terminals must equal G1T's `SHUFFLED_w70` and `SLEEP_w70`
EXACTLY. This proves that the setting machinery is a no-op at G1T's values and that the inlined sleep is
`o3.run_sleep`.

# Measurements

Per cell: terminal median over the canonical 64 (the registered statistic), per-task terminals, seconds; `WAKE`: depth-1
identity ARI of the wake routes; `SLEEP`: per-depth medians over the 188 stream tasks (60 / 64 / 64) and the slept
depth-1 identity ARI.

# Reading (written before any cell; exploratory, chooses and never confirms)

Among settings whose slept vocabulary keeps operation identity (`d1_ari >= 0.8`), CHOOSE the lowest slept terminal
median, and label by where it lands:
- A0 fails -> `ANCHOR_MISS`: nothing is read; the machinery is debugged first;
- no setting keeps identity -> `IDENTITY_LOST`;
- chosen `< 0.05` -> `FITS`. Next: a new frozen probe (G1T-v2) with the confirmed pipeline and G1T's other arms at
  the chosen setting on worlds 71-72 (paired with G1T's cells there) and fresh band-6 worlds; NOT world 70, which chose
  the setting;
- `0.05 <= chosen < 0.2` -> `IMPROVES`. Next: a second sizing axis on world 70 at the chosen setting: wake updates per
  example and sleep length;
- chosen `>= 0.2` -> `NO_GAIN`: learning rate and gain are not the lever. Next: Tier 0 "freeze one half" on world 70: a
  lifetime given the teacher's routes, which asks whether this slot family can fit `tanh2` compositions under the
  lifetime's optimizer at all.

Secondary reads: the per-depth profile of each setting (does depth-1 error fall, and does the compounding shrink with
it?), wake against slept terminal per setting, and the cost of each setting in seconds.

# Necessity

The behaviour is per-operation fit inside the lifetime. The arm that refuses the change is the baseline setting, G1T's;
its refusal cost is measured: slept terminal 0.405 on world 70 against 0.009 for the rotated learner there (scale: a
factor of ~45, the whole formation effect), and depth-1 error 0.13 against the 0.05 formation bar. The
expected impostor is a setting that lowers terminal error without separating operations (a few slots fitting
everything); it is excluded by the identity filter, which is part of the choice rule. Difficulty band: the target is
the rotated learner's regime, slept terminal 0.009-0.03 (G1), so a setting landing in `[0.005, 0.05]` is the band that
licenses a probe; one below 0.005 would be suspicious and checked against the teacher before use.

# Discriminating power

**The rule:** chosen = argmin slept terminal median among settings with `d1_ari >= 0.8`; `FITS` if `< 0.05`,
`IMPROVES` if `< 0.2`, else `NO_GAIN`.

Simulated (200,000 draws, seed 20261014) with each setting's terminal log-normal around its true value:
- null (no setting helps: the five non-baseline settings all centred on G1T's ~0.43, log-sd 0.25, far wider than the
  0.06 log-sd of G1T's three slept worlds): `FITS` false-fire 0.0000, `IMPROVES` false-fire 0.0055, `NO_GAIN` 0.9945;
- full effect (one setting centred on 0.02, log-sd 0.5): `FITS` detected 0.966, `IMPROVES` 0.034;
- partial effect (one setting centred on 0.10, log-sd 0.5): `IMPROVES` detected 0.854, `FITS` 0.082, `NO_GAIN` 0.064.

So the rule separates "these knobs fix fit" from "they do not" on one world. It does NOT rank close settings: two
settings within ~0.06 of each other are inside single-stream noise, and the choice between them is a coin flip that
the next probe, not this rung, has to survive.

**What the checks do not cover:**
- one world, one stream; the chosen setting is fitted to world 70, which is why the next probe excludes it;
- the grid is coarse (two gains, three rates); an optimum between points is not found;
- cells near 0.05 are fragile (SO1: unconverged single-stream cells spread up to 0.68), so a `FITS` at 0.04 is read
  with its neighbours;
- task-code learning rate, weight decay, slot count and budgets are held at v1.yaml's;
- re-routing is not run; a setting could interact with it;
- the thresholds 0.05 and 0.2 are judgements: 0.05 is the program's formation bar, 0.2 roughly half the baseline.
- the optimizer is AdamW with decoupled weight decay, so raising `lr` also raises the effective decay per step
  (`lr x 1e-4`); a setting's effect is the two together, not the step size alone.

# Operational

12 cells, pool 3: the six `WAKE` cells first, each `SLEEP` submitted when its `WAKE` exists. Timing from G1T on this
host: wake 787-789 s, sleep ~205 s plus a short diagnostic; expected about 35-45 minutes. Each cell is written
atomically to `artifacts/g1s_tanh_sizing/cells/`, stamped with the protocol hash (this plan, config, the G1T report,
every reused module, commit) and reused on relaunch only under that stamp; the parent writes every `run.log` line;
`status.json`, `run.pid`, `exit.json`, `precondition.json`, `gates.json` and `manifest.json` are archived to
`reports/g1s_tanh_sizing_<date>/`. Dry run: world 99, scale 1/16, settings `(0.001, 1)` and `(0.01, 2)`, both arms,
through `validate`, with a restart test (stop after one cell, kill orphans, relaunch, verify reuse). Launched detached
from clean committed code. The scorer recomputes the reading from the cell records and G1T's world-70 cells and refuses
a report whose inputs changed.

**Write-time check (2026-10-14), item by item against the implementation learnings.** Opportunity: the settings can
move per-operation fit or not, so the comparison can come out either way. Discrimination: the rule was run against
null and effect samplers (above). Thresholds against their baseline: the baseline sits at 0.405 (`NO_GAIN`), 0.2 is
about half of it, 0.05 the formation bar. Denominators: the choice is over the six settings; per-depth medians carry
their counts (60 / 64 / 64), checked in `validate`. Estimands against code: the terminal is the final model on the
canonical 64 (`score`), not end-of-task; the record list above matches the runner's output. Fit and score: world 70
chooses and is excluded from the next probe. Arms are stated as constructions. Non-vacuity: gate S1 has a test in
which it fails when the gain is not applied; each cell checks the learning rate it actually ran at; sleep must change
the library; A0 can fail. Found one omission while checking, now added: the learning rate also scales AdamW's
decoupled weight decay.

**Dry run disclosure.** World 99, scale 1/16, settings `(0.001, 1)` and `(0.01, 2)`, both arms. The first attempt
FAILED at the first `WAKE` cell: the runner relabels the arm `WAKE` before calling G1T's `o2.validate_cell`, which knows
the construction as `SHUFFLED`, so a valid cell was refused. Fixed by validating under the construction's name; that
dry directory was deleted (its stamp was the old code's). Then the restart test: stopped after one cell, two orphan
workers killed by `parent_pid`, relaunched; the finished cell was reused byte-identical (sha256 `866e67fc...`) and the
other three ran; exit 0. The dry-run log prints terminals for world 99 at 1/16 budget (1.02-1.84), which mean nothing at
that budget. No `tanh` learner has been run on world 70 at a non-baseline setting.

