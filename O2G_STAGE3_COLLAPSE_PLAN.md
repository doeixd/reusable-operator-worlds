# O2G: is the curriculum's stage-3 collapse determined by the stage-2 library, or a stochastic transition event? (Tier 1, exploratory)

Status: FROZEN 2026-09-26 before any O2G cell ran. **Tier 1, EXPLORATORY: it
produces no verdict.** It uses O2's saved `STAGED` stage-2 models (worlds
13-19): no new world.

# Why, and what was already seen (disclosed)

A descriptive census of O2's lifetimes (2026-09-26, not registered) found that
all three `STAGED` collapses (w14 s1, w14 s2, w16 s1) have normal stages 1-2
(stage-2 end-of-task median 0.08-0.12). They then fail at the very start of
stage 3: the first length-3 task is fine (0.02-0.08), the second or third is
1.7-2.4, and the lifetime never recovers. Non-collapsed cells start stage 3 at
0.04-0.88, with occasional recovering spikes.

If this collapse is STOCHASTIC, an online learner could detect it from its own
end-of-task error within a few tasks, roll back to the stage-2 library, and
retry. If it is DETERMINED by the library entering stage 3, the fix must act
before the transition. This rung asks which, and the answer decides between
those two successor designs. The census that prompted the question is on these
worlds, so this rung tests a different quantity (re-run outcomes) on them, and
any detector would be registered on fresh worlds.

# Arms, as constructions

For each of O2's 21 `STAGED` cells (w, s), stage 3 is re-run from the saved
stage-2 model (`work/STAGED_w{w}_s{s}/stage2/model.pt`, reloaded strictly),
through `so2.carry_library` and the same `learned_lifetime.run` with LEAN, as O2
ran it. Only the stage-3 replay seed changes: `SeedSequence([7600, w, s, k])`,
`k = 1, 2`. That gives 42 re-runs.

**Gate G0 (bitwise):** re-running stage 3 with O2's ORIGINAL stage-3 replay seed
reproduces O2's stage-3 library sha256 and per-task terminal exactly. It runs on
the three collapsed cells first: if the reload or carry is not O2's
construction, nothing is scored.

# Estimand and rule

A re-run COLLAPSES if its terminal median is `>= 1.0`, O2's collapse definition.
- `C`: the 3 cells that collapsed in O2 (6 re-runs).
- `H`: the 18 that did not (36 re-runs).

| condition | label |
|---|---|
| `>= 5` of 6 `C` re-runs collapse and `<= 3` of 36 `H` re-runs collapse | `SYSTEMATIC` |
| `<= 2` of 6 `C` re-runs collapse | `STOCHASTIC` |
| otherwise | `MIXED` |

The labels partition all outcomes. Also reported, descriptively: every re-run's
terminal, and each re-run's stage-3 first-16 end-of-task median, the early
signature seen in the census.

# Necessity

**Refusal comparison:** O2's own stage-3 outcome from the same library.
**Impostor:** the replay stream, which is the only thing varied. **Band:**
collapse (`>= 1.0`) against everything else. Near-miss re-runs are reported and
enter no clause.

# Discriminating power

Exact binomials (`reports/o2g_design/rates.py`, output `rates_output.txt`).

| world | `SYSTEMATIC` fires | `STOCHASTIC` fires |
|---|---|---|
| stochastic, hazard 0.10 / 0.143 (O2's rate) / 0.25 | **0.00003 / 0.00007 / 0.00005** | **0.984 / 0.958 / 0.831** |
| systematic, q = 0.9, h0 = 0.03 | **0.866** | 0.0013 |
| systematic, q = 0.8, h0 = 0.03 | 0.641 | 0.017 |
| systematic, q = 0.9, h0 = 0.06 | 0.737 | 0.0013 |

**What the checks do not cover:** a moderately systematic world (q = 0.8) is
labelled `MIXED` about half the time. Only three collapsed libraries exist, so
the `C` side rests on 3 libraries times 2 seeds. One protocol (`STAGED`) only:
the order-free and `MIXED_L1` collapses have no clean transition to re-run.

# Operational

45 stage-3 lifetimes (42 plus 3 gate runs), ~6.5 min each, pool of 3: about
1.7 h. Durable stamped cells, fingerprint, `run.log`, `status.json`, 8 GiB
precondition. The independent scorer is committed before launch.
