# M0: does a motif task distribution give a library-growth cycle something to buy? (Tier 0 census, rotated teacher operators)

Status: FROZEN on commit (2026-10-27). Tier 0, no learner. Module `row.experiments.census_m0_motif_opportunity`;
output `reports/m0_motif_opportunity_census.json`, run records `reports/m0_motif_opportunity_20261027/` (run.log,
status.json, one durable cell per (generator, world)). Opens PI decision 23 (CF1 PREDICTIONS entry): the PI's
"continue research" (2026-10-27) is read as approval of its recommended default, a hierarchical (motif) task generator
on the rotated family, by the same convention as decisions 15-17 ("Ok continue").

# Question

CF1 sizing found that on the rotated teacher library straight-line search binds from ~7 steps for every cheap
incumbent (relaxed descent + polish 0/6 at 7 steps), while exhaustive search stays exact at ~6x cost per step. Every
generator so far draws programs uniformly, so no learned macro can help (B3). This census builds the generator on
which one could, and checks on the teacher's exact operators that it has the intended profile before any learner is
put on it:

1. deep (depth-9) motif programs are NOT found by the incumbent;
2. macros mined from the solutions of shallow tasks of the same distribution find them, cheaply;
3. that help is specific: random macros of the same number do not find them, and macros mined from a UNIFORM
   distribution with the same depths do not find uniform deep programs.

# Generator (`make_programs`)

Rotated worlds 80, 81, 82 (development band 6, verified unused as world seeds 2026-10-27), `configs/v1.yaml` world
(6 primitives, d = 16, alpha 0.35). Seeds `SeedSequence([6300, world])` for programs, `[6301, world]` for the null,
`[6302, world, generator, kind, i]` for data, `[6303, world, generator]` for the random alphabet.

- MOTIF: 6 distinct non-constant length-3 primitive sequences. 48 SHALLOW tasks: one motif drawn uniformly, with
  f ~ U{0,1,2} random filler primitives split uniformly to its left and right (depth 3-5). 12 DEEP tasks: three motifs
  drawn with replacement (depth 9; 216 possible programs against 6^9 = 1.0e7 straight programs).
- UNIFORM (null): the same task counts and per-task depths, every primitive uniform.
- 128 support and 256 query examples per task, x ~ N(0, I), targets from the teacher (noiseless).

# Arms

Shallow tasks are solved by exhaustive prefix-shared enumeration at their known length. MINED alphabet: 3-grams at
every offset of the found shallow routes with count >= 3, most frequent first, at most 12. On each deep task
(success = support MSE <= 1e-12; the true program scores 0):

- `MINED`: enumerate every triple over the mined alphabet, then CF1's one/two-position primitive polish if not exact;
- `RANDOM`: the same with as many random distinct non-constant 3-grams as `MINED` found;
- `ORACLE` (MOTIF only): the same over the true motifs; an IMPLEMENTATION check, must be 36/36;
- `INCUMBENT` (MOTIF only): CF1's relaxed descent (2,100 Adam steps, 3 restarts, lr 0.05) + polish, length 9.

Recorded beside: motifs recovered, top grams, shallow solve rate and exactness, one-position aliases of each deep
truth, enumeration and polish evaluations, CPU seconds, query NMSE, exact-program rate.

# Decision rule

Rates over the 36 deep tasks of a (generator, arm) (denominator 36; all six cells must be complete), in order:

- `INVALID_ORACLE`: `ORACLE` < 36/36;
- `CONSTRUCT_FAILS`: MOTIF `MINED` < 0.9, OR MOTIF `RANDOM` > 0.2, OR UNIFORM `MINED` > 0.2;
- `NO_SEARCH_NECESSITY`: MOTIF `INCUMBENT` > 0.8;
- `MOTIF_OPPORTUNITY`: MOTIF `INCUMBENT` <= 0.5;
- `GRADED` otherwise (0.5 < `INCUMBENT` <= 0.8).

The branches are exhaustive and exclusive. Baselines, computed first: CF1 incumbent 0/6 at 7 uniform steps (so the
0.5 bar is far above the expected rate, and a motif-specific easing of search would show); pilot on world 79 (not a
census world, 4 deep tasks, sizing only): `MINED` 4/4 with all 6 motifs recovered in 0.05 s, `RANDOM` 0/4, `INCUMBENT`
0/4 at ~26 s, UNIFORM `MINED` 0/4 (3 noise grams at count 3), aliases 0.

# Necessity

The behaviour is solving deep programs by composing reusable subprograms. The arm that refuses it is `INCUMBENT`
(primitive-level search); its refusal cost is the solve rate it loses at depth 9, on the scale of all 36 tasks
(pilot: 0/4 against 4/4, query NMSE ~1.6 against 0). Impostors scored: `RANDOM` (any macro-structured reduced search)
and UNIFORM `MINED` (gram statistics without recurrence); exhaustive enumeration is the exact but expensive impostor,
priced from CF1 (13.5 s at depth 7, ~6x/step, so ~8 min per depth-9 task on 6 teacher operators and ~12^9 = 5e9
routes with a 12-slot learned library). Difficulty band: deep tasks sit above the incumbent's failure depth (7) and
below exhaustive's practical limit on teacher operators; shallow tasks sit inside exhaustive's cheap range (<= 7,776
routes).

# Opportunity gate (could it come out another way?)

- `ORACLE` cannot fail except through an implementation error: the true decomposition is in its alphabet.
- `MINED` on teacher operators is NEAR-GUARANTEED: shallow tasks are solved exactly (pilot 48/48) and each motif occurs
  ~8 times against ~0.4 expected for a random gram. It can fail only if aliasing or filler statistics defeat the
  count rule. It is a generator check, not evidence that a learner can mine motifs: in a learned library the routes
  are learner slots with drift and stale assignments, and that is the Tier 1 question this census licenses.
- `INCUMBENT` and the two controls CAN come out either way: motif programs are a 216-element subset of the 1e7 space,
  and a relaxation might find them more easily than uniform programs; random alphabets might cover deep programs if
  polish repairs them.

# Discriminating power

Binomial sampling of 36 tasks per rate (`discrimination()`, 100,000 draws, seed 2710). The null for each clause is
the rate on its wrong side of the bar:

| clause | true rate -> P(clause fires) |
|---|---|
| `MINED` >= 0.9 | 0.99: 0.999; 0.97: 0.978; 0.95: 0.898; 0.85: 0.191; 0.80: 0.052; 0.70: 0.002 |
| control > 0.2 (fails construct) | 0.05: 0.0003; 0.10: 0.023; 0.30: 0.889; 0.40: 0.992 |
| `INCUMBENT` <= 0.5 | 0.2: 1.000; 0.3: 0.996; 0.4: 0.919; 0.6: 0.147; 0.7: 0.009 |
| `INCUMBENT` > 0.8 | 0.7: 0.112; 0.75: 0.289; 0.9: 0.977; 0.95: 1.000 |

Each row applies one clause of the registered decision rule above to a null sampler (true rate on the wrong side of
the bar) and an effect sampler (true rate on the right side). False-fire is <= 5% one step beyond each bar (0.8 for
`MINED`, 0.1 for controls, 0.7 for `INCUMBENT` <= 0.5); a true incumbent rate of 0.6 still reads `MOTIF_OPPORTUNITY`
15% of the time, which is why 0.5-0.8 is `GRADED`, not a pass. Detection at the expected effect sizes (pilot and CF1):
`MINED` at 0.97 passes with probability 0.978, controls at 0.05 stay below the bar with probability 0.9997, and an
incumbent at 0.2 reads as binding with probability 1.000.

What these checks do not cover: tasks are simulated as independent Bernoulli draws, but the 12 deep tasks of a world
share one library and one motif set, so a world-level failure (for example a world whose motifs alias each other)
moves 12 tasks together and is more likely than the binomial rates suggest; per-world rates are recorded so this is
visible. Systematic errors (an aliasing deep program scored as a success, a polish that repairs random alphabets by
brute force) are not sampling questions; aliases and polish evaluations are recorded to expose them.

Re-freeze note: the first freeze commit (`5308f0f`) was made before `check_adequacy` passed (its exit code was hidden
by a pipe); this section was completed in a second freeze commit before anything ran. Only this section changed.

# What it licenses, and limits

- `MOTIF_OPPORTUNITY` licenses a Tier 1 design (M1) with a LEARNED library: does the confirmed online pipeline form a
  vocabulary on a motif stream, can motifs be mined from learner routes, and does macro search beat the incumbent on
  deep tasks at matched compute, charging the mining (learning A12).
- `NO_SEARCH_NECESSITY` or `GRADED` would mean motif structure itself eases primitive search; the decision 23 premise
  is then weakened and goes back to the PI.
- Limits: teacher operators (no drift), noiseless targets, motif length and count fixed (3, 6), deep programs are
  exactly motif concatenations (no filler), one generator family, three worlds, known program length.
