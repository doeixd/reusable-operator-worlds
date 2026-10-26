# CF0: is finding the structure of a composite loop program hard? (Tier 0 census, control-flow synthesis)

Status: FROZEN on commit (2026-10-26). Tier 0, no learner. Module `row.experiments.census_cf0_loop_structure`; output
`reports/cf0_loop_structure_census.json`. Follows SY1 (straight-line search has no opportunity: descent + polish
solves 96/96 at depths 7-10) and `SYNTHESIS_CYCLE_PROPOSAL.md`'s alternative: synthesis where gradient routing is not
available or not exact, i.e. control flow. The PI's "continue with the research" (2026-10-26) covers this Tier 0 step.

# Question

Before building any learned structure search or library-growth cycle for programs with loops, does a CHEAP incumbent
already find the structure? Tasks: `z = A(x); repeat at most 6 times: stop if w.z <= 0 else z = P(z); y = B(z)`, with A,
B random teacher-primitive sequences of length a = b in {0..4}, P one primitive, w a random unit hyperplane (rotated
worlds 76, 77, 78, development band 6, verified unused; 8 tasks per (world, size); seeds [6100, world, a, b, i]).

The census uses the TEACHER's exact operators, so it measures search with execution drift removed: the easiest case
for the searcher. If search binds even here, structure search is necessary; if it does not, the remaining difficulty
of composite loops on this generator is execution and predicate estimation, not search.

# Incumbent

Skeleton score: for (A, P, B), apply A, iterate P 0..6 times on every example, apply B after each count, take each
example's best count; score = mean minimum error (count inference per example; exact in a noiseless world).
Search: coordinate descent over the a + 1 + b positions from random starts, restarts until the score is ~0 or 3,000
skeleton scores are spent. Exhaustive enumeration as a reference where the space is <= 7,776. Then the predicate w is
fitted by logistic regression on the states the inferred counts label, and the query is executed with it.

# Decision rule

STRUCTURE success per task: the found skeleton explains every support example (support score <= 1e-12). Read at
support 128 over the 24 tasks of each size (denominator 24; every size must be complete):

- `NO_SEARCH_NECESSITY`: success >= 0.9 at every size AND median search seconds <= 60 at every size;
- `SEARCH_BINDS`: success < 0.6 at the largest size (9 positions, 6^9 = 1.0e7 skeletons) OR median seconds > 600 at
  any size;
- `GRADED` otherwise.

Recorded beside it: exact-skeleton rate, evaluations, the same at support 16, query NMSE with the fitted predicate at
supports 128 and 16, enumeration reference.

Design history (before the full run): a 1-task-per-cell smoke showed the structure found exactly in 1-290 evaluations
while query NMSE with the fitted predicate was 0.07-0.16 (128 examples) and 0.5-0.75 (16). The first draft's success
criterion (query NMSE <= 0.01) therefore measured predicate estimation, a sample-size question B0b already answered
(~512 examples), not search; it was replaced by the structure criterion above before the full run.

# Discriminating power

Binomial sampling of 24 tasks per size (`discrimination()`, 100,000 draws, seed 2611):

| true per-size structure success | P(all five sizes pass >= 0.9) | P(largest size < 0.6) |
|---|---|---|
| 0.99 | 0.991 | - |
| 0.97 | 0.841 | - |
| 0.90 | - | 0.00005 |
| 0.80 | 0.00002 | 0.013 |
| 0.70 | 0.0000 | - |
| 0.50 | - | 0.846 |
| 0.40 | - | 0.979 |

A searcher that succeeds 97%+ of the time is labelled `NO_SEARCH_NECESSITY` with probability >= 0.84; one at 80% is
almost never (2e-5). A searcher at 50% or below at the largest size reads `SEARCH_BINDS` with probability >= 0.85, and
one at 90% essentially never.

# Gate check and limits

- NECESSITY is the question. Refusal cost of loops is already known (straight routes ~1.0, B0/B2).
- OPPORTUNITY: if `NO_SEARCH_NECESSITY`, a learned structure searcher or macro cycle has nothing to buy on this
  generator; the open problem for loops is the predicate's sample cost and execution drift (C0), both measured.
- Limits: noiseless targets (count inference is exact; G3a's lesson), one loop per program, single-primitive bodies,
  teacher operators (a learned library adds drift, which makes scores approximate but does not change search
  combinatorics), K = 6.
