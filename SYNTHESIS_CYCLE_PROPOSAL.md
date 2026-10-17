# Proposal: a library-growth synthesis cycle, and the census that decides whether it can exist here

Status: PROPOSAL for the PI (2026-10-17). Registers nothing. Source: reviewer 86 (`reviews/reviewer-feedback-86.txt`),
issue #2 phase D. Written by Claude after G1S2; to be decided after G1R.

# The proposal

Reviewer 86's cycle, in this project's terms:

1. a formed vocabulary and a fixed SEARCH BUDGET per task (route optimization steps or enumerated routes);
2. solve a corpus of new tasks; some fail under the budget;
3. extract recurring subprograms from the solved ones and add them as macros, charging their storage and the
   alphabet tax (E6's accounting);
4. re-solve; count newly solved tasks; repeat for a second generation;
5. endpoint: NEW TASKS SOLVED PER TOTAL COMPUTE (search + extraction + storage), against the original library,
   ordinary search with the same total budget, and a library grown without selection (random grams).

# What the record already says (read before building anything)

- **Macros already pay, by a predicted amount.** E6B: substituting a length-3 macro cut route-optimization cost by
  +4.41 s at depth 6, x0.78 of the saving forecast from E5.1's cost law, 3/3 worlds, no quality loss. E6F: a gated
  criterion decides WHEN to create one. So steps 3-4 of the cycle are, on this testbed, a confirmed law plus a
  confirmed creation rule. Running them again would mostly re-measure E6B.
- **Search does not fail here, it only costs.** E5.1: gradient route optimization stayed at oracle parity from depth 3
  to depth 10 while the program space grew 3.58e7-fold; its cost grew 3.30-fold, about linearly in program LENGTH.
  Exhaustive enumeration is the search that blows up (4.66x gradient's cost at depth 5), and it is not the incumbent
  past depth 4.
- **So a macro buys a constant factor.** Against a search whose cost is linear in length, a length-L macro shortens a
  program by L-1 steps and saves a fixed fraction of cost. "Tasks solved per compute" can then rise by a factor of
  order D / (D - (L-1)) per substitution, which E6B's law already predicts with no free coefficient. That is an
  implementation check, not a discovery (the AGENTS rule on defining invariants), unless the budget makes tasks FAIL.

The cycle is therefore interesting only in a regime where, under a realistic budget, the incumbent search FAILS on a
meaningful fraction of tasks, and abstraction turns failures into successes. That is a necessity question, and it
gets answered first, cheaply.

# The census (Tier 0, proposed): where, if anywhere, does a fixed search budget fail?

On existing formed libraries (the staged/export libraries E5.1 used, rebuilt bitwise, or D2's depth-5 vocabularies):

- depths 6, 8, 10, 12, 16 (16 is past C0's ~8-application clean execution, so an execution-eligibility gate as in E5.1
  is required, or the failures are executor drift, not search);
- budgets from 125 to 2,000 gradient steps, and enumeration capped at a fixed number of route evaluations;
- record the SOLVED FRACTION per (depth, budget, search), with the oracle route as the ceiling;
- and, for the same tasks, the solved fraction when the program is shortened by an ORACLE macro (the teacher's own
  recurring subprogram): the ceiling on what any learned library could buy.

Reading (to be registered properly if the PI wants it):
- if gradient search at a moderate budget solves nearly everything to the depth where execution still works -> NO
  OPPORTUNITY for the cycle on this testbed; record it, and the cycle needs a different generator (e.g. noisy support,
  or tasks whose route cannot be relaxed: branches or loops, where gradient routing is not the incumbent);
- if there is a band of (depth, budget) where search fails AND oracle macros rescue a large share -> the cycle has an
  opportunity there, and a Tier 1 with LEARNED macros follows, scored on tasks per total compute against all three
  baselines above.

Cost: small (E5.1 measured 7.7-25.4 s of route optimization per task at depths 3-10), no new lifetime.

# Claude's recommendation

Run the census before any cycle. On the evidence above I expect NO OPPORTUNITY for straight-line programs on this
generator, as the inference line found. If so, the honest next direction for synthesis is search in spaces where
gradient routing is not available or not exact: programs with control flow (C-seal's loops and B1's branches already
form online), where the choice of structure is discrete and enumeration is the incumbent. That is where library growth
plausibly changes what can be solved, not only how fast.
