# TQBF on the stacked engine (L8, demonstrator 1)

## Result

Pi2 QBF (forall X exists Y . CNF) decided by flat counterexample-guided
expansion (the RAReQS/CEGIS shape, non-recursive per arXiv 1611.01553) with
the engine doing both sub-searches per round. Verdicts are EXACT against a
recursive oracle evaluator on every battery run:

| driver | battery | verdicts |
|---|---|---|
| v0 (driver strata) | random 5+5/6, 6+6/8, 8+8/10, 10+10/14, 12+12/16, 14+14/18 + planted 8+8/12 | 227/227 |
| v1 (barrier-staged single program per round) | random 5+5/6, 6+6/8, 8+8/10 + planted 6+6/10, 8+8/12, 10+10/14 | 100/100 triple-match (v1 == v0 == oracle) |

## Layer composition (all measured on one planted 8+8/12 round)

- L3 guarded emit as the NOGOOD mechanism: clause-derived forbidden schemas
  stored as De Bruijn patterns; the guard walk kills falsifying extensions at
  O(path) each. Counters: 127 candidates consulted, 29 dropped in the round.
- L4-style schema pruning: per-depth truncated schemas make pruning bite at
  every stratum. Before the per-depth fix the search explored the full
  binary tree until the last level (2,4,...,128 then 28 = 283 nodes); after
  it, 71 nodes vs the 510-node full tree on the same instance.
- L2 stratified quiescence: v1 stages one whole CEGIS round inside a single
  program -- clsat facts, forbid materialization, k Y-depth strata, the
  A-side member fold, and verdict extraction -- all sequenced by
  quiesce-headed barrier execs (30 exec steps for the metrics round).
- Guard-as-negation: (O (guard (clsat $j) (forbid $pat))) emits exactly when
  (clsat $j) is ABSENT: stratified negation-as-absence, used again for
  dead-clause (X-only, unsatisfied) detection.
- L6 retrieval join: the A-side per-member fold joins the k-column merge
  tables (mrg cells) and impose patterns -- small-table factors of exactly
  the shape the retrieval partition memoizes.

## Mechanism

Refutation form: decide not-(exists X forall Y not-phi) by expansion over Y.
S = expansion set of Y-assignments. Per round, ONE engine program:
- B-side (needs a fresh y for the last x*): clause satisfaction under x*
  (clsat), forbidden schemas for unsatisfied clauses, then k depth strata
  growing Y-assignments as cons lists under the guards; full-depth survivors
  are the satisfying y's. A dead X-only clause short-circuits to FALSE.
- A-side (needs an x falsifying every y in S): per-member clause choice with
  X-cell patterns merged through (mrg u/0/1) tables; the member cursor
  advances by barrier; survivors at cursor |S| are the falsifying x's.
- No survivors on the A-side: TRUE. No fresh y on the B-side: FALSE. Else
  install y, next round. Driver work between rounds is bookkeeping only
  (member tables and the next program file).

## Two MM2 authoring rules this build discovered (now in ai-todo.md)

1. A guard spec is (TABLE key) with ONE key argument; multi-column keys must
   wrap in a single compound, or the guard silently never drops (the kernel
   now warns once; commit ce72a8f).
2. Under stratified_quiescence, an exec firing that emits ONLY exec facts is
   nullified as exec-only churn (snapshot restore). Every init/re-arm
   template needs a non-exec witness fact -- the (armed X) idiom.

## Honesty

These instance sizes are trivial for any real QBF solver (DepQBF, CAQE
class); the claim is NOT QBF supremacy. The claim is that a PSPACE-complete
search runs as saturation on the fact engine with every stack layer doing
its published job -- nogoods via guards, staging via quiescence barriers,
pruning via schema generalization, small-table joins via retrieval -- and
that the verdicts are oracle-exact at every size tried. Scaling further is
an instance-family and constants question, not a mechanism question.
