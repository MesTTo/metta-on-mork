# Exponent halving on the meet-in-the-middle prover

## Result

Moving the meet point of a bidirectional proof search deeper into the forward
antichain collapses the searched state space, because the forward side stays
small under subsumption while the backward side branches. Every proof found
is independently type-checked against the three axioms (`verify_proof.py`,
which never consults the search).

| target | proof size | original split | rebalanced | state collapse | proof |
|---|---:|---:|---:|---:|---|
| jarr | 13 | 1,294 states | 83 states | 15.6x | verified |
| imim1 | 15 | 6,108 states | 918 states | 6.7x | verified |
| loowoz | 19 | 25,501 states | 2,170 states | 11.8x | verified |

## Why it works

A size-D proof is found by meeting a forward closure (built to size Hf)
against a backward search (covering the remaining steps). The meet rule
discharges `mk` units of the remaining budget for a forward proof of size
`mk`: `addFn meet_remaining mk ski` reads `meet_remaining = ski - mk`. A
deeper forward table discharges more per meet, so fewer backward steps are
needed. The backward branching factor is what makes search exponential, so
cutting backward depth is a direct exponent reduction -- the meet-in-the-
middle b^d -> b^(d-Hf) with Hf pushed as high as the forward side affords.

The forward antichain is cheap and, crucially, TARGET-INDEPENDENT: it is the
closure of the axioms under modus ponens, the same for every theorem. Built
once, it is a shared lemma base. Its growth under subsumption (measured):

| Hf | forward schemas | build time |
|---:|---:|---:|
| 12 | 490 | 0.1s |
| 15 | 4,120 | 0.7s |
| 18 | 12,068 | 2.7s |
| 21 | 111,887 | 34.6s |

This is only affordable because the sink route now gets semi-naive and the
WCO join (the forward closure went 7.0s -> 0.10s in the same session); before
that, a deeper forward table cost more than it saved.

## Honesty

- The state counts are the searched `sol` space; the collapse is the whole
  point (fewer states reached, same proof).
- Completeness is conditional: the meet only fires when the proof BISECTS at
  the chosen Hf -- some reachable backward context whose antecedent a
  size<=Hf forward schema proves. The three targets here bisect. Two harder
  targets in the working set (loolin, D=26; pm2.83, D=25) did NOT bisect at
  the meet points tried and returned no proof; that is a property of
  bidirectional search (the meet must exist), not an engine limitation, and
  completing them means sweeping meet points to find a bisection.
- This does not make proof search polynomial. Backward branching still
  dominates the exponent; halving a constant slice of the depth is a large
  constant-and-base win on real instances, not a complexity-class change.

## Reproducing

`MORK_BIN=<kernel built with witness_select> python3 run.py`. The kernel
features needed: `semi_naive_ic,leapfrog,stratified_quiescence,guarded_emit,
retrieval_join,witness_select`. `rebalance_meet.py` regenerates a rebalanced
program from a base MITM program and a forward-antichain dump;
`verify_proof.py` type-checks any dump's finals against the axioms and has a
negative control (a corrupted proof is rejected).
