# Forward subsumption of proof-search strata (L4)

## Result

The strata layer is a proven, sound, multiplied asymptotic win. The runtime meet
remains scan-bound and converts none of it yet; the missing engine operation is
now precisely characterized (see The wall, below).

## Strata compression (guarded emit, kernel a23b656 + fc71e65)

Emit-time forward subsumption on the MITM forward closure: a candidate theorem
is dropped when a stored theorem at an equal-or-lower stratum generalizes it
(inverted-bound encoding bound' = 26 - stratum turns the guard sink's
stored >= candidate test into stratum_G <= stratum_K). Index entries are guarded
by the same key, keeping the index an antichain (transitively complete).

| Hf | unguarded closure | subsumed closure | time ratio | facts | coverage |
|---|---|---|---|---|---|
| 6 | 6,816 ms / 33 | 11 ms / 24 | 620x | 1.4x | 0 missing |
| 9 | 66,643 ms / 894 | 98 ms / 189 | 680x | 4.7x | 0 missing |
| 12 | 310,512 ms / 5,522 | 377 ms / 490 | 823x | 11.3x | 0 missing |

The ratio grows with Hf: subsumption keeps the semi-naive frontier an antichain,
so the delta joins never see the proof-term multiplication that dominates the
unguarded closure. Coverage differential (coverage_check.py): every baseline
theorem is generalized by some subsumed-run theorem at an equal-or-lower
stratum; 0 missing at all splits after the kernel identity fix.

## The soundness bug found on the way (kernel fc71e65)

The guard walk compared candidate variable spans by raw bytes; in the De Bruijn
encoding every first occurrence is the same NewVar byte, so the repeated-variable
schema (-> $a (-> $b (-> $c $a))) covered candidates with DISTINCT variables in
the repeated positions, dropping sound theorems (1 at Hf=9, 4 at Hf=12 — found
by the coverage differential, pinned by index bisection, minimized to a
micro-repro). Fixed by resolving candidate variable occurrences to canonical
VarRef indices in a shadow copy used for binding comparisons; the fix also lets
a NewVar first occurrence match a VarRef re-occurrence of the same variable, so
the antichain got tighter and sound simultaneously.

## Downstream (MITM with the 490-fact Hf=12 table)

- Ladder holds: jarr 2 finals / 412 ms; imim1 1 final / 3.40 s.
- loowoz: 3 finals, 385,466 atoms (7.5x fewer than plain bfc's 2.87M) in 335 s
  (13.5x slower than bfc's 24.8 s).
- pm2.83 (size 25): timeout at 900 s.

## The wall, precisely

The meet joins (sol x fwd x arithmetic guards) SCAN: the join variable (the
theorem) sits inside compound columns on both sides.

- MORK_LEAPFROG=all does not change the time (333 s): the body is not routable,
  so this is not a dispatch-gate policy issue but an engine capability gap.
- The theorem-led data layout (fwd-bythm THEOREM K PROOF with the seek factor
  directly after sol) TIMES OUT (>900 s): the seek key $ma is SCHEMATIC (bound to
  a context subterm containing variables), so a leading column buys no prefix
  seek, and the reorder demoted the stratum guards that pre-filtered contexts.

Missing operation: UNIFIABILITY RETRIEVAL as a join factor — seeking a schema
table by a bound-but-schematic key. The guarded-emit sink already implements the
one-way version of this walk (discrimination-tree retrieval with variable
identity); L6 should lift it from sink-side filter to join-side factor.

## Files

- build_subsume_programs.py — guarded fwd-closure generator (invFn inverted-bound
  table, guarded fact + index emission).
- coverage_check.py — the coverage differential.
- fwd-sub-only-hf{6,9,12}.mm2/.dump/.log — subsumed strata runs.
- mitm-sub-*.mm2, mitm-subled-*.mm2 — downstream MITM over subsumed strata
  (compound-interior and theorem-led variants).
- micro-freevar.mm2, micro-realindex.mm2, bisect.mm2 — the soundness-bug hunt.


## Reproducing from this directory

`MORK_BIN=<kernel binary> python3 run_coverage.py 6 9` regenerates the
baseline and subsumed programs, runs both on the kernel, and checks
antichain coverage per stratum. The numbers in this report came from
working runs at Hf up to 12; the committed runner reproduces the
mechanism and coverage law at any Hf you give it (time grows with Hf).
