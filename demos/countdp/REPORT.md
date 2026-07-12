# Derivation counting without enumeration (L5)

## Result

Counting proofs by dynamic programming over (theorem-schema, size) states runs
in O(states + contributions) and is EXACT against the enumeration oracle at
every stratum it can be checked at. At the last enumerable scale the separation
is 6,210x; beyond it, counting continues polynomially in states while the
counted quantity grows exponentially.

| scale | enumeration (measured) | DP (measured) | derivations | states |
|---|---|---|---|---|
| Hf=9 | 66.6 s | 0.02 s | 729 @ K=9 | 166 |
| Hf=12 | 310.5 s | 0.05 s | 4,628 @ K=11 | 516 |
| Hf=21 | not measurable | 27.3 s | 67,373,289 @ K=21 | 188,637 |

Oracle (PASS, exact at K = 1..11): the baseline forward enumeration stores one
fact per distinct proof term, so per-K fact counts equal per-K derivation
totals; the DP reproduces 3 / 6 / 24 / 132 / 729 / 4,628 exactly.

## Mechanism

dcount(T, 1) = 1 per axiom; dcount(T, k) = sum over i + j + 1 = k of
dcount(F, i) * dcount(X, j) over unifying pairs (F = (-> A B), X ~ A) with
result schema T. Each stratum is one MM2 program: the kernel join unifies the
(fwd-count I THM SERIAL N) families and emits one contribution fact per
derivation pair with the count product computed in-engine by product_i64. The
driver (count_dp.py) sums contributions per canonical result schema between
strata and seeds the next stratum -- the staged-pipeline form whose equivalence
to barrier programs is the quiescence feature's own differential law.

## The bug the oracle caught

First version undercounted (129 vs 132 at K=7): the contribution fact recorded
the post-unification instance, so two DISTINCT schemas that unify to the same
instance with equal count products collapsed into one fact under trie set
semantics. Fixed by adding driver-assigned SERIAL columns (pre-unification
schema identity) to the count facts and contribution keys; the oracle then
matches exactly at every stratum. Same lesson as the guard-walk fix: identity
of schematic things must be carried explicitly, never inferred from bytes that
alpha-collapse.

## Reproducing

`MORK_BIN=<kernel binary> python3 count_dp.py 9` checks the DP against the
committed enumeration dump (oracle/fwd-only-hf9.dump) and must print PASS
with exact per-K matches. Larger scales need the corresponding
fwd-only-hfN.dump enumerations (generate the program with
../subsume/build_subsume_programs.py N, run it with the kernel, point
MITM_DUMPS at the directory); without them the oracle step reports itself
skipped rather than passing vacuously.

## Notes

- i64 products in-engine; driver sums in Python bigints. Max observed count
  67.4M, far below overflow; the boundary is per-pair products exceeding i64,
  which the driver can detect from magnitudes before seeding a stratum.
- States here are UNSUBSUMED alpha-distinct schemas; combining counting with
  the L4 antichain changes what is counted (derivations reroute through general
  lemmas), so the two layers compose for existence search but not for exact
  counting -- kept separate by design.
- Promotion to a shipped example belongs with the L8 packaging pass.
