# MeTTa-on-MORK

MeTTa-on-MORK runs a Hyperon atomspace directly on the MORK kernel. `MorkSpace` implements
Hyperon's `Space` and `SpaceMut` traits over MORK's PathMap byte-trie and its multi-pattern
matcher, so a MeTTa program evaluates against MORK in the same process, with no network hop and
no serialization boundary between the language and its store.

## Why this matters

The reason to build this is scale: making the size of the knowledge base stop being the thing
that decides what you can attempt. Stock Hyperon's `GroundingSpace` slows down under load and,
past about 2,000 atoms, its trie index panics outright on an ordinary conjunctive query
(hyperon-experimental #1076 — reproduced in this repo, see below). The wins that matter at
scale are asymptotic, not constant-factor, and that is what this bridge goes after: queries
that seek instead of scan, joins that cost what their output costs, counts that never
enumerate, and fixpoints that only re-derive the delta.

Every number below was measured on this machine against this exact tree (nightly Rust,
`-C target-cpu=native`), with the command that reproduces it.

## The MORK base

This crate builds against **upstream [trueagi-io/MORK](https://github.com/trueagi-io/MORK)
main with the full set of 28 open MesTTo PRs merged** (branch `upstream-plus-prs` of the
[MesTTo/MORK](https://github.com/MesTTo/MORK) fork), on clean upstream
[PathMap](https://github.com/Adam-Vandervorst/PathMap). Upstream-plus-PRs is the semantically
accurate kernel; the deeper private-fork optimizations return here either as opt-in features
or as bridge-level machinery in this crate. The hyperon dependency
(`../hyperon-experimental`) must be on its `send-sync-atoms` branch, which makes atoms and
spaces thread-safe (`GroundedAtom: Send + Sync`, lock-based `Shared`/`SpaceCommon`/`DynSpace`)
with hyperon's full workspace suite passing — the refactor upstream issue #410 asks for.

The kernel's complexity opt-ins pass through as cargo features, each byte-identical to the
default path by the kernel's own differential suites:

- `semi-naive` — `metta_calculus` fixpoints match only each round's delta. Measured here on a
  chain transitive closure (`cargo run --release --features semi-naive --example
  semi_naive_step`): 3–6× (104.7 s naive → 30.1 s at N=800), bounded on that shape by its
  quadratic output, which both paths must insert; PR #128's own workload shows the redundancy
  the delta removes (98.8% of naive match candidates on `process_calculus`, 201,401
  unifications naive against 2,377).
- `leapfrog` — flat conjunctive exec bodies route through the worst-case-optimal
  leapfrog-unification join (PR #124).
- `factorized-aggregate` — COUNT/SUM/MIN/MAX/AND exec sinks fold the join instead of
  enumerating it (PR #130), and this crate's conjunctive `count_matches` rides the same
  engine (below).

## Asymptotics at the bridge

### Conjunctive queries are native worst-case-optimal joins

`Space::query`'s contract admits sub-queries glued by the comma symbol. MorkSpace encodes each
conjunct of `(, q1 .. qn)` as its own factor of one kernel multi-pattern query, so a variable
shared across conjuncts becomes a join variable the worst-case-optimal join matches natively —
instead of the per-conjunct query-and-thread fold an interpreter pays, with its
intermediate-product blowup. The 2-hop join `(, (edge $x $m) (edge $m $y))` over an N-edge
chain (`cargo run --release --example conjunctive_join`):

| N | GroundingSpace | MorkSpace |
|---|---|---|
| 500 | 2.03 ms | 927 µs |
| 1,000 | 2.88 ms | 1.65 ms |
| 2,000 | **panics (#1076)** | 3.06 ms |
| 32,000 | panics | 53.5 ms |
| 512,000 | panics | 954 ms |

MorkSpace is output-linear across the whole range (1024× the edges, 1029× the time). The same
ordinary conjunctive query is a faithful reproduction of hyperon-experimental #1076:
GroundingSpace's trie index panics from 2,000 atoms on. Results are asserted identical below
the panic threshold, and a randomized differential seals the conjunction semantics against
GroundingSpace's fold — the same equivalence LeaTTa 1.0.8 states as its encoded-backend
conjunctive-query law.

### Selective queries seek a column index instead of scanning

A query bound only on a non-leading argument — `(edge $x nK)` — defeats the trie's prefix
descent, so the matcher scans the relation. MorkSpace maintains a permuted-key
argument-position index per (relation, arity, position), built on first use, invalidated in
O(1) by a mutation generation counter, and admitted on the fork's proven fragment (ground
functor, each argument a fresh variable or fully ground, repeated variables declined). This is
the clause index MeTTaLingo ships, ported from the optimized fork
(`cargo run --release --example arg_index`):

| N | matcher scan | steady indexed query | ratio |
|---|---|---|---|
| 100,000 | 4.52 ms | 881 ns | 5,128× |
| 400,000 | 16.9 ms | 822 ns | 20,539× |
| 1,600,000 | 60.4 ms | 742 ns | 81,415× |

The scan grows linearly; the steady indexed query does not grow at all. The one O(N) build
(409 ms at 1.6M) amortizes over every query — and mutations maintain the index incrementally
(one O(1) permuted-key update per add/remove of the relation's facts) instead of invalidating
it, so interleaved add/query workloads keep seeking. Snapshots carry the fresh indexes as
copy-on-write clones: a `Send + Sync` `MorkSnapshot` answers the same selective query in
~730 ns at every measured N, so parallel workers seek too.

### Conjunctive counts never enumerate the join

With the `factorized-aggregate` feature, `count_matches` on a conjunctive query folds the join
over its hypertree decomposition (`mork::ghd`, PR #130): O(N^fhtw) against the enumeration's
O(join output) — an exponent drop, not a factor. Counting the 2-hop join over a K+K double
star whose output is K² (`cargo run --release --features factorized-aggregate --example
factorized_count`):

| K | join output | enumerate | factorized count | ratio |
|---|---|---|---|---|
| 250 | 62,500 | 76.5 ms | 756 µs | 101× |
| 1,000 | 1,000,000 | 1.17 s | 3.05 ms | 384× |
| 4,000 | 16,000,000 | 19.5 s | 12.6 ms | 1,556× |

The ratio doubles with K because the fold is linear while the enumeration is quadratic; a
16-million-result join counts in 12.6 ms. Two admissions keep it exact, both found by this
crate's differentials: variable-bearing stores stay on the enumerating path (a stored bare
`$x` unifies with any factor but is invisible to a prefix seek — pinned as a test), and a
variable nested in a compound column declines (the documented #130 fragment).

### Repeated queries replay in O(answers)

The space tables the matcher's raw result rows per encoded pattern (an
alpha-invariant key), invalidated by the mutation generation. A replay decodes
the rows afresh — new variable ids per result, the mutable-grounded live filter
re-applied — so it is indistinguishable from a live match, and it costs the
answers, not the store. The worst-case shape is a variable-functor pattern
`($x mid $y)`, which neither the trie descent nor the column index can take
(`cargo run --release --example query_tabling`):

| N | first call (scan + fill) | tabled replay | ratio |
|---|---|---|---|
| 100,000 | 3.12 ms | 1.68 µs | 1,855× |
| 400,000 | 11.5 ms | 1.68 µs | 6,834× |
| 1,600,000 | 45.7 ms | 1.99 µs | 22,981× |

Unlike the byte-seek paths, tabling needs no var-freeness latch: the replayed
rows are the unifier's own output, schematic data included. Conjunctive
queries ride the same cache, so a repeated join replays without re-running the
join. Caps: 256 shapes per space, 4,096 rows per shape.

## Against stock GroundingSpace

Same workload (load N `(edge nK nK+1)` atoms, then a point query), measured back to back
(`cargo run --release --example scale_showcase` here, the `grounding_bench` example in
`hyperon-experimental/lib` for the baseline):

| N | load (Grounding → Mork) | point query (Grounding → Mork warm) |
|---|---|---|
| 100,000 | 114 ms → **12.7 ms** | ~16 µs → **~3.7 µs** |
| 500,000 | 935 ms → **67.4 ms** | ~16 µs → **~3.7 µs** |

Load is 9–14× faster and a warm point query about 4×; a 1M-atom load lands in 154 ms. A cold
first query (a shape MORK has not seen) is ~30 µs against GroundingSpace's ~16 µs, both flat
in N.

## Parallel querying, on one shared space

`MorkSpace` itself is now `Send + Sync` (a compile-time assertion in the test suite): one
live space object is shared by plain reference across threads — no snapshots, no clones.
`cargo run --release --example shared_space_parallel` alternates trie-descent point queries
and column-index seeks from every thread against the same space: 2.75 µs/query at 1 thread
to 1.00 µs/query at 16 threads (320,000 queries, every result asserted). `MorkSnapshot`
remains the frozen-view option (`--example parallel_query`), carrying the column indexes for
~730 ns seeks. Matcher-path thread scaling is sublinear because the upstream kernel keeps
process-global matcher counters that all threads write; the private fork's per-thread
accumulation is not yet in any upstream PR.

`fork()` is the copy-on-write branch-and-explore primitive: an O(1) clone of the whole space
(trie, registry, indexes, tabled queries share structure until a side mutates) where a
per-atom copy is O(N). `union_with`/`intersect_with`/`subtract_space` run PathMap's
join/meet/subtract on the trie structure itself, sharing common subtrees with both operands
instead of iterating atoms; they decline on stores holding mutable grounded atoms (space-local
identity ids), and a proptest differential holds each equal to per-atom set semantics.

## WILLIAM, carried in-crate

`compression_gain_index(ref_cost)` builds the whitepaper-5.12 term-boundary compression-gain
index over the stored atoms (every whole-subexpression prefix shared by ≥2 atoms weighted by
the bytes factoring it would save), and `frequent_subpatterns(k, ref_cost)` reports the k
heaviest patterns as a prefix-free antichain, rendered as readable MeTTa. The upstream kernel's
`weighted_paths` sidecar (PR #101) stops at weight bookkeeping, so the compression-gain
builder, the maximal top-k, and the pattern renderer live in this crate (`src/william.rs`),
byte-compatible with the optimized fork's index.

## Use

```rust
use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

let mut space = MorkSpace::new();
space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Bob")]));

let q = Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]);
let results = space.query(&q); // child = Bob

// Conjunctions join natively:
let two_hop = Atom::expr([
    Atom::sym(","),
    Atom::expr([Atom::sym("parent"), Atom::var("g"), Atom::var("p")]),
    Atom::expr([Atom::sym("parent"), Atom::var("p"), Atom::var("c")]),
]);
let grandparents = space.query(&two_hop);
```

Build and test with the flags MORK needs:

```
RUSTFLAGS="-C target-cpu=native" cargo +nightly test
RUSTFLAGS="-C target-cpu=native" cargo +nightly test --features "semi-naive,leapfrog,factorized-aggregate"
RUSTFLAGS="-C target-cpu=native" cargo +nightly run --release --example scale_showcase
```

## How it works

`encode_atom` walks a Hyperon `Atom` into MORK's preorder byte encoding
(`Arity`/`SymbolSize`/`NewVar`/`VarRef`), tracking variables in introduction order, and
`decode_atom` walks the bytes back. `add` and `remove` insert and remove those bytes in the
trie. `query` encodes the pattern (each conjunct of a `(, ...)` query as its own factor),
calls `query_multi`, and reads binding `(0, i)` for the i-th variable, decoding each bound
sub-expression back into an `Atom`. `atom_count` is `val_count`, and `visit` iterates the trie
with a read zipper. Symbols are stored as raw bytes, which is MORK's default.

The byte-level fast paths (the column index, the factorized count) are gated by a one-way
var-freeness latch: while every stored atom is variable-free they are admitted, and the first
variable-bearing `add`, `add_sexpr_text` containing `$`, or `step()` sends all queries back to
the general matcher, which unifies stored variables correctly. Routing changes complexity,
never answers — a proptest invariant holds the routed query equal to the raw matcher on every
query shape.

## Limitations

- `MorkSpace` is `Send + Sync` on the `send-sync-atoms` hyperon branch; sharing it across
  threads gives reader parallelism (`query` is `&self`). Mutation needs `&mut` or an outer
  lock, as usual.
- Grounded atom boundaries. Immutable grounded atoms are content-addressed by display string.
  Mutable grounded atoms such as `State` are stored by per-instance identity and matched by
  current live value. Snapshots and sharded spaces carry no grounded registry, so they are for
  immutable content-addressed data.
- `remove` of a mutable-grounded atom uses the content key, so an atom stored by mutable
  identity id cannot be removed by reconstructing the value key.
- The var-freeness latch is one-way: after the first variable-bearing add, text load with
  `$`, or `step()`, the byte-level fast paths stay off for that space's lifetime (correctness
  first; the matcher path handles every shape).
- Symbol and arity at most 63, from MORK's 6-bit fields; a conjunction takes at most 62
  conjuncts. `add` rejects atoms outside the encoding and increments `rejected_atom_count()`.

## Layout

- `src/lib.rs` — `MorkSpace`, the `Space`/`SpaceMut` impls, the byte-level codec, conjunctive
  encoding, the factorized count, and the differential test suite.
- `src/argindex.rs` — the argument-position (column) index: build, seek, classify.
- `src/william.rs` — the WILLIAM compression-gain index and pattern report.
- `examples/conjunctive_join.rs` — WCO join scaling and the #1076 reproduction.
- `examples/arg_index.rs` — column-index scaling against the matcher scan.
- `examples/factorized_count.rs` — factorized versus enumerating conjunctive counts.
- `examples/query_tabling.rs` — tabled replay against the live scan.
- `examples/shared_space_parallel.rs` — one `Send + Sync` space shared across threads.
- `examples/semi_naive_step.rs` — naive versus semi-naive fixpoint stepping.
- `examples/scale_showcase.rs`, `examples/query_warmup.rs`, `examples/parallel_query.rs` —
  load, cold/warm query, and parallel snapshot benchmarks.

## License

MIT (`SPDX-License-Identifier: MIT`). See [LICENSE](LICENSE); each source file carries an SPDX
header. The dependencies keep their own licenses: Hyperon (`hyperon-atom`, `hyperon-space`,
`hyperon-common`) and MORK/PathMap.
