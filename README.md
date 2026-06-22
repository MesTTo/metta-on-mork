# MeTTa-on-MORK

MeTTa-on-MORK runs a Hyperon atomspace directly on the MORK kernel. `MorkSpace` implements
Hyperon's `Space` and `SpaceMut` traits over MORK's PathMap byte-trie and its
worst-case-optimal-join matcher, so a MeTTa program evaluates against MORK in the same
process, with no network hop and no serialization boundary between the language and its store.

## Why this matters

The reason to build this is speed, and speed large enough to change what you can attempt.
MORK gives Hyperon a content-addressed byte-trie for storage and a zipper-based,
multi-threaded matcher for evaluation, built to stay fast across the full range of Space
sizes and shapes that MeTTa produces. Stock Hyperon slows down under load in four places: the
query planner, the matcher, the result-emit path, and the locks between them. Rewriting those
turns work that took seconds into work that takes microseconds.

The measurements bear it out. On this machine a warm point query that stock `GroundingSpace`
answers in about 16µs answers in about 2.4µs. A 500,000-atom load drops from about 979ms to
about 55ms. A whole-space join that the unoptimized path grinds through runs on the order of a
thousand times faster. Read-only point queries scale to 17.6 million per second across sixteen
cores, once a per-query lock comes off the hot path.

A speedup of that size is a qualitative change, not a quantitative one. It is the difference
between running a single training step and finishing the training in the same wall-clock time.
It is the difference between feeding a model a thousand samples and feeding it millions. Deep
learning advanced in part because its software platforms finally exposed the full capability of
the hardware beneath them, and the aim here is to do the same for symbolic AI: make the
substrate fast enough that the size of the knowledge base stops being the thing that decides
what you can build.

## The optimized MORK fork

This crate targets a personal, optimized fork of the MORK kernel
([github.com/MesTTo/MORK](https://github.com/MesTTo/MORK)) and of PathMap
([github.com/MesTTo/PathMap](https://github.com/MesTTo/PathMap)). The fork keeps MORK's design
and adds the work that makes it hold up at scale.

- A worst-case-optimal join for multi-pattern queries, so a conjunctive match costs what the
  output costs and not what the cross-product costs. This is where the largest wins come from.
- A compiled discrimination-trie matcher that walks the pattern and the trie together in a
  single descent.
- A streamed, factorized emit that writes results back into the Space without materializing the
  full product first.
- A single-factor fast path that skips the planner and the per-query metric locks for the
  common one-pattern shape, with per-thread metric accumulation so parallel queries stop
  serializing on a global mutex.
- WILLIAM-on-MORK: a compression-gain weighted index over the trie that returns the heaviest,
  most compressible subpatterns from any prefix without a scan. This is the substrate the
  Hyperon whitepaper (5.12) describes for compression-guided cognition, where the patterns
  worth remembering are the ones that compress experience best.

Measured against stock upstream MORK with `mork bench` (native build, minimum of eight runs),
the fork is faster on every core benchmark:

| benchmark | speedup vs upstream MORK |
|-----------|--------------------------|
| clique-style whole-space join | ~1200× |
| process-calculus rewriting | 2.0× |
| transitive closure | 1.5–2.4× |
| finite-domain solving | 1.8× |
| counter-machine | 1.35× |

These are correctness-preserving wins, not shortcuts. The single-factor and per-thread-metric
change is byte-identical to the planned path: the same results, the same unification and
instruction counts, verified by a differential. And WILLIAM's top-k iterator returns the 16
heaviest of 70,000 compressible prefixes in about 20µs, against about 3.8ms for the equivalent
full scan, a 190× difference, with a test that checks the fast answer equals the scan.

## Against stock GroundingSpace

Same workload (load N `(edge nK nK+1)` atoms, then a point query), MeTTa-on-MORK's byte-level
codec against Hyperon's stock `GroundingSpace`:

| N | load (Grounding → Mork) | warm query (Grounding → Mork) |
|---|---|---|
| 100,000 | 132 ms → **11 ms** | ~16 µs → **~2.4 µs** |
| 500,000 | 979 ms → **55 ms** | ~16 µs → **~2.4 µs** |

Load is about 18× faster and a warm point query about 6×. A cold first query, meaning a query
shape MORK has not seen before, is about 38µs against GroundingSpace's ~16µs, and both stay
constant in N. Run `cargo run --release --example scale_showcase` for the MORK side, and the
`grounding_bench` example in `hyperon-experimental/lib` for the baseline.

On hyperon-experimental #1076: that issue is the GroundingSpace trie panicking on the first
query past about 2k atoms for a specific workload (Project Dagaz). This synthetic edge workload
does not trigger the panic, so stock GroundingSpace handles it, and the win here is throughput
rather than crash-avoidance. A faithful #1076 repro is still future work.

## Parallel querying

A `MorkSnapshot` is `Send + Sync`, so read-only point queries parallelize across threads. The
match path used to take a global metrics mutex on every query, which collapsed sixteen-thread
throughput to about 3.6M q/s, below even the eight-thread number. Accumulating those metrics
per-thread takes the contention off the hot path, and throughput then scales cleanly on a
Ryzen 9950X (16 cores, 32 threads):

| threads | 1 | 8 | 16 | 32 |
|---------|----|----|-----|-----|
| q/s | 1.9M | 13.2M | 17.6M | 26.6M |

`cargo run --release --example parallel_query` exercises the `Send + Sync` snapshot.

## Use

```rust
use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

let mut space = MorkSpace::new();
space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Bob")]));

let q = Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]);
let results = space.query(&q); // child = Bob
```

Build and test with the flags MORK needs:

```
RUSTFLAGS="-C target-cpu=native" cargo +nightly test
RUSTFLAGS="-C target-cpu=native" cargo +nightly run --release --example scale_showcase
```

## How it works

`encode_atom` walks a Hyperon `Atom` into MORK's preorder byte encoding
(`Arity`/`SymbolSize`/`NewVar`/`VarRef`), tracking variables in introduction order, and
`decode_atom` walks the bytes back. `add` and `remove` insert and remove those bytes in the
trie. `query` encodes the pattern, wraps it as `(, pattern)`, calls `query_multi`, and reads
binding `(0, i)` for the i-th variable, decoding each bound sub-expression back into an `Atom`.
`atom_count` is `val_count`, and `visit` iterates the trie with a read zipper. Symbols are
stored as raw bytes, which is MORK's default; the `interning` feature is incomplete, currently
breaks correctness, and would enlarge short symbols anyway.

## Limitations (honest)

- Full `MorkSpace` is not `Sync`. `query` is `&self`, but Hyperon's `SpaceCommon`, the MORK and
  PathMap internals, and grounded atoms carry non-`Sync` state. Use `MorkSnapshot` for
  `Send + Sync` read-only parallel querying.
- Grounded atom boundaries. Immutable grounded atoms are content-addressed by display string.
  Mutable grounded atoms such as `State` are stored by per-instance identity and matched by
  current live value. Snapshots and sharded spaces carry no grounded registry, so they are for
  immutable content-addressed data.
- `remove` of a mutable-grounded atom uses the content key, so an atom stored by mutable
  identity id cannot be removed by reconstructing the value key.
- Single-pattern queries. Conjunctive (`,`-glued) sub-queries are not yet split into a native
  MORK multi-factor join from the bridge.
- Symbol and arity at most 63, from MORK's 6-bit fields. `add` rejects atoms outside that
  encoding and increments `rejected_atom_count()`.

## Layout

- `src/lib.rs` carries `MorkSpace`, the `Space`/`SpaceMut` impls, and the byte-level codec.
- `examples/scale_showcase.rs` is the load and query benchmark.
- `examples/query_warmup.rs` times cold versus warm queries.
- `examples/parallel_query.rs` runs parallel querying on a `Send + Sync` snapshot.

## License

MIT (`SPDX-License-Identifier: MIT`). See [LICENSE](LICENSE); each source file carries an SPDX
header. The dependencies keep their own licenses: Hyperon (`hyperon-atom`, `hyperon-space`,
`hyperon-common`) and MORK/PathMap.
