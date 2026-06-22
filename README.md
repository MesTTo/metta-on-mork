# MeTTa-On-Mork

The MORK kernel as an in-process [Hyperon](https://github.com/trueagi-io/hyperon-experimental)
atomspace backend. `MorkSpace` implements Hyperon's `Space`/`SpaceMut` traits over
MORK's PathMap trie and worst-case-optimal-join matcher, so a Hyperon atomspace runs
on MORK's kernel with no network hop or serialization boundary.

## Head-to-head vs stock GroundingSpace

Same workload (load N `(edge nK nK+1)` atoms, then a point query). MeTTa-On-Mork
byte-level codec vs the stock Hyperon `GroundingSpace`:

| N        | load (Grounding → Mork) | warm query (Grounding → Mork) |
|----------|-------------------------|-------------------------------|
| 100,000  | 132 ms → **11 ms**      | ~16 µs → **~2.4 µs**          |
| 500,000  | 979 ms → **55 ms**      | ~16 µs → **~2.4 µs**          |

Load is ~18× faster and a warm point query ~6× faster. A *cold* first query (a new
query shape) is ~38 µs vs GroundingSpace's ~16 µs, constant in N either way.
(`cargo run --release --example scale_showcase`, and the `grounding_bench` example
in hyperon-experimental/lib for the baseline.)

Note on #1076: that issue is the GroundingSpace trie panicking on the first query
after ~2k atoms for a specific workload (Project Dagaz). This synthetic edge workload
does not trigger the panic — stock GroundingSpace handles it — so the win here is
throughput, not crash-avoidance. A faithful #1076 repro is future work.

## Two fixes that made it fast

1. **Kernel (general):** `query_factor_plan` computed `btm.val_count()` — an O(space)
   whole-trie count — on every query, to seed a shape side-index used only for
   multi-factor ordering. A single-pattern query (the most common MeTTa shape) never
   needs it. Computing it only for `sources.len() > 1` drops a 500k point query from
   ~3 ms to ~3 µs. The multi-factor path is byte-identical, so the six core
   benchmarks are unchanged. (Committed in the MORK kernel.)
2. **Codec (this crate):** a direct byte-level `Atom`↔trie-bytes codec, no text
   round-trip.

## Parallel querying (read-only snapshots)

A `MorkSnapshot` is `Send + Sync`, so read-only point queries parallelize across threads.
The kernel's match path used to take a global query-metrics mutex on every query, which
collapsed 16-thread throughput to ~3.6M q/s (below the 8-thread number). Accumulating those
metrics per-thread instead removed the contention, and point-query throughput then scales
cleanly on a Ryzen 9950X (16 cores / 32 threads):

| threads | 1    | 8     | 16    | 32    |
|---------|------|-------|-------|-------|
| q/s     | 1.9M | 12.7M | 17.6M | 26.1M |

(`cargo run --release --example parallel_query` exercises the `Send + Sync` snapshot.)

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

Build/test with the flags MORK needs:

```
RUSTFLAGS="-C target-cpu=native" cargo +nightly test
RUSTFLAGS="-C target-cpu=native" cargo +nightly run --release --example scale_showcase
```

## How it works

`encode_atom` walks a Hyperon `Atom` into MORK's preorder byte encoding
(`Arity`/`SymbolSize`/`NewVar`/`VarRef`), tracking variables in introduction order;
`decode_atom` walks the bytes back. `add`/`remove` insert/remove those bytes in the
trie; `query` encodes the pattern, wraps it as `(, pattern)`, calls `query_multi`,
and reads binding `(0, i)` for the i-th variable, decoding each bound sub-expression
into an `Atom`; `atom_count` is `val_count`; `visit` iterates the trie with a read
zipper. Symbols are stored as raw bytes (MORK's default; the `interning` feature is
incomplete and currently breaks correctness, and would enlarge short symbols anyway).

## Limitations (honest)

- **Full `MorkSpace` is not `Sync`.** `query` is `&self`, but Hyperon's
  `SpaceCommon`, MORK/PathMap internals, and grounded atoms carry non-`Sync` state.
  Use `MorkSnapshot` for `Send + Sync` read-only parallel querying.
- **Grounded atom boundaries.** Immutable grounded atoms are content-addressed by
  display string. Mutable grounded atoms, such as `State`, are stored by per-instance
  identity and matched by current live value. Snapshots and sharded spaces carry no
  grounded registry, so they are for immutable content-addressed data.
- **`remove` of mutable-grounded atoms.** `remove` uses the content key. An atom
  stored by mutable identity id cannot be removed by reconstructing the value key.
- **Single-pattern queries.** Conjunctive (`,`-glued) sub-queries are not yet split
  into a native MORK multi-factor join.
- **Symbol/arity <= 63** (MORK's 6-bit fields). `add` rejects atoms outside that
  encoding and increments `rejected_atom_count()`.

## Layout

- `src/lib.rs` — `MorkSpace`, the `Space`/`SpaceMut` impls, the byte-level codec.
- `examples/scale_showcase.rs` — the load + query benchmark.
- `examples/query_warmup.rs` — cold-vs-warm query timing.
- `examples/parallel_query.rs` — parallel querying on a `Send + Sync` snapshot.

## License

MIT (`SPDX-License-Identifier: MIT`). See [LICENSE](LICENSE). Each source file carries an
SPDX header. The dependencies keep their own licenses: Hyperon (`hyperon-atom`,
`hyperon-space`, `hyperon-common`) and MORK/PathMap.
