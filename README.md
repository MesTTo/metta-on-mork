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

- **`RefCell`, so `!Sync`.** `query` is `&self` while MORK matching needs `&mut` to
  the trie cursor. A `&self` matching path would restore `Sync`.
- **Symbolic atoms.** Symbols, expressions, and variables are supported. Grounded
  atoms encode via their `Display` (as a symbol), not as native grounded values.
- **Single-pattern queries.** Conjunctive (`,`-glued) sub-queries are not yet split
  into a join.
- **Symbol/arity ≤ 63** (MORK's 6-bit fields).

## Layout

- `src/lib.rs` — `MorkSpace`, the `Space`/`SpaceMut` impls, the byte-level codec.
- `examples/scale_showcase.rs` — the load + query benchmark.
- `examples/query_warmup.rs` — cold-vs-warm query timing.
