# mork-hyperon-space

The MORK kernel as an in-process [Hyperon](https://github.com/trueagi-io/hyperon-experimental)
atomspace backend. `MorkSpace` implements Hyperon's `Space`/`SpaceMut` traits over
MORK's PathMap trie and worst-case-optimal-join matcher, so a Hyperon atomspace
gains MORK's scale and speed without a network hop or serialization boundary.

## Why

hyperon-experimental issue #1076: the default `GroundingSpace` trie panics on the
first query after roughly 1.9k–3k atoms. The same workload runs fine on MORK.

Measured here (`cargo run --release --example scale_showcase`):

| atoms   | load    | first query | result  |
|---------|---------|-------------|---------|
| 10,000  | ~1.0 ms | ~76 µs      | correct |
| 100,000 | ~11 ms  | ~0.5 ms     | correct |
| 500,000 | ~54 ms  | ~3.0 ms     | correct |

500,000 atoms is ~250× the scale that crashes the trie.

## Use

```rust
use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use mork_hyperon_space::MorkSpace;

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

## How it works (v1)

Atoms cross the boundary as S-expressions through MORK's own tested parser and
serializer. `add`/`remove` go through `add_all_sexpr`/`remove_all_sexpr`; `query`
encodes the pattern together with a variable-tuple template in one parse (so the
template references the pattern's variables), runs MORK's `dump_sexpr`, and decodes
each substituted tuple back into Hyperon `Bindings`. `atom_count` is
`PathMap::val_count`; `visit` walks `dump_all_sexpr`.

## v1 limitations (honest)

- **Text-mediated codec.** Atoms round-trip through S-expression text rather than a
  direct byte-level `Atom`↔`Expr` codec. Already fast; a byte codec is the next
  optimization.
- **`RefCell`, so `!Sync`.** Hyperon's `query` is `&self` while MORK's parser interns
  through `&mut`; v1 borrows mutably for a query. A `&self` interning path (via the
  shared symbol-table handle) would restore `Sync`.
- **Symbolic atoms only.** Symbols, expressions, and variables are fully supported.
  Grounded atoms serialize via their `Display` (a symbol), not as native grounded
  values; faithful grounded support needs a side table and execution semantics.
- **Single-pattern queries.** Conjunctive (`,`-glued) sub-queries are not yet split.
- **Symbol/arity limits.** MORK symbols are 1..=63 bytes and arity < 64.

## Layout

- `src/lib.rs` — `MorkSpace`, the `Space`/`SpaceMut` impls, the text codec.
- `examples/scale_showcase.rs` — the load + first-query benchmark above.
