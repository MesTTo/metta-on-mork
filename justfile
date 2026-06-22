set shell := ["bash", "-cu"]

rustflags := "-C target-cpu=native -Awarnings"
hyperon := "../hyperon-experimental"

check:
    RUSTFLAGS="{{rustflags}}" cargo +nightly test
    just kani
    just verus
    just panic-probe
    just conformance
    just chaining-compat
    just duplication

test:
    RUSTFLAGS="{{rustflags}}" cargo +nightly test

panic-probe:
    cd "{{hyperon}}" && RUSTFLAGS="{{rustflags}}" cargo +nightly run --release -p mork-demo --example panic_probe

conformance:
    cd "{{hyperon}}" && RUSTFLAGS="{{rustflags}}" cargo +nightly run --release -p mork-demo --example conformance

chaining-compat:
    cd "{{hyperon}}" && RUSTFLAGS="{{rustflags}}" cargo +nightly run --release -p mork-demo --example chaining_compat

kani:
    kani formal/kani_decode_atom.rs --harness decode_atom_head_does_not_panic_on_short_grounded_refs --unwind 12

verus:
    verus formal/verus/BindingsWellFormed.rs

duplication:
    jscpd --reporters ai --min-lines 8 --min-tokens 80 src ../hyperon-experimental/hyperon-atom/src/matcher.rs
