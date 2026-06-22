# MeTTa-On-Mork — Hyperon's MeTTa, on the MORK kernel

One command, the whole stack, live:

```
RUSTFLAGS="-C target-cpu=native" cargo +nightly run --release --example showcase
```

```
[1] SCALE     loaded 1,000,000 atoms in 119 ms        (~18x faster than GroundingSpace)
[2] QUERY     point query over 1M atoms: 2.1 µs        (indexed O(prefix); GroundingSpace ~16 µs)
[3] COMPUTE   transitive closure to fixpoint, 129 µs   real MM2 forward-chaining on MORK
[4] PARALLEL  match-count of 1M atoms: 207 ms -> 37 ms (5.6x, 32 shards)
[5] KERNEL    all six core benchmarks beat stock MORK  (clique 5-way 1234x)
```

## What this is

`MorkSpace` implements Hyperon's `Space`/`SpaceMut` over the optimized MORK kernel,
so Hyperon's own MeTTa runner executes against MORK (see the `mork-demo` crate:
`!(match &self (parent Tom $x) $x)` runs on the trie). On top of the atomspace, it
exposes MORK's **MM2 exec engine** (`step`, à la CeTTa's `mork:step!`) — so it is a
*computation* engine, not just storage — and a **hash-prefix sharded** space for
data-parallel sweeps (ShardZipper's symbolic-CPU path).

This is the **MORK lane** of the agreed roadmap **HE-MeTTa → MeTTa-IL → {MORK + rholang}**.

## Why each line matters to a MeTTa/MORK person

1. **Scale.** The default Hyperon `GroundingSpace` trie panics on the first query
   after ~2k atoms on the #1076 workload, and loads this synthetic 1M-edge set ~18x
   slower. MORK's PathMap handles it without ceremony.
2. **Query.** A point query is O(prefix) on the trie, not an O(N) scan. The kernel
   fix behind this — `query_factor_plan` was doing an O(space) `val_count` per query
   for a cache key it only needs when ordering multiple factors — is a general win
   for every single-pattern query (3 ms → 3 µs on 500k), benchmarks byte-identical.
3. **Compute.** `(exec <loc> (, <src>) (, <tpl>))` rules run forward-chaining to
   fixpoint inside the live space, on the optimized exec path the six benchmarks
   accelerate. The whole thing is driven from Hyperon `Atom`s through a byte-level
   `Atom`↔trie-bytes codec — no text round-trip.
4. **Parallel.** Hyperon's atomspace is `Rc<RefCell>` (issue #410) and interned every
   symbol through one global `Mutex` (~13% `lock_contended` at 8 threads). MORK's
   `PathMap` is `Send + Sync`; sharding the space (and sharding that interner 64 ways,
   committed to hyperon-common) gives real multi-core reasoning. Data-parallel branch
   evaluation is exactly what eval-control issue #448 asks for.
5. **Kernel.** All six core benchmarks beat stock MORK: clique 5-way **1234x**,
   finite_domain 1.67x, transitive 1.5–2.4x, process_calculus 2.0x, counter_machine
   1.32x (was 1.55x *slower* — fixed by compiling match programs in one linear
   flatterm pass instead of O(depth²)).

## Honest scope

- The #1076 *crash* is a specific Hyperon-trie bug on a particular workload; the
  synthetic edge set here doesn't trigger it on GroundingSpace (the win above is
  throughput). A faithful Dagaz #1076 repro is future work.
- Parallel here is data-parallel *querying/sweeping*; parallel *exec* (sharded
  forward-chaining) is the next step.
- Grounded atoms are encoded as symbols; symbolic atoms are full-fidelity.

## Layout

- `src/lib.rs` — `MorkSpace` (Space/SpaceMut + byte codec + `step`), `MorkSnapshot`
  (Send+Sync concurrent queries), `ShardedMorkSpace` (parallel sweep), `priority`
  (the #448 eval-control planner grabbed from MeTTaTron).
- `examples/showcase.rs` — the run above. Also `mm2_exec`, `sharded_sweep`,
  `parallel_query`, `scale_showcase`.
- `../hyperon-experimental/mork-demo` — Hyperon's MeTTa runner on a MorkSpace.
