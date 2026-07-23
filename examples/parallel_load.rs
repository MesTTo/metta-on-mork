// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 MesTTo
//! Parallel bulk load on PathMap's own architecture: the trie is a prefix-
//! partitioned structure of copy-on-write nodes, so per-thread private tries
//! build without contention and merge by structural join (shared subtrees,
//! no deep copies). Atoms are Send + Sync on the thread-safe base, so the
//! input slice shards by reference.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    const N: usize = 1_000_000;
    let atoms: Vec<Atom> = (0..N)
        .map(|i| {
            Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("n{i}")),
                Atom::sym(format!("n{}", i + 1)),
            ])
        })
        .collect();

    let t = Instant::now();
    let mut seq = MorkSpace::new();
    for a in &atoms {
        seq.add(a.clone());
    }
    let sequential = t.elapsed();
    assert_eq!(seq.atom_count(), Some(N));

    println!("loading {N} atoms | sequential add loop: {sequential:.2?}");
    for threads in [2usize, 4, 8, 16] {
        let t = Instant::now();
        let mut par = MorkSpace::new();
        par.extend_parallel(&atoms, threads);
        let elapsed = t.elapsed();
        assert_eq!(par.atom_count(), Some(N));
        let speedup = sequential.as_secs_f64() / elapsed.as_secs_f64();
        println!("{threads:2} threads: {elapsed:>10.2?}  ({speedup:4.1}x)");
    }
}
