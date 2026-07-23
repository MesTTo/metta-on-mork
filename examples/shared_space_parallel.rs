// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! The Send+Sync payoff: ONE MorkSpace shared by reference across threads.
//! No snapshots, no copies -- queries are `&self`, atoms are Send+Sync on the
//! thread-safe hyperon base, and every cache in the space is lock-based. Each
//! thread runs point queries (trie descent) and selective queries (column-
//! index seeks) against the same live space object.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    const N: usize = 200_000;
    const QUERIES_PER_THREAD: usize = 20_000;
    let mut space = MorkSpace::new();
    for i in 0..N {
        space.add(Atom::expr([
            Atom::sym("edge"),
            Atom::sym(format!("n{i}")),
            Atom::sym(format!("n{}", i + 1)),
        ]));
    }
    // Warm the column index for the selective shape once, so threads seek.
    let warm = Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::sym("n77")]);
    assert_eq!(space.query(&warm).len(), 1);

    let space = &space; // shared by reference: the whole point
    println!("one shared &MorkSpace, {N} atoms, {QUERIES_PER_THREAD} queries/thread");
    for threads in [1usize, 2, 4, 8, 16] {
        let t = Instant::now();
        std::thread::scope(|s| {
            for t_id in 0..threads {
                s.spawn(move || {
                    for k in 0..QUERIES_PER_THREAD {
                        let i = (t_id * QUERIES_PER_THREAD + k) % N;
                        // Alternate: leading-bound point query (trie descent)
                        // and non-leading-bound selective query (index seek).
                        let q = if k % 2 == 0 {
                            Atom::expr([
                                Atom::sym("edge"),
                                Atom::sym(format!("n{i}")),
                                Atom::var("y"),
                            ])
                        } else {
                            Atom::expr([
                                Atom::sym("edge"),
                                Atom::var("x"),
                                Atom::sym(format!("n{}", i + 1)),
                            ])
                        };
                        assert_eq!(space.query(&q).len(), 1);
                    }
                });
            }
        });
        let elapsed = t.elapsed();
        let total = threads * QUERIES_PER_THREAD;
        let per = elapsed.as_nanos() as f64 / total as f64;
        println!(
            "{threads:2} thread(s): {total:7} queries in {elapsed:>9.2?}  ({per:7.0} ns/query)"
        );
    }
}
