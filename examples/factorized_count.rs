// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! The asymptotic case for the factorized conjunctive count: counting the 2-hop
//! join (, (edge $x $m) (edge $m $y)) over a K-K bipartite double star, whose
//! output is K² while the store holds only 2K edges.
//!
//! The enumerating count walks every one of the K² join results. The factorized
//! count (feature `factorized-aggregate`) folds counts up the join tree instead
//! -- O(N^fhtw), linear here -- so the gap grows linearly with K.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::SpaceMut;
use metta_on_mork::MorkSpace;

fn main() {
    println!("count of (, (edge $x $m) (edge $m $y)) over a K+K double star; join output = K^2");
    for k in [250usize, 500, 1_000, 2_000, 4_000] {
        let mut space = MorkSpace::new();
        for i in 0..k {
            space.add(Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("a{i}")),
                Atom::sym("hub"),
            ]));
            space.add(Atom::expr([
                Atom::sym("edge"),
                Atom::sym("hub"),
                Atom::sym(format!("b{i}")),
            ]));
        }
        let query = Atom::expr([
            Atom::sym(","),
            Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("m")]),
            Atom::expr([Atom::sym("edge"), Atom::var("m"), Atom::var("y")]),
        ]);
        let snap = space.snapshot();

        let t = Instant::now();
        let count = snap.count_matches(&query);
        let factorized_time = t.elapsed();

        // The enumerating reference: the same join, counted by walking it.
        let t = Instant::now();
        let enumerated = snap.query(&query).len();
        let enumerate_time = t.elapsed();

        assert_eq!(count, enumerated, "counts diverge");
        // The 2-hop join also pairs each a-edge with itself when $m chains through
        // shared endpoints; on this shape the exact output is K*K plus nothing else
        // only if a/b names never collide, which they don't.
        assert_eq!(count, k * k, "expected K^2 join results");
        let ratio = enumerate_time.as_secs_f64() / factorized_time.as_secs_f64().max(1e-9);
        println!(
            "K={k:5} | join output {count:9} | enumerate {enumerate_time:>10.2?} | factorized count {factorized_time:>10.2?} | {ratio:8.1}x"
        );
    }
}
