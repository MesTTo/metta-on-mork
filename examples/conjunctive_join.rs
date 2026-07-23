// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 MesTTo
//! The asymptotic case for native conjunctive queries: a 2-hop join
//! `(, (edge $x $m) (edge $m $y))` over an N-edge chain.
//!
//! GroundingSpace folds a conjunction: query the first conjunct (N results),
//! then re-query the space once per result — O(N²) on this shape even before
//! its per-query scan. MorkSpace hands the whole conjunction to MORK's
//! worst-case-optimal join, which costs what the output costs: O(N) here.
//! The gap therefore *grows* with N; watch the ratio climb.

use std::time::Instant;

use hyperon::space::grounding::GroundingSpace;
use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn edges(n: usize) -> Vec<Atom> {
    (0..n)
        .map(|i| {
            Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("n{i}")),
                Atom::sym(format!("n{}", i + 1)),
            ])
        })
        .collect()
}

fn two_hop() -> Atom {
    Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("m")]),
        Atom::expr([Atom::sym("edge"), Atom::var("m"), Atom::var("y")]),
    ])
}

fn main() {
    println!("2-hop join (, (edge $x $m) (edge $m $y)) over an N-edge chain; expected results = N-1");
    for n in [500usize, 1_000, 2_000, 8_000, 32_000, 128_000, 512_000] {
        let atoms = edges(n);

        let mut mork = MorkSpace::new();
        for a in &atoms {
            mork.add(a.clone());
        }
        let t = Instant::now();
        let mork_results = mork.query(&two_hop()).len();
        let mork_time = t.elapsed();
        assert_eq!(mork_results, n - 1, "join must return one result per interior node");

        // GroundingSpace's trie index panics on this workload past ~2k atoms:
        // hyperon-experimental #1076, reproduced here by an ordinary conjunctive
        // query. Catch it so the MORK side can keep scaling.
        let ground_report = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ground = GroundingSpace::new();
            for a in &atoms {
                ground.add(a.clone());
            }
            let t = Instant::now();
            let ground_results = ground.query(&two_hop()).len();
            (ground_results, t.elapsed())
        }));

        match ground_report {
            Ok((ground_results, ground_time)) => {
                assert_eq!(mork_results, ground_results, "join results diverge");
                let ratio = ground_time.as_secs_f64() / mork_time.as_secs_f64().max(1e-9);
                println!(
                    "N={n:6} | results {mork_results:6} | GroundingSpace {ground_time:>10.2?} | MorkSpace {mork_time:>10.2?} | {ratio:8.1}x"
                );
            }
            Err(_) => {
                println!(
                    "N={n:6} | results {mork_results:6} | GroundingSpace PANIC (#1076)   | MorkSpace {mork_time:>10.2?} |"
                );
            }
        }
    }
}
