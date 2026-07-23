// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! Showcase: load far past the hyperon-experimental #1076 trie crash (~2k atoms)
//! and run the first query, with timings. The default GroundingSpace trie panics
//! on the first query after ~1.9-3k atoms; MorkSpace handles orders of magnitude
//! more, fast.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    for n in [10_000u32, 100_000, 500_000] {
        let mut space = MorkSpace::new();

        let mut text = String::with_capacity(n as usize * 16);
        for i in 0..n {
            text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
        }

        let t = Instant::now();
        space.add_sexpr_text(&text).unwrap();
        let load = t.elapsed();

        let mid = n / 2;
        let q = Atom::expr([
            Atom::sym("edge"),
            Atom::sym(format!("n{}", mid)),
            Atom::var("dst"),
        ]);
        let t = Instant::now();
        let results = space.query(&q);
        let query = t.elapsed();

        println!(
            "{:>7} atoms | load {:>8.2?} | first query {:>10.2?} | {} result(s) -> {:?}",
            space.len(),
            load,
            query,
            results.len(),
            results
                .iter()
                .next()
                .and_then(|b| b.resolve(&hyperon_atom::VariableAtom::new("dst")))
                .map(|a| a.to_string()),
        );
    }
}
