// SPDX-License-Identifier: MIT
//! Is MorkSpace's query cost per-query or a cold first-query (COW) artifact?
//! Load 500k, then time successive queries.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    let n = 500_000u32;
    let mut space = MorkSpace::new();
    let mut text = String::with_capacity(n as usize * 16);
    for i in 0..n {
        text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
    }
    space.add_sexpr_text(&text).unwrap();

    for k in 0..8 {
        let q = Atom::expr([
            Atom::sym("edge"),
            Atom::sym(format!("n{}", 100_000 + k)),
            Atom::var("dst"),
        ]);
        let t = Instant::now();
        let r = space.query(&q);
        println!("query #{}: {:>10.2?}  ({} result)", k, t.elapsed(), r.len());
    }
}
