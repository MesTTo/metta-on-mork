// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 MesTTo
//! MeTTa computation on MORK: load a forward-chaining `exec` rule, step MORK's MM2
//! engine to fixpoint, and read the transitive closure back through the integration.
//! This is the dream's computation engine (CeTTa's `mork:step!`) on the optimized
//! kernel, driven entirely from Hyperon `Atom`s.
use hyperon_atom::{Atom, VariableAtom};
use hyperon_space::Space;
use metta_on_mork::MorkSpace;

fn main() {
    let mut space = MorkSpace::new();
    space
        .add_sexpr_text(
            "(edge a b)\n(edge b c)\n(edge c d)\n(edge d e)\n\
             (path a b)\n(path b c)\n(path c d)\n(path d e)\n",
        )
        .unwrap();
    println!("seed: {} atoms", space.len());

    let (mut prev, mut rounds) = (0usize, 0u32);
    loop {
        space
            .add_sexpr_text("(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n")
            .unwrap();
        space.step(1);
        rounds += 1;
        let now = space.len();
        if now == prev {
            break;
        }
        prev = now;
    }
    println!("MM2 exec reached fixpoint in {} rounds, {} atoms", rounds, space.len());

    let q = Atom::expr([Atom::sym("path"), Atom::sym("a"), Atom::var("z")]);
    let mut zs: Vec<String> = space
        .query(&q)
        .iter()
        .filter_map(|b| b.resolve(&VariableAtom::new("z")))
        .map(|a| a.to_string())
        .collect();
    zs.sort();
    zs.dedup();
    println!("transitive closure (path a $z) => {:?}", zs);
}
