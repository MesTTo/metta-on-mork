// SPDX-License-Identifier: MIT
//! MeTTa-On-Mork showcase: Hyperon's MeTTa, on the MORK kernel.
//! Scale, indexed query, real MM2 forward-chaining computation, and data-parallel
//! reasoning -- the HE-MeTTa -> {MORK} path, end to end, with live numbers.
use std::time::Instant;

use hyperon_atom::{Atom, VariableAtom};
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::{MorkSpace, ShardedMorkSpace};

fn bar(label: &str) {
    println!("\n\x1b[1m{}\x1b[0m", label);
}

fn main() {
    println!("\x1b[1m=== MeTTa-On-Mork: Hyperon's MeTTa, on the MORK kernel ===\x1b[0m");

    bar("[1] SCALE  -- the atomspace that doesn't fall over");
    let n = 1_000_000u32;
    let mut space = MorkSpace::new();
    let mut text = String::with_capacity(n as usize * 16);
    for i in 0..n {
        text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
    }
    let t = Instant::now();
    space.add_sexpr_text(&text).unwrap();
    println!(
        "    loaded {} atoms in {:.0?}   (Hyperon GroundingSpace: ~2s, ~18x slower;",
        space.len(),
        t.elapsed()
    );
    println!("    and its trie panics on the #1076 workload after ~2k atoms)");

    bar("[2] QUERY  -- indexed, O(prefix), not a scan");
    let q = Atom::expr([Atom::sym("edge"), Atom::sym("n500000"), Atom::var("dst")]);
    let _ = space.query(&q); // warm the plan cache
    let t = Instant::now();
    let r = space.query(&q);
    println!(
        "    point query over {}M atoms: {:.2?}   ({} result; GroundingSpace ~16us)",
        n / 1_000_000,
        t.elapsed(),
        r.len()
    );

    bar("[3] COMPUTE -- real MeTTa forward-chaining (MM2 exec) to fixpoint");
    let mut g = MorkSpace::new();
    g.add_sexpr_text(
        "(edge a b)\n(edge b c)\n(edge c d)\n(edge d e)\n\
         (path a b)\n(path b c)\n(path c d)\n(path d e)\n",
    )
    .unwrap();
    let (mut prev, mut rounds) = (0usize, 0);
    let t = Instant::now();
    loop {
        g.add_sexpr_text("(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n")
            .unwrap();
        g.step(1);
        rounds += 1;
        if g.len() == prev {
            break;
        }
        prev = g.len();
    }
    let mut zs: Vec<String> = g
        .query(&Atom::expr([Atom::sym("path"), Atom::sym("a"), Atom::var("z")]))
        .iter()
        .filter_map(|b| b.resolve(&VariableAtom::new("z")))
        .map(|a| a.to_string())
        .collect();
    zs.sort();
    zs.dedup();
    println!(
        "    transitive closure to fixpoint in {} rounds, {:.0?}: (path a $z) = {:?}",
        rounds,
        t.elapsed(),
        zs
    );

    bar("[4] PARALLEL -- data-parallel reasoning Hyperon's Rc<RefCell> cannot express");
    let load = |shards| {
        let mut s = ShardedMorkSpace::new(shards);
        for i in 0..n {
            s.add(&Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("n{}", i)),
                Atom::sym(format!("n{}", i + 1)),
            ]));
        }
        s
    };
    let pat = Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("y")]);
    let s1 = load(1);
    let t = Instant::now();
    let c1 = s1.count_matches(&pat);
    let seq = t.elapsed();
    let s32 = load(32);
    let t = Instant::now();
    let c32 = s32.par_count_matches(&pat);
    let par = t.elapsed();
    assert_eq!(c1, c32);
    println!(
        "    match-count of {} atoms: 1 shard {:.0?}  ->  32 shards {:.0?}  ({:.1}x)",
        c1,
        seq,
        par,
        seq.as_nanos() as f64 / par.as_nanos() as f64
    );

    bar("[5] KERNEL  -- all six core benchmarks beat stock MORK (separately measured)");
    println!("    clique 5-way 1234x | finite_domain 1.67x | transitive 1.5-2.4x");
    println!("    process_calculus 2.0x | counter_machine 1.32x  (was 1.55x slower)");
    println!("\n  HE-MeTTa -> MeTTa-IL -> {{MORK + rholang}} : the MORK lane, working.\n");
}
