//! ShardZipper symbolic-CPU slice: hash-prefix-shard the atomspace and sweep all
//! shards in parallel. Whole-space match-count of (edge $x $y) -- sequential on one
//! trie vs across N shards in parallel (rayon).
use std::time::Instant;

use hyperon_atom::Atom;
use metta_on_mork::ShardedMorkSpace;

fn load(n: u32, shards: usize) -> ShardedMorkSpace {
    let mut space = ShardedMorkSpace::new(shards);
    for i in 0..n {
        space.add(&Atom::expr([
            Atom::sym("edge"),
            Atom::sym(format!("n{}", i)),
            Atom::sym(format!("n{}", i + 1)),
        ]));
    }
    space
}

fn main() {
    let n = 1_000_000u32;
    let pat = Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("y")]);

    let base = load(n, 1);
    let t = Instant::now();
    let base_count = base.count_matches(&pat);
    let base_t = t.elapsed();
    println!("baseline (1 shard, sequential): {} matches in {:.2?}", base_count, base_t);

    for shards in [4usize, 8, 16, 32] {
        let space = load(n, shards);
        let t = Instant::now();
        let par = space.par_count_matches(&pat);
        let par_t = t.elapsed();
        assert_eq!(par, base_count);
        println!(
            "{:>2} shards (parallel sweep): {} in {:>9.2?}  ({:.1}x vs baseline)",
            shards,
            par,
            par_t,
            base_t.as_nanos() as f64 / par_t.as_nanos() as f64
        );
    }
}
