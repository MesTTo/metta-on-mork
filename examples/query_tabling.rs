// SPDX-License-Identifier: MIT
//! The asymptotic case for query tabling: a variable-functor pattern
//! ($x mid $y) defeats both the trie's prefix descent and the column index
//! (no ground functor), so every live match is a relation scan, O(N). The
//! space tables the matcher's raw rows per pattern: from the second call on,
//! an unchanged space replays in O(answers) -- flat in N -- with per-replay
//! decode keeping variable hygiene identical to a live match.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    println!("repeated ($x mid $y) -- variable functor, so a live match scans O(N); replays are O(answers)");
    for n in [100_000usize, 400_000, 1_600_000] {
        let mut space = MorkSpace::new();
        for i in 0..n {
            space.add(Atom::expr([
                Atom::sym(format!("f{i}")),
                Atom::sym(format!("a{i}")),
                Atom::sym(format!("b{i}")),
            ]));
        }
        // The needles: three atoms whose middle argument is `mid`.
        for k in 0..3 {
            space.add(Atom::expr([
                Atom::sym(format!("g{k}")),
                Atom::sym("mid"),
                Atom::sym(format!("c{k}")),
            ]));
        }
        let q = Atom::expr([Atom::var("x"), Atom::sym("mid"), Atom::var("y")]);

        let t = Instant::now();
        let first = space.query(&q).len();
        let scan = t.elapsed();

        let t = Instant::now();
        let mut replay_results = 0usize;
        const RUNS: u32 = 100;
        for _ in 0..RUNS {
            replay_results = space.query(&q).len();
        }
        let replay = t.elapsed() / RUNS;

        assert_eq!(first, 3);
        assert_eq!(replay_results, 3);
        let ratio = scan.as_secs_f64() / replay.as_secs_f64().max(1e-9);
        println!(
            "N={n:8} | first (scan+fill) {scan:>10.2?} | tabled replay {replay:>10.2?} | {ratio:9.1}x"
        );
    }
}
