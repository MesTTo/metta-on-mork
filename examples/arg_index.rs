// SPDX-License-Identifier: MIT
//! The asymptotic case for the argument-position index: the point query
//! (edge $x nK) binds only the NON-leading argument, so the primary trie
//! cannot seek it -- the matcher scans the whole relation, O(N). MorkSpace
//! maintains a permuted-key column index instead: one O(N) build on first
//! use (amortized over every later query and rebuilt only on mutation), then
//! each query is a prefix seek whose cost tracks the answer, not the store.
//!
//! The comparator is a MorkSnapshot of the same trie: snapshots carry no
//! index cache, so their query is the plain matcher scan.

use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    println!("(edge $x nK) -- bound only on the second argument; scan is O(N), the index seeks");
    for n in [100_000usize, 400_000, 1_600_000] {
        let mut space = MorkSpace::new();
        for i in 0..n {
            space.add(Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("n{i}")),
                Atom::sym(format!("n{}", i + 1)),
            ]));
        }
        let q = Atom::expr([
            Atom::sym("edge"),
            Atom::var("x"),
            Atom::sym(format!("n{}", n / 2)),
        ]);

        let snap = space.snapshot();
        let t = Instant::now();
        let scan_results = snap.query(&q).len();
        let scan = t.elapsed();

        let t = Instant::now();
        let first = space.query(&q).len();
        let build = t.elapsed();

        let t = Instant::now();
        let mut steady_results = 0usize;
        const STEADY_RUNS: u32 = 100;
        for _ in 0..STEADY_RUNS {
            steady_results = space.query(&q).len();
        }
        let steady = t.elapsed() / STEADY_RUNS;

        assert_eq!(scan_results, 1);
        assert_eq!(first, 1);
        assert_eq!(steady_results, 1);
        let ratio = scan.as_secs_f64() / steady.as_secs_f64().max(1e-9);
        println!(
            "N={n:8} | matcher scan {scan:>10.2?} | index build+query {build:>10.2?} | steady query {steady:>10.2?} | steady {ratio:9.1}x"
        );
    }
}
