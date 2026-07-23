// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 MesTTo
//! Parallel concurrent queries on a MORK-backed space. MORK's PathMap is Send+Sync,
//! so one shared space serves many querying threads at once -- the parallelism that
//! Hyperon's Rc<RefCell> DynSpace (issue #410) cannot express.
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::{MorkSnapshot, MorkSpace};

fn query_batch(space: &MorkSnapshot, base: u32, n: u32) -> usize {
    let mut total = 0;
    for k in 0..n {
        let q = Atom::expr([
            Atom::sym("edge"),
            Atom::sym(format!("n{}", (base + k) % 450_000)),
            Atom::var("dst"),
        ]);
        total += space.query(&q).len();
    }
    total
}

fn main() {
    let mut space = MorkSpace::new();
    let mut text = String::with_capacity(500_000 * 16);
    for i in 0..500_000u32 {
        text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
    }
    space.add_sexpr_text(&text).unwrap();
    let space = Arc::new(space.snapshot());

    let per_thread = 20_000u32;
    for threads in [1usize, 2, 4, 8] {
        let t = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let space = Arc::clone(&space);
                thread::spawn(move || query_batch(&space, tid as u32 * per_thread, per_thread))
            })
            .collect();
        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let elapsed = t.elapsed();
        let qcount = threads as u32 * per_thread;
        println!(
            "{} thread(s): {} queries in {:>8.2?}  ({:>6.0} ns/query, {} results)",
            threads,
            qcount,
            elapsed,
            elapsed.as_nanos() as f64 / qcount as f64,
            total
        );
    }
}
