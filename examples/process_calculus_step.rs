// SPDX-License-Identifier: MIT
//! The rule-rich fixpoint where semi-naive stepping earns its exponent: the
//! kernel's process_calculus dish (rho-style message passing computing Peano
//! x+y), where most naive match candidates are redundant re-derivation (PR
//! #128 counts 98.8% on this shape). Unlike a chain closure, the dish's
//! output stays small while the match work compounds -- so the naive side
//! grows superlinearly and the semi-naive side tracks the delta.
//!
//! Requires the `semi-naive` feature; the naive column is the same build with
//! the kernel's SNI_DISARM revert set for the run.

use std::time::{Duration, Instant};

use hyperon_atom::Atom;
use hyperon_space::Space;
use metta_on_mork::MorkSpace;
use mork::space::SNI_DISARM;

fn peano(n: usize) -> String {
    let mut out = String::with_capacity(n * 4 + 1);
    for _ in 0..n {
        out.push_str("(S ");
    }
    out.push('Z');
    for _ in 0..n {
        out.push(')');
    }
    out
}

fn peano_atom(n: usize) -> Atom {
    let mut a = Atom::sym("Z");
    for _ in 0..n {
        a = Atom::expr([Atom::sym("S"), a]);
    }
    a
}

fn program(steps: usize, x: usize, y: usize) -> String {
    format!(
        r#"
(exec (IC 0 1 {})
               (, (exec (IC $x $y (S $c)) $sp $st)
                  ((exec $x) $p $t))
               (, (exec (IC $y $x $c) $sp $st)
                  (exec (R $x) $p $t)))

((exec 0)
      (, (petri (? $channel $payload $body))
         (petri (! $channel $payload)) )
      (, (petri $body)))
((exec 1)
      (, (petri (| $lprocess $rprocess)))
      (, (petri $lprocess)
         (petri $rprocess)))

(petri (? (add $ret) ((S $x) $y) (| (! (add (PN $x $y)) ($x $y))
                                    (? (PN $x $y) $z (! $ret (S $z)))  )  ))
(petri (? (add $ret) (Z $y) (! $ret $y)))
(petri (! (add result) ({} {})))
    "#,
        peano(steps),
        peano(x),
        peano(y)
    )
}

fn run(steps: usize, x: usize, y: usize, disarm: bool) -> (Duration, Atom) {
    SNI_DISARM.with(|c| c.set(disarm));
    let mut space = MorkSpace::new();
    space
        .add_sexpr_text(&program(steps, x, y))
        .expect("program loads");
    let t = Instant::now();
    space.step(usize::MAX / 2);
    let elapsed = t.elapsed();
    SNI_DISARM.with(|c| c.set(false));

    // Exactly one (petri (! result <sum>)) must exist, bound to Peano x+y.
    let q = Atom::expr([
        Atom::sym("petri"),
        Atom::expr([Atom::sym("!"), Atom::sym("result"), Atom::var("z")]),
    ]);
    let results = space.query(&q);
    assert_eq!(results.len(), 1, "expected one result in the dish");
    let z = results
        .iter()
        .next()
        .unwrap()
        .resolve(&hyperon_atom::VariableAtom::new("z"))
        .expect("result bound");
    (elapsed, z)
}

fn main() {
    println!("process_calculus dish: Peano x+y by message passing; naive vs semi-naive step()");
    for (steps, x, y) in [(100usize, 20usize, 20usize), (400, 80, 80), (1000, 200, 200)] {
        let (naive_time, naive_sum) = run(steps, x, y, true);
        let (sni_time, sni_sum) = run(steps, x, y, false);
        assert_eq!(naive_sum, sni_sum, "modes disagree");
        assert_eq!(naive_sum, peano_atom(x + y), "wrong sum");
        let ratio = naive_time.as_secs_f64() / sni_time.as_secs_f64().max(1e-9);
        println!(
            "{x:3}+{y:3} ({steps:4} sched steps) | naive {naive_time:>10.2?} | semi-naive {sni_time:>10.2?} | {ratio:7.1}x"
        );
    }
}
