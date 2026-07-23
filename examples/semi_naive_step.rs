// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! Semi-naive MM2 fixpoint over a chain transitive closure.
//!
//! The control rule is the known IC scheduler shape from the MORK kernel tests:
//! it turns one dormant `((exec 0) ...)` rule into repeated ordinary `exec`
//! firings inside a single `MorkSpace::step(...)` call. That keeps the kernel's
//! semi-naive frontier armed across repeated firings of the same closure rule.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use metta_on_mork::MorkSpace;
use mork::space::SNI_DISARM;

// 800 is the ceiling for a runnable example: the NAIVE side is the slow one
// (105s here), and it only gets worse with N -- which is the point.
const CHAIN_LENGTHS: [usize; 3] = [200, 400, 800];

struct SniDisarmGuard;

impl SniDisarmGuard {
    fn set(disarmed: bool) -> Self {
        SNI_DISARM.with(|cell| cell.set(disarmed));
        Self
    }
}

impl Drop for SniDisarmGuard {
    fn drop(&mut self) {
        SNI_DISARM.with(|cell| cell.set(false));
    }
}

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

fn program_text(n: usize) -> String {
    let steps = peano(n.saturating_sub(1));
    let mut text = String::with_capacity(n * 40 + steps.len() + 512);

    for i in 0..n {
        writeln!(&mut text, "(edge n{i} n{})", i + 1).expect("writing to String cannot fail");
    }

    writeln!(
        &mut text,
        r#"
(exec (A)
  (, (edge $x $y))
  (, (path $x $y)))

(exec (IC 0 0 {steps})
  (, (exec (IC $x $y (S $c)) $sp $st)
     ((exec $x) $p $t))
  (, (exec (IC $y $x $c) $sp $st)
     (exec (R $x) $p $t)))

((exec 0)
  (, (edge $x $y) (path $y $z))
  (, (path $x $z)))
"#
    )
    .expect("writing to String cannot fail");

    text
}

fn build_space(text: &str) -> Result<MorkSpace, String> {
    let mut space = MorkSpace::new();
    space.add_sexpr_text(text)?;
    Ok(space)
}

fn run_to_fixpoint(space: &mut MorkSpace, step_limit: usize, disarm: bool) -> (Duration, usize) {
    let _guard = SniDisarmGuard::set(disarm);
    let start = Instant::now();
    let steps = space.step(step_limit);
    let elapsed = start.elapsed();
    assert!(
        steps < step_limit,
        "step limit {step_limit} reached before fixpoint"
    );
    (elapsed, space.len())
}

fn expected_atom_count(n: usize) -> usize {
    let path_closure = n * (n + 1) / 2;
    n + path_closure + 1
}

fn main() -> Result<(), String> {
    println!("N | naive | semi-naive | ratio");
    println!("--:|--:|--:|--:");

    for n in CHAIN_LENGTHS {
        let text = program_text(n);
        let step_limit = n * 2 + 16;

        let mut naive = build_space(&text)?;
        let mut semi_naive = build_space(&text)?;

        let (naive_time, naive_count) = run_to_fixpoint(&mut naive, step_limit, true);
        let (semi_time, semi_count) = run_to_fixpoint(&mut semi_naive, step_limit, false);

        assert_eq!(naive_count, semi_count, "atom counts diverged for N={n}");
        assert_eq!(
            naive_count,
            expected_atom_count(n),
            "transitive closure did not reach the expected fixpoint for N={n}"
        );

        let ratio = naive_time.as_secs_f64() / semi_time.as_secs_f64().max(1e-9);
        println!("{n:4} | {naive_time:>10.2?} | {semi_time:>10.2?} | {ratio:8.1}x");
    }

    Ok(())
}
