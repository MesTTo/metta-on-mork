// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! Runs an MM2 program file on `MorkSpace`, the way `mork run <file>` does on the
//! kernel binary, but purely through this crate's Space API: `add_sexpr_text` to
//! load, `step` to run the exec scheduler to fixpoint, `count_matches` to inspect
//! the result.
//!
//! Usage:
//!   cargo run --release --example run_mm2 -- <file.mm2> [--steps N] [--naive] [--count "<pattern>"]...
//!
//! `--count` takes an s-expression pattern (`$name` is a variable) and reports how
//! many atoms in the final space match it. `--naive` disarms the semi-naive
//! frontier for the run (builds with `--features semi-naive` only), giving an
//! in-process A/B against the kernel's default full re-derivation.

use std::time::Instant;

use hyperon_atom::Atom;
use metta_on_mork::MorkSpace;

/// Parses one s-expression into an `Atom`: `(...)` nests, `$name` is a variable,
/// any other token is a symbol. This is the same surface MORK's own text parser
/// accepts for ground/pattern data, minus string literals, which none of the MM2
/// benchmark programs use in patterns.
fn parse_sexpr(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<Atom, String> {
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
    match chars.peek() {
        None => Err("unexpected end of pattern".into()),
        Some('(') => {
            chars.next();
            let mut items = Vec::new();
            loop {
                while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
                    chars.next();
                }
                match chars.peek() {
                    None => return Err("unclosed '('".into()),
                    Some(')') => {
                        chars.next();
                        return Ok(Atom::expr(items));
                    }
                    Some(_) => items.push(parse_sexpr(chars)?),
                }
            }
        }
        Some(')') => Err("unexpected ')'".into()),
        Some(_) => {
            let mut tok = String::new();
            while matches!(chars.peek(), Some(c) if !c.is_whitespace() && *c != '(' && *c != ')') {
                tok.push(chars.next().unwrap());
            }
            match tok.strip_prefix('$') {
                Some(name) => Ok(Atom::var(name)),
                None => Ok(Atom::sym(tok)),
            }
        }
    }
}

fn parse_pattern(text: &str) -> Result<Atom, String> {
    let mut chars = text.chars().peekable();
    let atom = parse_sexpr(&mut chars)?;
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
    match chars.peek() {
        None => Ok(atom),
        Some(_) => Err(format!("trailing input after pattern in {text:?}")),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut file = None;
    let mut steps = usize::MAX / 2;
    let mut naive = false;
    let mut counts: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                steps = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--steps takes a number");
            }
            "--count" => counts.push(args.next().expect("--count takes a pattern")),
            "--naive" => naive = true,
            _ => {
                assert!(file.is_none(), "unexpected extra argument {arg:?}");
                file = Some(arg);
            }
        }
    }
    let file = file.expect(
        "usage: run_mm2 <file.mm2> [--steps N] [--naive] [--count \"<pattern>\"]...",
    );
    let patterns: Vec<Atom> = counts
        .iter()
        .map(|c| parse_pattern(c).expect("count pattern parses"))
        .collect();

    #[cfg(feature = "semi-naive")]
    mork::space::SNI_DISARM.with(|c| c.set(naive));
    #[cfg(not(feature = "semi-naive"))]
    assert!(
        !naive,
        "--naive contrasts against the semi-naive frontier; build with --features semi-naive"
    );

    let text = std::fs::read_to_string(&file).expect("program file reads");
    let mut space = MorkSpace::new();
    let t = Instant::now();
    space.add_sexpr_text(&text).expect("program loads");
    let load = t.elapsed();
    println!(
        "loaded {} ({} atoms, {} rejected) in {load:.2?}",
        file,
        space.len(),
        space.rejected_atom_count()
    );

    #[cfg(feature = "guarded-emit")]
    mork::reset_guarded_emit_stats();
    let t = Instant::now();
    let performed = space.step(steps);
    let run = t.elapsed();
    println!(
        "ran {performed} exec steps in {run:.2?} ({}); {} atoms in the final space",
        if naive { "naive" } else { "default frontier" },
        space.len()
    );
    #[cfg(feature = "guarded-emit")]
    {
        let stats = mork::guarded_emit_stats();
        println!(
            "guarded_emit consulted {} candidates; dropped {}",
            stats.consulted, stats.dropped
        );
    }

    if !patterns.is_empty() {
        let snapshot = space.snapshot();
        for (text, pattern) in counts.iter().zip(&patterns) {
            let t = Instant::now();
            let n = snapshot.count_matches(pattern);
            println!("{n} matches of {text} (counted in {:.2?})", t.elapsed());
        }
    }
}
