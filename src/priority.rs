// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! Priority ordering for evaluation control (Hyperon issue #448).
//!
//! Grabbed from F1R3FLY MeTTaTron's `backend/eval/priority.rs` and ported to Hyperon
//! `Atom`s -- the representation on the HE-MeTTa -> MeTTa-IL -> {MORK + rholang}
//! path. A planner uses this total order to decide which nondeterministic branch or
//! exec rule to fire/schedule first. Low to high:
//!
//!   grounded values  <  symbols  <  expressions  <  variables
//!
//! with Peano numerals (`Z`, `(S Z)`, ...) ordered by depth, grounded numbers
//! numerically when they parse as integers (else lexicographically by display), and
//! tuples compared lexicographically by their children.

use std::cmp::Ordering;

use hyperon_atom::Atom;

/// Rank of an atom's meta-type, used as the coarse key before structural comparison.
fn type_rank(a: &Atom) -> u8 {
    match a {
        Atom::Grounded(_) => 0,
        Atom::Symbol(_) => 1,
        Atom::Expression(_) => 2,
        Atom::Variable(_) => 3,
    }
}

/// Whether `a` is a Peano numeral: `Z`, or `(S p)` with `p` a numeral.
pub fn is_peano(a: &Atom) -> bool {
    match a {
        Atom::Symbol(s) => s.name() == "Z",
        Atom::Expression(e) => {
            let c = e.children();
            c.len() == 2 && matches!(&c[0], Atom::Symbol(s) if s.name() == "S") && is_peano(&c[1])
        }
        _ => false,
    }
}

/// Depth of a Peano numeral (`Z` -> 0, `(S Z)` -> 1, ...). Non-numerals return 0.
pub fn peano_depth(a: &Atom) -> u64 {
    match a {
        Atom::Symbol(s) if s.name() == "Z" => 0,
        Atom::Expression(e) => {
            let c = e.children();
            if c.len() == 2 && matches!(&c[0], Atom::Symbol(s) if s.name() == "S") {
                1 + peano_depth(&c[1])
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Total priority order over atoms (see module docs).
pub fn compare_priority(a: &Atom, b: &Atom) -> Ordering {
    match (a, b) {
        (Atom::Grounded(_), Atom::Grounded(_)) => {
            let (sa, sb) = (a.to_string(), b.to_string());
            match (sa.parse::<i64>(), sb.parse::<i64>()) {
                (Ok(na), Ok(nb)) => na.cmp(&nb),
                _ => sa.cmp(&sb),
            }
        }
        (Atom::Symbol(x), Atom::Symbol(y)) => x.name().cmp(y.name()),
        (Atom::Variable(x), Atom::Variable(y)) => x.name().cmp(&y.name()),
        (Atom::Expression(_), Atom::Expression(_)) => {
            match (is_peano(a), is_peano(b)) {
                (true, true) => peano_depth(a).cmp(&peano_depth(b)),
                (true, false) => Ordering::Less, // a numeral is "smaller" than a tuple
                (false, true) => Ordering::Greater,
                (false, false) => {
                    let (ca, cb) = (
                        a_children(a),
                        a_children(b),
                    );
                    compare_children(ca, cb)
                }
            }
        }
        // Mixed meta-types: order by type rank.
        _ => type_rank(a).cmp(&type_rank(b)),
    }
}

fn a_children(a: &Atom) -> &[Atom] {
    match a {
        Atom::Expression(e) => e.children(),
        _ => &[],
    }
}

/// Lexicographic comparison of two child slices under `compare_priority`.
fn compare_children(xs: &[Atom], ys: &[Atom]) -> Ordering {
    for (x, y) in xs.iter().zip(ys.iter()) {
        let o = compare_priority(x, y);
        if o != Ordering::Equal {
            return o;
        }
    }
    xs.len().cmp(&ys.len())
}

/// Sorts `(priority_atom, payload)` pairs ascending by priority. Stable, so equal
/// priorities keep input order.
pub fn sort_by_priority<T>(mut items: Vec<(Atom, T)>) -> Vec<(Atom, T)> {
    items.sort_by(|a, b| compare_priority(&a.0, &b.0));
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperon_atom::Atom;

    fn s(n: &str) -> Atom {
        Atom::sym(n)
    }
    /// `(S (S ... Z))` with `n` successors.
    fn peano(n: u64) -> Atom {
        let mut a = s("Z");
        for _ in 0..n {
            a = Atom::expr([s("S"), a]);
        }
        a
    }

    #[test]
    fn peano_orders_by_depth() {
        assert!(is_peano(&peano(3)));
        assert_eq!(peano_depth(&peano(3)), 3);
        assert_eq!(compare_priority(&peano(2), &peano(5)), Ordering::Less);
        assert_eq!(compare_priority(&peano(5), &peano(2)), Ordering::Greater);
        assert_eq!(compare_priority(&s("Z"), &peano(1)), Ordering::Less);
    }

    #[test]
    fn type_precedence_and_symbols() {
        // symbol < expression
        assert_eq!(compare_priority(&s("a"), &Atom::expr([s("a")])), Ordering::Less);
        // symbols lexicographic
        assert_eq!(compare_priority(&s("a"), &s("b")), Ordering::Less);
    }

    #[test]
    fn sort_by_priority_orders_branches() {
        let items = vec![(peano(3), "c"), (peano(1), "a"), (peano(2), "b")];
        let sorted = sort_by_priority(items);
        assert_eq!(
            sorted.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }
}
