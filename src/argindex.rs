// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 MesTTo
//! The argument-position (column) index: a permuted-key secondary trie per
//! (relation, argument position), so a single-factor query bound only on
//! non-leading positions becomes a prefix seek instead of a relation scan --
//! the shipped MeTTaLingo multi-argument clause index, ported from the
//! optimized MORK fork (its build_arg_index / arg_index_seek /
//! indexed_match decline rules) onto upstream PathMap traversal.
//!
//! The index stores each fact's argument columns re-emitted with the indexed
//! column first and variables renumbered canonically, so a fact's coreference
//! survives the permutation and the original fact bytes reconstruct exactly.
//! [`classify_single_factor`] admits the fork's fragment: ground symbol
//! functor, arity >= 2, every argument either one fresh variable or fully
//! ground; anything else (a repeated variable, a compound carrying a variable,
//! a variable functor, a fully-unbound query) declines to the general matcher.

use mork::__mork_expr::{byte_item, item_byte, maybe_byte_item, Tag};
use mork::zipper_join::{first_subterm_is_ground, first_subterm_len};
use pathmap::zipper::{ZipperIteration, ZipperMoving, ZipperValues};
use pathmap::PathMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArgClass {
    Free,
    Bound(Vec<u8>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Classified {
    pub functor_prefix: Vec<u8>,
    pub args: Vec<ArgClass>,
}

impl Classified {
    pub fn ncols(&self) -> usize {
        self.args.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Item {
    Byte(u8),
    Var(usize),
}

/// Length in bytes of the first complete subterm of `bytes` (re-exported for
/// offset arithmetic over whole encoded atoms).
pub fn fact_subterm_len(bytes: &[u8]) -> usize {
    first_subterm_len(bytes)
}

/// Split a fact's argument bytes into `ncols` encoded subterms.
pub fn split_columns(bytes: &[u8], ncols: usize) -> Vec<&[u8]> {
    let mut cols = Vec::with_capacity(ncols);
    let mut i = 0;
    for _ in 0..ncols {
        let len = first_subterm_len(&bytes[i..]);
        cols.push(&bytes[i..i + len]);
        i += len;
    }
    cols
}

fn columns_to_items(cols: &[&[u8]]) -> Vec<Vec<Item>> {
    let mut next_orig = 0usize;
    let mut out = Vec::with_capacity(cols.len());
    for col in cols {
        let mut items = Vec::new();
        let mut i = 0usize;
        while i < col.len() {
            let b = col[i];
            let tag = maybe_byte_item(b).unwrap_or_else(|reserved| {
                panic!("reserved mork expression tag byte {reserved}");
            });
            match tag {
                Tag::Arity(_) => {
                    items.push(Item::Byte(b));
                    i += 1;
                }
                Tag::VarRef(var) => {
                    items.push(Item::Var(var as usize));
                    i += 1;
                }
                Tag::NewVar => {
                    items.push(Item::Var(next_orig));
                    next_orig += 1;
                    i += 1;
                }
                Tag::SymbolSize(size) => {
                    let end = i + 1 + size as usize;
                    items.push(Item::Byte(b));
                    items.extend(col[i + 1..end].iter().copied().map(Item::Byte));
                    i = end;
                }
            }
        }
        out.push(items);
    }
    out
}

/// Re-emit columns in `new_order`, renumbering variables canonically.
fn emit_reordered(items_by_col: &[Vec<Item>], new_order: &[usize]) -> Vec<u8> {
    use std::collections::HashMap;

    let mut out = Vec::new();
    let mut renum: HashMap<usize, usize> = HashMap::new();
    for &c in new_order {
        for item in &items_by_col[c] {
            match item {
                Item::Byte(b) => out.push(*b),
                Item::Var(orig) => match renum.get(orig) {
                    Some(&new_id) => out.push(item_byte(Tag::VarRef(new_id as u8))),
                    None => {
                        renum.insert(*orig, renum.len());
                        out.push(item_byte(Tag::NewVar));
                    }
                },
            }
        }
    }
    out
}

/// The encoded bytes of every argument of an atom.
pub fn columns(atom: &[u8], plen: usize, ncols: usize) -> Option<Vec<Vec<u8>>> {
    if atom.len() < plen {
        return None;
    }
    let cols = split_columns(&atom[plen..], ncols);
    if cols.len() != ncols {
        return None;
    }
    Some(cols.into_iter().map(|c| c.to_vec()).collect())
}

fn index_order(ncols: usize, pos: usize) -> Vec<usize> {
    let mut new_order = Vec::with_capacity(ncols);
    new_order.push(pos);
    new_order.extend((0..ncols).filter(|&c| c != pos));
    new_order
}

/// The index key for one fact's argument bytes: its columns re-emitted with
/// `pos` first and variables renumbered canonically. `None` when the bytes do
/// not split into `ncols` columns. This is the unit of incremental index
/// maintenance: one fact added or removed from the relation is one key
/// inserted into or removed from each of its indexes.
pub fn permuted_fact_key(args: &[u8], ncols: usize, pos: usize) -> Option<Vec<u8>> {
    let cols = split_columns(args, ncols);
    if cols.len() != ncols {
        return None;
    }
    let items = columns_to_items(&cols);
    Some(emit_reordered(&items, &index_order(ncols, pos)))
}

/// Build an argument-position index for one relation column.
pub fn build_arg_index(
    map: &PathMap<()>,
    functor_prefix: &[u8],
    ncols: usize,
    pos: usize,
) -> PathMap<()> {
    debug_assert!(pos < ncols);

    let mut index = PathMap::<()>::new();
    let mut rz = map.read_zipper_at_path(functor_prefix);
    while rz.to_next_val() {
        if let Some(key) = permuted_fact_key(rz.path(), ncols, pos) {
            index.insert(key, ());
        }
    }
    index
}

/// Seek a column index and return original full fact bytes.
pub fn arg_index_seek(
    index: &PathMap<()>,
    functor_prefix: &[u8],
    ncols: usize,
    pos: usize,
    value: &[u8],
) -> Vec<Vec<u8>> {
    let new_order = index_order(ncols, pos);
    let mut inv = vec![0usize; ncols];
    for (i, &c) in new_order.iter().enumerate() {
        inv[c] = i;
    }

    let mut out = Vec::new();
    let mut rz = index.read_zipper_at_path(value);
    if rz.val().is_some() {
        let reordered_args = value.to_vec();
        push_original_fact(&mut out, functor_prefix, ncols, &inv, &reordered_args);
    }
    while rz.to_next_val() {
        let mut reordered_args = value.to_vec();
        reordered_args.extend_from_slice(rz.path());
        push_original_fact(&mut out, functor_prefix, ncols, &inv, &reordered_args);
    }
    out
}

fn push_original_fact(
    out: &mut Vec<Vec<u8>>,
    functor_prefix: &[u8],
    ncols: usize,
    inv: &[usize],
    reordered_args: &[u8],
) {
    let cols = split_columns(reordered_args, ncols);
    if cols.len() != ncols {
        return;
    }
    let items = columns_to_items(&cols);
    let mut fact = functor_prefix.to_vec();
    fact.extend_from_slice(&emit_reordered(&items, inv));
    out.push(fact);
}

/// Classify the single-factor shapes accepted by the fork's indexed match path.
pub fn classify_single_factor(pattern_bytes: &[u8]) -> Option<Classified> {
    let (&first, rest) = pattern_bytes.split_first()?;
    let Tag::Arity(k) = byte_item(first) else {
        return None;
    };
    let k = k as usize;
    if k < 2 {
        return None;
    }
    if !first_subterm_is_ground(rest) {
        return None;
    }

    let plen = 1 + first_subterm_len(rest);
    let ncols = k - 1;
    let cols = columns(pattern_bytes, plen, ncols)?;
    let newvar = item_byte(Tag::NewVar);
    let mut args = Vec::with_capacity(ncols);
    let mut has_bound = false;
    for col in cols {
        if col.len() == 1 && col[0] == newvar {
            args.push(ArgClass::Free);
        } else if first_subterm_is_ground(&col) {
            has_bound = true;
            args.push(ArgClass::Bound(col));
        } else {
            return None;
        }
    }
    has_bound.then(|| Classified {
        functor_prefix: pattern_bytes[..plen].to_vec(),
        args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn tag(tag: Tag) -> u8 {
        let byte = item_byte(tag);
        assert!(maybe_byte_item(byte).is_ok());
        byte
    }

    fn sym(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        assert!(bytes.len() < 64);
        let mut out = Vec::with_capacity(1 + bytes.len());
        out.push(tag(Tag::SymbolSize(bytes.len() as u8)));
        out.extend_from_slice(bytes);
        out
    }

    fn var_ref(i: u8) -> Vec<u8> {
        vec![tag(Tag::VarRef(i))]
    }

    fn new_var() -> Vec<u8> {
        vec![tag(Tag::NewVar)]
    }

    fn compound(functor: &str, args: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(tag(Tag::Arity((args.len() + 1) as u8)));
        out.extend_from_slice(&sym(functor));
        for arg in args {
            out.extend_from_slice(arg);
        }
        out
    }

    fn fact(functor: &str, args: &[Vec<u8>]) -> Vec<u8> {
        compound(functor, args)
    }

    fn relation_prefix(functor: &str, ncols: usize) -> Vec<u8> {
        let mut prefix = Vec::new();
        prefix.push(tag(Tag::Arity((ncols + 1) as u8)));
        prefix.extend_from_slice(&sym(functor));
        prefix
    }

    fn relation(facts: &[Vec<u8>]) -> PathMap<()> {
        let mut map = PathMap::<()>::new();
        for fact in facts {
            map.insert(fact, ());
        }
        map
    }

    fn scan_filter(
        map: &PathMap<()>,
        functor_prefix: &[u8],
        ncols: usize,
        bound: &[(usize, Vec<u8>)],
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut rz = map.read_zipper_at_path(functor_prefix);
        while rz.to_next_val() {
            let args = rz.path();
            let cols = split_columns(args, ncols);
            let matches = bound
                .iter()
                .all(|(pos, value)| cols.get(*pos).is_some_and(|col| *col == value.as_slice()));
            if matches {
                let mut fact = functor_prefix.to_vec();
                fact.extend_from_slice(args);
                out.push(fact);
            }
        }
        out.sort();
        out
    }

    fn indexed_filter(
        map: &PathMap<()>,
        functor_prefix: &[u8],
        ncols: usize,
        seek_pos: usize,
        seek_value: &[u8],
        bound: &[(usize, Vec<u8>)],
    ) -> Vec<Vec<u8>> {
        let index = build_arg_index(map, functor_prefix, ncols, seek_pos);
        let mut out: Vec<Vec<u8>> =
            arg_index_seek(&index, functor_prefix, ncols, seek_pos, seek_value)
                .into_iter()
                .filter(|fact| {
                    let cols = split_columns(&fact[functor_prefix.len()..], ncols);
                    bound.iter().all(|(pos, value)| {
                        cols.get(*pos).is_some_and(|col| *col == value.as_slice())
                    })
                })
                .collect();
        out.sort();
        out
    }

    fn assert_index_matches_scan(
        map: &PathMap<()>,
        functor_prefix: &[u8],
        ncols: usize,
        seek_pos: usize,
        seek_value: Vec<u8>,
        bound: &[(usize, Vec<u8>)],
    ) {
        let got = indexed_filter(map, functor_prefix, ncols, seek_pos, &seek_value, bound);
        let want = scan_filter(map, functor_prefix, ncols, bound);
        assert_eq!(got, want);
    }

    #[test]
    fn arg_index_seek_matches_scan_for_flat_relation() {
        let facts = vec![
            fact("edge", &[sym("a"), sym("b")]),
            fact("edge", &[sym("b"), sym("c")]),
            fact("edge", &[sym("a"), sym("c")]),
            fact("edge", &[sym("c"), sym("a")]),
            fact("edge", &[sym("b"), sym("a")]),
            fact("edge", &[sym("a"), sym("a")]),
        ];
        let map = relation(&facts);
        let prefix = relation_prefix("edge", 2);

        assert_index_matches_scan(&map, &prefix, 2, 0, sym("a"), &[(0, sym("a"))]);
        assert_index_matches_scan(&map, &prefix, 2, 1, sym("c"), &[(1, sym("c"))]);
        assert_index_matches_scan(
            &map,
            &prefix,
            2,
            1,
            sym("c"),
            &[(0, sym("a")), (1, sym("c"))],
        );
    }

    #[test]
    fn arg_index_seek_matches_scan_for_compound_argument_relation() {
        let px = compound("p", &[sym("x")]);
        let qy = compound("q", &[sym("y")]);
        let facts = vec![
            fact("t", &[sym("1"), px.clone(), sym("z")]),
            fact("t", &[sym("2"), px.clone(), sym("w")]),
            fact("t", &[sym("1"), qy, sym("z")]),
            fact("t", &[sym("3"), px.clone(), sym("z")]),
        ];
        let map = relation(&facts);
        let prefix = relation_prefix("t", 3);

        assert_index_matches_scan(&map, &prefix, 3, 0, sym("1"), &[(0, sym("1"))]);
        assert_index_matches_scan(&map, &prefix, 3, 1, px.clone(), &[(1, px.clone())]);
        assert_index_matches_scan(
            &map,
            &prefix,
            3,
            2,
            sym("z"),
            &[(0, sym("1")), (2, sym("z"))],
        );
        assert_index_matches_scan(&map, &prefix, 3, 1, px.clone(), &[(1, px), (2, sym("z"))]);
    }

    #[test]
    fn arg_index_seek_returns_every_distinct_column_value_as_byte_identical_facts() {
        let facts = vec![
            fact("edge", &[sym("a"), sym("b")]),
            fact("edge", &[sym("b"), sym("c")]),
            fact("edge", &[sym("a"), sym("c")]),
            fact("edge", &[sym("c"), sym("a")]),
        ];
        let map = relation(&facts);
        let prefix = relation_prefix("edge", 2);
        let all = scan_filter(&map, &prefix, 2, &[]);

        for pos in 0..2 {
            let index = build_arg_index(&map, &prefix, 2, pos);
            let values: BTreeSet<Vec<u8>> = all
                .iter()
                .map(|fact| split_columns(&fact[prefix.len()..], 2)[pos].to_vec())
                .collect();
            for value in values {
                let mut got = arg_index_seek(&index, &prefix, 2, pos, &value);
                got.sort();
                let mut want: Vec<Vec<u8>> = all
                    .iter()
                    .filter(|fact| split_columns(&fact[prefix.len()..], 2)[pos] == value.as_slice())
                    .cloned()
                    .collect();
                want.sort();
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn classify_single_factor_accepts_ground_and_free_arguments() {
        let pattern = fact("edge", &[sym("a"), new_var()]);

        let classified = classify_single_factor(&pattern).unwrap();

        assert_eq!(
            classified,
            Classified {
                functor_prefix: relation_prefix("edge", 2),
                args: vec![ArgClass::Bound(sym("a")), ArgClass::Free],
            }
        );
    }

    #[test]
    fn classify_single_factor_declines_var_ref_argument() {
        let pattern = fact("edge", &[new_var(), var_ref(0)]);

        assert!(classify_single_factor(&pattern).is_none());
    }

    #[test]
    fn classify_single_factor_declines_partially_bound_compound_argument() {
        let pattern = fact("edge", &[sym("a"), compound("p", &[new_var()])]);

        assert!(classify_single_factor(&pattern).is_none());
    }

    #[test]
    fn classify_single_factor_declines_fully_unbound_pattern() {
        let pattern = fact("edge", &[new_var(), new_var()]);

        assert!(classify_single_factor(&pattern).is_none());
    }

    #[test]
    fn classify_single_factor_declines_variable_functor() {
        let mut pattern = Vec::new();
        pattern.push(tag(Tag::Arity(3)));
        pattern.extend_from_slice(&new_var());
        pattern.extend_from_slice(&sym("a"));
        pattern.extend_from_slice(&sym("b"));

        assert!(classify_single_factor(&pattern).is_none());
    }
}
