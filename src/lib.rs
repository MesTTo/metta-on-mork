//! MORK kernel as an in-process Hyperon atomspace backend.
//!
//! [`MorkSpace`] implements Hyperon's [`Space`]/[`SpaceMut`] over the optimized MORK
//! kernel (a PathMap trie plus the worst-case-optimal-join matcher), so a Hyperon
//! atomspace gains MORK's scale and speed. The motivating case is
//! hyperon-experimental issue #1076: the default `GroundingSpace` trie panics after
//! ~2k atoms on the first query; MORK handles far more without crashing.
//!
//! v1 is text-mediated: atoms cross the boundary as S-expressions through MORK's
//! own tested parser/serializer (`add_all_sexpr`, `parse_sexpr`, `dump_sexpr`). A
//! direct byte-level codec is a later optimization.

use std::borrow::Cow;
use std::cell::RefCell;

use hyperon_atom::matcher::{Bindings, BindingsSet};
use hyperon_atom::{Atom, VariableAtom};
use hyperon_common::FlexRef;
use hyperon_space::{Space, SpaceCommon, SpaceMut, SpaceVisitor};

use mork::space::Space as MorkKernel;
use mork::__mork_expr::ExprEnv;

/// A Hyperon atomspace backed by the MORK kernel.
///
/// The kernel sits behind a `RefCell` because Hyperon's `Space::query` is `&self`
/// while MORK's parser interns symbols through a `&mut` entry point; v1 borrows
/// mutably for the duration of a query. (A `&self` interning path keeps the type
/// `Sync` and is a planned refinement.)
pub struct MorkSpace {
    kernel: RefCell<MorkKernel>,
    common: SpaceCommon,
}

impl MorkSpace {
    /// Creates an empty space.
    pub fn new() -> Self {
        Self {
            kernel: RefCell::new(MorkKernel::new()),
            common: SpaceCommon::default(),
        }
    }

    /// Number of atoms currently stored (MORK `PathMap::val_count`).
    pub fn len(&self) -> usize {
        self.kernel.borrow().btm.val_count()
    }

    /// Whether the space holds no atoms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bulk-loads atoms from an S-expression source (whitespace-separated atoms),
    /// bypassing the per-atom `Atom` round-trip.
    pub fn add_sexpr_text(&mut self, text: &str) -> Result<usize, String> {
        self.kernel.get_mut().add_all_sexpr(text.as_bytes())
    }

    /// The query core: encode `pattern` together with a variable-tuple template in
    /// one parse (so the template references the pattern's variables), run MORK's
    /// `dump_sexpr`, and decode each substituted tuple back into a `Bindings`.
    fn run_query(&self, query: &Atom) -> BindingsSet {
        let mut vars: Vec<VariableAtom> = Vec::new();
        collect_vars(query, &mut vars);

        let var_str = vars
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        // `_p`/`_t` are throwaway wrapper heads; only children matter on decode.
        let combined = format!("(_p {} (_t {}))", query, var_str);

        let mut kernel = self.kernel.borrow_mut();
        let mut buf = vec![0u8; combined.len() * 8 + 256];
        let combined_expr = match kernel.parse_sexpr(combined.as_bytes(), buf.as_mut_ptr()) {
            Ok((e, _len)) => e,
            Err(_) => return BindingsSet::empty(),
        };

        let mut parts = Vec::new();
        ExprEnv::new(0, combined_expr).args(&mut parts);
        // parts = [_p, pattern, template]
        if parts.len() < 3 {
            return BindingsSet::empty();
        }
        let pattern = parts[1].subsexpr();
        let template = parts[2].subsexpr();

        let mut out = Vec::new();
        kernel.dump_sexpr(pattern, template, &mut out);
        drop(kernel);

        decode_result_tuples(&out, &vars)
    }
}

impl Default for MorkSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Collects the distinct variables of `atom` in first-occurrence order.
fn collect_vars(atom: &Atom, out: &mut Vec<VariableAtom>) {
    match atom {
        Atom::Variable(v) => {
            if !out.contains(v) {
                out.push(v.clone());
            }
        }
        Atom::Expression(e) => {
            for child in e.children() {
                collect_vars(child, out);
            }
        }
        _ => {}
    }
}

/// Decodes MORK's serialized output (one `(_t v1 v2 ...)` tuple per match) into a
/// `BindingsSet`, pairing each tuple position with the query variable in `vars`.
fn decode_result_tuples(out: &[u8], vars: &[VariableAtom]) -> BindingsSet {
    let text = String::from_utf8_lossy(out);
    let mut set = BindingsSet::empty();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(tuple) = parse_sexpr_atom(line) else {
            continue;
        };
        let values: Vec<Atom> = match tuple {
            Atom::Expression(e) => e.children().iter().skip(1).cloned().collect(),
            _ => Vec::new(),
        };
        let mut acc = Some(Bindings::new());
        for (var, value) in vars.iter().zip(values.iter()) {
            acc = acc.and_then(|b| b.add_var_binding(var.clone(), value.clone()).ok());
        }
        if let Some(bindings) = acc {
            set.push(bindings);
        }
    }
    set
}

/// Minimal S-expression -> Atom decoder for MORK's symbolic output (symbols,
/// `$variables`, and nested compounds). Does not handle quoted strings/escapes,
/// which the v1 symbolic showcase does not produce.
fn parse_sexpr_atom(s: &str) -> Option<Atom> {
    let toks = tokenize(s);
    let mut pos = 0;
    let atom = parse_tokens(&toks, &mut pos)?;
    Some(atom)
}

enum Tok {
    Open,
    Close,
    Word(String),
}

fn tokenize(s: &str) -> Vec<Tok> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, toks: &mut Vec<Tok>| {
        if !cur.is_empty() {
            toks.push(Tok::Word(std::mem::take(cur)));
        }
    };
    for c in s.chars() {
        match c {
            '(' => {
                flush(&mut cur, &mut toks);
                toks.push(Tok::Open);
            }
            ')' => {
                flush(&mut cur, &mut toks);
                toks.push(Tok::Close);
            }
            c if c.is_whitespace() => flush(&mut cur, &mut toks),
            _ => cur.push(c),
        }
    }
    flush(&mut cur, &mut toks);
    toks
}

fn parse_tokens(toks: &[Tok], pos: &mut usize) -> Option<Atom> {
    match toks.get(*pos)? {
        Tok::Open => {
            *pos += 1;
            let mut children = Vec::new();
            loop {
                match toks.get(*pos)? {
                    Tok::Close => {
                        *pos += 1;
                        break;
                    }
                    _ => children.push(parse_tokens(toks, pos)?),
                }
            }
            Some(Atom::expr(children))
        }
        Tok::Close => None,
        Tok::Word(w) => {
            *pos += 1;
            Some(match w.strip_prefix('$') {
                Some(name) => Atom::var(name),
                None => Atom::sym(w.as_str()),
            })
        }
    }
}

impl Space for MorkSpace {
    fn common(&self) -> FlexRef<'_, SpaceCommon> {
        FlexRef::from_simple(&self.common)
    }

    fn query(&self, query: &Atom) -> BindingsSet {
        self.run_query(query)
    }

    fn atom_count(&self) -> Option<usize> {
        Some(self.len())
    }

    fn visit(&self, v: &mut dyn SpaceVisitor) -> Result<(), ()> {
        let mut out = Vec::new();
        if self.kernel.borrow().dump_all_sexpr(&mut out).is_err() {
            return Err(());
        }
        let text = String::from_utf8_lossy(&out);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(atom) = parse_sexpr_atom(line) {
                v.accept(Cow::Owned(atom));
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl SpaceMut for MorkSpace {
    fn add(&mut self, atom: Atom) {
        let _ = self.kernel.get_mut().add_all_sexpr(atom.to_string().as_bytes());
    }

    fn remove(&mut self, atom: &Atom) -> bool {
        match self
            .kernel
            .get_mut()
            .remove_all_sexpr(atom.to_string().as_bytes())
        {
            Ok(n) => n > 0,
            Err(_) => false,
        }
    }

    fn replace(&mut self, from: &Atom, to: Atom) -> bool {
        if self.remove(from) {
            self.add(to);
            true
        } else {
            false
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for MorkSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MorkSpace({} atoms)", self.len())
    }
}

impl std::fmt::Display for MorkSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MorkSpace({} atoms)", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperon_atom::Atom;

    #[test]
    fn add_and_count() {
        let mut space = MorkSpace::new();
        assert_eq!(space.atom_count(), Some(0));
        space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Bob")]));
        space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Bob"), Atom::sym("Ann")]));
        assert_eq!(space.atom_count(), Some(2));
    }

    #[test]
    fn query_binds_one_variable() {
        let mut space = MorkSpace::new();
        space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Bob")]));
        space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Bob"), Atom::sym("Ann")]));

        // (parent Tom $child)  ->  $child = Bob
        let q = Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 1);
        let bound = results
            .iter()
            .next()
            .unwrap()
            .resolve(&VariableAtom::new("child"));
        assert_eq!(bound, Some(Atom::sym("Bob")));
    }

    #[test]
    fn query_two_variables_multiple_results() {
        let mut space = MorkSpace::new();
        space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Bob")]));
        space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Bob"), Atom::sym("Ann")]));

        let q = Atom::expr([Atom::sym("parent"), Atom::var("p"), Atom::var("c")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn bulk_load_scales_past_the_trie_crash() {
        // hyperon #1076: GroundingSpace trie panics ~2k atoms on the first query.
        let mut space = MorkSpace::new();
        let mut text = String::new();
        for i in 0..20_000u32 {
            text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
        }
        space.add_sexpr_text(&text).unwrap();
        assert_eq!(space.atom_count(), Some(20_000));

        // The first query after a large load is exactly what crashes the trie.
        let q = Atom::expr([Atom::sym("edge"), Atom::sym("n10000"), Atom::var("dst")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 1);
    }
}
