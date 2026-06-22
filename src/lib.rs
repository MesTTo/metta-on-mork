//! MORK kernel as an in-process Hyperon atomspace backend (MeTTa-On-Mork).
//!
//! [`MorkSpace`] implements Hyperon's [`Space`]/[`SpaceMut`] over the optimized MORK
//! kernel (a PathMap trie plus the worst-case-optimal-join matcher), so a Hyperon
//! atomspace gains MORK's scale and speed. Motivated by hyperon-experimental #1076
//! (the GroundingSpace trie panics on the first query past ~2k atoms).
//!
//! The codec is **byte-level**: a Hyperon `Atom` is walked directly into MORK's
//! expression-zipper byte encoding (`Arity`/`SymbolSize`/`NewVar`/`VarRef`) and
//! inserted into the trie; a query encodes its pattern, calls `query_multi`, and
//! decodes each bound sub-expression's bytes straight back into an `Atom`. No text
//! round-trip, no per-query parsing. (Build uses MORK's default features, where
//! symbols are stored as raw bytes rather than interned tokens.)

use std::borrow::Cow;
use std::cell::RefCell;

use hyperon_atom::matcher::{Bindings, BindingsSet};
use hyperon_atom::{Atom, VariableAtom};
use hyperon_common::FlexRef;
use hyperon_space::{Space, SpaceCommon, SpaceMut, SpaceVisitor};

use mork::__mork_expr::{byte_item, item_byte, Expr, Tag};
use mork::space::Space as MorkKernel;

/// MORK's 6-bit `SymbolSize`/`Arity` fields cap symbol length and arity at 63.
const MAX_FIELD: usize = 63;

/// A Hyperon atomspace backed by the MORK kernel.
///
/// The kernel is behind a `RefCell` because `query` is `&self` while MORK matching
/// needs `&mut` access to the trie cursor; a query borrows mutably for its duration.
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

    /// Bulk-loads atoms from an S-expression source (whitespace-separated atoms)
    /// through MORK's parser. Convenient for large loads.
    pub fn add_sexpr_text(&mut self, text: &str) -> Result<usize, String> {
        self.kernel.get_mut().add_all_sexpr(text.as_bytes())
    }

    fn query_inner(&self, query: &Atom) -> BindingsSet {
        // Encode the pattern, tracking variables in introduction order so binding
        // key (0, i) maps to the i-th variable.
        let mut vars: Vec<VariableAtom> = Vec::new();
        let mut pat = Vec::new();
        if encode_atom(query, &mut vars, &mut pat).is_err() {
            return BindingsSet::empty();
        }
        // Wrap as the single-factor conjunction `(, <pattern>)` query_multi expects.
        let mut wrapped = Vec::with_capacity(pat.len() + 3);
        wrapped.push(item_byte(Tag::Arity(2)));
        wrapped.push(item_byte(Tag::SymbolSize(1)));
        wrapped.push(b',');
        wrapped.extend_from_slice(&pat);

        let kernel = self.kernel.borrow();
        let mut set = BindingsSet::empty();
        let pat_expr = Expr {
            ptr: wrapped.as_mut_ptr(),
        };
        MorkKernel::query_multi(&kernel.btm, pat_expr, |res, _loc| {
            if let Ok(_) = res {
                return true;
            }
            let Err(bindings) = res else { unreachable!() };
            let mut acc = Some(Bindings::new());
            for (i, var) in vars.iter().enumerate() {
                let Some(env) = bindings.get(&(0u8, i as u8)) else {
                    continue;
                };
                let span = unsafe { env.subsexpr().span().as_ref().unwrap() };
                let mut pos = 0usize;
                if let Some(atom) = decode_atom(span, &mut pos) {
                    acc = acc.and_then(|b| b.add_var_binding(var.clone(), atom).ok());
                }
            }
            if let Some(b) = acc {
                set.push(b);
            }
            true
        });
        set
    }
}

impl Default for MorkSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Walks a Hyperon `Atom` into MORK's preorder byte encoding, recording variables
/// in first-occurrence order (`NewVar` introduces, later occurrences `VarRef` back).
/// Errors when a symbol or arity exceeds MORK's 63 limit.
fn encode_atom(atom: &Atom, vars: &mut Vec<VariableAtom>, out: &mut Vec<u8>) -> Result<(), ()> {
    match atom {
        Atom::Symbol(s) => encode_symbol(s.name(), out),
        Atom::Grounded(g) => encode_symbol(&g.to_string(), out),
        Atom::Expression(e) => {
            let children = e.children();
            if children.len() > MAX_FIELD {
                return Err(());
            }
            out.push(item_byte(Tag::Arity(children.len() as u8)));
            for child in children {
                encode_atom(child, vars, out)?;
            }
            Ok(())
        }
        Atom::Variable(v) => {
            match vars.iter().position(|x| x == v) {
                Some(i) if i <= MAX_FIELD => out.push(item_byte(Tag::VarRef(i as u8))),
                Some(_) => return Err(()),
                None => {
                    out.push(item_byte(Tag::NewVar));
                    vars.push(v.clone());
                }
            }
            Ok(())
        }
    }
}

fn encode_symbol(name: &str, out: &mut Vec<u8>) -> Result<(), ()> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_FIELD {
        return Err(());
    }
    out.push(item_byte(Tag::SymbolSize(bytes.len() as u8)));
    out.extend_from_slice(bytes);
    Ok(())
}

/// Walks MORK's preorder byte encoding back into a Hyperon `Atom`. `pos` advances
/// past the consumed bytes. Returns `None` on a malformed/short buffer.
fn decode_atom(bytes: &[u8], pos: &mut usize) -> Option<Atom> {
    let tag = byte_item(*bytes.get(*pos)?);
    *pos += 1;
    match tag {
        Tag::SymbolSize(s) => {
            let s = s as usize;
            let end = pos.checked_add(s)?;
            let name = std::str::from_utf8(bytes.get(*pos..end)?).ok()?;
            *pos = end;
            Some(Atom::sym(name))
        }
        Tag::Arity(k) => {
            let mut children = Vec::with_capacity(k as usize);
            for _ in 0..k {
                children.push(decode_atom(bytes, pos)?);
            }
            Some(Atom::expr(children))
        }
        Tag::NewVar => Some(Atom::var("_")),
        Tag::VarRef(i) => Some(Atom::var(format!("_{}", i))),
    }
}

impl Space for MorkSpace {
    fn common(&self) -> FlexRef<'_, SpaceCommon> {
        FlexRef::from_simple(&self.common)
    }

    fn query(&self, query: &Atom) -> BindingsSet {
        self.query_inner(query)
    }

    fn atom_count(&self) -> Option<usize> {
        Some(self.len())
    }

    fn visit(&self, v: &mut dyn SpaceVisitor) -> Result<(), ()> {
        use pathmap::zipper::{ZipperIteration, ZipperMoving};
        let kernel = self.kernel.borrow();
        let mut rz = kernel.btm.read_zipper();
        while rz.to_next_val() {
            let atom_bytes = rz.path();
            let mut pos = 0;
            if let Some(atom) = decode_atom(atom_bytes, &mut pos) {
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
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        if encode_atom(&atom, &mut vars, &mut bytes).is_ok() {
            self.kernel.get_mut().btm.insert(&bytes, ());
        }
    }

    fn remove(&mut self, atom: &Atom) -> bool {
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        if encode_atom(atom, &mut vars, &mut bytes).is_err() {
            return false;
        }
        self.kernel.get_mut().btm.remove(&bytes).is_some()
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

    fn parent(a: &str, b: &str) -> Atom {
        Atom::expr([Atom::sym("parent"), Atom::sym(a), Atom::sym(b)])
    }

    #[test]
    fn add_and_count() {
        let mut space = MorkSpace::new();
        assert_eq!(space.atom_count(), Some(0));
        space.add(parent("Tom", "Bob"));
        space.add(parent("Bob", "Ann"));
        assert_eq!(space.atom_count(), Some(2));
    }

    #[test]
    fn round_trip_nested_atom() {
        let mut space = MorkSpace::new();
        let a = Atom::expr([
            Atom::sym("f"),
            Atom::expr([Atom::sym("g"), Atom::sym("a")]),
            Atom::sym("b"),
        ]);
        space.add(a.clone());
        let q = Atom::var("x");
        // (match $x) -- a bare variable matches every stored atom.
        let results = space.query(&q);
        let got: Vec<Atom> = results
            .iter()
            .filter_map(|b| b.resolve(&VariableAtom::new("x")))
            .collect();
        assert_eq!(got, vec![a]);
    }

    #[test]
    fn query_binds_one_variable() {
        let mut space = MorkSpace::new();
        space.add(parent("Tom", "Bob"));
        space.add(parent("Bob", "Ann"));
        let q = Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results.iter().next().unwrap().resolve(&VariableAtom::new("child")),
            Some(Atom::sym("Bob"))
        );
    }

    #[test]
    fn query_two_variables_multiple_results() {
        let mut space = MorkSpace::new();
        space.add(parent("Tom", "Bob"));
        space.add(parent("Bob", "Ann"));
        let q = Atom::expr([Atom::sym("parent"), Atom::var("p"), Atom::var("c")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn remove_atom() {
        let mut space = MorkSpace::new();
        space.add(parent("Tom", "Bob"));
        assert_eq!(space.len(), 1);
        assert!(space.remove(&parent("Tom", "Bob")));
        assert_eq!(space.len(), 0);
        assert!(!space.remove(&parent("Tom", "Bob")));
    }

    #[test]
    fn bulk_load_scales_past_the_trie_crash() {
        let mut space = MorkSpace::new();
        let mut text = String::new();
        for i in 0..20_000u32 {
            text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
        }
        space.add_sexpr_text(&text).unwrap();
        assert_eq!(space.atom_count(), Some(20_000));
        let q = Atom::expr([Atom::sym("edge"), Atom::sym("n10000"), Atom::var("dst")]);
        assert_eq!(space.query(&q).len(), 1);
    }
}
