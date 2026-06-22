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
use std::collections::HashMap;

use hyperon_atom::matcher::{Bindings, BindingsSet};
use hyperon_atom::{next_variable_id, Atom, VariableAtom};
use hyperon_common::FlexRef;
use hyperon_space::{Space, SpaceCommon, SpaceMut, SpaceVisitor};

use mork::__mork_expr::{byte_item, item_byte, Expr, Tag};
use mork::space::Space as MorkKernel;
use pathmap::PathMap;

/// Priority ordering for evaluation control (Hyperon #448), grabbed from MeTTaTron.
pub mod priority;

/// MORK's 6-bit `SymbolSize`/`Arity` fields cap symbol length and arity at 63.
const MAX_FIELD: usize = 63;

/// Reserved first byte of a grounded atom's symbol encoding. No real MeTTa symbol
/// starts with NUL, so it cleanly separates a grounded `Atom` from a plain symbol in
/// the trie. The bytes after it are the grounded atom's display string, the key into
/// [`GroundedRegistry`].
const GROUNDED_MARK: u8 = 0x00;

/// Side table that makes grounded atoms round-trip losslessly through the byte trie.
///
/// MORK's encoding has no grounded type, so a Hyperon `Grounded` atom (a number, a
/// bool, an arithmetic op like `<=`) is stored as a marker symbol carrying its
/// display string, and the original `Atom` is kept here keyed by that string.
/// `decode` rebuilds the exact grounded atom by cloning the canonical instance.
/// Filled on `add`, read on `decode`; encoding is deterministic from the display
/// string and needs no lookup, so query patterns encode to the same bytes as stored
/// data without touching the table.
///
/// Identity is by display string: two grounded atoms that print the same are treated
/// as equal (true for `Number`/`Bool`/`String`/op symbols). Snapshots and sharded
/// shards carry no registry, so they decode grounded atoms as bare symbols.
#[derive(Default, Clone)]
struct GroundedRegistry {
    by_display: HashMap<String, Atom>,
}

impl GroundedRegistry {
    /// Records every grounded atom in `atom` (recursing into expressions) so it can
    /// be reconstructed on decode. First instance per display string wins.
    fn register(&mut self, atom: &Atom) {
        match atom {
            Atom::Grounded(g) => {
                self.by_display.entry(g.to_string()).or_insert_with(|| atom.clone());
            }
            Atom::Expression(e) => {
                for c in e.children() {
                    self.register(c);
                }
            }
            _ => {}
        }
    }

    fn get(&self, display: &str) -> Option<Atom> {
        self.by_display.get(display).cloned()
    }
}

/// Decode-time state: the namespace these bytes belong to (so variables are named by
/// `(namespace, index)` and a data variable can't collide with a pattern variable), a
/// running counter for first-occurrence variables, the optional grounded registry, and
/// the query's own variables. Namespace 0 is the query pattern, so a namespace-0 variable
/// decodes back to the caller's actual `VariableAtom` (preserving `$n` through the byte
/// round-trip) rather than a fresh name; without this, evaluating `(Add $n (S Z))` returns
/// an alpha-equivalent `(S $v0_0)` that the script's `assertEqual` against `(S $n)` rejects.
struct DecodeCtx<'a> {
    ns: u8,
    var_counter: usize,
    grounded: Option<&'a GroundedRegistry>,
    query_vars: &'a [VariableAtom],
    /// Globally unique id stamped on every data variable decoded for one query result,
    /// so two results (or two separate query calls, e.g. recursion levels) never produce
    /// the same `VariableAtom` and alias when the interpreter threads their bindings.
    result_id: usize,
}

/// A Hyperon atomspace backed by the MORK kernel.
///
/// The byte-level matcher (`query_multi`) takes the trie by shared reference, so
/// `query`/`visit` read the kernel through `&self` and mutation goes through
/// `&mut self`. No interior `RefCell` is needed (DynSpace already provides one),
/// which keeps `MorkSpace` itself `Sync` and removes a borrow per operation.
pub struct MorkSpace {
    kernel: MorkKernel,
    common: SpaceCommon,
    grounded: GroundedRegistry,
}

impl MorkSpace {
    /// Creates an empty space.
    pub fn new() -> Self {
        Self {
            kernel: MorkKernel::new(),
            common: SpaceCommon::default(),
            grounded: GroundedRegistry::default(),
        }
    }

    /// Number of atoms currently stored (MORK `PathMap::val_count`).
    pub fn len(&self) -> usize {
        self.kernel.btm.val_count()
    }

    /// Whether the space holds no atoms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bulk-loads atoms from an S-expression source (whitespace-separated atoms)
    /// through MORK's parser. Convenient for large loads.
    pub fn add_sexpr_text(&mut self, text: &str) -> Result<usize, String> {
        self.kernel.add_all_sexpr(text.as_bytes())
    }

    /// Runs MORK's MM2 exec engine -- the forward-chaining `(exec <loc> (, <src>)
    /// (, <tpl>))` rules in the space -- for up to `steps` exec steps (each runs a
    /// rule to fixpoint), returning the steps taken. This is the *computation*
    /// engine (CeTTa's `mork:step!`): the optimized kernel exec path that the six
    /// benchmarks accelerate. Load facts and `exec` rules with `add`/`add_sexpr_text`
    /// first, then `step`, then `query` the results.
    pub fn step(&mut self, steps: usize) -> usize {
        self.kernel.metta_calculus(steps)
    }

    fn query_inner(&self, query: &Atom) -> BindingsSet {
        query_btm(&self.kernel.btm, query, Some(&self.grounded))
    }

    /// A `Send + Sync` read-only snapshot for data-parallel querying: a cheap
    /// copy-on-write clone of the trie that many threads can query concurrently.
    /// MORK's `PathMap` is `Send + Sync`, so this is the parallel querying that
    /// Hyperon's `Rc<RefCell>` spaces (issue #410) cannot express.
    pub fn snapshot(&self) -> MorkSnapshot {
        MorkSnapshot {
            btm: self.kernel.btm.clone(),
        }
    }
}

/// Adds a binding the way Hyperon's matcher does (matcher.rs `Bindings::from`): a
/// variable bound to another variable is a variable *equality* (so equivalence classes
/// merge and otherwise-equal results collapse instead of multiplying), and anything else
/// is a value binding. Returns `None` if the addition splits or conflicts.
fn bind_or_equate(b: Bindings, var: VariableAtom, atom: Atom) -> Option<Bindings> {
    match atom {
        Atom::Variable(v) => b.add_var_equality(&var, &v).ok(),
        _ => b.add_var_binding(var, atom).ok(),
    }
}

/// Decodes the variable at `index` in the current namespace. Namespace 0 is the query,
/// so its variables are the caller's own (`query_vars[index]`), preserving `$n` through
/// the round-trip; other namespaces (matched data atoms / factors) get `v{ns}_{index}`.
fn decode_var(ctx: &DecodeCtx, index: usize) -> Atom {
    if ctx.ns == 0 {
        if let Some(v) = ctx.query_vars.get(index) {
            return Atom::Variable(v.clone());
        }
    }
    Atom::Variable(VariableAtom::new_id(format!("v{}_{}", ctx.ns, index), ctx.result_id))
}

/// The byte-level query against a bare trie, shared by `MorkSpace` and the
/// `Send + Sync` `MorkSnapshot`.
fn query_btm(btm: &PathMap<()>, query: &Atom, grounded: Option<&GroundedRegistry>) -> BindingsSet {
    let Some((mut wrapped, vars)) = wrap_pattern(query) else {
        return BindingsSet::empty();
    };
    // Decode must recover grounded atoms that appear in the query itself (the `4` in
    // `(= (sqr 4) $X)`), not only previously-stored ones, so a data-side binding like
    // $x<-4 round-trips as the grounded Number rather than a bare symbol "4" that `*`
    // cannot reduce. Merge the query's grounded atoms with the space registry here.
    let mut reg = grounded.cloned().unwrap_or_default();
    reg.register(query);
    let mut set = BindingsSet::empty();
    let pat_expr = Expr {
        ptr: wrapped.as_mut_ptr(),
    };
    MorkKernel::query_multi(btm, pat_expr, |res, _loc| {
        let Err(bindings) = res else { return true };
        let mut acc = Some(Bindings::new());
        // One fresh id for this whole result: every data variable decoded below shares it,
        // so the result is internally coreferent by name yet globally distinct from every
        // other result and query call. Without this, a data var `v1_0` from one match
        // aliased a `v1_0` from a deeper recursion's match, and the interpreter's threaded
        // bindings collapsed the branch to empty (b2 backchaining: mortal<-human<-And).
        let result_id = next_variable_id();
        // Iterate every binding the unifier produced. The key (ns, idx) names a variable
        // by its namespace -- 0 is the query pattern, >=1 a matched data atom / factor --
        // and its index; the bound value's own variables live in the value's namespace
        // `env.n`. Decode names a variable `v{ns}_{idx}`, so a data variable and a pattern
        // variable can never collide. (Without this, evaluating (Add (S $n) Z) bound the
        // data variable $x to the compound (S $n); both decoded to `v0`, yielding the
        // spurious occurs-failing binding `v0 <- (S v0)` that dropped the whole result and
        // stalled b1_equal_chain.) Query variables (ns 0) bind their real names so the
        // result is in terms the caller asked for; data/factor variables bind `v{ns}_{idx}`
        // for Hyperon's transitive `resolve` to chase.
        for (&(key_ns, key_idx), env) in bindings.iter() {
            let span = unsafe { env.subsexpr().span().as_ref().unwrap() };
            let mut pos = 0usize;
            // Seed the NewVar counter with `env.v`: the count of NewVars preceding this
            // span in its base atom (maintained by ExprEnv::args). A captured sub-span is
            // carved from the middle of an atom, so its `VarRef(i)` bytes index variables
            // in the atom's *global* scope, and a `NewVar` inside it is the (env.v)-th
            // variable of that scope, not the 0th. Decoding from 0 gave a binder var a
            // local index that collided with an outer VarRef (e.g. (part-appl $f $x) ->
            // (lambda $y ($f $x $y)) decoded the binder $y as v1_0 == $f, so $y absorbed
            // $f's value `+`). Seeding from env.v aligns NewVar names with VarRef names,
            // so a variable's binder and body occurrences share one name (consistent VBTO).
            let mut ctx = DecodeCtx {
                var_counter: env.v as usize,
                grounded: Some(&reg),
                ns: env.n,
                query_vars: &vars,
                result_id,
            };
            let Some(atom) = decode_atom(span, &mut pos, &mut ctx) else {
                continue;
            };
            let var = if key_ns == 0 {
                match vars.get(key_idx as usize) {
                    Some(v) => v.clone(),
                    None => continue,
                }
            } else {
                VariableAtom::new_id(format!("v{}_{}", key_ns, key_idx), result_id)
            };
            acc = acc.and_then(|b| bind_or_equate(b, var, atom));
        }
        if let Some(b) = acc {
            set.push(b);
        }
        true
    });
    set
}

/// A `Send + Sync` read-only snapshot of a space's atoms (a copy-on-write clone of
/// the MORK trie). Construct with [`MorkSpace::snapshot`]; share one across threads
/// (e.g. `Arc<MorkSnapshot>`) for concurrent queries.
pub struct MorkSnapshot {
    btm: PathMap<()>,
}

impl MorkSnapshot {
    /// Query the snapshot; safe to call concurrently from many threads.
    pub fn query(&self, query: &Atom) -> BindingsSet {
        query_btm(&self.btm, query, None)
    }

    /// Number of atoms in the snapshot.
    pub fn len(&self) -> usize {
        self.btm.val_count()
    }
}

/// Encodes `query` into the `(, <pattern>)` wrapper `query_multi` expects, returning
/// the bytes and the variables in introduction order.
fn wrap_pattern(query: &Atom) -> Option<(Vec<u8>, Vec<VariableAtom>)> {
    let mut vars = Vec::new();
    let mut wrapped = Vec::with_capacity(64);
    wrapped.push(item_byte(Tag::Arity(2)));
    wrapped.push(item_byte(Tag::SymbolSize(1)));
    wrapped.push(b',');
    if encode_atom(query, &mut vars, &mut wrapped).is_err() {
        return None;
    }
    Some((wrapped, vars))
}

/// A hash-prefix-sharded MORK space for data-parallel whole-space sweeps -- the
/// ShardZipper symbolic-CPU path (Goertzel 2025). Atoms are partitioned across
/// `n_shards` PathMap tries by a hash of their byte encoding; each shard is a
/// locally-sweepable sub-trie, and a whole-space match-count sweeps all shards in
/// parallel with rayon. (A ground point query lands in one shard; a pattern that can
/// match anywhere sweeps all shards, which is where the parallelism pays.)
pub struct ShardedMorkSpace {
    shards: Vec<PathMap<()>>,
}

impl ShardedMorkSpace {
    /// Creates an empty sharded space with `n_shards` shards (at least one).
    pub fn new(n_shards: usize) -> Self {
        Self {
            shards: (0..n_shards.max(1)).map(|_| PathMap::new()).collect(),
        }
    }

    fn shard_of(&self, bytes: &[u8]) -> usize {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        (h.finish() as usize) % self.shards.len()
    }

    /// Adds an atom to its hash-determined shard. Returns false on an unencodable atom.
    pub fn add(&mut self, atom: &Atom) -> bool {
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        if encode_atom(atom, &mut vars, &mut bytes).is_err() {
            return false;
        }
        let s = self.shard_of(&bytes);
        self.shards[s].insert(&bytes, ());
        true
    }

    /// Total atoms across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.val_count()).sum()
    }

    /// Whether the space holds no atoms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of shards.
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// Counts atoms matching `pattern`, sweeping every shard in parallel (rayon).
    pub fn par_count_matches(&self, pattern: &Atom) -> usize {
        use rayon::prelude::*;
        let Some((mut wrapped, _vars)) = wrap_pattern(pattern) else {
            return 0;
        };
        // Pass the pattern-buffer address as a usize (Send) and rebuild the Expr in
        // each task; the buffer is alive for the whole sweep and read-only.
        let pat_ptr = wrapped.as_mut_ptr() as usize;
        self.shards
            .par_iter()
            .map(|shard| {
                let pat_expr = Expr {
                    ptr: pat_ptr as *mut u8,
                };
                MorkKernel::query_multi(shard, pat_expr, |_, _| true)
            })
            .sum()
    }

    /// Sequential baseline of [`par_count_matches`].
    pub fn count_matches(&self, pattern: &Atom) -> usize {
        let Some((mut wrapped, _vars)) = wrap_pattern(pattern) else {
            return 0;
        };
        let pat_expr = Expr {
            ptr: wrapped.as_mut_ptr(),
        };
        self.shards
            .iter()
            .map(|shard| MorkKernel::query_multi(shard, pat_expr, |_, _| true))
            .sum()
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
        Atom::Grounded(g) => encode_grounded(&g.to_string(), out),
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

/// Encodes a grounded atom as a marker symbol: [`GROUNDED_MARK`] followed by the
/// atom's display string. `decode_atom` recognises the marker and rebuilds the
/// original `Atom` from the [`GroundedRegistry`]. The display plus marker must fit
/// MORK's 63-byte symbol field.
fn encode_grounded(display: &str, out: &mut Vec<u8>) -> Result<(), ()> {
    let bytes = display.as_bytes();
    if bytes.is_empty() || bytes.len() + 1 > MAX_FIELD {
        return Err(());
    }
    out.push(item_byte(Tag::SymbolSize((bytes.len() + 1) as u8)));
    out.push(GROUNDED_MARK);
    out.extend_from_slice(bytes);
    Ok(())
}

/// Walks MORK's preorder byte encoding back into a Hyperon `Atom`. `pos` advances
/// past the consumed bytes. Returns `None` on a malformed/short buffer.
fn decode_atom(bytes: &[u8], pos: &mut usize, ctx: &mut DecodeCtx) -> Option<Atom> {
    let tag = byte_item(*bytes.get(*pos)?);
    *pos += 1;
    match tag {
        Tag::SymbolSize(s) => {
            let s = s as usize;
            let end = pos.checked_add(s)?;
            let raw = bytes.get(*pos..end)?;
            *pos = end;
            if let Some((&GROUNDED_MARK, disp_bytes)) = raw.split_first() {
                let disp = std::str::from_utf8(disp_bytes).ok()?;
                // Rebuild the grounded atom from the registry; without one (snapshot or
                // sharded shard) fall back to a bare symbol of the display string.
                return Some(
                    ctx.grounded
                        .and_then(|reg| reg.get(disp))
                        .unwrap_or_else(|| Atom::sym(disp)),
                );
            }
            let name = std::str::from_utf8(raw).ok()?;
            Some(Atom::sym(name))
        }
        Tag::Arity(k) => {
            let mut children = Vec::with_capacity(k as usize);
            for _ in 0..k {
                children.push(decode_atom(bytes, pos, ctx)?);
            }
            Some(Atom::expr(children))
        }
        // The n-th introduced variable and its back-references share one name, so the
        // coreference in stored atoms like (-> (→ $p $q) $p $q) survives the round-trip.
        Tag::NewVar => {
            let idx = ctx.var_counter;
            ctx.var_counter += 1;
            Some(decode_var(ctx, idx))
        }
        Tag::VarRef(i) => Some(decode_var(ctx, i as usize)),
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
        let mut rz = self.kernel.btm.read_zipper();
        while rz.to_next_val() {
            let atom_bytes = rz.path();
            let mut pos = 0;
            let mut ctx = DecodeCtx {
                ns: 1,
                var_counter: 0,
                grounded: Some(&self.grounded),
                query_vars: &[],
                result_id: next_variable_id(),
            };
            if let Some(atom) = decode_atom(atom_bytes, &mut pos, &mut ctx) {
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
            self.grounded.register(&atom);
            self.kernel.btm.insert(&bytes, ());
        }
    }

    fn remove(&mut self, atom: &Atom) -> bool {
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        if encode_atom(atom, &mut vars, &mut bytes).is_err() {
            return false;
        }
        self.kernel.btm.remove(&bytes).is_some()
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

    #[test]
    fn sharded_parallel_sweep_matches_sequential() {
        let mut space = ShardedMorkSpace::new(8);
        for i in 0..1000u32 {
            space.add(&Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("a{}", i)),
                Atom::sym(format!("b{}", i)),
            ]));
        }
        assert_eq!(space.len(), 1000);
        let all = Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("y")]);
        assert_eq!(space.count_matches(&all), 1000);
        assert_eq!(space.par_count_matches(&all), 1000);
        // A partial pattern still sweeps every shard and finds the one match.
        let one = Atom::expr([Atom::sym("edge"), Atom::sym("a500"), Atom::var("y")]);
        assert_eq!(space.par_count_matches(&one), 1);
    }

    #[test]
    fn mm2_exec_runs_transitive_closure() {
        // Load facts + a forward-chaining exec rule, run MORK's MM2 exec to fixpoint,
        // then read the transitive closure back through the byte-level query.
        let mut space = MorkSpace::new();
        space
            .add_sexpr_text(
                "(edge a b)\n(edge b c)\n(edge c d)\n\
                 (path a b)\n(path b c)\n(path c d)\n\
                 (exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n",
            )
            .unwrap();
        // The plain exec rule is one-shot, so re-add it each round and step until the
        // closure stops growing (a tiny driver; semi-naive fixpoint is feature-gated).
        let mut prev = 0;
        for _ in 0..8 {
            space
                .add_sexpr_text("(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n")
                .unwrap();
            space.step(1);
            let now = space.len();
            if now == prev {
                break;
            }
            prev = now;
        }
        let q = Atom::expr([Atom::sym("path"), Atom::sym("a"), Atom::var("z")]);
        let mut zs: Vec<String> = space
            .query(&q)
            .iter()
            .filter_map(|b| b.resolve(&VariableAtom::new("z")))
            .map(|a| a.to_string())
            .collect();
        zs.sort();
        zs.dedup();
        assert_eq!(zs, vec!["b", "c", "d"]); // transitive closure from a: b, c, d
    }

    /// hyperon-experimental #1079: GroundingSpace::visit undercounts atoms when many
    /// share a head symbol (its reproducer enumerated 1293 of 1500). MORK walks every
    /// trie leaf, so visit / get-atoms / atom_count are exact. Same reproducer workload.
    #[test]
    fn visit_counts_all_same_head_atoms() {
        struct Counter(usize);
        impl SpaceVisitor for Counter {
            fn accept(&mut self, _: Cow<'_, Atom>) {
                self.0 += 1;
            }
        }
        let mut space = MorkSpace::new();
        let target = 1500usize;
        for n in 0..target {
            space.add(Atom::expr([
                Atom::sym("item-shape-signal"),
                Atom::sym(format!("pattern-{n}")),
                Atom::sym(format!("target-shape-{n}")),
                Atom::sym(format!("{n}")),
            ]));
        }
        assert_eq!(space.atom_count(), Some(target));
        let mut counter = Counter(0);
        space.visit(&mut counter).unwrap();
        assert_eq!(counter.0, target, "MORK visit enumerated {} of {}", counter.0, target);
    }
}
