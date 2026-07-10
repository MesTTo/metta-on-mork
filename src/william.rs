// SPDX-License-Identifier: MIT
//! WILLIAM's compression-gain index and frequent-subpattern report (whitepaper
//! 5.12), carried by this crate.
//!
//! The MORK kernel's `weighted_paths` sidecar (upstream PR #101) provides the
//! weight bookkeeping and offset selection, but the compression-gain builder,
//! the prefix-free maximal top-k, and the pattern renderer live only in the
//! optimized fork and in no upstream PR. [`MorkSpace`](crate::MorkSpace) needs
//! exactly those three, so they are ported here over the same `PathMap<i64>`
//! shape, byte-compatible with the fork's index.

use std::collections::BTreeMap;

use mork::__mork_expr::{maybe_byte_item, Tag};
use pathmap::PathMap;
use pathmap::morphisms::Catamorphism;
use pathmap::zipper::{ZipperValues, ZipperWriting};

/// Byte length of a definition/reference id header in the fork's validated
/// factoring loop: one `SymbolSize` tag plus an 8-byte id payload. Building the
/// gain index with this `ref_cost` makes its weights that factoring's predicted
/// gains.
pub const REF_COST: u64 = 9;

/// Derived compression-gain index over encoded MORK paths.
///
/// Weights live outside the authoritative `PathMap<()>` atom store, so building
/// or dropping the index never changes byte-path semantics.
#[derive(Clone, Debug, Default)]
pub struct WeightedPathIndex {
    weights: PathMap<i64>,
    total_positive_weight: u64,
    updates: usize,
}

/// Errors from weighted-index maintenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightedPathError {
    /// Signed path weight arithmetic overflowed while applying a delta.
    WeightOverflow { current: i64, delta: i64 },
    /// Positive sampling-weight aggregation overflowed.
    TotalPositiveWeightOverflow { left: u64, right: u64 },
    /// Positive sampling-weight aggregation underflowed, which indicates a
    /// broken index invariant.
    TotalPositiveWeightUnderflow { current: u64, decrement: u64 },
}

/// Aggregate positive-weight snapshot for structural descent.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct WeightedSelectionTree {
    total_positive_weight: u64,
    nodes: BTreeMap<Vec<u8>, WeightedSelectionNode>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct WeightedSelectionNode {
    self_weight: u64,
    total_weight: u64,
    children: Box<[(u8, u64)]>,
}

impl WeightedPathIndex {
    /// Creates an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Boundary-restricted compression-gain index: a prefix is weighted only when
    /// its byte offset falls on a MORK term boundary (the start of an item, never
    /// inside a symbol's raw payload). Every retained pattern is then a whole
    /// MeTTa subexpression spine, so [`decode_pattern`] renders it and no
    /// candidate is a mid-symbol byte cut. For every prefix shared by
    /// `count >= 2` stored atoms the weight is the bytes factoring it would save:
    /// `(count - 1) * len - count * ref_cost`. One bottom-up catamorphism
    /// computes every subtrie's occurrence count, so the whole index is built in
    /// a single pass over the store.
    pub fn from_compression_gain_on_boundaries(atoms: &PathMap<()>, ref_cost: u64) -> Self {
        let mut index = Self::new();
        atoms.read_zipper().into_cata_side_effect(
            |_mask, children: &mut [usize], value: Option<&()>, path: &[u8]| -> usize {
                let count = value.is_some() as usize + children.iter().copied().sum::<usize>();
                let admit = count >= 2 && !path.is_empty() && path_ends_on_term_boundary(path);
                if admit {
                    let gain = boundary_gain(count as i64, path.len(), ref_cost);
                    if gain > 0 {
                        // Overflow only at extreme scales; a dropped weight just omits
                        // one compressible prefix from the derived index, never
                        // corrupts the store.
                        let _ = index.set_weight(path, gain);
                    }
                }
                count
            },
        );
        index
    }

    /// Returns the signed weight stored for `path`, or zero when absent.
    pub fn weight(&self, path: &[u8]) -> i64 {
        self.weights.get_val_at(path).copied().unwrap_or(0)
    }

    /// Returns the total positive weight retained by the index.
    pub fn total_positive_weight(&self) -> u64 {
        self.total_positive_weight
    }

    /// Sets the signed weight for `path`. Zero removes the entry; negative values
    /// are retained as signed maintenance state but never surface in top-k.
    pub fn set_weight(&mut self, path: &[u8], weight: i64) -> Result<(), WeightedPathError> {
        let current_total = self.total_positive_weight;
        let mut zipper = self.weights.write_zipper_at_path(path);
        let previous = zipper.val().copied().unwrap_or(0);
        let total_positive_weight = updated_total(current_total, previous, weight)?;

        if weight == 0 {
            zipper.remove_val(true);
        } else {
            zipper.set_val(weight);
        }

        self.total_positive_weight = total_positive_weight;
        self.updates += 1;
        Ok(())
    }

    /// WILLIAM maximal top-k: the `k` heaviest patterns forming a prefix-free
    /// antichain, so nested prefixes of one hot chain collapse to a single
    /// representative. See [`WeightedSelectionTree::top_k_maximal`].
    pub fn iter_any_topk_maximal(&self, k: usize) -> Result<Vec<(Vec<u8>, u64)>, WeightedPathError> {
        Ok(WeightedSelectionTree::from_weights(&self.weights)?.top_k_maximal(k))
    }
}

impl WeightedSelectionTree {
    fn from_weights(weights: &PathMap<i64>) -> Result<Self, WeightedPathError> {
        let mut nodes = BTreeMap::new();
        let total_positive_weight = weights.read_zipper().into_cata_side_effect(
            |mask, children: &mut [Result<u64, WeightedPathError>], value, path| {
                let self_weight = value.copied().map(positive_weight).unwrap_or(0);
                let mut total_weight = self_weight;
                let mut retained_children = Vec::new();

                for (byte, child_total) in mask.iter().zip(children.iter().copied()) {
                    let child_total = child_total?;
                    total_weight = checked_add_positive_weight(total_weight, child_total)?;
                    if child_total > 0 {
                        retained_children.push((byte, child_total));
                    }
                }

                if total_weight > 0 {
                    nodes.insert(
                        path.to_vec(),
                        WeightedSelectionNode {
                            self_weight,
                            total_weight,
                            children: retained_children.into_boxed_slice(),
                        },
                    );
                }

                Ok(total_weight)
            },
        );
        let total_positive_weight = total_positive_weight?;

        Ok(Self {
            total_positive_weight,
            nodes,
        })
    }

    /// The `k` highest-weight patterns forming a prefix-free antichain (no chosen
    /// pattern is a byte-prefix of another). Greedy by descending weight, so along
    /// any root-to-leaf chain only its single heaviest node survives. Tie-break:
    /// path ascending. Considers every weighted node: the antichain constraint
    /// couples a pick to its whole chain, so subtree pruning cannot bound it; this
    /// is the report-time query, not an inner-loop sampler.
    fn top_k_maximal(&self, k: usize) -> Vec<(Vec<u8>, u64)> {
        if k == 0 {
            return Vec::new();
        }
        let mut cands: Vec<(u64, &Vec<u8>)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.self_weight > 0)
            .map(|(path, node)| (node.self_weight, path))
            .collect();
        // Heaviest first; smaller path wins a weight tie.
        cands.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

        let mut chosen: Vec<Vec<u8>> = Vec::new();
        let mut out: Vec<(Vec<u8>, u64)> = Vec::new();
        for (weight, path) in cands {
            if out.len() >= k {
                break;
            }
            let overlaps = chosen
                .iter()
                .any(|c| path.starts_with(c) || c.starts_with(path.as_slice()));
            if !overlaps {
                chosen.push(path.clone());
                out.push((path.clone(), weight));
            }
        }
        out
    }
}

/// Bytes saved by factoring a `len`-byte pattern shared by `count` atoms into one
/// definition plus `count` references: `(count - 1) * len - count * ref_cost`.
fn boundary_gain(count: i64, len: usize, ref_cost: u64) -> i64 {
    (count - 1) * len as i64 - count * ref_cost as i64
}

fn positive_weight(weight: i64) -> u64 {
    if weight > 0 { weight as u64 } else { 0 }
}

fn checked_add_positive_weight(left: u64, right: u64) -> Result<u64, WeightedPathError> {
    left.checked_add(right)
        .ok_or(WeightedPathError::TotalPositiveWeightOverflow { left, right })
}

fn updated_total(current_total: u64, previous: i64, next: i64) -> Result<u64, WeightedPathError> {
    let previous_positive = positive_weight(previous);
    let next_positive = positive_weight(next);

    if next_positive >= previous_positive {
        checked_add_positive_weight(current_total, next_positive - previous_positive)
    } else {
        let decrement = previous_positive - next_positive;
        current_total.checked_sub(decrement).ok_or(
            WeightedPathError::TotalPositiveWeightUnderflow {
                current: current_total,
                decrement,
            },
        )
    }
}

/// Whether `path` (a prefix of a MORK-encoded atom) ends exactly between items.
///
/// MORK encodes each item as a single tag byte; `SymbolSize(s)` is followed by `s`
/// raw payload bytes and `Arity(a)` opens `a` complete subterms. A prefix that
/// ends inside a symbol's payload is a mid-symbol cut, not a subexpression
/// boundary. Total on arbitrary stores: a byte outside the tag range makes the
/// whole prefix a non-boundary rather than panicking, so the boundary-restricted
/// index simply skips foreign paths.
fn path_ends_on_term_boundary(path: &[u8]) -> bool {
    let mut pos = 0usize;
    while pos < path.len() {
        match maybe_byte_item(path[pos]) {
            Ok(Tag::NewVar) | Ok(Tag::VarRef(_)) | Ok(Tag::Arity(_)) => pos += 1,
            Ok(Tag::SymbolSize(s)) => {
                let next = pos + 1 + s as usize;
                if next > path.len() {
                    // path ends inside this symbol's payload: not a boundary.
                    return false;
                }
                pos = next;
            }
            Err(_) => return false,
        }
    }
    pos == path.len()
}

/// Frame for one open compound while decoding: how many argument slots remain, and
/// whether the compound has emitted its first element yet (for spacing).
struct DecodeFrame {
    remaining: usize,
    first: bool,
}

/// Renders a MORK-encoded path prefix as readable MeTTa.
///
/// The prefix must end on a term boundary (as produced by
/// [`WeightedPathIndex::from_compression_gain_on_boundaries`]). A compound whose
/// trailing arguments the prefix cut off shows the missing slots as `…`, so the
/// pattern reads as the whole subexpression it factors: the encoding of
/// `(rule (when $x))` truncated after `rule` decodes to `(rule …)`, and an arity
/// opened with no head yet decodes to `(…)`. Non-UTF-8 symbol bytes are
/// hex-escaped so the rendering never panics.
pub fn decode_pattern(path: &[u8]) -> String {
    let mut out = String::new();
    let mut stack: Vec<DecodeFrame> = Vec::new();
    let mut pos = 0usize;

    while pos < path.len() {
        let Ok(item) = maybe_byte_item(path[pos]) else {
            break; // foreign byte: render what decoded so far.
        };
        match item {
            Tag::Arity(a) => {
                pos += 1;
                before_element(&mut out, &mut stack);
                out.push('(');
                if a == 0 {
                    out.push(')');
                    complete_element(&mut out, &mut stack);
                } else {
                    stack.push(DecodeFrame { remaining: a as usize, first: true });
                }
            }
            Tag::SymbolSize(s) => {
                let start = pos + 1;
                let end = start + s as usize;
                if end > path.len() {
                    break; // defensive: a non-boundary prefix slipped in.
                }
                before_element(&mut out, &mut stack);
                emit_symbol(&mut out, &path[start..end]);
                pos = end;
                complete_element(&mut out, &mut stack);
            }
            Tag::NewVar => {
                pos += 1;
                before_element(&mut out, &mut stack);
                out.push('$');
                complete_element(&mut out, &mut stack);
            }
            Tag::VarRef(i) => {
                pos += 1;
                before_element(&mut out, &mut stack);
                out.push('_');
                out.push_str(&i.to_string());
                complete_element(&mut out, &mut stack);
            }
        }
    }

    // Truncation: close every still-open compound, filling its unfilled argument
    // slots with `…`. A closed compound occupies one slot of its parent, so
    // decrement the parent before flushing it (that slot is the in-progress
    // child, not a fresh `…`).
    while let Some(frame) = stack.pop() {
        for slot in 0..frame.remaining {
            if !(frame.first && slot == 0) {
                out.push(' ');
            }
            out.push('…');
        }
        out.push(')');
        if let Some(parent) = stack.last_mut() {
            parent.remaining = parent.remaining.saturating_sub(1);
            parent.first = false;
        }
    }

    out
}

/// Emit s-expression spacing before an element: a space when the current compound
/// has already emitted an element, nothing when this is its head (or at top level).
fn before_element(out: &mut String, stack: &mut [DecodeFrame]) {
    if let Some(top) = stack.last_mut() {
        if top.first {
            top.first = false;
        } else {
            out.push(' ');
        }
    }
}

/// Register that one element finished: decrement the enclosing compound's
/// remaining slots and cascade-close every compound that reaches zero.
fn complete_element(out: &mut String, stack: &mut Vec<DecodeFrame>) {
    loop {
        match stack.last_mut() {
            None => break,
            Some(top) => {
                top.remaining -= 1;
                if top.remaining == 0 {
                    out.push(')');
                    stack.pop();
                } else {
                    break;
                }
            }
        }
    }
}

/// Append a symbol's raw bytes as text, hex-escaping when they are not valid UTF-8.
fn emit_symbol(out: &mut String, sym: &[u8]) {
    match std::str::from_utf8(sym) {
        Ok(s) => out.push_str(s),
        Err(_) => {
            out.push_str("0x");
            for b in sym {
                out.push_str(&format!("{b:02x}"));
            }
        }
    }
}
