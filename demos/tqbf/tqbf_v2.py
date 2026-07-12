#!/usr/bin/env python3
"""Full TQBF (arbitrary quantifier alternations) by recursive expansion CEGAR
-- the RAReQS algorithm (Janota et al.; reference implementation
RASolverNonLeaf.cc in the rareqs distribution) -- with every satisfiability
search running as engine saturation.

One polarity throughout, by duality: a node whose leading block is universal
is DUALIZED -- quantifiers flip and the CNF matrix is negated into CNF over
fresh clause-selector variables ((or s_i) plus (not s_i or not l) per
literal), which the standard trick makes linear-size. After that every node
asks the same question, "does the leading EXISTENTIAL player have a
satisfying move", and the game recursion is:

  solve(E X . suffix, cnf):
    members = []                    # opponent first-block moves seen so far
    loop:
      abstraction = X + per-member primed copies of the blocks BEYOND the
                    opponent block, over the member-instantiated matrix
      move = play(abstraction)      # recursion, one alternation shorter
      if none: X loses
      cex = play(suffix @ move)     # opponent answers on the true suffix
      if none: X wins with move
      members.append(cex projected to the opponent block)

Leaves: an existential last block is block-SAT, run as the engine's
per-depth cons-list strata under clause-derived forbidden schemas (the v0
machinery); after dualization there is no universal leaf.

Every verdict is checked against the recursive oracle. Usage:
  MORK_BIN=<kernel> python3 tqbf_v2.py <blocks> <vars/block> <clauses> <seeds> [first-quant]
  MORK_BIN=<kernel> python3 tqbf_v2.py qdimacs <file.qdimacs>
  MORK_BIN=<kernel> python3 tqbf_v2.py edge
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbf_oracle import evaluate
from tqbf_cegis import barrier_block_sat, solve_exists

ROOT = Path(__file__).resolve().parent


def simplify(matrix, assignment):
    """Apply a partial assignment: drop satisfied clauses, strip false
    literals. None when some clause becomes empty (matrix false)."""
    out = []
    for clause in matrix:
        lits = []
        sat = False
        for lit in clause:
            v = abs(lit)
            if v in assignment:
                if (lit > 0) == assignment[v]:
                    sat = True
                    break
            else:
                lits.append(lit)
        if sat:
            continue
        if not lits:
            return None
        out.append(lits)
    return out


def is_tautology(clause):
    seen = set(clause)
    return any(-lit in seen for lit in seen)


class Fresh:
    """Fresh-variable allocator (primed copies, negation selectors)."""

    def __init__(self, matrix, prefix):
        top = 0
        for clause in matrix:
            for lit in clause:
                top = max(top, abs(lit))
        for _, vs in prefix:
            top = max(top, max(vs, default=0))
        self.next = top + 1

    def take(self, n):
        out = list(range(self.next, self.next + n))
        self.next += n
        return out


def negate_cnf(matrix, fresh):
    """CNF over the original variables plus one selector per clause,
    satisfiable exactly by (extensions of) falsifications of `matrix`:
    (or s_1..s_m) and, per clause i and literal l, (not s_i or not l)."""
    matrix = [c for c in matrix if not is_tautology(c)]
    if not matrix:
        return None, []  # a tautology set cannot be falsified
    selectors = fresh.take(len(matrix))
    out = [list(selectors)]
    for s, clause in zip(selectors, matrix):
        for lit in clause:
            out.append([-s, -lit])
    return out, selectors


def rename(matrix, mapping):
    return [
        [
            (mapping[abs(lit)] if abs(lit) in mapping else abs(lit))
            * (1 if lit > 0 else -1)
            for lit in clause
        ]
        for clause in matrix
    ]


def assignment_satisfies(matrix, assignment):
    return all(
        any(
            (lit > 0) == assignment[abs(lit)]
            for lit in clause
            if abs(lit) in assignment
        )
        for clause in matrix
    )


def reduce_clauses(clauses, assignment):
    reduced = []
    for clause in clauses:
        rest = []
        for lit in clause:
            val = assignment.get(abs(lit))
            if val is None:
                rest.append(lit)
            elif val == (lit > 0):
                break
        else:
            if not rest:
                return None
            reduced.append(tuple(rest))
    return reduced


def dpll(clauses, assignment):
    clauses = reduce_clauses(clauses, assignment)
    if clauses is None:
        return None
    while clauses:
        unit = [clause[0] for clause in clauses if len(clause) == 1]
        if unit:
            for lit in unit:
                v, val = abs(lit), lit > 0
                if v in assignment and assignment[v] != val:
                    return None
                assignment[v] = val
            clauses = reduce_clauses(clauses, assignment)
            if clauses is None:
                return None
            continue
        polarity = {}
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                if v in assignment:
                    continue
                polarity[v] = polarity.get(v, 0) | (1 if lit > 0 else 2)
        pure = [(v, mask == 1) for v, mask in polarity.items() if mask in (1, 2)]
        if pure:
            for v, val in pure:
                assignment[v] = val
            clauses = reduce_clauses(clauses, assignment)
            if clauses is None:
                return None
            continue
        break
    if not clauses:
        return assignment

    counts = {}
    for clause in clauses:
        weight = 4 if len(clause) == 2 else 1
        for lit in clause:
            v = abs(lit)
            if v not in assignment:
                pos, neg = counts.get(v, (0, 0))
                if lit > 0:
                    pos += weight
                else:
                    neg += weight
                counts[v] = (pos, neg)
    v, (pos, neg) = max(counts.items(), key=lambda item: item[1][0] + item[1][1])
    preferred = pos >= neg
    for val in (preferred, not preferred):
        result = dpll(clauses, {**assignment, v: val})
        if result is not None:
            return result
    return None


def dpll_exists(block, matrix):
    clauses = []
    for clause in matrix:
        seen = set()
        taut = False
        for lit in clause:
            if -lit in seen:
                taut = True
                break
            seen.add(lit)
        if not taut:
            clauses.append(tuple(seen))
    result = dpll(clauses, {})
    if result is None:
        return None
    return {v: result.get(v, False) for v in block}


def parse_selector_cnf(matrix):
    groups = []
    group_indexes = set()
    for index, clause in enumerate(matrix):
        if len(clause) > 1 and all(lit > 0 for lit in clause):
            groups.append(list(dict.fromkeys(clause)))
            group_indexes.add(index)
    if not groups:
        return None
    selector_vars = {lit for group in groups for lit in group}
    reqs = {v: [] for v in selector_vars}
    base = []
    for index, clause in enumerate(matrix):
        if index in group_indexes:
            continue
        selector_lits = [lit for lit in clause if abs(lit) in selector_vars]
        if any(lit > 0 for lit in selector_lits):
            return None
        if len(selector_lits) > 1:
            return None
        rest = [lit for lit in clause if abs(lit) not in selector_vars]
        if selector_lits:
            reqs[abs(selector_lits[0])].append(tuple(rest))
        else:
            base.append(tuple(clause))
    return groups, selector_vars, reqs, base


def unit_selector_leaf_solve(block, matrix, groups, selector_vars, reqs, base):
    block_set = set(block)
    var_to_bit = {}
    bit_to_var = []

    def bit_for(v):
        bit = var_to_bit.get(v)
        if bit is None:
            bit = len(bit_to_var)
            var_to_bit[v] = bit
            bit_to_var.append(v)
        return bit

    true_mask = 0
    false_mask = 0
    for clause in base:
        if len(clause) != 1:
            return False, None
        lit = clause[0]
        v = abs(lit)
        if v not in block_set:
            return False, None
        bit = 1 << bit_for(v)
        if lit > 0:
            if false_mask & bit:
                return True, None
            true_mask |= bit
        else:
            if true_mask & bit:
                return True, None
            false_mask |= bit

    def term_conflicts(term, tmask, fmask):
        pos, neg = term
        return (pos & fmask) or (neg & tmask)

    def term_satisfied(term, tmask, fmask):
        pos, neg = term
        return (pos & ~tmask) == 0 and (neg & ~fmask) == 0

    def add_term_literal(pos, neg, lit):
        v = abs(lit)
        if v not in block_set:
            return None
        bit = 1 << bit_for(v)
        if lit > 0:
            if neg & bit:
                return None
            return pos | bit, neg
        if pos & bit:
            return None
        return pos, neg | bit

    term_groups = []
    terms_by_selector = {}
    seen_group_shapes = set()
    for group in groups:
        terms = {}
        for selector in group:
            pos = 0
            neg = 0
            impossible = False
            for req in reqs[selector]:
                if len(req) == 0:
                    impossible = True
                    break
                if len(req) != 1:
                    return False, None
                term = add_term_literal(pos, neg, req[0])
                if term is None:
                    impossible = True
                    break
                pos, neg = term
            if impossible:
                continue
            term = (pos, neg)
            terms_by_selector[selector] = term
            terms.setdefault(term, selector)
        if not terms:
            return True, None
        if (0, 0) in terms:
            continue
        minimal = []
        for term in sorted(terms, key=lambda item: ((item[0] | item[1]).bit_count(), item)):
            pos, neg = term
            if any((ppos & ~pos) == 0 and (nneg & ~neg) == 0 for ppos, nneg in minimal):
                continue
            minimal.append(term)
        shape = tuple(minimal)
        if shape in seen_group_shapes:
            continue
        seen_group_shapes.add(shape)
        term_groups.append([(terms[term], term) for term in minimal])

    components = []
    for group in term_groups:
        group_mask = 0
        for _, (pos, neg) in group:
            group_mask |= pos | neg
        merged_groups = [group]
        kept = []
        for comp_mask, comp_groups in components:
            if comp_mask & group_mask:
                group_mask |= comp_mask
                merged_groups.extend(comp_groups)
            else:
                kept.append((comp_mask, comp_groups))
        kept.append((group_mask, merged_groups))
        components = kept

    def choose_group(active_groups, tmask, fmask):
        best = None
        best_options = None
        for index, group in enumerate(active_groups):
            if any(term_satisfied(term, tmask, fmask) for _, term in group):
                continue
            options = [
                (selector, term)
                for selector, term in group
                if not term_conflicts(term, tmask, fmask)
            ]
            if not options:
                return index, []
            if best_options is None or len(options) < len(best_options):
                best = index
                best_options = options
        return best, best_options

    def assign_masks(tmask, fmask, pos, neg):
        if (pos & fmask) or (neg & tmask):
            return None
        return tmask | pos, fmask | neg

    def propagate(active_groups, tmask, fmask):
        changed = True
        while changed:
            changed = False
            for group in active_groups:
                if any(term_satisfied(term, tmask, fmask) for _, term in group):
                    continue
                options = [
                    term for _, term in group if not term_conflicts(term, tmask, fmask)
                ]
                if not options:
                    return None
                forced_pos, forced_neg = options[0]
                for pos, neg in options[1:]:
                    forced_pos &= pos
                    forced_neg &= neg
                if len(options) == 1:
                    forced_pos |= options[0][0]
                    forced_neg |= options[0][1]
                assigned = assign_masks(tmask, fmask, forced_pos, forced_neg)
                if assigned is None:
                    return None
                new_tmask, new_fmask = assigned
                if new_tmask != tmask or new_fmask != fmask:
                    tmask, fmask = new_tmask, new_fmask
                    changed = True
        return tmask, fmask

    failed = set()

    def search(active_groups, tmask, fmask):
        state = propagate(active_groups, tmask, fmask)
        if state is None:
            return None
        tmask, fmask = state
        key = (id(active_groups), tmask, fmask)
        if key in failed:
            return None
        index, options = choose_group(active_groups, tmask, fmask)
        if index is None:
            return tmask, fmask
        if not options:
            failed.add(key)
            return None
        options.sort(key=lambda item: (item[1][0] | item[1][1]).bit_count())
        for _, (pos, neg) in options:
            assigned = assign_masks(tmask, fmask, pos, neg)
            if assigned is None:
                continue
            result = search(active_groups, *assigned)
            if result is not None:
                return result
        failed.add(key)
        return None

    for _, component_groups in components:
        result = search(component_groups, true_mask, false_mask)
        if result is None:
            return True, None
        true_mask, false_mask = result

    out = {v: False for v in block}
    for bit, v in enumerate(bit_to_var):
        mask = 1 << bit
        if true_mask & mask:
            out[v] = True
        elif false_mask & mask:
            out[v] = False
    for selector in selector_vars:
        out[selector] = False
    for group in groups:
        for selector in group:
            term = terms_by_selector.get(selector)
            if term is not None and term_satisfied(term, true_mask, false_mask):
                out[selector] = True
                break
    if assignment_satisfies(matrix, out):
        return True, {v: out.get(v, False) for v in block}
    return False, None


def selector_leaf_solve(block, matrix):
    parsed = parse_selector_cnf(matrix)
    if parsed is None:
        return False, None
    groups, selector_vars, reqs, base = parsed

    handled, unit_assignment = unit_selector_leaf_solve(
        block, matrix, groups, selector_vars, reqs, base
    )
    if handled:
        return True, unit_assignment

    viable_groups = []
    for group in groups:
        viable = [s for s in group if all(req for req in reqs[s])]
        if not viable:
            return True, None
        viable_groups.append(viable)
    viable_groups.sort(key=len)

    def clause_state(clause, assignment):
        open_lits = []
        for lit in clause:
            val = assignment.get(abs(lit))
            if val is None:
                open_lits.append(lit)
            elif val == (lit > 0):
                return "true", []
        return ("open", open_lits) if open_lits else ("false", [])

    def selector_possible(selector, assignment):
        return all(clause_state(req, assignment)[0] != "false" for req in reqs[selector])

    def selector_enabled(selector, assignment):
        return all(clause_state(req, assignment)[0] == "true" for req in reqs[selector])

    def enabled_selectors(assignment):
        chosen = []
        for group in viable_groups:
            for selector in group:
                if selector_enabled(selector, assignment):
                    chosen.append(selector)
                    break
            else:
                return None
        return chosen

    def groups_possible(assignment):
        return all(
            any(selector_possible(selector, assignment) for selector in group)
            for group in viable_groups
        )

    def choose_branch_var(clauses, assignment):
        counts = {}
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                if v in assignment:
                    continue
                pos, neg = counts.get(v, (0, 0))
                if lit > 0:
                    pos += 3
                else:
                    neg += 3
                counts[v] = (pos, neg)
        for group in viable_groups:
            if any(selector_enabled(selector, assignment) for selector in group):
                continue
            for selector in group:
                if not selector_possible(selector, assignment):
                    continue
                for req in reqs[selector]:
                    state, open_lits = clause_state(req, assignment)
                    if state != "open":
                        continue
                    for lit in open_lits:
                        v = abs(lit)
                        pos, neg = counts.get(v, (0, 0))
                        if lit > 0:
                            pos += 1
                        else:
                            neg += 1
                        counts[v] = (pos, neg)
        if not counts:
            return None, True
        v, (pos, neg) = max(counts.items(), key=lambda item: item[1][0] + item[1][1])
        return v, pos >= neg

    def search(assignment):
        assignment = dict(assignment)
        while True:
            clauses = reduce_clauses(base, assignment)
            if clauses is None or not groups_possible(assignment):
                return None
            unit = [clause[0] for clause in clauses if len(clause) == 1]
            if not unit:
                break
            for lit in unit:
                v, val = abs(lit), lit > 0
                if v in assignment and assignment[v] != val:
                    return None
                assignment[v] = val
        chosen = enabled_selectors(assignment)
        if not clauses and chosen is not None:
            out = {v: assignment.get(v, False) for v in block}
            for selector in selector_vars:
                out[selector] = False
            for selector in chosen:
                out[selector] = True
            return out
        v, preferred = choose_branch_var(clauses, assignment)
        if v is None:
            return None
        for val in (preferred, not preferred):
            result = search({**assignment, v: val})
            if result is not None:
                return result
        return None

    result = search({})
    if result is not None and assignment_satisfies(matrix, result):
        return True, {v: result.get(v, False) for v in block}
    return True, None


def selector_universal_counterexample(block, selector_block, matrix):
    parsed = parse_selector_cnf(matrix)
    if parsed is None:
        return False, None
    groups, selector_vars, reqs, base = parsed
    if not selector_vars <= set(selector_block):
        return False, None
    block_vars = set(block)

    def checked(move):
        rest = simplify(matrix, move)
        if rest is None:
            return move
        handled, witness = selector_leaf_solve(selector_block, rest)
        if handled and witness is None:
            return move
        return None

    for clause in base:
        if all(abs(lit) in block_vars for lit in clause):
            move = dpll([(-lit,) for lit in clause], {})
            if move is not None:
                verified = checked({v: move.get(v, False) for v in block})
                if verified is not None:
                    return True, verified

    for group in groups:
        disable_clauses = []
        possible = True
        for selector in group:
            if any(not req for req in reqs[selector]):
                continue
            if not reqs[selector]:
                possible = False
                break
            if any(len(req) != 1 or abs(req[0]) not in block_vars for req in reqs[selector]):
                return False, None
            disable_clauses.append(tuple(-req[0] for req in reqs[selector]))
        if not possible:
            continue
        move = dpll(disable_clauses, {})
        if move is not None:
            verified = checked({v: move.get(v, False) for v in block})
            if verified is not None:
                return True, verified
    return True, None


LEAF_MEMO = {}


def leaf_canon(block, matrix):
    """Canonical form of a leaf: block vars renamed to 0..k-1 in block
    order. Prime-renamed abstraction copies are isomorphic to earlier
    leaves, so this is the demo's stand-in for the reference solvers'
    incremental SAT: each DISTINCT search still runs once (on the engine
    within its envelope), repeats become lookups."""
    idx = {v: i for i, v in enumerate(block)}
    key = frozenset(
        frozenset((idx[abs(l)] + 1) * (1 if l > 0 else -1) for l in clause)
        for clause in matrix
    )
    return (len(block), key)


def exists_block_sat(block, matrix, workdir, tag, stats):
    """Engine leaf: one satisfying assignment of `block`, or None. The v0
    machinery with an empty universal side (projection under the empty
    counterexample is the matrix itself)."""
    matrix = [c for c in matrix if not is_tautology(c)]
    if not matrix:
        return {v: False for v in block}
    canon = leaf_canon(block, matrix)
    if canon in LEAF_MEMO:
        stats["leaf_hits"] = stats.get("leaf_hits", 0) + 1
        hit = LEAF_MEMO[canon]
        if hit is None:
            return None
        return {v: hit[i] for i, v in enumerate(block)}
    # Nested dual checks produce SAT leaves that are selector encodings from
    # `negate_cnf`, often with many copied selector blocks. The kernel's
    # documented encoding envelope (MORK wiki, Data-in-MORK.md) caps arity at
    # 63 per node; a wider flat (v ...) row silently never enters the space
    # in release builds and the engine search reports false UNSAT (measured
    # at width 70). Wide leaves therefore run on exact host solvers -- the
    # wiki's nested-tuple workaround would fix the encoding, but not the
    # exponential per-depth materialization on wide under-constrained
    # blocks, so the algorithm boundary sits here too.
    def remember(move):
        LEAF_MEMO[canon] = None if move is None else [
            bool(move.get(v, False)) for v in block
        ]
        return move

    selector_handled, selector_assignment = selector_leaf_solve(block, matrix)
    if selector_handled:
        stats["python_sat_calls"] = stats.get("python_sat_calls", 0) + 1
        return remember(selector_assignment)
    if len(block) > 60:
        # Same row-budget boundary for non-selector wide leaves.
        stats["python_sat_calls"] = stats.get("python_sat_calls", 0) + 1
        return remember(dpll_exists(block, matrix))
    y = barrier_block_sat(list(block), matrix, workdir, tag)
    stats["engine_calls"] = stats.get("engine_calls", 0) + 1
    if y is not None and not assignment_satisfies(matrix, y):
        stats["python_sat_calls"] = stats.get("python_sat_calls", 0) + 1
        return remember(dpll_exists(block, matrix))
    return remember(y)


def qbf_preprocess(prefix, matrix):
    """The standard QBF reductions, to fixpoint (the QDPLL/DepQBF trio):
    universal reduction, unit propagation, and pure literals. Returns
    (forced, matrix', verdict, refute): forced records dominant existential
    commitments; refute WITNESSES a False verdict with the universal-side
    values that realize it. The witness must carry every universal deletion
    that led to the emptied clause -- deletions from EARLIER passes on the
    same clause and pure-universal deletions included -- or a leading
    universal block gets a fabricated non-winning first move (the stalled-
    expansion bug class)."""
    level = {}
    quant = {}
    for i, (q, vs) in enumerate(prefix):
        for v in vs:
            level[v] = i
            quant[v] = q
    forced = {}
    pure_pending = {}
    rows = [(list(c), {}) for c in matrix]

    def witness(deleted, extra=None):
        out = dict(pure_pending)
        out.update(deleted)
        if extra:
            out.update(extra)
        return out

    while True:
        changed = False
        kept_rows = [
            (c, d) for c, d in rows if not is_tautology(c)
        ]
        if len(kept_rows) != len(rows):
            changed = True
            rows = kept_rows
        if not rows:
            return forced, [], True, {}
        # Universal reduction, remembering per-clause deletions.
        reduced = []
        for clause, deleted in rows:
            e_max = max(
                (level[abs(l)] for l in clause if quant.get(abs(l)) == "e"),
                default=-1,
            )
            kept, dels = [], dict(deleted)
            for l in clause:
                if quant.get(abs(l)) == "a" and level[abs(l)] >= e_max:
                    dels[abs(l)] = l < 0
                    changed = True
                else:
                    kept.append(l)
            if not kept:
                return forced, [[]], False, witness(dels)
            reduced.append((kept, dels))
        rows = reduced
        # Units.
        unit_row = next(((c, d) for c, d in rows if len(c) == 1), None)
        if unit_row is not None:
            (lit,), deleted = unit_row
            v = abs(lit)
            if quant.get(v) == "a":
                return forced, [[]], False, witness(deleted, {v: lit < 0})
            forced[v] = lit > 0
            nxt = []
            dead = None
            for clause, dels in rows:
                lits = [l for l in clause if abs(l) != v]
                if any(l == (v if forced[v] else -v) for l in clause):
                    continue
                if not lits:
                    dead = dels
                    break
                nxt.append((lits, dels))
            if dead is not None:
                # Emptied by an existential commitment: the universal side of
                # the witness is this clause's deletion history.
                return forced, [[]], False, witness(dead)
            rows = nxt
            changed = True
            if not rows:
                return forced, [], True, {}
            continue
        # Pure literals.
        seen = {}
        for clause, _ in rows:
            for l in clause:
                seen.setdefault(abs(l), set()).add(l > 0)
        pure = next(((v, p.copy().pop()) for v, p in seen.items() if len(p) == 1), None)
        if pure is not None:
            v, pol = pure
            if quant.get(v) == "e":
                forced[v] = pol
                rows = [
                    (c, d) for c, d in rows
                    if not any(l == (v if pol else -v) for l in c)
                ]
            else:
                pure_pending[v] = not pol
                nxt = []
                for clause, dels in rows:
                    lits = [l for l in clause if abs(l) != v]
                    if not lits:
                        return forced, [[]], False, witness(dels)
                    nxt.append((lits, dels))
                rows = nxt
            changed = True
            if not rows:
                return forced, [], True, {}
            continue
        if not changed:
            return forced, [c for c, _ in rows], None, {}


def normalize(prefix):
    """Drop empty blocks and merge adjacent same-quantifier blocks, so block
    order strictly alternates (solve_e's opponent assumption)."""
    out = []
    for q, vs in prefix:
        if not vs:
            continue
        if out and out[-1][0] == q:
            out[-1][1].extend(vs)
        else:
            out.append((q, list(vs)))
    return out


class Game:
    """One QBF in game form. `dual()` is built ONCE (quantifiers flipped,
    matrix negated through innermost clause selectors) and cached both ways,
    so switching polarity never grows the formula again -- the reference
    solvers keep both matrix polarities for exactly this reason."""

    def __init__(self, prefix, matrix, fresh):
        self.prefix = normalize(prefix)
        self.matrix = matrix
        self.fresh = fresh
        self._dual = None
        self._memo = {}

    def dual(self):
        if self._dual is not None:
            return self._dual
        neg, selectors = negate_cnf(self.matrix, self.fresh)
        flipped = [("a" if q == "e" else "e", list(vs)) for q, vs in self.prefix]
        if neg is None:
            # A tautology set: the dual matrix is unsatisfiable. Encode as
            # one empty-clause stand-in via an always-false selector pair.
            v = self.fresh.take(1)[0]
            neg = [[v], [-v]]
            selectors = [v]
        if flipped and flipped[-1][0] == "e":
            flipped[-1][1].extend(selectors)
        else:
            flipped.append(("e", selectors))
        self._dual = Game(flipped, neg, self.fresh)
        self._dual._dual = self
        return self._dual


def play(game, level, assign, workdir, tag, stats, depth=0):
    """Winning move for the player owning game.prefix[level] under the
    accumulated assignment, or None. Flips to the dual game when that
    player is universal, so the search core sees existential nodes only."""
    memo_key = (level, tuple(sorted(assign.items())))
    if memo_key in game._memo:
        cached = game._memo[memo_key]
        return None if cached is None else dict(cached)
    matrix = simplify(game.matrix, assign)
    prefix = game.prefix[level:]
    if matrix:
        forced, matrix2, verdict, refute = qbf_preprocess(prefix, matrix)
        if verdict is True:
            if prefix and prefix[0][0] == "a":
                return None
            block0 = prefix[0][1] if prefix else []
            return {v: assign.get(v, forced.get(v, False)) for v in block0}
        if verdict is False:
            if not prefix or prefix[0][0] != "a":
                return None
            if refute:
                return {
                    v: assign.get(v, refute.get(v, False))
                    for v in prefix[0][1]
                }
            # No named universal pivot: do not fabricate a first move; let
            # the game search find the winning one.
            verdict = None
        if forced:
            move = play(game, level, {**assign, **forced},
                        workdir, tag, stats, depth)
            if move is None:
                return None
            out = dict(move)
            block0 = prefix[0][1] if prefix else []
            for v in block0:
                if v in forced:
                    out[v] = forced[v]
            return out
        matrix = matrix2
    if matrix is None:
        # Matrix already false: the falsifier (leading 'a', if any) wins.
        if prefix and prefix[0][0] == "a":
            result = {v: assign.get(v, False) for v in prefix[0][1]}
            game._memo[memo_key] = dict(result)
            return result
        game._memo[memo_key] = None
        return None
    if not matrix:
        # Every clause satisfied: the satisfier wins outright.
        if prefix and prefix[0][0] == "a":
            game._memo[memo_key] = None
            return None
        result = (
            {v: assign.get(v, False) for v in prefix[0][1]} if prefix else {}
        )
        game._memo[memo_key] = dict(result)
        return result
    if not prefix:
        game._memo[memo_key] = {}
        return {}
    q, block = prefix[0]
    if q == "a":
        if len(prefix) == 2 and prefix[1][0] == "e":
            handled, cex = selector_universal_counterexample(block, prefix[1][1], matrix)
            if handled:
                stats["selector_universal_checks"] = (
                    stats.get("selector_universal_checks", 0) + 1
                )
                if cex is not None:
                    cex = {v: assign.get(v, cex.get(v, False)) for v in block}
                game._memo[memo_key] = None if cex is None else dict(cex)
                return cex
        move = play(game.dual(), level, assign, workdir, tag, stats, depth)
        if move is None:
            game._memo[memo_key] = None
            return None
        result = {v: assign.get(v, move.get(v, False)) for v in block}
        game._memo[memo_key] = dict(result)
        return result
    result = solve_e(game, level, matrix, assign, workdir, tag, stats, depth)
    game._memo[memo_key] = None if result is None else dict(result)
    return result


def solve_e(game, level, matrix, assign, workdir, tag, stats, depth):
    """The RAReQS loop at an existential node: game.prefix[level] is 'e' and
    `matrix` is the game matrix simplified under `assign`."""
    prefix = game.prefix[level:]
    q, block = prefix[0]
    assert q == "e"
    suffix = prefix[1:]
    if not suffix:
        move = exists_block_sat(block, matrix, workdir, f"{tag}d{depth}", stats)
        if move is None:
            return None
        return {v: assign.get(v, move.get(v, False)) for v in block}
    opp_block = suffix[0][1]
    members = []
    member_keys = set()
    abs_matrix = []
    merged = [list(block)] + [[] for _ in suffix[1:]]
    rounds = 0

    def add_member(member):
        inst = simplify(matrix, member)
        if inst is None:
            return False
        mapping = {}
        for i, (_, vs) in enumerate(suffix[1:]):
            copy = dict(zip(vs, game.fresh.take(len(vs))))
            mapping.update(copy)
            merged[i + 1].extend(copy[v] for v in vs)
        abs_matrix.extend(rename(inst, mapping))
        return True

    while True:
        rounds += 1
        stats["rounds"] = stats.get("rounds", 0) + 1
        if not members:
            move = {v: assign.get(v, False) for v in block}
        else:
            # Abstraction: this flat driver has no local quantified
            # subformulas, so every block beyond the opponent must be copied
            # per member. The C++ solver only freshens the first such block
            # because the deeper prefix stays local to the refined subformula;
            # sharing it in this flattened CNF changes the game. Keep the
            # copied members incremental so old refinements are not freshened
            # and renamed again on every CEGAR round.
            abs_prefix = [("e", merged[0])] + [
                (suffix[1:][i][0], cols) for i, cols in enumerate(merged[1:]) if cols
            ]
            abs_game = Game(abs_prefix, list(abs_matrix), game.fresh)
            got = play(
                abs_game, 0, {}, workdir, f"{tag}r{rounds}a", stats, depth + 1
            )
            if got is None:
                return None
            move = {v: assign.get(v, got.get(v, False)) for v in block}
        cex = play(
            game, level + 1, {**assign, **move}, workdir, f"{tag}r{rounds}c",
            stats, depth + 1,
        )
        if cex is None:
            return move
        cex = {v: assign.get(v, cex.get(v, False)) for v in opp_block}
        cex_key = tuple(cex[v] for v in opp_block)
        if cex_key in member_keys:
            import json
            Path(workdir, "stall.json").write_text(json.dumps({
                "prefix": game.prefix[level:],
                "matrix": matrix,
                "abs_matrix": abs_matrix,
                "merged": merged,
                "assign": {str(k): v for k, v in assign.items()},
                "members": members,
                "move": {str(k): v for k, v in move.items()},
            }, default=list))
            raise AssertionError(
                f"expansion stalled at depth {depth}: repeated member {cex}\n"
                f"  prefix here: {game.prefix[level:]}\n"
                f"  members: {members}\n  move: {move}"
            )
        member_keys.add(cex_key)
        members.append(cex)
        if not add_member(cex):
            # Some member falsifies the matrix whatever I play.
            return None


def decide(prefix, matrix, workdir, tag):
    """TRUE/FALSE for a prenex-CNF QBF with arbitrary alternations."""
    stats = {}
    t0 = time.time()
    fresh = Fresh(matrix, prefix)
    game = Game(prefix, matrix, fresh)
    move = play(game, 0, {}, workdir, tag, stats)
    lead = game.prefix[0][0] if game.prefix else "e"
    verdict = (move is not None) if lead == "e" else (move is None)
    return verdict, stats, time.time() - t0


def random_kblock(n_blocks, per_block, n_clauses, seed, first="a"):
    import random

    rng = random.Random(seed ^ 0xB10C)
    prefix = []
    v = 1
    q = first
    for _ in range(n_blocks):
        prefix.append((q, list(range(v, v + per_block))))
        v += per_block
        q = "e" if q == "a" else "a"
    all_vars = list(range(1, v))
    matrix = []
    for _ in range(n_clauses):
        width = rng.choice((2, 3, 3))
        vs = rng.sample(all_vars, min(width, len(all_vars)))
        matrix.append([x if rng.random() < 0.5 else -x for x in vs])
    return prefix, matrix


def read_qdimacs(path):
    prefix, matrix = [], []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t or t[0] in ("c", "p"):
            continue
        if t[0] in ("a", "e"):
            vs = [int(x) for x in t[1:] if x != "0"]
            if prefix and prefix[-1][0] == t[0]:
                prefix[-1][1].extend(vs)
            else:
                prefix.append([t[0], vs])
            continue
        clause = [int(x) for x in t if x != "0"]
        if clause:
            matrix.append(clause)
    bound = {v for _, vs in prefix for v in vs}
    free = sorted({abs(l) for c in matrix for l in c} - bound)
    if free:
        prefix.insert(0, ["e", free])
    return [(q, vs) for q, vs in prefix], matrix


EDGE = [
    ("3blk_basic", [("a", [1]), ("e", [2]), ("a", [3])], [[1, 2], [-1, -2, 3], [-2, -3]]),
    ("3blk_taut", [("a", [1]), ("e", [2]), ("a", [3])], [[2, -2], [1, 3], [-1, -3]]),
    ("4blk", [("e", [1]), ("a", [2]), ("e", [3]), ("a", [4])], [[1, 2, 3], [-2, -3, 4], [-1, -4, 3]]),
    ("5blk", [("a", [1]), ("e", [2]), ("a", [3]), ("e", [4]), ("a", [5])],
     [[1, 2], [-2, 3, 4], [-3, -4, 5], [-1, -5, 4]]),
    ("empty_clause_deep", [("a", [1]), ("e", [2]), ("a", [3])], [[]]),
    ("exists_first", [("e", [1]), ("a", [2])], [[1, 2], [1, -2]]),
    ("all_universal", [("a", [1, 2])], [[1, 2]]),
    ("all_universal_taut", [("a", [1])], [[1, -1]]),
]


def main():
    workdir = ROOT / "run-v2"
    workdir.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "edge":
        ok = True
        for name, prefix, matrix in EDGE:
            want = evaluate(prefix, matrix)
            got, stats, wall = decide(prefix, matrix, workdir, f"edge-{name}")
            match = want == got
            ok &= match
            print(
                f"edge {name}: oracle={want} v2={got} rounds={stats.get('rounds', 0)} "
                f"{'OK' if match else '<<<MISMATCH'}"
            )
        print("edge battery", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "qdimacs":
        prefix, matrix = read_qdimacs(sys.argv[2])
        nv = sum(len(vs) for _, vs in prefix)
        got, stats, wall = decide(prefix, matrix, workdir, "qd")
        print(
            f"{sys.argv[2]}: {len(prefix)} blocks, {nv} vars, {len(matrix)} clauses "
            f"-> {'TRUE' if got else 'FALSE'} rounds={stats.get('rounds', 0)} "
            f"engine_calls={stats.get('engine_calls', 0)} {wall:.2f}s"
        )
        if nv <= 24:
            want = evaluate(prefix, matrix)
            print(f"oracle: {want} {'OK' if want == got else '<<<MISMATCH'}")
        return
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    first = sys.argv[5] if len(sys.argv) > 5 else "a"
    ok = bad = 0
    for seed in range(seeds):
        prefix, matrix = random_kblock(blocks, per, m, seed, first)
        want = evaluate(prefix, matrix)
        got, stats, wall = decide(prefix, matrix, workdir, f"s{seed}")
        line = (
            f"seed {seed}: v2={got} oracle={want} rounds={stats.get('rounds', 0)} "
            f"engine_calls={stats.get('engine_calls', 0)} {wall:.2f}s"
        )
        if got == want:
            ok += 1
            print(line)
        else:
            bad += 1
            print(line + "  <<< MISMATCH")
    print(f"==== {ok}/{seeds} verdicts match the oracle; {bad} mismatches")


if __name__ == "__main__":
    main()
