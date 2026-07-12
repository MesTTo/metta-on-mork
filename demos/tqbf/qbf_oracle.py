#!/usr/bin/env python3
"""Trusted QBF evaluator + instance generator for the L8 demonstrator.

The evaluator decides a prenex-CNF QBF by direct recursion over the prefix
(exponential, fine at demonstrator sizes; it is the ORACLE, so clarity beats
speed). Instances are (prefix, matrix): prefix = [('a'|'e', [vars...])...],
matrix = list of clauses, clause = list of signed ints (DIMACS-style).
"""

import random
import sys


def evaluate(prefix, matrix, assignment=None):
    if assignment is None:
        assignment = {}
    if not prefix:
        return all(
            any(
                (lit > 0) == assignment[abs(lit)]
                for lit in clause
                if abs(lit) in assignment
            )
            or any(abs(lit) not in assignment for lit in clause)
            and _clause_open_sat(clause, assignment)
            for clause in matrix
        )
    quant, vars_ = prefix[0]
    rest = prefix[1:]
    if not vars_:
        return evaluate(rest, matrix, assignment)
    v, tail = vars_[0], vars_[1:]
    outcomes = (
        evaluate([(quant, tail)] + rest, matrix, {**assignment, v: b})
        for b in (False, True)
    )
    return all(outcomes) if quant == "a" else any(outcomes)


def _clause_open_sat(clause, assignment):
    # A clause with unassigned literals cannot be judged false; at the leaf
    # every variable is assigned, so this only guards partial evaluation.
    return any(
        abs(lit) not in assignment or (lit > 0) == assignment[abs(lit)]
        for lit in clause
    )


def random_qbf(n_forall, n_exists, n_clauses, seed, order="ae"):
    """Random 2-block QBF. order='ae' gives forall-exists (Pi2, the CEGIS
    target); 'ea' gives exists-forall."""
    rng = random.Random(seed)
    a_vars = list(range(1, n_forall + 1))
    e_vars = list(range(n_forall + 1, n_forall + n_exists + 1))
    all_vars = a_vars + e_vars
    matrix = []
    for _ in range(n_clauses):
        width = rng.choice((2, 3, 3))
        vs = rng.sample(all_vars, min(width, len(all_vars)))
        matrix.append([v if rng.random() < 0.5 else -v for v in vs])
    if order == "ae":
        prefix = [("a", a_vars), ("e", e_vars)]
    else:
        prefix = [("e", e_vars), ("a", a_vars)]
    return prefix, matrix


def planted_qbf(n_forall, n_exists, n_clauses, seed):
    """TRUE-by-construction Pi2 instance: plant a Skolem function
    y_j = x_{s_j} XOR b_j, then keep only random clauses consistent with it
    (checked exhaustively over X at generator sizes)."""
    rng = random.Random(seed ^ 0x5EED)
    a_vars = list(range(1, n_forall + 1))
    e_vars = list(range(n_forall + 1, n_forall + n_exists + 1))
    plant = {y: (rng.choice(a_vars), rng.random() < 0.5) for y in e_vars}

    def y_val(y, xbits):
        src, flip = plant[y]
        return xbits[src] ^ flip

    def clause_ok(clause):
        for mask in range(1 << n_forall):
            xbits = {a_vars[i]: bool(mask >> i & 1) for i in range(n_forall)}
            sat = False
            for lit in clause:
                v = abs(lit)
                val = xbits[v] if v in xbits else y_val(v, xbits)
                if (lit > 0) == val:
                    sat = True
                    break
            if not sat:
                return False
        return True

    all_vars = a_vars + e_vars
    matrix = []
    guard = 0
    while len(matrix) < n_clauses and guard < 10000:
        guard += 1
        width = rng.choice((2, 3, 3))
        vs = rng.sample(all_vars, min(width, len(all_vars)))
        clause = [v if rng.random() < 0.5 else -v for v in vs]
        if clause_ok(clause):
            matrix.append(clause)
    return [("a", a_vars), ("e", e_vars)], matrix


def main():
    # Self-check: the evaluator on hand-verifiable instances.
    # forall x . exists y . (x or y) and (not x or not y)  -- y := not x. TRUE.
    assert evaluate([("a", [1]), ("e", [2])], [[1, 2], [-1, -2]]) is True
    # forall x . exists y . x  -- x is universal and unsupported. FALSE.
    assert evaluate([("a", [1]), ("e", [2])], [[1]]) is False
    # exists y . forall x . (x or y)  -- y := true. TRUE.
    assert evaluate([("e", [2]), ("a", [1])], [[1, 2]]) is True
    # forall x exists y . (x iff y) as CNF. TRUE.
    assert evaluate([("a", [1]), ("e", [2])], [[-1, 2], [1, -2]]) is True
    counts = {True: 0, False: 0}
    for seed in range(40):
        pfx, mat = random_qbf(5, 5, 18, seed)
        counts[evaluate(pfx, mat)] += 1
    print(f"oracle self-check OK; random Pi2 5+5/18: {counts}")


if __name__ == "__main__":
    main()
