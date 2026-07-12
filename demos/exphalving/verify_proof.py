#!/usr/bin/env python3
"""Independently check that an emitted proof term really derives its theorem.

The MITM rules encode a small calculus, and this re-derives the type of a
proof term from the axiom schemas alone -- it never consults the search, so
it is a genuine oracle over the engine's answer.

Forward proofs (of a formula):
    ax1 : (→ a (→ b a))
    ax2 : (→ (→ a (→ b c)) (→ (→ a b) (→ a c)))
    ax3 : (→ (→ (¬ a) (¬ b)) (→ b a))
    (ax-mp P Q) : B          when P : (→ A B) and Q : A

Context proofs (of a meta-arrow `(-> A B)`: "give me a proof of A, get B"),
which is what the backward search carries:
    I        : (-> A A)                                  the seed
    (mpⁱ F)  : (-> (→ a b) (-> a c))   when F : (-> b c) the backward split
    (F P)    : B                       when F : (-> A B) and P : A, the meet

A final is a context proof whose meta-arrow has been fully discharged, so its
type is the bare target formula.
"""

import sys
from itertools import count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_check import tokenize, parse

fresh = count()


def V():
    return ("?", next(fresh))


def is_var(t):
    return isinstance(t, tuple) and len(t) == 2 and t[0] == "?"


AXIOMS = {
    "ax₁": lambda: (lambda a, b: ("→", a, ("→", b, a)))(V(), V()),
    "ax₂": lambda: (lambda a, b, c: ("→", ("→", a, ("→", b, c)),
                                      ("→", ("→", a, b), ("→", a, c))))(V(), V(), V()),
    "ax₃": lambda: (lambda a, b: ("→", ("→", ("¬", a), ("¬", b)), ("→", b, a)))(V(), V()),
}


def walk(t, s):
    while is_var(t) and t in s:
        t = s[t]
    return t


def occurs(v, t, s):
    t = walk(t, s)
    if t == v:
        return True
    if isinstance(t, tuple) and not is_var(t):
        return any(occurs(v, x, s) for x in t)
    return False


def unify(x, y, s):
    x, y = walk(x, s), walk(y, s)
    if x == y:
        return s
    if is_var(x):
        if occurs(x, y, s):
            return None
        s = dict(s)
        s[x] = y
        return s
    if is_var(y):
        return unify(y, x, s)
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        for a, b in zip(x, y):
            s = unify(a, b, s)
            if s is None:
                return None
        return s
    return None


def resolve(t, s):
    t = walk(t, s)
    if isinstance(t, tuple) and not is_var(t):
        return tuple(resolve(x, s) for x in t)
    return t


def infer(term, s):
    """(type, substitution) or raise."""
    if isinstance(term, str):
        if term in AXIOMS:
            return AXIOMS[term](), s
        if term == "I":
            a = V()
            return ("->", a, a), s
        raise ValueError(f"unknown atom {term}")

    head = term[0]
    if head == "ax-mp" and len(term) == 3:
        pt, s = infer(term[1], s)
        qt, s = infer(term[2], s)
        b = V()
        s2 = unify(pt, ("→", qt, b), s)
        if s2 is None:
            raise ValueError(f"ax-mp does not apply: {pt} to {qt}")
        return b, s2
    if head == "mpⁱ" and len(term) == 2:
        ft, s = infer(term[1], s)
        b, c = V(), V()
        s2 = unify(ft, ("->", b, c), s)
        if s2 is None:
            raise ValueError(f"mpi needs a context, got {ft}")
        a = V()
        return ("->", ("→", a, b), ("->", a, c)), s2
    if len(term) == 2:  # application: (F P)
        ft, s = infer(term[0], s)
        pt, s = infer(term[1], s)
        b = V()
        s2 = unify(ft, ("->", pt, b), s)
        if s2 is None:
            raise ValueError(f"cannot apply {ft} to {pt}")
        return b, s2
    raise ValueError(f"bad term {term}")


def check(dump, target):
    finals = []
    for line in Path(dump).read_text().splitlines():
        if line.startswith("(final "):
            t, _ = parse(tokenize(line))
            finals.append((t[3][1], t[3][2]))
    if not finals:
        return False, "no proof emitted"
    for thm, proof in finals:
        try:
            ty, s = infer(proof, {})
        except (ValueError, RecursionError) as e:
            return False, f"proof does not type-check: {e}"
        got = resolve(ty, s)
        # The derived type must be an INSTANCE of nothing -- it must match the
        # claimed theorem exactly (the target is ground).
        s2 = unify(got, thm, s)
        if s2 is None:
            return False, f"proof derives {got}, not {thm}"
        if resolve(got, s2) != thm:
            return False, f"proof derives {resolve(got, s2)}, not {thm}"
        if thm != target:
            return False, f"final proves {thm}, not the target {target}"
    return True, f"{len(finals)} proof(s) type-check against the axioms and equal the target"


def target_of(prog):
    for line in Path(prog).read_text().splitlines():
        if line.startswith("(target "):
            t, _ = parse(tokenize(line))
            return t[2][1]
    return None


if __name__ == "__main__":
    prog, dump = sys.argv[1], sys.argv[2]
    ok, msg = check(dump, target_of(prog))
    print(("PASS: " if ok else "FAIL: ") + msg)
    sys.exit(0 if ok else 1)
