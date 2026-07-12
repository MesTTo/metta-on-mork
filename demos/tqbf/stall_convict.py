#!/usr/bin/env python3
"""Debugging aid: replay a stall.json bundle (written by solve_e's repeated-
member assertion) and brute-force the position to convict a side: the check
(returned a non-winning or spurious counterexample) or the abstraction
(admitted a refuted move)."""

import itertools
import json
import sys
from pathlib import Path


def val(pfx, mat, asg):
    m2 = []
    for c in mat:
        lits = []
        sat = False
        for l in c:
            v = abs(l)
            if v in asg:
                if (l > 0) == asg[v]:
                    sat = True
                    break
            else:
                lits.append(l)
        if sat:
            continue
        if not lits:
            return False
        m2.append(lits)
    if not m2:
        return True
    if not pfx:
        return False
    (q, vs), rest = pfx[0], pfx[1:]
    res = [
        val(rest, m2, {**asg, **dict(zip(vs, bits))})
        for bits in itertools.product([False, True], repeat=len(vs))
    ]
    return all(res) if q == "a" else any(res)


def main(path):
    b = json.loads(Path(path).read_text())
    pfx = [(q, [int(v) for v in vs]) for q, vs in b["prefix"]]
    mat = [[int(l) for l in c] for c in b["matrix"]]
    assign = {int(k): v for k, v in b["assign"].items()}
    move = {int(k): v for k, v in b["move"].items()}
    mem = [
        {int(k): v for k, v in (m.items() if isinstance(m, dict) else m)}
        for m in b["members"]
    ]
    print("prefix:", pfx)
    print("assign:", assign)
    print("matrix:", mat)
    print("move:", move, "members:", mem)
    print("node value (leading player wins?):", val(pfx, mat, {}))
    print("after move (opp to play):", val(pfx[1:], mat, move))
    m0 = mem[-1]
    print("after move+repeated-member:", val(pfx[2:], mat, {**move, **m0}))
    opp = pfx[1][1]
    for bits in itertools.product([False, True], repeat=len(opp)):
        o = dict(zip(opp, bits))
        print(f"  opp {o}: value {val(pfx[2:], mat, {**move, **o})}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run-v2/stall.json")
