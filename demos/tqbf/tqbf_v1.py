#!/usr/bin/env python3
"""L8 demonstrator v1: one engine program per CEGIS round. Inside the
program, quiesce barriers stage the whole round: clause-satisfaction facts,
forbidden schemas by guard-negation (a stored (clsat j) drops the schema
emission -- negation as absence), k depth strata growing Y-candidates as
cons lists under nogood guards, then the A-side member fold over the
expansion set via a driver-built member chain, and verdict facts. The
driver's between-round work is bookkeeping only: install the found y as
member facts and re-run.

Differential law: v1 verdicts == v0 verdicts == the recursive oracle.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbf_oracle import evaluate, planted_qbf, random_qbf
from tqbf_cegis import cegis as cegis_v0

ROOT = Path(__file__).resolve().parent
MORK = os.environ.get("MORK_BIN", "mork")


def cons_term(cells):
    """cells outermost-first = y_k .. y_1 (head is the LAST bit appended)."""
    t = "nil"
    for c in cells:
        t = f"(cons {c} {t})"
    return t


def clause_tables(matrix, a_index, e_index, k_e):
    """Static per-instance tables: clause ids, X-literal rows, the clause's
    Y-side forbid schema (cons pattern; None when the clause has no Y-part),
    and the clause's X-side impose pattern (cells over 0/1/u; None when some
    X-var repeats with conflicting polarity)."""
    rows = []
    for j, clause in enumerate(matrix):
        xlits, cells, ok = [], ["u"] * len(a_index), True
        constrained = {}
        taut = False
        for lit in clause:
            v = abs(lit)
            if v in a_index:
                want = "0" if lit > 0 else "1"
                i = a_index[v]
                if cells[i] not in ("u", want):
                    ok = False
                cells[i] = want
                xlits.append((i, "1" if lit > 0 else "0"))
            else:
                want = "0" if lit > 0 else "1"
                i = e_index[v]
                if constrained.get(i, want) != want:
                    taut = True
                constrained[i] = want
        if taut:
            # Both polarities of a Y-variable: the clause is always satisfied.
            # No forbid schema, no dead-clause marker, no impose row (the
            # member loop's y_saved check already skips it).
            rows.append((f"c{j}", xlits, [], None, True))
            continue
        if constrained:
            # One nil-terminated pattern per depth d from (top constrained
            # position + 1) to k: the guard key at depth d is exactly d deep,
            # and pruning must bite at EVERY depth past the last constrained
            # position, not only at full depth (measured: full-depth-only
            # schemas explore the whole 2^d tree until the final level).
            top = max(constrained)
            schemas = []
            for d in range(top + 1, k_e + 1):
                t = "nil"
                for pos in range(d):
                    t = f"(cons {constrained.get(pos, f'$w{pos}')} {t})"
                schemas.append(t)
        else:
            schemas = None
        rows.append((f"c{j}", xlits, schemas, cells if ok else None, bool(constrained)))
    return rows


def round_program(matrix, a_list, e_list, members, x_probe):
    """One CEGIS round as a single barrier-staged program.

    Inputs: members = list of y-assignments (dicts) in installation order;
    x_probe = the X-assignment to project B against (previous round's x*).
    Outputs read from the dump: (xsurv (v ...)) A-side survivors,
    (ycand K (cons ...)) B-side full-depth candidates.
    """
    a_index = {v: i for i, v in enumerate(a_list)}
    e_index = {v: i for i, v in enumerate(e_list)}
    k_a, k_e = len(a_list), len(e_list)
    rows = clause_tables(matrix, a_index, e_index, k_e)

    facts = ["(bit 0)", "(bit 1)"]
    top = max(k_a, k_e, len(members)) + 2
    facts += [f"(incFn {i} {i + 1})" for i in range(top)]
    facts += [f"(lte {i} {j})" for i in range(top) for j in range(i, top)]
    facts += ["(mrg u u u)", "(mrg u 0 0)", "(mrg u 1 1)",
              "(mrg 0 u 0)", "(mrg 1 u 1)", "(mrg 0 0 0)", "(mrg 1 1 1)"]

    # Instance tables.
    for cid, xlits, schemas, _cells, has_y in rows:
        facts.append(f"(cl {cid})")
        for i, b in xlits:
            facts.append(f"(xlit {cid} {i} {b})")
        if schemas is None:
            facts.append(f"(fempty {cid})")
        else:
            for schema in schemas:
                facts.append(f"(fschema {cid} {schema})")

    # The probe X (previous x*): clause satisfaction under it.
    for v, val in x_probe.items():
        facts.append(f"(xcell {a_index[v]} {'1' if val else '0'})")

    # Members: per member m, per clause, the X-impose pattern is ACTIVE
    # unless the member's y satisfies the clause's Y-part. Both computed
    # here (driver bookkeeping, static per round).
    for m, y in enumerate(members):
        for cid, xlits, _schema, cells, _has_y in rows:
            if cells is None:
                continue
            y_saved = False
            for lit in matrix[int(cid[1:])]:
                v = abs(lit)
                if v in e_index and (lit > 0) == y[v]:
                    y_saved = True
                    break
            if not y_saved:
                facts.append(f"(imp {m} (v {' '.join(cells)}))")
        facts.append(f"(mnext {m} {m + 1})")

    n_members = len(members)
    xcols = " ".join(f"$c{i}" for i in range(k_a))
    pcols = " ".join(f"$p{i}" for i in range(k_a))
    ocols = " ".join(f"$o{i}" for i in range(k_a))
    mrgs = "\n".join(f"     (mrg $c{i} $p{i} $o{i})" for i in range(k_a))
    ucells = " ".join(["u"] * k_a)

    rules = f"""
(ycand 0 nil)
(depth 0)
(xpart 0 (v {ucells}))
((clsat rule)
  (, ((clsat rule) $sp $st)
     (xcell $i $b)
     (xlit $j $i $b))
  (, (clsat $j)))
((forbid rule)
  (, ((forbid rule) $sp $st)
     (fschema $j $pat))
  (O (guard (clsat $j) (forbid $pat))))
((deadclause rule)
  (, ((deadclause rule) $sp $st)
     (fempty $j))
  (O (guard (clsat $j) (deadx $j))))
((extend rule)
  (, ((extend rule) $sp $st)
     (ycand $d $prev)
     (depth $d)
     (incFn $d $d1)
     (lte $d1 {k_e})
     (bit $nb))
  (O (guard (forbid (cons $nb $prev)) (ycand $d1 (cons $nb $prev)))))
((bstep rule)
  (, ((bstep rule) $ap $at)
     ((extend rule) $ep $et)
     (depth $d)
     (lte $d {k_e - 1})
     (incFn $d $d1))
  (O (- (depth $d))
     (+ (depth $d1))
     (+ (exec (30 extend) $ep $et))
     (+ (exec (quiesce 31 bstep) $ap $at))))
((achoose rule)
  (, ((achoose rule) $sp $st)
     (xpart $m (v {xcols}))
     (imp $m (v {pcols}))
{mrgs}
     (incFn $m $m1))
  (, (xpart $m1 (v {ocols}))))
((astep rule)
  (, ((astep rule) $ap $at)
     ((achoose rule) $cp $ct)
     (acursor $m)
     (lte $m {max(n_members - 1, 0)})
     (incFn $m $m1))
  (O (- (acursor $m))
     (+ (acursor $m1))
     (+ (exec (50 achoose) $cp $ct))
     (+ (exec (quiesce 51 astep) $ap $at))))
(acursor 0)
((xfinal rule)
  (, ((xfinal rule) $sp $st)
     (xpart {n_members} (v {xcols})))
  (, (xsurv (v {xcols}))))
(exec (10 init)
  (, ((clsat rule) $c1 $c2)
     ((forbid rule) $f1 $f2)
     ((deadclause rule) $d1 $d2))
  (, (armed init)
     (exec (11 clsat) $c1 $c2)
     (exec (12 forbid) $f1 $f2)
     (exec (13 deadclause) $d1 $d2)))
(exec (20 binit)
  (, ((extend rule) $ep $et)
     ((bstep rule) $bp $bt))
  (, (armed binit)
     (exec (30 extend) $ep $et)
     (exec (quiesce 31 bstep) $bp $bt)))
(exec (40 ainit)
  (, ((achoose rule) $cp $ct)
     ((astep rule) $ap $at))
  (, (armed ainit)
     (exec (50 achoose) $cp $ct)
     (exec (quiesce 51 astep) $ap $at)))
(exec (90 finalize)
  (, ((xfinal rule) $xp $xt))
  (, (armed finalize)
     (exec (quiesce 91 xfinal) $xp $xt)))
"""
    return "\n".join(facts) + rules


def parse_cons(term):
    bits = []
    t = term.strip()
    while t.startswith("(cons "):
        inner = t[len("(cons "):-1]
        bits.append(inner[0])
        t = inner[2:].strip()
    return list(reversed(bits))  # position 0 first


def cegis_v1(prefix, matrix, workdir, tag):
    (qa, a_list), (qe, e_list) = prefix
    assert qa == "a" and qe == "e"
    if not a_list or not e_list:
        # Degenerate blocks make zero-column programs; v0's arms handle both
        # exactly, so route them there.
        got, _, _, _ = cegis_v0(prefix, matrix, workdir, f"{tag}-degen")
        return got, 0, 0.0
    members = [{v: False for v in e_list}]
    x_probe = {v: False for v in a_list}
    rounds = 0
    t0 = time.time()
    while True:
        rounds += 1
        src = workdir / f"{tag}-v1r{rounds}.mm2"
        dump = workdir / f"{tag}-v1r{rounds}.dump"
        src.write_text(round_program(matrix, a_list, e_list, members, x_probe))
        subprocess.run([MORK, "run", str(src), str(dump)], check=True, capture_output=True)
        lines = dump.read_text().splitlines()
        xsurvs = [l for l in lines if l.startswith("(xsurv ")]
        if not xsurvs:
            return True, rounds, time.time() - t0
        cells = xsurvs[0][len("(xsurv (v "):-2].split()
        x_star = {a_list[i]: cells[i] == "1" for i in range(len(a_list))}
        # Semantic invariant: x* must falsify the matrix under EVERY member.
        for y_i in members:
            assign = {**x_star, **y_i}
            falsified = any(
                all((lit > 0) != assign[abs(lit)] for lit in clause)
                for clause in matrix
            )
            assert falsified, (
                f"A-side unsound: x*={x_star} does not falsify member {y_i} "
                f"(cells={cells})"
            )
        # B-answers were computed against x_probe (LAST round's x*), so run
        # one more round with x_probe = x_star to get its fresh y.
        src2 = workdir / f"{tag}-v1r{rounds}b.mm2"
        dump2 = workdir / f"{tag}-v1r{rounds}b.dump"
        src2.write_text(round_program(matrix, a_list, e_list, members, x_star))
        subprocess.run([MORK, "run", str(src2), str(dump2)], check=True, capture_output=True)
        lines2 = dump2.read_text().splitlines()
        if any(l.startswith("(deadx ") for l in lines2):
            # Some X-only clause is false under x*: no y can save the matrix.
            return False, rounds, time.time() - t0
        ycands = [
            l for l in lines2
            if l.startswith(f"(ycand {len(e_list)} ")
        ]
        if not ycands:
            return False, rounds, time.time() - t0
        bits = parse_cons(ycands[0][len(f"(ycand {len(e_list)} "):-1])
        y_new = {e_list[i]: bits[i] == "1" for i in range(len(e_list))}
        if y_new in members:
            raise AssertionError(f"v1 progress violated: repeated member {y_new}")
        members.append(y_new)
        x_probe = x_star


EDGE_CASES = [
    ("taut_y", [("a", [1]), ("e", [2, 3])], [[1, 2], [-1, -2], [-3], [-3, 3]]),
    ("no_forall_empty", [("a", []), ("e", [1])], [[]]),
    ("no_forall_sat", [("a", []), ("e", [1])], [[1]]),
    ("empty_clause", [("a", [1]), ("e", [2])], [[]]),
    ("dup_same_pol", [("a", [1]), ("e", [2])], [[2, 2, 1], [-1, -2]]),
    ("taut_x", [("a", [1]), ("e", [2])], [[1, -1], [2]]),
    ("no_exists", [("a", [1]), ("e", [])], [[1]]),
]


def edge_battery(workdir):
    """Adversarial clause shapes (found by review): tautologies, duplicates,
    empty clauses, degenerate quantifier blocks. Every driver change must
    keep this green."""
    ok = True
    for name, prefix, matrix in EDGE_CASES:
        want = evaluate(prefix, matrix)
        v0 = cegis_v0(prefix, matrix, workdir, f"edge-{name}-v0")[0]
        v1, _, _ = cegis_v1(prefix, matrix, workdir, f"edge-{name}-v1")
        match = want == v0 == v1
        ok &= match
        print(f"edge {name}: oracle={want} v0={v0} v1={v1} {'OK' if match else '<<<MISMATCH'}")
    return ok


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    family = sys.argv[4] if len(sys.argv) > 4 else "random"
    workdir = ROOT / "run-v1"
    workdir.mkdir(exist_ok=True)
    if family == "edge":
        sys.exit(0 if edge_battery(workdir) else 1)
    gen = planted_qbf if family == "planted" else random_qbf
    ok = bad = 0
    for seed in range(seeds):
        prefix, matrix = gen(n, n, m, seed)
        want = evaluate(prefix, matrix)
        v0, _, _, _ = cegis_v0(prefix, matrix, workdir, f"x{seed}")
        got, rounds, wall = cegis_v1(prefix, matrix, workdir, f"s{seed}")
        line = f"seed {seed}: v1={got} v0={v0} oracle={want} rounds={rounds} {wall:.2f}s"
        if got == want == v0:
            ok += 1
            print(line)
        else:
            bad += 1
            print(line + "  <<< MISMATCH")
    print(f"==== {ok}/{seeds} triple-match (v1 == v0 == oracle); {bad} mismatches")


if __name__ == "__main__":
    main()
