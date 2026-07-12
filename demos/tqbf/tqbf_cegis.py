#!/usr/bin/env python3
"""L8 demonstrator v0: Pi2 QBF (forall X exists Y . CNF) by flat CEGIS on the
MORK engine, one stratum program per step, verdict checked against the
recursive oracle.

The engine does the hard half of each round: deciding whether some
Y-assignment satisfies the matrix under EVERY learned counterexample. That
sub-search runs as per-depth saturation strata with clause-derived FORBIDDEN
SCHEMAS enforced by the guarded-emit sink: a stored schema (forbid D $ 0 $ 1)
generalizes exactly the extensions that falsify a clause, so the guard walk
kills them at O(path) each -- nogood learning (L3) with schema subsumption
(L4) as one mechanism. The Pi2 check step is linear, no search: a clause
whose Y-part is false under the candidate y* yields a counterexample by
setting its X-literals false.

Layers on display: guarded emit (nogood tables), quiescence-style staging
(driver strata, the barrier law), retrieval joins (clause-table lookups),
and the oracle differential as the gate.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qbf_oracle import evaluate, planted_qbf, random_qbf

ROOT = Path(__file__).resolve().parent
MORK = os.environ.get("MORK_BIN", "mork")


def project_clause(clause, cex, a_vars, e_vars):
    """Clause under a universal assignment: None if satisfied by cex (or a
    tautology -- both polarities of some Y-variable make it always true),
    else the list of Y-literals that must save it."""
    y_lits = []
    for lit in clause:
        v = abs(lit)
        if v in a_vars:
            if (lit > 0) == cex[v]:
                return None
            continue
        if -lit in y_lits:
            return None
        y_lits.append(lit)
    return y_lits


def forbid_schema(y_lits, e_index, depth):
    """The assignment pattern that falsifies all y_lits: position i fixed to
    the falsifying bit, $ elsewhere. Only meaningful once every constrained
    position is <= depth."""
    cells = []
    top = 0
    constrained = {}
    for lit in y_lits:
        i = e_index[abs(lit)]
        constrained[i] = 0 if lit > 0 else 1
        top = max(top, i)
    if top >= depth:
        return None
    for i in range(depth):
        cells.append(str(constrained[i]) if i in constrained else f"$w{i}")
    # One compound key: the guard spec is (TABLE key), a single key argument.
    return "(forbid (v " + " ".join(cells) + "))"


def solve_exists(matrix, cexes, a_vars, e_vars, workdir, tag):
    """Engine strata: find one Y-assignment satisfying matrix under every
    counterexample, or report none. Returns (assignment dict | None, stats)."""
    k = len(e_vars)
    e_index = {v: i for i, v in enumerate(e_vars)}
    # Collect projected clauses over Y across ALL counterexamples; an empty
    # projection means that cex already falsifies the matrix outright.
    projected = []
    for cex in cexes:
        for clause in matrix:
            yl = project_clause(clause, cex, a_vars, e_vars)
            if yl is None:
                continue
            if not yl:
                return None, {"reason": "empty-projection"}
            projected.append(tuple(sorted(set(yl))))
    projected = sorted(set(projected))

    survivors = ["(ycand)"]
    stats = {"strata": 0, "candidates": 0, "dropped": 0}
    for depth in range(1, k + 1):
        forbids = []
        for yl in projected:
            f = forbid_schema(yl, e_index, depth)
            if f is not None:
                forbids.append(f)
        forbids = sorted(set(forbids))
        prev = "\n".join(survivors)
        prev_cols = " ".join(f"$b{i}" for i in range(depth - 1))
        out_cols = (prev_cols + " " if prev_cols else "") + "$nb"
        program = f"""{prev}
{chr(10).join(forbids)}
(bit 0)
(bit 1)
(exec (10 extend)
      (, (ycand{(' ' + prev_cols) if prev_cols else ''})
         (bit $nb))
      (O (guard (forbid (v {out_cols})) (ycand {out_cols}))))
"""
        src = workdir / f"{tag}-d{depth}.mm2"
        dump = workdir / f"{tag}-d{depth}.dump"
        src.write_text(program)
        subprocess.run([MORK, "run", str(src), str(dump)], check=True, capture_output=True)
        survivors = [
            l for l in dump.read_text().splitlines() if l.startswith("(ycand ")
            and len(l.split()) == depth + 1
        ]
        stats["strata"] += 1
        stats["candidates"] += len(survivors)
        if not survivors:
            return None, stats
    bits = survivors[0].strip("()").split()[1:]
    return {e_vars[i]: bits[i] == "1" for i in range(k)}, stats


def check_forall(matrix, y_star, a_vars, e_vars):
    """Pi2 check, linear: a clause whose Y-part is false under y_star gives a
    counterexample (X-literals set false, rest False)."""
    for clause in matrix:
        y_ok = any(
            (lit > 0) == y_star[abs(lit)]
            for lit in clause
            if abs(lit) in y_star
        )
        if y_ok:
            continue
        x_lits = [lit for lit in clause if abs(lit) in a_vars]
        cex = {v: False for v in a_vars}
        for lit in x_lits:
            cex[abs(lit)] = lit < 0
        return cex
    return None


def impose_pattern(clause, y_assign, a_index, k, a_vars):
    """The X-cell pattern that falsifies `clause` given y_assign: None if the
    clause is already saved by its Y-part (cannot be the falsified one), else
    cells over {0,1,u} forcing every X-literal false."""
    cells = ["u"] * k
    for lit in clause:
        v = abs(lit)
        if v in a_vars:
            want = "0" if lit > 0 else "1"
            i = a_index[v]
            if cells[i] not in ("u", want):
                return None
            cells[i] = want
        else:
            if (lit > 0) == y_assign[v]:
                return None
    return cells


def solve_forall(matrix, y_set, a_list, e_vars, workdir, tag):
    """Engine strata (sub-search A): find one X-assignment pattern falsifying
    the matrix under EVERY y in y_set -- per-y clause choice, merged cell-wise
    through (mrg ...) tables so incompatible choices die inside the join."""
    k = len(a_list)
    a_index = {v: i for i, v in enumerate(a_list)}
    a_vars = set(a_list)
    if k == 0:
        # Degenerate universal block: the empty x falsifies a member iff some
        # clause is all-false under it (no X-literals exist to falsify).
        for y_assign in y_set:
            if not any(
                all((lit > 0) != y_assign[abs(lit)] for lit in clause)
                for clause in matrix
            ):
                return None, {"reason": "y-unfalsifiable"}
        return {}, {"strata": 0, "candidates": 0}
    per_y_imposes = []
    for y_assign in y_set:
        imposes = []
        for clause in matrix:
            cells = impose_pattern(clause, y_assign, a_index, k, a_vars)
            if cells is not None:
                imposes.append(cells)
        if not imposes:
            return None, {"reason": "y-unfalsifiable"}
        per_y_imposes.append(sorted(set(map(tuple, imposes))))

    survivors = ["(xpart " + " ".join(["u"] * k) + ")"]
    stats = {"strata": 0, "candidates": 0}
    cols = " ".join(f"$c{i}" for i in range(k))
    outs = " ".join(f"$o{i}" for i in range(k))
    mrg_rows = ["(mrg u u u)", "(mrg u 0 0)", "(mrg u 1 1)",
                "(mrg 0 u 0)", "(mrg 1 u 1)", "(mrg 0 0 0)", "(mrg 1 1 1)"]
    for depth, imposes in enumerate(per_y_imposes, 1):
        imp_rows = [
            "(imp " + " ".join(c) + ")" for c in imposes
        ]
        mrgs = "\n".join(
            f"         (mrg $c{i} $p{i} $o{i})" for i in range(k)
        )
        pcols = " ".join(f"$p{i}" for i in range(k))
        program = "\n".join(survivors) + "\n" + "\n".join(imp_rows) + "\n" + "\n".join(mrg_rows) + f"""
(exec (10 choose)
      (, (xpart {cols})
         (imp {pcols})
{mrgs})
      (, (xnext {outs})))
"""
        src = workdir / f"{tag}-a{depth}.mm2"
        dump = workdir / f"{tag}-a{depth}.dump"
        src.write_text(program)
        subprocess.run([MORK, "run", str(src), str(dump)], check=True, capture_output=True)
        survivors = [
            "(xpart " + l[len("(xnext "):]
            for l in dump.read_text().splitlines()
            if l.startswith("(xnext ")
        ]
        stats["strata"] += 1
        stats["candidates"] += len(survivors)
        if not survivors:
            return None, stats
    cells = survivors[0].strip("()").split()[1:]
    return {a_list[i]: cells[i] == "1" for i in range(k)}, stats


def cegis(prefix, matrix, workdir, tag):
    """RAReQS polarity for forall-X exists-Y: refute by expansion over Y.
    S is the expansion set; sub-search A (engine) finds x falsifying every
    y in S; sub-search B (engine) answers x with a fresh y; no fresh y means
    FALSE, no falsifying x means TRUE."""
    (qa, a_list), (qe, e_list) = prefix
    assert qa == "a" and qe == "e"
    a_vars, e_vars = set(a_list), list(e_list)
    y_set = [{v: False for v in e_list}]
    rounds = 0
    t0 = time.time()
    agg = {"strata": 0, "candidates": 0}
    while True:
        rounds += 1
        x_star, stats_a = solve_forall(matrix, y_set, list(a_list), e_vars, workdir, f"{tag}-r{rounds}")
        for key in agg:
            agg[key] += stats_a.get(key, 0)
        if x_star is None:
            return True, rounds, agg, time.time() - t0
        y_new, stats_b = solve_exists(matrix, [x_star], a_vars, e_vars, workdir, f"{tag}-r{rounds}b")
        for key in agg:
            agg[key] += stats_b.get(key, 0)
        if y_new is None:
            return False, rounds, agg, time.time() - t0
        if y_new in y_set:
            raise AssertionError(
                f"CEGIS progress violated: repeated expansion {y_new} "
                f"(x*={x_star})"
            )
        y_set.append(y_new)


def main():
    workdir = ROOT / "run"
    workdir.mkdir(exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    ok = mismatches = 0
    family = sys.argv[4] if len(sys.argv) > 4 else "random"
    for seed in range(seeds):
        gen = planted_qbf if family == "planted" else random_qbf
        prefix, matrix = gen(n, n, m, seed)
        want = evaluate(prefix, matrix)
        got, rounds, agg, wall = cegis(prefix, matrix, workdir, f"s{seed}")
        tagline = f"seed {seed}: engine={got} oracle={want} rounds={rounds} strata={agg['strata']} cands={agg['candidates']} {wall:.2f}s"
        if got == want:
            ok += 1
            print(tagline)
        else:
            mismatches += 1
            print(tagline + "  <<< MISMATCH")
    print(f"==== {ok}/{seeds} verdicts match the oracle; {mismatches} mismatches")


if __name__ == "__main__":
    main()
