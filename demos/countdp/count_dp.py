#!/usr/bin/env python3
"""Derivation counting without enumeration (L5): dynamic programming over
(theorem-schema, size) states.

dcount(T, k) = number of distinct mp-derivations of schema T at exact size k.
Recurrence: dcount(T, 1) = 1 for each axiom; for k > 1,
  dcount(T, k) = sum over i + j + 1 = k, over unifying pairs
                 (F at i with F = (-> A B), X at j with X ~ A, result B' = T)
                 of dcount(F, i) * dcount(X, j).

The ENGINE does the hard part: each stratum runs one MM2 program whose join
unifies (fwd-count I FTHM NF) x (fwd-count J XTHM NX) under i + j + 1 = K and
emits one contribution fact per derivation pair, with the count product
computed in-engine by product_i64. The driver only sums contribution numbers
per resulting theorem between strata (O(contributions) trivial arithmetic) and
seeds the next stratum -- the staged pipeline whose equivalence to barrier
programs is the quiescence feature's own differential law.

Oracle: in the baseline forward enumeration every stored fact is one distinct
derivation (proof terms are derivation trees), so sum_T dcount(T, K) must equal
the baseline dump's per-K fact count exactly.
"""

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MITM = Path(os.environ.get("MITM_DUMPS", ROOT / "oracle"))
MORK = os.environ.get("MORK_BIN", "mork")

AXIOMS = [
    "(→ $𝜑 (→ $𝜓 $𝜑))",
    "(→ (→ $𝜑 (→ $𝜓 $𝜒)) (→ (→ $𝜑 $𝜓) (→ $𝜑 $𝜒)))",
    "(→ (→ (¬ $𝜑) (¬ $𝜓)) (→ $𝜓 $𝜑))",
]

STRATUM_PROGRAM = """;; One DP stratum: contributions to size {k} from pairs i + j + 1 = {k}.
{count_facts}
{pairs}
(exec (10 contrib)
      (, (dp-pair $i $j)
         (fwd-count $i (→ $fa $fb) $fs $nf)
         (fwd-count $j $fa $xs $nx))
      (O (pure (contrib $fb $i $j $fs $xs $prod) $prod
               (i64_to_string (product_i64 (i64_from_string $nf)
                                           (i64_from_string $nx))))))
"""


def tokenize(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "()":
            toks.append(c)
            i += 1
        else:
            j = i
            while j < n and not s[j].isspace() and s[j] not in "()":
                j += 1
            toks.append(s[i:j])
            i = j
    return toks


def parse(toks, pos=0):
    if toks[pos] == "(":
        items, pos = [], pos + 1
        while toks[pos] != ")":
            item, pos = parse(toks, pos)
            items.append(item)
        return tuple(items), pos + 1
    return toks[pos], pos + 1


def render(t):
    if isinstance(t, tuple):
        return "(" + " ".join(render(x) for x in t) + ")"
    return t


def canon(t, env=None):
    """Alpha-canonical rendering so driver-side keys match trie dedup."""
    if env is None:
        env = {}
    if isinstance(t, str) and t.startswith("$"):
        return env.setdefault(t, f"$v{len(env)}")
    if isinstance(t, tuple):
        return tuple(canon(x, env) for x in t)
    return t


def run_stratum(k, counts, workdir):
    """counts: {stratum: {theorem-term: n}} -> contributions for stratum k."""
    fact_lines = []
    serial = 0
    for i, per in sorted(counts.items()):
        for thm, n in sorted(per.items(), key=lambda kv: render(kv[0])):
            # The serial is the schema's PRE-unification identity: two distinct
            # schemas unifying to the same instance with equal products must not
            # collapse into one contribution fact under trie set semantics.
            fact_lines.append(f"(fwd-count {i} {render(thm)} s{serial} {n})")
            serial += 1
    pair_lines = [
        f"(dp-pair {i} {k - 1 - i})"
        for i in sorted(counts)
        if (k - 1 - i) in counts
    ]
    prog = STRATUM_PROGRAM.format(
        k=k, count_facts="\n".join(fact_lines), pairs="\n".join(pair_lines)
    )
    src = workdir / f"stratum-{k}.mm2"
    dump = workdir / f"stratum-{k}.dump"
    src.write_text(prog)
    subprocess.run(
        [MORK, "run", str(src), str(dump)], check=True, capture_output=True
    )
    per_thm = defaultdict(int)
    contribs = 0
    for line in dump.read_text().splitlines():
        if not line.startswith("(contrib "):
            continue
        term, _ = parse(tokenize(line))
        thm = canon(term[1])
        per_thm[thm] += int(term[6])
        contribs += 1
    return per_thm, contribs


def dp(hf, workdir):
    counts = {1: {canon(parse(tokenize(a))[0]): 1 for a in AXIOMS}}
    stats = []
    t0 = time.time()
    for k in range(3, hf + 1, 2):
        per_thm, contribs = run_stratum(k, counts, workdir)
        if per_thm:
            counts[k] = dict(per_thm)
        stats.append(
            (k, len(per_thm), sum(per_thm.values()), contribs)
        )
    wall = time.time() - t0
    return counts, stats, wall


def oracle_check(hf, counts):
    """Baseline enumeration per-K fact counts == per-K derivation totals."""
    dump = MITM / f"fwd-only-hf{hf}.dump"
    if not dump.exists():
        return None
    base = defaultdict(int)
    for line in dump.read_text().splitlines():
        if line.startswith("(fwd "):
            k = int(line.split()[1])
            base[k] += 1
    ok = True
    rows = []
    for k in sorted(base):
        dp_total = sum(counts.get(k, {}).values())
        rows.append((k, base[k], dp_total, base[k] == dp_total))
        ok &= base[k] == dp_total
    return ok, rows


def main():
    hf = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    workdir = ROOT / f"run-hf{hf}"
    workdir.mkdir(exist_ok=True)
    counts, stats, wall = dp(hf, workdir)
    print(f"DP to Hf={hf}: {wall:.2f}s total")
    print("K | states(theorems) | derivations | contributions")
    for k, states, derivs, contribs in stats:
        print(f"{k:2} | {states:6} | {derivs:12} | {contribs}")
    oracle = oracle_check(hf, counts)
    if oracle is None:
        print(f"(no baseline enumeration dump for Hf={hf}; oracle skipped)")
    else:
        ok, rows = oracle
        print(f"oracle vs enumeration: {'PASS' if ok else 'FAIL'}")
        for k, b, d, match in rows:
            print(f"  K={k}: enumerated {b}, DP {d}, {'==' if match else 'MISMATCH'}")


if __name__ == "__main__":
    main()
