#!/usr/bin/env python3
"""Coverage differential: every theorem in the baseline forward strata must be
generalized by some theorem in the subsumed strata at an equal-or-lower stratum.
Generalization = one-way matching of a schema onto an instance with a consistent
variable mapping (variables in the instance are constants for this direction)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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


def is_var(t):
    return isinstance(t, str) and t.startswith("$")


def generalizes(schema, instance, env):
    if is_var(schema):
        if schema in env:
            return env[schema] == instance
        env[schema] = instance
        return True
    if isinstance(schema, tuple):
        if not isinstance(instance, tuple) or len(schema) != len(instance):
            return False
        return all(generalizes(s, i, env) for s, i in zip(schema, instance))
    return schema == instance


def load_theorems(dump_path):
    """stratum -> set of theorem terms (from (fwd K (: PROOF THM)) lines)."""
    out = {}
    for line in Path(dump_path).read_text().splitlines():
        if not line.startswith("(fwd "):
            continue
        term, _ = parse(tokenize(line))
        k = int(term[1])
        thm = term[2][2]
        out.setdefault(k, set()).add(thm)
    return out


def check(base_path, sub_path):
    base = load_theorems(base_path)
    sub = load_theorems(sub_path)
    sub_flat = [(k, t) for k, ts in sub.items() for t in ts]
    missing = []
    total = 0
    for k, thms in sorted(base.items()):
        for thm in thms:
            total += 1
            if not any(
                ks <= k and generalizes(ts, thm, {}) for ks, ts in sub_flat
            ):
                missing.append((k, thm))
    print(f"{base_path} vs {sub_path}:")
    print(f"  baseline theorems: {total}; uncovered: {len(missing)}")
    for k, thm in missing[:5]:
        print(f"  MISSING at K={k}: {thm}")
    for k in sorted(set(base) | set(sub)):
        print(
            f"  K={k}: baseline {len(base.get(k, ()))} theorems,"
            f" subsumed {len(sub.get(k, ()))}"
        )
    return not missing


if __name__ == "__main__":
    hf = sys.argv[1] if len(sys.argv) > 1 else "9"
    ok = check(
        ROOT.parent / "mitm" / f"fwd-only-hf{hf}.dump"
        if (ROOT.parent / "mitm" / f"fwd-only-hf{hf}.dump").exists()
        else ROOT / f"fwd-only-hf{hf}.dump",
        ROOT / f"fwd-sub-only-hf{hf}.dump",
    )
    sys.exit(0 if ok else 1)
