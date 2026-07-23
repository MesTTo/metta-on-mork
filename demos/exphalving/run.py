#!/usr/bin/env python3
"""Exponent halving on the meet-in-the-middle prover: move the meet point deeper
into the (now cheap) forward antichain and watch the searched state space
collapse, with every proof independently type-checked against the axioms.

A proof of size D is found by meeting a forward closure (built to size Hf,
target-independent) against a backward search (covering the remaining D-Hf).
The forward antichain stays small under subsumption while the backward search
branches, so the searched space is roughly b^(D-Hf): pushing Hf up is a direct
exponent reduction. The forward side is the SAME for every theorem, so it is a
shared lemma base -- built once, reused.

Completeness caveat, honestly: this only finds the proof when it BISECTS at the
meet point (some reachable backward context whose antecedent a size<=Hf forward
schema proves). The three targets here bisect; a proof that does not decompose
at the chosen Hf yields no meet, which is a property of bidirectional search,
not a failure of the engine.

Usage: MORK_BIN=<kernel with witness_select> python3 run.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_proof import check, target_of

ROOT = Path(__file__).resolve().parent
MORK = os.environ.get("MORK_BIN", "mork")

# (target, proof size, original-split program, rebalanced program)
CASES = [
    ("jarr", 13, "mitm-ws-jarr-hf6", "mitm-balws-jarr-hf12"),
    ("imim1", 15, "mitm-ws-imim1-hf6", "mitm-balws-imim1-hf12"),
    ("loowoz", 19, "mitm-ws-loowoz-hf12", "mitm-balws-loowoz-hf18"),
]


def run(stem):
    src = ROOT / f"{stem}.mm2"
    dump = ROOT / f"{stem}.dump"
    t = time.time()
    subprocess.run([MORK, "run", str(src), str(dump)], check=True, capture_output=True)
    wall = time.time() - t
    sol = sum(1 for line in dump.open() if line.startswith("(sol "))
    return sol, wall, dump


def build_act_tables():
    """Build the fromNumber and lte ACT tables the mitm programs read.

    The kernel materializes ACTs under /dev/shm (its ACT_PATH), so the two
    generators write /dev/shm/fromNumber.act and /dev/shm/lte.act. gen-lte
    reads fromNumber, so the order matters. Both are vendored from
    trueagi-io/chaining (Nil Geisweiller); see their file headers.
    """
    for gen in ("gen-fromNumber", "gen-lte"):
        subprocess.run(
            [MORK, "run", str(ROOT / f"{gen}.mm2"), str(ROOT / f"{gen}.dump")],
            check=True, capture_output=True,
        )


def main():
    build_act_tables()
    print(f"{'target':8} {'D':>3}  {'original states':>16}  {'rebalanced states':>18}  "
          f"{'collapse':>9}  proof")
    for target, d, orig, bal in CASES:
        so, _, _ = run(orig)
        sb, wb, bdump = run(bal)
        ok, msg = check(bdump, target_of(ROOT / f"{bal}.mm2"))
        collapse = f"{so / max(sb, 1):.1f}x"
        print(f"{target:8} {d:>3}  {so:>16}  {sb:>18}  {collapse:>9}  "
              f"{'PASS' if ok else 'FAIL: ' + msg}")


if __name__ == "__main__":
    main()
