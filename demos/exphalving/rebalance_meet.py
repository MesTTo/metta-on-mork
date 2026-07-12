#!/usr/bin/env python3
"""Move the meet point of a MITM program deeper into the forward side.

The forward closure is cheap (subsumption keeps it an antichain: 490 schemas
at Hf=12, 12,068 at Hf=18) while the backward search is the exponential one.
A target of proof size D split as (Hf forward, D-Hf backward) therefore wants
Hf as LARGE as the forward side can afford -- that is the exponent halving,
and the original split (11 forward, 15 backward) is on the wrong side of it.
"""
import re
import sys
from pathlib import Path

src, new_hf, fwd_dump, dst = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
text = src.read_text()
old_hf = int(re.search(r"\(forward-bound (\d+)\)", text).group(1))

# The forward table: replace wholesale with the deeper antichain.
fwd = [l for l in fwd_dump.read_text().splitlines() if l.startswith("(fwd ")]
lines = [l for l in text.splitlines() if not l.startswith("(fwd ")]
out, inserted = [], False
for l in lines:
    if not inserted and l.startswith("(forward-bound"):
        out.append(f"(forward-bound {new_hf})")
        out.extend(fwd)
        inserted = True
        continue
    out.append(l)
text = "\n".join(out) + "\n"

# The cap appears in the three transition rules as the split between
# "above the forward bound" (split + axiom discharge) and "at or below it"
# (close the tail with any forward theorem).
text = text.replace(f"(gtFn $ski {old_hf})", f"(gtFn $ski {new_hf})")
text = text.replace(f"(lte $ski {old_hf})", f"(lte $ski {new_hf})")
text = re.sub(r";; Forward bound Hf: \d+", f";; Forward bound Hf: {new_hf}", text)
dst.write_text(text)
print(f"{dst.name}: Hf {old_hf} -> {new_hf}, {len(fwd)} forward schemas")
