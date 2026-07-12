#!/usr/bin/env python3
"""One-command reproduction: generate the baseline and subsumption programs,
run both on the kernel, and check antichain coverage (no fact of the baseline
closure escapes the subsumed closure's generalization).

Usage: MORK_BIN=<mork binary> python3 run_coverage.py [Hf ...]
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MORK = os.environ.get("MORK_BIN", "mork")


def main():
    hfs = [int(a) for a in sys.argv[1:]] or [6, 9]
    subprocess.run(
        [sys.executable, str(ROOT / "build_subsume_programs.py"), *map(str, hfs)],
        check=True,
    )
    for hf in hfs:
        for stem in (f"fwd-only-hf{hf}", f"fwd-sub-only-hf{hf}"):
            src = ROOT / f"{stem}.mm2"
            dump = ROOT / f"{stem}.dump"
            t = time.time()
            subprocess.run([MORK, "run", str(src), str(dump)], check=True, capture_output=True)
            n = sum(1 for _ in dump.open())
            print(f"{stem}: {n} atoms in {time.time() - t:.2f}s")
        subprocess.run(
            [sys.executable, str(ROOT / "coverage_check.py"), str(hf)], check=True
        )


if __name__ == "__main__":
    main()
