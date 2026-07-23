#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 MesTTo
"""Render docs/asymptotics.svg from the measured values in README.md.

Every number here is copied from the README's tables, which record runs on
this machine against this tree; the plot is a view of those tables, not a new
measurement. Regenerate after re-measuring:

    python3 docs/plot_readme_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "asymptotics.svg"

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})
# colour-blind-safe (Wong)
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))

# -- selective query: scan vs column seek vs tabled replay (README tables) --
ax = axes[0][0]
n = [100_000, 400_000, 1_600_000]
ax.loglog(n, [4.52e-3, 16.9e-3, 60.4e-3], "o-", color=RED, label="matcher scan")
ax.loglog(n, [881e-9, 822e-9, 742e-9], "s-", color=BLUE, label="column-index seek")
ax.loglog(n, [1.66e-6] * 3, "^-", color=GREEN, label="tabled replay")
ax.annotate("81,415x at 1.6M", xy=(1.55e6, 1.0e-6), xytext=(3.0e5, 3.5e-5),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
ax.set_title("Selective queries: seek, don't scan")
ax.set_xlabel("stored atoms")
ax.set_ylabel("seconds per query")
ax.legend(fontsize=7.5)

# -- conjunctive 2-hop join: output-linear where GroundingSpace panics --
ax = axes[0][1]
ax.loglog([500, 1_000, 2_000, 32_000, 512_000],
          [927e-6, 1.65e-3, 3.06e-3, 53.5e-3, 954e-3],
          "o-", color=BLUE, label="MorkSpace (WCO join)")
ax.loglog([500, 1_000], [2.03e-3, 2.88e-3], "o-", color=RED, label="GroundingSpace")
ax.plot([2_000], [4.1e-3], "x", color=RED, markersize=9, markeredgewidth=2.4)
ax.annotate("panics (#1076)", xy=(2_000, 4.1e-3), xytext=(4_000, 1.3e-3),
            fontsize=8, color=RED)
ax.set_title("2-hop conjunctive join over an N-edge chain")
ax.set_xlabel("edges")
ax.set_ylabel("seconds")
ax.legend(fontsize=7.5)

# -- factorized conjunctive count vs enumeration --
ax = axes[1][0]
k = [250, 1_000, 4_000]
ax.loglog(k, [76.5e-3, 1.17, 19.5], "o-", color=RED, label="enumerate the join")
ax.loglog(k, [756e-6, 3.05e-3, 12.6e-3], "s-", color=BLUE, label="factorized count")
ax.annotate("16M-row join\ncounted in 12.6 ms", xy=(3_800, 14e-3),
            xytext=(1_050, 0.10), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
ax.set_title("Counting a K x K join output")
ax.set_xlabel("K (double-star arm)")
ax.set_ylabel("seconds")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=7.5)

# -- semi-naive fixpoints on process_calculus --
ax = axes[1][1]
steps = [100, 400, 1_000]
ax.loglog(steps, [54.1e-3, 2.38, 35.9], "o-", color=RED, label="naive")
ax.loglog(steps, [6.60e-3, 122e-3, 1.23], "s-", color=BLUE, label="semi-naive")
for x, y, r in [(100, 6.60e-3, "8.2x"), (400, 122e-3, "19.6x"), (1_000, 1.23, "29.1x")]:
    ax.annotate(r, xy=(x, y), xytext=(x, y * 3.4), fontsize=8, ha="center", color="0.3")
ax.set_title("process_calculus fixpoint: the ratio grows with size")
ax.set_xlabel("steps (workload grows with steps)")
ax.set_ylabel("seconds")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=7.5)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
