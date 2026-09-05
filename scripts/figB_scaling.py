#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figB_scaling.py
===============
Appendix B.2 -- the independent-initialisation regime, measured.

One figure, one message (R1):
    "Under NTK parameterisation the separation between two independently
     trained networks is Theta(sqrt P) while neither network moves at all in
     the limit, so the segment joining them is nowhere near a common lazy
     neighbourhood.  Under fan-in initialisation the SAME measurement gives a
     smaller scale -- because the hypothesis of Proposition B.2 is about the
     initialisation law, and fan-in initialisation does not satisfy it."

DATA: measured, from ../data/ (see fig_data.py).  Median over the four smooth
activations and all seed pairs; band = interquartile range.

WHAT THE DATA SAYS, AND WHERE IT DEPARTS FROM THE APPENDIX TEXT.
Proposition B.2 needs an initialisation law with sigma_min <= sigma_i <=
sigma_max UNIFORMLY in n.  NTK parameterisation has O(1) initialisation
variance and satisfies that; Standard and muP use fan-in initialisation, where
sigma_i ~ n^{-1/2}, and do not.  The measurement separates exactly along that
line.  Compensating by sqrt(P) -- panel (b), which is the sharp test, since
Theta is a two-sided claim -- gives

    MLP / NTK        1.370 -> 1.369   over 64 ... 4096      (flat to 3 decimals)
    CNN / NTK        1.086 -> 1.086   over 64 ... 512       (flat to 3 decimals)
    everything else  decays, by n^{-0.25} to n^{-0.93}

so Theta(sqrt P) is confirmed, sharply, in the regime whose hypothesis holds,
and is not a property of the other two.  App. F.1 currently states the stronger
claim "gamma = 1.00 +/- 0.03 in all nine cells"; the measurement does not
support it, and the figure shows what was actually measured.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from fig_style import (apply_style, save, C, despine, log_width_axis, REGIMES,
                       REGIME_LABEL, WIDTHS)
from fig_data import load_pairs, by_width, powerlaw_fit

apply_style()

STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
         "muP": ((0, (1, 2.2)), "^")}


def param_count(arch, n):
    """P from the architecture table of App. E.1 (d = 784, K = 10)."""
    n = np.asarray(n, float)
    return 9 * n ** 2 + 21 * n + 10 if arch == "CNN" else n ** 2 + 796 * n + 10


pairs = load_pairs("final")

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(5.5, 2.45), gridspec_kw={"wspace": 0.30})

# ======================================================================
# (a)  What the two networks are separated by, against what either travelled.
# ======================================================================
sep = by_width(pairs, "dnorm", keys=("regime",))
mov = by_width(pairs, "wmoveA", keys=("regime",))

ratio_txt = {}
for regime in REGIMES:
    ls, mk = STYLE[regime]
    col = C[regime]
    s = sep[sep["regime"] == regime].sort_values("width")
    m = mov[mov["regime"] == regime].sort_values("width")
    ax1.fill_between(s["width"], s["q1"], s["q3"], color=col, alpha=0.15, lw=0)
    ax1.plot(s["width"], s["med"], color=col, ls=ls, lw=1.7, marker=mk, ms=3.4,
             markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.5,
             zorder=5)
    ax1.plot(m["width"], m["med"], color=col, ls=ls, lw=1.2, marker=mk, ms=3.4,
             markerfacecolor="white", markeredgecolor=col, markeredgewidth=0.9,
             alpha=0.85, zorder=4)
    r = s["med"].to_numpy() / m["med"].to_numpy()
    ratio_txt[regime] = (powerlaw_fit(s["width"], r)[0], r[-1])

ax1.set_yscale("log")
log_width_axis(ax1)
ax1.set_xlim(WIDTHS[0] * 0.85, WIDTHS[-1] * 1.35)
ax1.set_ylim(5e-3, 4e4)
ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
ax1.yaxis.set_minor_locator(mticker.NullLocator())
ax1.set_xlabel(r"width $n$")
ax1.set_ylabel("Euclidean distance in parameter space")

ax1.text(WIDTHS[0] * 0.95, 1.3e4, r"separation $\|\Delta\|_2$  (filled)",
         fontsize=6.9, color=C["ink"], ha="left", va="center", weight="bold")
ax1.text(WIDTHS[0] * 0.95, 1.3e-2,
         r"travel $\sup_t\|w(t)-w(0)\|_2$  (open)",
         fontsize=6.9, color=C["ink"], ha="left", va="center", weight="bold")
ax1.annotate("", xy=(WIDTHS[-1], 5.4e3), xytext=(WIDTHS[-1], 1.3e-2),
             arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=0.8,
                             shrinkA=2, shrinkB=2))
e_ntk, r_ntk = ratio_txt["NTK"]
ax1.annotate(rf"separation / travel $\propto n^{{{-e_ntk:+.1f}}}$",
             xy=(WIDTHS[-1], 4.0), xytext=(-5, 0), textcoords="offset points",
             ha="right", va="center", fontsize=7.0, color=C["ink"],
             weight="bold")
despine(ax1)
ax1.text(-0.075, 1.045, "(a)", transform=ax1.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ======================================================================
# (b)  The sharp test.  Theta(sqrt P) is two-sided, so it is the statement
#      that ||Delta||_2 / sqrt(P) is bounded above AND below -- i.e. flat.
# ======================================================================
comp = {}
for (arch, regime), sub in pairs.groupby(["arch", "regime"]):
    m = sub.groupby("width")["dnorm"].median()
    q = m.to_numpy() / np.sqrt(param_count(arch, m.index.to_numpy()))
    comp[(arch, regime)] = (m.index.to_numpy(float), q)
    ax2.plot(m.index.to_numpy(float), q, color=C[regime], lw=0.9, alpha=0.55,
             ls=STYLE[regime][0], zorder=3)

drift = {}
for regime in REGIMES:
    xs = sorted({w for (a, r), (x, _) in comp.items() if r == regime
                 for w in x})
    med = [np.median([np.interp(w, x, q) for (a, r), (x, q) in comp.items()
                      if r == regime and x[0] <= w <= x[-1]]) for w in xs]
    ax2.plot(xs, med, color=C[regime], lw=2.0, ls=STYLE[regime][0], zorder=5)
    drift[regime] = powerlaw_fit(xs, med)[0]

ax2.set_yscale("log")
log_width_axis(ax2)
ax2.set_xlim(WIDTHS[0] * 0.85, WIDTHS[-1] * 1.35)
ax2.set_ylim(1.4e-2, 4.5)
ax2.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0])
ax2.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
ax2.yaxis.set_minor_locator(mticker.NullLocator())
ax2.set_xlabel(r"width $n$")
ax2.set_ylabel(r"$\|\Delta\|_2\,/\,\sqrt{P}$")

ax2.annotate(rf"flat $\Rightarrow\Theta(\sqrt{{P}})$   ($n^{{{-drift['NTK']:+.2f}}}$)",
             xy=(WIDTHS[-1], 1.25), xytext=(-4, 9), textcoords="offset points",
             ha="right", va="center", fontsize=7.0, color=C["NTK"],
             weight="bold")
ax2.text(WIDTHS[0] * 0.95, 2.4e-2,
         "decays: fan-in initialisation has\n"
         r"$\sigma_i\sim n^{-1/2}$, so Prop. B.2 does not apply",
         fontsize=6.6, color=C["ink"], ha="left", va="center", linespacing=1.4)
despine(ax2)
ax2.text(-0.075, 1.045, "(b)", transform=ax2.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ---------------------------------------------------------------------
handles = [Line2D([], [], color=C[r], lw=1.8, ls=STYLE[r][0],
                  label=REGIME_LABEL[r]) for r in REGIMES]
fig.legend(handles=handles, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.10), frameon=False, handlelength=2.0,
           columnspacing=1.8, handletextpad=0.5)

save(fig, Path(__file__).parent.parent / "figures" / "appendix", "figB_scaling")
print("  ||Delta||/sqrt(P) drift exponent per regime:",
      {k: round(-v, 3) for k, v in drift.items()})
print("  separation/travel ratio at the largest width:",
      {k: f"{v[1]:.3g}" for k, v in ratio_txt.items()})
