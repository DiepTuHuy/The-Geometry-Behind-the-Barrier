#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figD_vacuous.py
===============
Appendix D.2 -- why the Fisher--barrier bound is vacuous at the measured scale.

One figure, one message (R1):
    "Three of the four terms of R(w_0) grow like P; the third-order remainder
     grows like P^{3/2}.  Since every term is nonnegative, the sum inherits the
     worst one, so the interval [delta, R] that Theorem A.3 permits for the
     barrier widens without limit while the floor delta stays O(1).  A diverging
     upper bound certifies nothing."

Why this figure exists (figures/SURVEY_APPENDIX.md, criterion K3(a)):
"the bound is vacuous" is a statement about competing RATES, and a rate
comparison is the one thing a log-log plot says better than a paragraph.  The
precedent is Dinh et al. (ICML 2017), whose limitation results are carried
almost entirely by figures rather than by prose.

Panel (a) is the term-by-term rate comparison, with the admissible interval for
the barrier shaded; panel (b) is the same data as shares of the total, which is
the actual content of the proof's final line, O(P) + Theta(P^{3/2}) =
Theta(P^{3/2}).

Constants: the magnitudes are illustrative (the paper reports exponents, not
magnitudes -- App. E.1), but the EXPONENTS are exactly those proved in D.2, and
they are what the figure is about.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from fig_style import apply_style, save, C, despine
from fig_data import load_pairs

apply_style()

INK, REF = C["ink"], C["ref"]

# ----------------------------------------------------------------------
# MEASURED inputs (MLP / MNIST, NTK-lazy, medians over the four smooth
# activations and all seed pairs):
#     ||Delta||_2          data/final/mlp_pairs.csv      321 -> 6128
#     Delta^T F Delta      data/geodesic/mlp_pairs.csv   2.05 -> 4.30
#     rho*(w_A)            data/final/mlp_pairs.csv      0.170 -> 0.240
#     delta = |L_A-L_B|/2  data/final/mlp_pairs.csv      ~1e-3, i.e. O(1)
#
# NOT measured, and therefore carried as the Theta(1) constants the proposition
# assumes: M_2 and M_3, both set to 1 here.  The endpoint-gradient term needs
# ||grad L(w_0)||, which the released tables do not carry, so it is omitted --
# it is O(P), the same rate as the fit-residual term already drawn, so leaving
# it out changes no exponent and cannot rescue the bound.
#
# The Fisher term is the one worth watching: the proof only bounds it by O(P),
# but MEASURED it is essentially flat, because the Rayleigh quotient decays
# fast enough to cancel the growth of ||Delta||_2^2.  The divergence of R is
# therefore carried entirely by the two remainder terms.
# ----------------------------------------------------------------------
_pf, _geo = load_pairs("final"), load_pairs("geo")
_p = _pf[(_pf.arch == "MLP") & (_pf.regime == "NTK")]
_g = _geo[(_geo.arch == "MLP") & (_geo.regime == "NTK")]

n = _p.groupby("width")["dnorm"].median().index.to_numpy(float)
d = _p.groupby("width")["dnorm"].median().to_numpy()
flen = _g.groupby("width")["flen_A"].median().reindex(n).to_numpy()
rho = _p.groupby("width")["rho_A"].median().to_numpy()
DELTA = float((0.5 * (_p.L_A - _p.L_B).abs()).groupby(_p.width).median().median())
P = n ** 2 + 796 * n + 10

M2 = M3 = 1.0        # Theta(1) by hypothesis; not measured

terms = [
    ("fisher", r"$\frac{1}{2}\Delta^{\top}F(w_0)\Delta$", "measured",
     0.5 * flen, C["term_fisher"], 1.7),
    ("resid", r"$\frac{1}{2}\rho^{*}M_2\|\Delta\|_2^{2}$", r"$O(P)$",
     0.5 * rho * M2 * d ** 2, C["term_resid"], 1.0),
    ("cubic", r"$\frac{M_3}{6}\|\Delta\|_2^{3}$", r"$\Theta(P^{3/2})$",
     (M3 / 6.0) * d ** 3, C["term_cubic"], 1.7),
]
R = sum(t[3] for t in terms)
B_meas = _p.groupby("width")["B"].median().to_numpy()   # what R is supposed to bound

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(5.5, 2.62),
    gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.34})

# ======================================================================
# (a)  Rates, and the interval the theorem leaves open for the barrier.
# ======================================================================
axA.fill_between(n, DELTA, R, color=REF, alpha=0.15, linewidth=0, zorder=1)
axA.axhline(DELTA, color=REF, lw=1.0, linestyle=(0, (4, 2)), zorder=3)

for _, label, rate, y, col, lw in terms:
    axA.plot(n, y, color=col, lw=lw, zorder=4)
axA.plot(n, R, color=INK, lw=1.0, linestyle=(0, (1.5, 1.5)), zorder=5)
axA.plot(n, B_meas, color=C["NTK"], lw=1.8, marker="o", ms=3.2,
         markerfacecolor=C["NTK"], markeredgecolor="white",
         markeredgewidth=0.5, zorder=6)
axA.annotate(r"measured barrier $B(\gamma_{\mathrm{lin}})$",
             xy=(n[-1], B_meas[-1]), xytext=(-4, -12),
             textcoords="offset points", fontsize=6.9, color=C["NTK"],
             ha="right", va="top", weight="bold")

axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlim(n[0] * 0.85, n[-1] * 1.25)
axA.set_ylim(3e-4, 3e11)
axA.set_xlabel(r"width $n$")
axA.set_ylabel(r"contribution to $R(w_0)$")
axA.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
axA.yaxis.set_minor_locator(mticker.NullLocator())

sec = axA.secondary_xaxis("top", functions=(np.square, np.sqrt))
sec.set_xticks([64 ** 2, 256 ** 2, 1024 ** 2, 4096 ** 2])
sec.set_xticklabels(["4k", "65k", "1M", "17M"])
sec.set_xlabel(r"$P$", labelpad=2.0)
sec.tick_params(labelsize=7.0)
sec.spines["top"].set_color("#4d4d4d")

# The three O(P) curves are near-coincident, and that IS the content, so they
# are labelled once as a bundle; only the term with the different rate gets a
# label of its own.  Identities are carried by the shared legend below.
def at(series, n_value):
    return series[np.argmin(np.abs(n - n_value))]


# Both labels sit in the two genuinely empty corners of the panel and reach
# their curve with a hairline leader, so nothing is drawn over data (R8).
axA.annotate(r"$O(P)$", xy=(2048.0, at(terms[1][3], 2048.0)),
             xytext=(4400.0, 1.5e3), textcoords="data", fontsize=6.9,
             color=INK, ha="right", va="center",
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))

axA.annotate(r"$\frac{M_3}{6}\|\Delta\|_2^{3}=\Theta(P^{3/2})$",
             xy=(400.0, at(terms[2][3], 400.0)),
             xytext=(70.0, 2.5e9), textcoords="data", fontsize=7.0,
             color=INK, ha="left", va="center", weight="bold",
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))

axA.annotate(r"floor $\delta$ (measured)", xy=(n[0] * 0.95, DELTA), xytext=(0, 4),
             textcoords="offset points", fontsize=6.8, color=REF, ha="left",
             va="bottom")
axA.text(P[0] * 1.35, 2.4e6,
         "interval Thm. A.3\n" + r"leaves open for $B$" + "\n"
         + r"widens as $\Theta(P^{3/2})$",
         fontsize=6.8, color=INK, ha="left", va="center", linespacing=1.5)

despine(axA)
axA.text(-0.075, 1.20, "(a)", transform=axA.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ======================================================================
# (b)  How vacuous, in orders of magnitude.  R is an upper bound on B; the
#      quantity that matters is how much room it leaves, and it leaves more
#      at every width.
# ======================================================================
slack = np.log10(R / B_meas)
axB.plot(n, slack, color=INK, lw=1.8, marker="o", ms=3.4,
         markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.5,
         zorder=5)
axB.fill_between(n, 0, slack, color=REF, alpha=0.18, lw=0)
axB.axhline(0, color=C["NTK"], lw=1.2, zorder=4)

axB.set_xscale("log")
axB.set_xlim(n[0] * 0.85, n[-1] * 1.25)
axB.set_ylim(0, slack.max() * 1.20)
axB.set_xlabel(r"width $n$")
axB.set_ylabel(r"orders of magnitude of slack,  $\log_{10}(R/B)$")

secB = axB.secondary_xaxis("top", functions=(np.square, np.sqrt))
secB.set_xticks([64 ** 2, 256 ** 2, 1024 ** 2, 4096 ** 2])
secB.set_xticklabels(["4k", "65k", "1M", "17M"])
secB.set_xlabel(r"$P$", labelpad=2.0)
secB.tick_params(labelsize=7.0)
secB.spines["top"].set_color("#4d4d4d")

axB.annotate(f"{slack[-1]:.0f} orders of magnitude\nat $n={int(n[-1])}$",
             xy=(n[-1], slack[-1]), xytext=(-6, -14),
             textcoords="offset points", fontsize=6.9, color=INK,
             ha="right", va="top", weight="bold")
axB.text(n[0] * 1.05, 0.6, r"a tight bound would sit here", fontsize=6.6,
         color=C["NTK"], ha="left", va="center")

despine(axB)
axB.text(-0.135, 1.20, "(b)", transform=axB.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ======================================================================
# Shared legend (R8): identities live here, not on top of the curves.
# ======================================================================
from matplotlib.lines import Line2D

handles = [Line2D([], [], color=t[4], lw=1.6 if t[0] == "cubic" else 1.2,
                  label=f"{t[1]}  {t[2]}") for t in terms]
handles += [
    Line2D([], [], color=INK, lw=1.0, linestyle=(0, (1.5, 1.5)),
           label=r"$R(w_0)$, the sum"),
    Line2D([], [], color=REF, lw=1.0, linestyle=(0, (4, 2)),
           label=r"floor $\delta=\frac{1}{2}|L(w_A)-L(w_B)|$"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.235), handlelength=1.7, columnspacing=1.6,
           handletextpad=0.5, fontsize=6.6)

save(fig, Path(__file__).parent.parent / "figures" / "appendix", "figD_vacuous")
