#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figF_exponents.py
=================
Appendix F.1 -- every measured width exponent, on one axis.

One figure, one message (R1):
    "Not one of the twelve lazy-flattening exponents reaches 2, the rate that
     would be needed to beat the Theta(n^2) growth of ||Delta||_2^2 -- and the
     four families of exponents sit at visibly different places, which is why
     spectral alignment and geodesic shape cannot be the same effect."

Why this figure exists (figures/SURVEY_APPENDIX.md, Part III, P2):
Table F.1 is twelve rows of "estimate, 95% CI, R^2" whose entire message is a
comparison against one threshold.  Turning exactly that kind of table into a
dot-and-whisker plot is the standard recommendation -- Gelman, Pasarica & Dodhia
(2002), "Let's practice what we preach: turning tables into graphs", and
Kastellec & Leoni (2007) -- because a threshold comparison across twelve rows is
a visual task, not a reading task.  The table stays in the appendix; this figure
is what a reader checks the claim against.

Every number here is MEASURED: the exponents are refitted from the released
CSVs through fig_data.py, not copied out of the table they illustrate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fig_style import (apply_style, save, C, despine, REGIMES, REGIME_LABEL,
                       ARCH_MARKERS, ARCH_LABEL)
from fig_data import (load_pairs, load_dF, boot_ci_over_seeds, powerlaw_fit,
                      ACT_LABEL)

apply_style()

INK, REF = C["ink"], C["ref"]
CRITICAL = 2.0        # the rate needed to beat ||Delta||_2^2 = Theta(n^2)
GAUSS_NEWTON = 0.5    # frozen Gauss-Newton prediction, before activation effects

# ----------------------------------------------------------------------
# Everything below is MEASURED (see fig_data.py), never transcribed.
#
#   alpha_op   OLS on seed medians of ||dF||_op in the NTK-lazy regime, per
#              architecture x activation -- the procedure App. E.1 specifies.
#              It reproduces all twelve point estimates of Table F.1 to two
#              decimals, which is how the loader was validated.
#   95% CI     bootstrap over SEEDS within each width, pushed through that same
#              median-then-fit procedure.  It is wider than the interval in the
#              table for the CNN, because the CNN grid has only four widths.
#   the other three families are fitted per cell in the same way.  The excess
#              barrier is B - delta with delta = |L(w_A) - L(w_B)| / 2, and is
#              fitted only where a power law is meaningful (Standard and muP).
# ----------------------------------------------------------------------
_ORDER = ["gelu", "tanh", "swish", "softplus"]

_dF = load_dF()
ALPHA_OP = []
for _arch in ("MLP", "TS", "CNN"):
    for _act in _ORDER:
        _sub = _dF[(_dF.arch == _arch) & (_dF.regime == "NTK")
                   & (_dF.act == _act)]
        _e, _lo, _hi, _r2, _n = boot_ci_over_seeds(_sub, "dF_op")
        ALPHA_OP.append((_arch, ACT_LABEL[_act], _e, _lo, _hi, _r2))

_pf, _geo = load_pairs("final"), load_pairs("geo")
_pf = _pf.assign(excess=_pf.B - 0.5 * (_pf.L_A - _pf.L_B).abs())


def _cells(df, col, regimes=None):
    d = df if regimes is None else df[df.regime.isin(regimes)]
    out = []
    for (arch, reg, _act), sub in (d.dropna(subset=[col])
                                    .groupby(["arch", "regime", "act"])):
        m = sub[sub[col] > 0].groupby("width")[col].median()
        e = powerlaw_fit(m.index.values, m.values)[0]
        if np.isfinite(e):
            out.append((arch, reg, float(e)))
    return out


D_REL = [(a_, None, v) for a_, _r, v in _cells(_geo, "dev_rel", ["NTK"])]
RAYLEIGH = _cells(_geo, "rq_mid")
BARRIER = _cells(_pf, "excess", ["Standard", "muP"])

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(5.5, 3.85), sharex=True,
    gridspec_kw={"height_ratios": [3.05, 1.0], "hspace": 0.16})

# ======================================================================
# (a)  The twelve lazy-flattening exponents, grouped by architecture.
# ======================================================================
rows, ys = [], []
y = 0.0
prev_arch = None
for arch, act, est, lo, hi, r2 in ALPHA_OP:
    if prev_arch is not None and arch != prev_arch:
        y += 0.9                       # a gap, not a rule: grouping without ink
    rows.append((arch, act, est, lo, hi, r2))
    ys.append(y)
    y += 1.0
    prev_arch = arch
ys = np.array(ys)
y_top = ys.max() + 1.0

for (arch, act, est, lo, hi, r2), yy in zip(rows, ys):
    axA.plot([lo, hi], [yy, yy], color=INK, lw=1.0, solid_capstyle="butt",
             zorder=3)
    axA.plot([lo, lo], [yy - 0.16, yy + 0.16], color=INK, lw=0.9, zorder=3)
    axA.plot([hi, hi], [yy - 0.16, yy + 0.16], color=INK, lw=0.9, zorder=3)
    axA.plot([est], [yy], marker=ARCH_MARKERS[arch], markersize=4.2,
             color=INK, markerfacecolor="white", markeredgewidth=1.0, zorder=4)
    axA.text(hi + 0.045, yy, f"{est:.2f}", fontsize=6.6, color=INK,
             ha="left", va="center")
    axA.text(hi + 0.185, yy, f"$R^2\\!=\\!{r2:.2f}$", fontsize=6.4, color=REF,
             ha="left", va="center")

axA.axvline(CRITICAL, color=INK, lw=1.1, zorder=2)
axA.axvline(GAUSS_NEWTON, color=REF, lw=0.9, linestyle=(0, (2, 2)), zorder=2)

axA.set_yticks(ys)
axA.set_yticklabels([f"{ARCH_LABEL[a] if a != 'TS' else 'T–S'}  {act}"
                     for a, act, *_ in rows], fontsize=6.8)
axA.set_ylim(-1.25, ys.max() + 1.30)
axA.invert_yaxis()

axA.text(GAUSS_NEWTON + 0.05, -1.02,
         r"$\alpha=\frac{1}{2}$: frozen Gauss–Newton",
         fontsize=6.5, color=REF, ha="left", va="center")
axA.text(CRITICAL - 0.05, -1.02, "critical rate " + r"$\alpha=2$",
         fontsize=6.9, color=INK, ha="right", va="center", weight="bold")
axA.text(CRITICAL - 0.05, ys.max() + 0.85,
         r"$12/12$ below the rate that would beat "
         r"$\|\Delta\|_2^{2}=\Theta(n^{2})$",
         fontsize=6.7, color=INK, ha="right", va="center", style="italic")

despine(axA)
axA.grid(axis="y", visible=False)
axA.grid(axis="x", visible=True)
axA.set_ylabel(r"$\alpha_{\mathrm{op}}$ by architecture $\times$ activation",
               fontsize=7.4)
axA.text(-0.135, 1.01, "(a)", transform=axA.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ======================================================================
# (b)  All four exponent families on the same axis.
# ======================================================================
families = [
    (r"$D_{\mathrm{rel}}$  (geodesic shape)",
     D_REL),
    (r"excess barrier $B-\delta$",
     BARRIER),
    (r"$\alpha_{\mathrm{op}}$  (operator norm)",
     [(a, None, e) for a, _act, e, *_ in ALPHA_OP]),
    (r"Rayleigh $\widehat\Delta^{\top}F\widehat\Delta$",
     RAYLEIGH),
]

for i, (name, pts) in enumerate(families):
    vals = np.array([v for _a, _g, v in pts])
    axB.plot([vals.min(), vals.max()], [i, i], color=REF, lw=0.8, zorder=2)
    for j, (arch, regime, v) in enumerate(pts):
        jitter = 0.13 * ((j % 3) - 1)
        axB.plot([v], [i + jitter], marker=ARCH_MARKERS[arch], markersize=3.6,
                 color=C[regime] if regime else INK,
                 markerfacecolor="white", markeredgewidth=0.9, zorder=4)

axB.axvline(CRITICAL, color=INK, lw=1.1, zorder=2)
axB.axvline(GAUSS_NEWTON, color=REF, lw=0.9, linestyle=(0, (2, 2)), zorder=2)

axB.set_yticks(range(len(families)))
axB.set_yticklabels([n for n, _ in families], fontsize=6.9)
axB.set_ylim(-0.80, len(families) - 0.45)
axB.invert_yaxis()
axB.set_xlim(-0.34, 2.72)
axB.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
axB.set_xlabel(r"width exponent  $\gamma$  in  (quantity) $\sim n^{-\gamma}$")

_dr = np.median([v for *_x, v in D_REL])
_rq = np.median([v for *_x, v in RAYLEIGH])
axB.annotate("", xy=(_dr, -0.38), xytext=(_rq, -0.38),
             arrowprops=dict(arrowstyle="<->", color=REF, lw=0.7))
axB.text(0.5 * (_dr + _rq), -0.50,
         f"spectral alignment decays {_rq/_dr:.1f}$\\times$ faster than shape"
         " (medians)", fontsize=6.5, color=REF, ha="center", va="bottom")

# The three Standard-MLP cells fit a NEGATIVE exponent: that barrier is
# non-monotonic in width (Fig. 4a), so no power law describes it.  They are
# plotted rather than dropped, and said so, because silently clipping the
# inconvenient cells off the axis would be the one unacceptable option.
axB.annotate("Standard MLP:\nnon-monotonic,\nno power law",
             xy=(-0.15, 1.0), xytext=(0.02, 3.32), textcoords="data",
             fontsize=6.2, color=C["Standard"], ha="left", va="center",
             linespacing=1.35,
             arrowprops=dict(arrowstyle="-", color=C["Standard"], lw=0.6))

despine(axB)
axB.grid(axis="y", visible=False)
axB.grid(axis="x", visible=True)
axB.text(-0.135, 1.03, "(b)", transform=axB.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ======================================================================
# Shared legend: marker = architecture, colour = parameterisation (R3).
# ======================================================================
handles = [Line2D([], [], color=INK, lw=0, marker=ARCH_MARKERS[a],
                  markersize=4.0, markerfacecolor="white", markeredgewidth=1.0,
                  label=ARCH_LABEL[a]) for a in ["MLP", "TS", "CNN"]]
handles += [Line2D([], [], color=C[r], lw=0, marker="o", markersize=4.0,
                   markerfacecolor="white", markeredgewidth=1.0,
                   label=REGIME_LABEL[r]) for r in REGIMES]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, -0.105), handlelength=1.0, columnspacing=1.2,
           handletextpad=0.35, fontsize=6.5,
           title="marker = architecture,   colour = parameterisation "
                 "(black = pooled)",
           title_fontsize=6.5)

save(fig, Path(__file__).parent.parent / "figures" / "appendix",
     "figF_exponents")
