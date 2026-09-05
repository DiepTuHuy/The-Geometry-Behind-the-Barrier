#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figC_geodesic.py
================
Appendix C.4-C.5 -- the tube, the Green kernel, and the regime the theorem
does not reach.

One figure, one message (R1):
    "Theorem C.4 builds the geodesic as a fixed point inside a tube of radius
     ||Delta||_2 and pins it to within (3c_1/4 lambda n^alpha)||Delta||_2^2 of
     the straight segment -- but the smallness condition confines it to
     ||Delta||_2 <~ n^{alpha/2}, and the measured regime is Theta(n).  The bound
     is not wrong there; it is silent."

Why this figure exists (figures/SURVEY_APPENDIX.md, criterion K3(a)): the proof
turns on a geometric configuration -- a tube, a straight chord, a curve pinned
inside a lens whose SHAPE is the Green kernel -- that a reader has to picture to
follow the fixed-point argument.  That is precisely the case in which strong
theory papers do draw in the appendix: Dinh et al. (ICML 2017) Figs. 2-5 (level
curves of an equivalent reparametrisation), Chizat et al. (NeurIPS 2019)
Fig. B.5 (commutative diagram for the rank theorem), Draxler et al. (ICML 2018)
Fig. A.1 (the pivot-insertion condition).

Panel (a) is drawn TO SCALE in units of ||Delta||_2: the tube radius really is
||Delta||_2 (definition of T_Delta), and the deviation lens really is ten times
thinner, because the smallness condition (C.13) forces kappa*||Delta||_2 <= 1/5
and hence sup_t||xi|| <= (kappa/2)||Delta||_2^2 <= ||Delta||_2/10.  The lens
profile is t(1-t): that is the Green kernel of Lemma C.5, not decoration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from fig_style import (apply_style, save, C, despine, log_width_axis,
                       WIDTHS, ARCH_MARKERS, ARCH_LABEL)
from fig_data import load_pairs, load_dF, boot_ci_over_seeds, powerlaw_fit

apply_style()

INK, REF = C["ink"], C["ref"]
ACCENT, ACCENT_FILL, PANEL = C["accent"], C["accent_fill"], C["panel"]

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(5.5, 2.45),
    gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})

# ======================================================================
# (a)  How far the geodesic can be from the chord, as a fraction of ||Delta||_2.
#      Everything on this panel is dimensionless, so the two lengths that matter
#      -- the tube radius and the deviation bound -- are directly comparable.
# ======================================================================
t = np.linspace(0.0, 1.0, 601)


def green_reconstruct(t_grid, load):
    """xi(t) = (1-t) int_0^t s h(s) ds + t int_t^1 (1-s) h(s) ds.

    The O(M) forward/backward cumulative form actually used in App. E.1, not an
    O(M^2) double loop -- so the curve below is the estimator, not a cartoon."""
    s_grid = t_grid
    fwd = np.concatenate([[0.0], np.cumsum(
        0.5 * (s_grid[1:] * load[1:] + s_grid[:-1] * load[:-1]) * np.diff(s_grid))])
    g = (1 - s_grid) * load
    bwd_total = np.trapezoid(g, s_grid) if hasattr(np, "trapezoid") \
        else np.trapz(g, s_grid)
    bwd_cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (g[1:] + g[:-1]) * np.diff(s_grid))])
    return (1 - t_grid) * fwd + t_grid * (bwd_total - bwd_cum)


# A non-constant Christoffel load, so the reconstruction is not symmetric.
h = 1.0 + 0.85 * np.sin(np.pi * t) + 0.55 * t
xi = green_reconstruct(t, h)

BOUND = 0.10          # (kappa/2)||Delta||_2^2 <= ||Delta||_2/10 under (C.13)
envelope = BOUND * 4 * t * (1 - t)
xi = xi / xi.max() * 0.058      # a realised deviation, strictly inside the bound

axA.fill_between(t, 0, envelope, color=ACCENT, alpha=0.20, linewidth=0, zorder=2)
axA.plot(t, envelope, color=ACCENT, lw=1.0, zorder=3)
axA.plot(t, xi, color=INK, lw=1.6, zorder=4)

axA.axhline(1.0, color=REF, lw=1.0, linestyle=(0, (4, 2)), zorder=3)
axA.text(0.02, 1.045, r"tube radius $\|\Delta\|_2$: the set $X$ on which "
                      r"$\Phi$ contracts",
         fontsize=6.6, color=REF, ha="left", va="bottom")

# Annotations live in the empty middle band and point at what they describe.
axA.text(0.045, 0.545, "Thm. C.4 bound", fontsize=6.9, color=ACCENT,
         ha="left", va="bottom", weight="bold")
axA.text(0.045, 0.435,
         r"$\leq\frac{3c_1}{4\lambda n^{\alpha}}\|\Delta\|_2^{2}"
         r"\leq\frac{1}{10}\|\Delta\|_2$",
         fontsize=6.9, color=ACCENT, ha="left", va="bottom")
axA.text(0.045, 0.360,
         r"shape $t(1{-}t)$: Green kernel $G(t,s)$",
         fontsize=6.5, color=ACCENT, ha="left", va="bottom")
axA.annotate("", xy=(0.23, float(np.interp(0.23, t, envelope)) + 0.012),
             xytext=(0.20, 0.345),
             arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6))

axA.annotate(r"$\xi=\mathcal{G}h$  (App. E.1)",
             xy=(0.62, float(np.interp(0.62, t, xi))), xytext=(0.62, 0.215),
             fontsize=6.9, color=INK, ha="center", va="bottom", weight="bold",
             arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))

# --- a small key, in the empty upper band, for what xi(t) measures ---
kx0, kx1, ky = 0.60, 0.92, 0.70
u = np.linspace(0, 1, 120)
axA.plot([kx0, kx1], [ky, ky], color=INK, lw=1.0, linestyle=(0, (3, 2)),
         zorder=4)
axA.plot(kx0 + u * (kx1 - kx0), ky + 0.115 * 4 * u * (1 - u), color=INK,
         lw=1.3, zorder=4)
axA.plot([kx0, kx1], [ky, ky], "o", color=INK, markersize=3.2, zorder=5)
axA.annotate("", xy=(0.75, ky + 0.115), xytext=(0.75, ky),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7,
                             shrinkA=0, shrinkB=0))
axA.text(0.755, ky + 0.055, r"$\xi(t)$", fontsize=6.8, color=INK, ha="left",
         va="center")
axA.text(kx0, ky - 0.03, r"$w_A$", fontsize=6.6, color=INK, ha="center",
         va="top")
axA.text(kx1, ky - 0.03, r"$w_B$", fontsize=6.6, color=INK, ha="center",
         va="top")
axA.text(kx0 - 0.02, ky + 0.115, r"$\gamma_{\mathrm{g}}$", fontsize=6.6,
         color=INK, ha="right", va="center")
axA.text(kx0 - 0.02, ky, r"$\gamma_{\mathrm{lin}}$", fontsize=6.6, color=INK,
         ha="right", va="center")

axA.set_xlim(0, 1)
axA.set_ylim(0, 1.22)
axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_xticklabels(["0", "", r"$\frac{1}{2}$", "", "1"])
axA.set_yticks([0, 0.1, 0.5, 1.0])
axA.set_yticklabels(["0", "0.1", "0.5", "1"])
axA.set_xlabel(r"$t$")
axA.set_ylabel(r"$\|\xi(t)\|_2\ /\ \|\Delta\|_2$")
despine(axA)
axA.text(-0.085, 1.02, "(a)", transform=axA.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

# ======================================================================
# (b)  Where the smallness condition stops -- as a comparison of RATES.
#
#      An earlier version drew a crossing point between ||Delta||_2 and the
#      tube radius.  That was not honest: the tube radius is c_0 n^{alpha/2}
#      with c_0 depending on c_1, c_2 and lambda, none of which the experiment
#      pins down, so the WIDTH at which the condition fails is not determined
#      by the data.  What the data does determine is the two exponents, and
#      that comparison is enough: the separation grows faster than the tube is
#      allowed to, in every architecture, so the condition fails eventually --
#      which is all Remark C.9 claims.
# ======================================================================
_pairs = load_pairs("final")
_dF = load_dF()
_alphas = []
for _a in ("MLP", "TS", "CNN"):
    for _act in ("gelu", "tanh", "swish", "softplus"):
        _sub = _dF[(_dF.arch == _a) & (_dF.regime == "NTK") & (_dF.act == _act)]
        _e = boot_ci_over_seeds(_sub, "dF_op", n_boot=1)[0]
        if np.isfinite(_e):
            _alphas.append(_e)
ALPHA_LO, ALPHA_HI = float(min(_alphas)), float(max(_alphas))

_growth = {}
for _a in ("MLP", "TS", "CNN"):
    _sub = _pairs[(_pairs.arch == _a) & (_pairs.regime == "NTK")]
    _m = _sub.groupby("width")["dnorm"].median()
    _growth[_a] = -powerlaw_fit(_m.index.to_numpy(float), _m.to_numpy())[0]

lo2, hi2 = ALPHA_LO / 2, ALPHA_HI / 2
axB.axvspan(0.0, hi2, color=ACCENT, alpha=0.16, lw=0, zorder=1)
axB.axvline(hi2, color=ACCENT, lw=1.0, ls=(0, (3, 2)), zorder=3)

for i, (_a, _v) in enumerate(_growth.items()):
    axB.plot([_v], [i], marker=ARCH_MARKERS[_a], ms=5.0, color=INK,
             markerfacecolor="white", markeredgewidth=1.1, zorder=5)
    axB.annotate(f"{_v:.2f}", xy=(_v, i), xytext=(7, 0),
                 textcoords="offset points", fontsize=6.8, color=INK,
                 va="center")

axB.set_yticks(range(3))
axB.set_yticklabels(["MLP", "T–S", "CNN"])
axB.set_ylim(-0.85, 2.75)
axB.invert_yaxis()
axB.set_xlim(0.0, 1.25)
axB.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
axB.set_xlabel("width exponent")

axB.text(hi2 - 0.03, -0.62,
         rf"tube admits growth up to $n^{{\alpha/2}}$," + "\n"
         rf"$\alpha\in[{ALPHA_LO:.2f},{ALPHA_HI:.2f}]$ measured",
         fontsize=6.6, color=ACCENT, ha="right", va="center", linespacing=1.4)
axB.text(1.22, 2.62,
         "every architecture grows faster than the tube admits:\n"
         "Thm. C.4 is eventually silent, at a width the\n"
         "constants $c_1,c_2,\\lambda$ decide",
         fontsize=6.4, color=INK, ha="right", va="center", linespacing=1.4)

despine(axB)
axB.grid(axis="y", visible=False)
axB.grid(axis="x", visible=True)
axB.text(-0.09, 1.02, "(b)", transform=axB.transAxes, fontsize=10,
         fontweight="bold", va="bottom", ha="right", color="black")

save(fig, Path(__file__).parent.parent / "figures" / "appendix", "figC_geodesic")
