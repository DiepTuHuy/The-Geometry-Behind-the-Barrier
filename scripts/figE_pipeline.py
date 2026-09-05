#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figE_pipeline.py
================
Appendix E.1 -- the measurement pipeline.

One figure, one message (R1):
    "Every geometric number in the paper comes out of ONE matrix-free primitive,
     v |-> Fv, at O(P) memory.  The barrier comes out of a branch that never
     touches it.  Loss-side and Fisher-side quantities are produced by disjoint
     machinery -- which is exactly why they are free to disagree (App. D.2)."

Why this figure exists (figures/SURVEY_APPENDIX.md, criteria K3(b) and K3(c)):
a *measurement* pipeline that every number of the paper passes through is one of
the few things strong theory papers do draw in an appendix -- Singh & Jaggi
(NeurIPS 2020) Fig. S1, Ghorbani et al. (NeurIPS 2020) Fig. A.7 -- and Ghorbani,
Krishnan & Xiao (ICML 2019) spend three appendix figures on the *validity of the
estimator itself*, which is the job of the call-out at the foot here.

Colour discipline (R3 + survey conclusion K4): the regime palette
(NTK-lazy / Standard / muP) is deliberately absent, because nothing in this
figure is a regime.  The single accent is the Okabe-Ito bluish green reserved in
fig_style.py for schematics, so a highlighted box can never be misread as a
parameterisation.

Layout note: the chain inside the primitive is laid out from MEASURED text
extents, so the three operation labels (JVP / O(K) / VJP) always sit under their
own arrow whatever fonts the rendering machine happens to have.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig_style import apply_style, save, C
from fig_schematic import new_canvas, box, arrow, chain

apply_style()

INK, REF = C["ink"], C["ref"]
ACCENT, ACCENT_FILL, PANEL = C["accent"], C["accent_fill"], C["panel"]

# ----------------------------------------------------------------------
# Canvas: an abstract 0-100 grid so the layout arithmetic stays readable.
# The figure is saved at exactly \linewidth = 5.5 in (R5), so every font size
# below is a true physical size on the printed page.
# ----------------------------------------------------------------------
FIG_W, FIG_H = 5.5, 3.30
fig, ax = new_canvas(FIG_W, FIG_H)


# Partial-application shims so the layout code below reads unchanged.
_box, _arrow, _chain = box, arrow, chain
box = lambda *a, **k: _box(ax, *a, **k)
arrow = lambda *a, **k: _arrow(ax, *a, **k)
chain = lambda *a, **k: _chain(ax, *a, **k)


# ======================================================================
# BAND 1 -- what goes in, and the gauge fixed before anything is measured.
# ======================================================================
B1_Y, B1_H = 89.5, 20.0
box(2, 98, B1_Y, B1_H,
    [r"$3$ architectures $\times\ 3$ parameterisations $\times\ 4$ activations"
     r"$\ \times\ 7$ widths $n\in\{64,\dots,4096\}\ \times\ S\geq3$ seed pairs",
     r"gauge-fixed post hoc by layerwise weight matching $\Pi$   "
     r"(never during training)",
     r"$\Delta=\Pi w_B-w_A$,$\qquad$"
     r"$\gamma_{\mathrm{lin}}(t)=w_A+t\Delta$,$\quad t\in[0,1]$"],
    fc=PANEL, ec=INK, lw=1.0, fs=6.9,
    title=r"Trained endpoints $w_A,\,w_B$", title_fs=7.6,
    title_drop=2.8, body_drop=7.4)

# ======================================================================
# BAND 2 -- the single primitive that carries every Fisher-side quantity.
# ======================================================================
P_Y, P_H = 62.0, 22.0
box(26, 98, P_Y, P_H, [], fc=ACCENT_FILL, ec=ACCENT, lw=1.3)

ax.text(62, P_Y + P_H / 2 - 2.8,
        "Matrix-free Fisher–vector product   " r"$v\mapsto Fv$",
        ha="center", va="top", fontsize=7.7, color=ACCENT, weight="bold",
        zorder=4)
chain(28.5, 95.5, P_Y + 1.5,
      segments=[r"$v$", r"$u=J(x;w)\,v$",
                r"$a=p\odot u-p\,(p^{\top}u)$", r"$J(x;w)^{\top}a=Fv$"],
      ops=["JVP", r"$O(K)$", "VJP"])
ax.text(62, P_Y - P_H / 2 + 3.0,
        r"$O(P)$ memory:  $J\in\mathbb{R}^{K\times P}$ and "
        r"$F\in\mathbb{R}^{P\times P}$ are never formed",
        ha="center", va="center", fontsize=6.7, color=ACCENT, zorder=4)

arrow(62, B1_Y - B1_H / 2, 62, P_Y + P_H / 2, color=INK, lw=1.1)

# ======================================================================
# BAND 3 -- the four estimator families built on that one primitive.
# ======================================================================
E_Y, E_H = 28.0, 23.0
est = [
    (26.0, 42.5, "Fisher length",
     [r"$Q_F=\Delta^{\top}F\Delta$", r"unregularised $F$"]),
    (44.5, 61.0, "Rayleigh quotient",
     [r"$\widehat\Delta^{\top}F\widehat\Delta$",
      r"$\widehat\Delta=\Delta/\|\Delta\|_2$"]),
    (63.0, 79.5, "Christoffel load",
     [r"$\Gamma_{\gamma(s)}(\Delta,\Delta)$",
      r"$+\ \nabla_w Q_F$",
      r"$+$ CG on $F{+}\lambda I$"]),
    (81.5, 98.0, "Operator norms",
     [r"$\|\partial_z F\|_{\mathrm{op}}$", r"power iteration"]),
]
for x0, x1, title, lines in est:
    box(x0, x1, E_Y, E_H, lines, fc="white", ec=INK, lw=0.9, fs=6.8,
        title=title, title_fs=7.1, title_drop=2.8, body_drop=8.0)
    arrow(0.5 * (x0 + x1), P_Y - P_H / 2, 0.5 * (x0 + x1), E_Y + E_H / 2,
          color=ACCENT, lw=0.9)

# The Christoffel load is the only estimator with a downstream reconstruction.
arrow(71.25, E_Y - E_H / 2, 71.25, E_Y - E_H / 2 - 2.6, color=INK, lw=0.8,
      scale=6)
ax.text(71.0, E_Y - E_H / 2 - 6.0,
        r"$\xi_G=\mathcal{G}h$  (Green kernel, $O(M)$ not $O(M^2)$) "
        r"$\Rightarrow\ D_{\mathrm{rel}}=\sup_t\|\xi_G(t)\|_2/\|\Delta\|_2$",
        ha="center", va="center", fontsize=6.7, color=INK, zorder=4)

# ======================================================================
# LEFT BRANCH -- the barrier bypasses the primitive entirely.  That bypass is
# the substantive content of the diagram, not decoration.
# ======================================================================
box(2, 23, E_Y, E_H,
    [r"$\geq101$-point grid", r"$+$ local refinement,",
     r"grid-doubling check,",
     r"floor $\frac{1}{2}|L(w_A){-}L(w_B)|$"],
    fc="white", ec=REF, lw=0.9, fs=6.6,
    title=r"Barrier $B(\gamma_{\mathrm{lin}})$", title_fs=7.1,
    title_drop=2.4, body_drop=7.0)

arrow(12.5, B1_Y - B1_H / 2, 12.5, E_Y + E_H / 2, color=REF, lw=1.1,
      dash=(0, (3.5, 2.5)))
ax.text(14.2, 60.0, "loss only:", ha="left", va="center", fontsize=6.6,
        color=REF, weight="bold")
ax.text(14.2, 55.4, r"never touches $F$", ha="left", va="center", fontsize=6.6,
        color=REF)

# ======================================================================
# FOOT -- the substitution the pipeline refuses to make.  (Ghorbani et al. put
# estimator validity in the appendix; this is its one-line statement.)
# ======================================================================
ax.text(50, 6.2,
        r"Not interchangeable: Hutchinson traces answer a typical-direction "
        r"question, power iteration a worst-case one. A random probe returns",
        ha="center", va="center", fontsize=6.5, color=INK, style="italic")
ax.text(50, 1.8,
        r"$\mathbb{E}\,z^{\top}Az=\operatorname{tr}A/P="
        r"\|A\|_{\mathrm{op}}\,r_{\mathrm{eff}}(A)/P$, so with $NK\ll P$ it "
        r"reports a flattening exponent that is too large.",
        ha="center", va="center", fontsize=6.5, color=INK, style="italic")

save(fig, Path(__file__).parent.parent / "figures" / "appendix", "figE_pipeline")
