#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figR_roadmap.py
===============
Appendix roadmap -- the dependency graph of the appendix results.

One figure, one message (R1):
    "Appendices A-C are one connected chain of positive results; Appendix D is
     not a weaker corollary of them but the place where two of them are stopped
     by the scale established in B.2; E-F measure, they do not prove."

Why this figure exists (figures/SURVEY_APPENDIX.md, criterion K3(d)):
opening a long appendix with a proof-dependency graph is an established
convention in recent theory papers with heavy appendices -- e.g. Tang, Li & Zou
(2025) place one before Appendix A, nodes = theorems and lemmas, a directed edge
A -> B meaning "B's proof uses A".  This appendix has thirteen subsections and
2.7k lines of proof with dense cross-references, which is exactly the case the
convention exists for.

The graph carries TWO edge types, and the distinction is the substantive point:

    solid   "used in the proof of"
    dashed  "bounds the regime in which it applies"

The dashed edges are the ones that matter.  Neither of them is a contradiction:
Remark B.6 says precisely that the local machinery does not extend to the scale
at which linear mode connectivity is posed, and drawing that as a different kind
of arrow keeps a reader from mistaking Appendix D for a corollary of A-C.

Colour discipline (R3, survey conclusion K4): no regime colours -- nothing here
is a parameterisation.  Grey = limitation results, bluish-green accent =
measurement, ink = positive results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from fig_style import apply_style, save, C
from fig_schematic import new_canvas, box, arrow

apply_style()

INK, REF = C["ink"], C["ref"]
ACCENT, ACCENT_FILL, PANEL = C["accent"], C["accent_fill"], C["panel"]

fig, ax = new_canvas(5.5, 3.80)

# ----------------------------------------------------------------------
# Level geometry.  Boxes never occupy the horizontal corridors between levels
# nor the vertical corridor at x = 50, so every edge below is routed through
# empty space and no edge crosses a node.
# ----------------------------------------------------------------------
L1, L2, L3, L4 = 87.0, 64.5, 41.5, 19.0     # level centres
H1, H2, H3, H4 = 15.0, 16.0, 16.0, 15.0     # level heights
COL = [14.0, 38.0, 62.0, 86.0]              # column centres, levels 1-2
W = 22.0                                    # box width, levels 1-2
CORR_A, CORR_B = 51.5, 28.4                 # horizontal corridors
CORR_X = 50.0                               # vertical corridor


def node(xc, yc, w, h, tag, title, lines, *, ec=INK, fc="white", fs=6.1,
         title_fs=6.9, title_col=None):
    """A result node: bold appendix tag + name, statement underneath.

    The frame colour carries the category (ink = positive result, grey =
    limitation, accent = measurement); the title stays near-black so a
    limitation node is not read as merely faded."""
    box(ax, xc - w / 2, xc + w / 2, yc, h, lines, fc=fc, ec=ec, lw=0.9, fs=fs,
        title=f"{tag}  {title}", title_fs=title_fs, title_col=title_col or ec,
        title_drop=2.6, body_drop=7.0, rounding=1.3)


def edge(x0, y0, x1, y1, *, dashed=False, color=None, lw=0.85):
    arrow(ax, x0, y0, x1, y1,
          color=color or (REF if dashed else INK), lw=lw, scale=7,
          dash=(0, (3.0, 2.2)) if dashed else None)


def elbow(points, *, dashed=False, color=None, lw=0.85):
    """Orthogonal multi-segment edge; arrowhead on the final segment."""
    col = color or (REF if dashed else INK)
    ls = (0, (3.0, 2.2)) if dashed else "solid"
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=col, lw=lw, linestyle=ls, zorder=2,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=7,
        color=col, linewidth=lw, shrinkA=0, shrinkB=0, zorder=2, linestyle=ls))


# ======================================================================
# LEVEL 1 -- the four inputs that are not proved from anything else here.
# ======================================================================
node(COL[0], L1, W, H1, "A.1", "Fisher properties",
     [r"$F\succeq0$;", r"$\operatorname{rank}F_N\leq N(K{-}1)$"])
node(COL[1], L1, W, H1, "A.2", "Fisher–Hessian",
     [r"$\Delta^{\top}\nabla^{2}\!L\,\Delta$", r"$=Q_F+\mathcal{E}$-term"])
node(COL[2], L1, W, H1, "B.1", "Lazy flattening",
     [r"$\|\partial_z F\|_{\mathrm{op}}$", r"$=O(n^{-\alpha})$"])
node(COL[3], L1, W, H1, "C.1", "Duality lemma",
     [r"$m_w(\Delta,\Delta)$", r"$=\nabla_w Q_F$"])

# ======================================================================
# LEVEL 2 -- the paper's positive results.
# ======================================================================
node(COL[0], L2, W, H2, "A.3", "Fisher–barrier",
     [r"$B\leq\min\{R(w_A),$", r"$R(w_B)\}+\delta$"])
node(COL[1], L2, W, H2, "B.2", "Independent init",
     [r"$\|\Delta\|_2=\Theta(\sqrt{P})$", r"$=\Theta(n)$  (MLP)"])
node(COL[2], L2, W, H2, "C.2–C.3", "Flattening",
     [r"$\|\Gamma\|=O(n^{-\alpha})$,", r"$|\mathrm{Sec}|=O(n^{-2\alpha})$"])
node(COL[3], L2, W, H2, "C.5", "Green operator",
     [r"$\xi=\mathcal{G}h$ solves $\ddot\xi=-h$,",
      r"$\|\mathcal{G}h\|_{C^1}\leq\frac{5}{8}\|h\|_{\infty}$"])

# ======================================================================
# LEVEL 3 -- one positive result with a restricted regime, one limitation.
# ======================================================================
node(24.0, L3, 42.0, H3, "D.1", "No lower bound on Fisher length",
     [r"constant $F_\varepsilon$ satisfies every flattening",
      r"conclusion, yet $Q_F=\varepsilon\|\Delta\|_2^{2}$ is free"],
     ec=REF, fc=PANEL, title_col=INK)
node(76.0, L3, 42.0, H3, "C.4", "Geodesic deviation & existence",
     [r"fixed point in the tube $T_\Delta$; valid only",
      r"while $\|\Delta\|_2\lesssim n^{\alpha/2}$"])

# ======================================================================
# LEVEL 4 -- the limitation the whole appendix is organised around, and the
# measurement branch, which proves nothing and is drawn apart from the graph.
# ======================================================================
node(24.0, L4, 42.0, H4, "D.2", "Flattening does not certify collapse",
     [r"$R(w_0)=\Theta(P^{3/2})\to\infty$ while the floor is $O(1)$;",
      r"the assumptions admit no lower bound at all"],
     ec=REF, fc=PANEL, title_col=INK)
node(76.0, L4, 42.0, H4, "E.1 $\\to$ F.1", "Measurement",
     [r"matrix-free estimators; width exponents",
      r"and normalised ratios only"],
     ec=ACCENT, fc=ACCENT_FILL)

# ======================================================================
# EDGES
# ======================================================================
b1, t2 = L1 - H1 / 2, L2 + H2 / 2
b2, t3 = L2 - H2 / 2, L3 + H3 / 2
b3, t4 = L3 - H3 / 2, L4 + H4 / 2

# "used in the proof of"
edge(COL[0], b1, COL[0], t2)                       # A.1 -> A.3
edge(COL[1] - 6, b1, COL[0] + 7, t2)               # A.2 -> A.3
edge(COL[2] - 6, b1, COL[1] + 7, t2)               # B.1 -> B.2
edge(COL[3] - 6, b1, COL[2] + 7, t2)               # C.1 -> C.2-C.3
edge(COL[2] + 2, b2, 70.0, t3)                     # C.2-C.3 -> C.4
edge(COL[3], b2, 84.0, t3)                         # C.5     -> C.4
edge(COL[0], b2, 16.0, t3)                         # A.3     -> D.1
edge(18.0, b3, 18.0, t4)                           # D.1     -> D.2

# "bounds the regime in which it applies" -- the two edges that matter
edge(COL[1] + 7, b2, 62.0, t3, dashed=True)        # B.2 -> C.4  (tube radius)
elbow([(COL[1] + 2, b2), (COL[1] + 2, CORR_A), (CORR_X, CORR_A),
       (CORR_X, CORR_B), (32.0, CORR_B), (32.0, t4)], dashed=True)  # B.2 -> D.2

# measurement branch, drawn in the accent so it reads as a different kind of link
edge(76.0, b3, 76.0, t4, dashed=True, color=ACCENT)

# ======================================================================
# LEGEND -- two edge types, stated once (R8: in empty space, no frame)
# ======================================================================
handles = [
    Line2D([], [], color=INK, lw=0.9, label="used in the proof of"),
    Line2D([], [], color=REF, lw=0.9, linestyle=(0, (3.0, 2.2)),
           label="bounds the regime in which it applies"),
    Line2D([], [], color=ACCENT, lw=0.9, linestyle=(0, (3.0, 2.2)),
           label="measured, not proved"),
]
ax.legend(handles=handles, loc="lower center", ncol=3,
          bbox_to_anchor=(0.5, 0.0), frameon=False, handlelength=2.0,
          columnspacing=1.6, handletextpad=0.5, fontsize=6.6)

save(fig, Path(__file__).parent.parent / "figures" / "appendix", "figR_roadmap")
