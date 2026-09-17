#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figD17_rayleigh_profile.py -- the midpoint is a dip, not a typical point.

Every Rayleigh and Fisher-length number in the paper is read at t = 0, 1/2 or 1,
and the scaling figures regress on the MIDPOINT.  Measuring R_F across the whole
path shows the midpoint is not a typical point of it but a sharp local MINIMUM
sitting between two humps -- and the dip deepens as width grows.

(a) the measured profile at n = 4096, three parameterisations, median over the
    activations present, each curve normalised by its own R_F(0).
(b) the dip depth  max_t R_F(t) / R_F(1/2)  against width, one curve per cell.

Why it matters: width is the axis every exponent in the paper is fitted along,
so a midpoint anchor whose relation to the path CHANGES with width is a moving
reference.  This figure does not show that any published exponent is wrong -- it
shows that the anchor deserves the check, and reports the size of the effect.

DATA.  The canonical table ../data/rayleigh/rayleigh_profile.csv, written by
`src/10_rayleigh/measure_rayleigh.py merge`.  Two runs measured this profile and
the merge already resolved them, keeping the finer grid wherever both exist:

  src/04_profile/measure_profile_length.py  84 cells, 840 pairs, 21-point grid
  src/10_rayleigh/measure_rayleigh.py       42 cells, 420 pairs,  9-point grid

They are independent and they agree: over the 1800 rows they share, the maximum
relative deviation is 7e-05 -- float summation order, nothing more.  So the
precedence is not about trust but coverage: the 21-point grid resolves where the
humps sit, and a 9-point grid understates the dip depth by up to 13%.  The `src`
column records which run each row came from.

ARCHITECTURE.  MLP only, because that is the only architecture whose R_F(t) has
been measured at all.  `measure_rayleigh.py audit` lists CNN and TS as gaps; when
they are filled, set ARCH or loop it.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fig_style import (apply_style, despine, panel_letter, save, C,
                       REGIMES, REGIME_LABEL, log_width_axis, legend_table)
from fig_data import load_rayleigh_canon

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"

ARCH = "MLP"        # the only architecture with an R_F(t) profile so far
REF_WIDTH = 4096
REG_STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
             "muP": ((0, (1, 2.2)), "^")}
# ReLU is here because the canonical table carries it: R_F takes no derivative
# of F, so the C^3 requirement that excluded ReLU from the Christoffel run does
# not apply, and dropping it would discard measured cells.
ACT_LS = {"relu": (0, (1, 1.2)), "gelu": "-", "tanh": (0, (5, 1.6)),
          "swish": (0, (4, 1.3, 1, 1.3)), "softplus": (0, (2.4, 1.2))}
ACT_ORDER = ["relu", "gelu", "tanh", "swish", "softplus"]


def panel_a(ax, prof):
    """R_F(t)/R_F(0) at the reference width, by regime."""
    ax.axvline(0.5, color=C["ref"], lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.annotate("anchor", xy=(0.5, 0.985), xytext=(3.0, 0.0),
                textcoords="offset points", xycoords=("data", "axes fraction"),
                fontsize=6.4, color=C["ref"], ha="left", va="top",
                rotation=90, zorder=3)
    ax.axhline(1.0, color=C["ref"], lw=0.7, ls=":", zorder=2)

    at_w = prof[prof["width"] == REF_WIDTH]
    rows = []
    for regime in REGIMES:
        g = at_w[at_w["regime"] == regime]
        if g.empty:
            continue
        m = g.groupby("t")["rq_t"].median()
        ls, marker = REG_STYLE[regime]
        style = dict(color=C[regime], ls=ls, lw=1.4)
        ax.plot(m.index.to_numpy(), (m / m.iloc[0]).to_numpy(),
                marker=marker, ms=2.8, markerfacecolor=C[regime],
                markeredgecolor="white", markeredgewidth=0.4, zorder=5,
                **style)
        rows.append((dict(style, marker=marker, ms=2.8),
                     REGIME_LABEL[regime], None))

    ax.set_yscale("log")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n$w_A$", "", r"$\frac{1}{2}$", "", "1\n$w_B$"])
    ax.set_xlabel("position along the linear path  $t$")
    ax.set_ylabel(r"$\mathcal{R}_F(t)\,/\,\mathcal{R}_F(0)$")
    ax.set_title(f"$n={REF_WIDTH}$, MLP", fontsize=7.5, pad=3, loc="left")
    despine(ax)
    # The curves peak either side of t = 1/2 and fall to 1 at both ends, so the
    # bottom-left corner is the one region all three have left.
    legend_table(ax, rows, loc="lower left", fontsize=6.2)


def panel_b(ax, prof):
    """Dip depth against width, one curve per cell."""
    per_cell = {}
    for (regime, act), g in prof.groupby(["regime", "act"]):
        ws, dep = [], []
        for w, gw in g.groupby("width"):
            m = gw.groupby("t")["rq_t"].median()
            if 0.5 not in m.index:
                continue
            ws.append(float(w))
            dep.append(float(m.max() / m.loc[0.5]))
        # A cell needs three widths before "the dip deepens with width" is a
        # statement about anything; the partial ReLU cells have one or two.
        if len(ws) < 3:
            continue
        order = np.argsort(ws)
        ws, dep = np.array(ws)[order], np.array(dep)[order]
        per_cell[(regime, act)] = (ws, dep)
        ax.plot(ws, dep, color=C[regime], ls=ACT_LS.get(act, "-"), lw=1.15,
                marker="o", ms=2.4, markerfacecolor=C[regime],
                markeredgecolor="white", markeredgewidth=0.35, zorder=5)

    ax.axhline(1.0, color=C["ref"], lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.annotate("no dip", xy=(4096, 1.0), xytext=(-1.5, 2.0),
                textcoords="offset points", fontsize=6.4, color=C["ref"],
                ha="right", va="bottom", zorder=3)
    ax.set_yscale("log")
    log_width_axis(ax)
    ax.set_xlabel("width $n$")
    ax.set_ylabel(r"dip depth  $\max_t \mathcal{R}_F(t)\,/\,\mathcal{R}_F(\frac{1}{2})$")
    despine(ax)
    return per_cell


def main():
    apply_style()
    prof = load_rayleigh_canon('profile')
    prof = prof[prof['arch'] == ARCH]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 2.70), gridspec_kw={"wspace": 0.40})
    panel_a(ax_a, prof)
    per_cell = panel_b(ax_b, prof)
    panel_letter(ax_a, "a")
    panel_letter(ax_b, "b")

    # Colour is the regime (R3); within a regime the four activations differ by
    # dash pattern only, so the activation key is shared and goes below.
    present = [a for a in ACT_ORDER if a in set(prof["act"])]
    handles = [Line2D([], [], color=C["ink"], ls=ACT_LS[a], lw=1.15) for a in present]
    labels = [{"relu": "ReLU", "gelu": "GELU", "tanh": r"$\tanh$",
               "swish": "Swish", "softplus": "Softplus"}[a] for a in present]
    fig.subplots_adjust(bottom=0.28)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=6.4,
               frameon=False, handlelength=2.2, columnspacing=1.6,
               handletextpad=0.5, bbox_to_anchor=(0.5, -0.01))
    save(fig, OUT, "figD17_rayleigh_profile")

    n_cells = len(per_cell)
    grew = sum(1 for (_, d) in per_cell.values() if d[-1] > d[0])
    print(f"n cells = {n_cells}  ({ARCH}; cells with >=3 widths)")
    print(f"  dip deepens with width in {grew}/{n_cells} cells")
    for (regime, act), (ws, d) in sorted(per_cell.items()):
        print(f"  {regime:9s} {act:9s} {d[0]:6.1f}x at n={int(ws[0])}"
              f"  ->{d[-1]:7.1f}x at n={int(ws[-1])}")


if __name__ == "__main__":
    main()
