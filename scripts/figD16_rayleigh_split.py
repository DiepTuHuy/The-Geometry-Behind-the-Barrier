#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figD16_rayleigh_split.py -- where the Rayleigh quotient's decay comes from.

Section 5 reads R_F -> 0 as evidence that the displacement "increasingly
concentrates in the low-eigenvalue subspace of F".  That reading has two
competing causes and R_F alone cannot separate them:

    directional   the spectrum stays put and Delta-hat rotates down it
    global        every eigenvalue shrinks and Delta-hat sits exactly where a
                  random direction would

The separator is an identity: for v uniform on the unit sphere,
E[v'Fv] = tr F / P exactly.  So the alignment ratio

    A = R_F / (tr F / P)

is 1 when Delta-hat is spectrally indistinguishable from a random direction,
and because R_F = (tr F/P) * A the width exponents are additive:

    alpha_{R_F} = alpha_{tr F/P} + alpha_A

(a) shows A itself against width, with A = 1 drawn as the random-direction
reference.  (b) shows the split: the bar is alpha_{tr F/P}, the marker is
alpha_{R_F}, and the gap between them IS alpha_A.

DATA: the canonical table ../data/rayleigh/rayleigh_cells.csv, written by
`src/10_rayleigh/measure_rayleigh.py merge`.  Only cells carrying tr F/P appear,
because the alignment ratio needs it and no earlier run measures a trace at all;
`rayleigh_canon_cells(need=...)` applies that filter together with the >=3-width
threshold a fitted exponent requires.  Everything annotated is computed here.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from fig_style import (apply_style, despine, panel_letter, save, C,
                       REGIME_LABEL, log_width_axis)
from fig_data import rayleigh_canon_cells

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"

# Colour carries the parameterisation (R3).  Three activations share the muP
# colour, so the activation has to be carried by the dash pattern instead.
ACT_LS = {"relu": (0, (1, 1.2)), "gelu": "-", "tanh": (0, (5, 1.6)),
          "swish": (0, (4, 1.3, 1, 1.3)), "softplus": (0, (2.4, 1.2))}
ACT_LABEL = {"relu": "ReLU", "gelu": "GELU", "tanh": r"$\tanh$",
             "swish": "Swish", "softplus": "Softplus"}
# Panel (a)'s key has to stay narrow enough not to sit on the muP/tanh curve,
# and the panel states the regime mapping through colour anyway.
SHORT = {"NTK": "NTK", "Standard": "SP", "muP": r"$\mu$P"}


def cell_rows(cells):
    """One record per cell, ordered so the legend reads NTK, Standard, muP."""
    order = {"NTK": 0, "Standard": 1, "muP": 2}
    out = []
    for (arch, regime, act), g in cells.groupby(["arch", "regime", "act"]):
        g = g.sort_values("width")
        out.append(dict(arch=arch, regime=regime, act=act,
                        width=g["width_paper"].to_numpy(float),
                        A=g["align_mid_rat"].to_numpy(float),
                        a_rq=float(g["alpha_rq"].iloc[0]),
                        a_trP=float(g["alpha_trP"].iloc[0]),
                        a_align=float(g["alpha_align"].iloc[0]),
                        r2=float(g["alpha_align_r2"].iloc[0])))
    out.sort(key=lambda r: (r["arch"], order[r["regime"]], r["act"]))
    return out


def panel_a(ax, rows):
    """A against width, with the random-direction reference at A = 1."""
    ref_style = dict(color=C["ref"], ls=(0, (4, 2)), lw=0.9)
    ax.axhline(1.0, zorder=2, **ref_style)

    # The key lives BELOW the figure, not inside this panel.  Measured: a
    # seven-row in-panel table is 0.62 of the axes height, and clearing the
    # rising muP/tanh curve under it would need seven decades of y range for
    # three decades of data -- the data would fill a third of the panel.  A
    # shared legend costs one strip of canvas and hands the panel back.
    handles = [Line2D([], [], **ref_style)]
    labels = ["random direction  ($A=1$)"]
    for r in rows:
        style = dict(color=C[r["regime"]], ls=ACT_LS[r["act"]], lw=1.25)
        ax.plot(r["width"], r["A"], marker="o", ms=2.6,
                markerfacecolor=C[r["regime"]], markeredgecolor="white",
                markeredgewidth=0.4, zorder=5, **style)
        handles.append(Line2D([], [], marker="o", ms=2.6,
                              markerfacecolor=C[r["regime"]],
                              markeredgecolor="white", markeredgewidth=0.4,
                              **style))
        labels.append(f"{SHORT[r['regime']]} {ACT_LABEL[r['act']]}")

    ax.set_yscale("log")
    log_width_axis(ax)
    ax.set_xlabel("width $n$")
    ax.set_ylabel(r"alignment ratio  $A=\mathcal{R}_F\,/\,(\mathrm{tr}\,F/P)$")
    despine(ax)
    allA = np.concatenate([r["A"] for r in rows])
    ax.set_ylim(allA.min() / 1.8, allA.max() * 1.8)
    return handles, labels


def panel_b(ax, rows):
    """alpha_{tr F/P} as the bar, alpha_{R_F} as the marker; the gap is alpha_A."""
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        col = C[r["regime"]]
        ax.barh(yi, r["a_trP"], height=0.52, color=col, alpha=0.80,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([r["a_rq"]], [yi], marker="D", ms=4.4, color=C["ink"],
                markerfacecolor="white", markeredgewidth=1.0, zorder=6,
                linestyle="none")
        # The gap is the whole point, so draw it whenever it is visible.
        if abs(r["a_align"]) > 0.08:
            ax.annotate("", xy=(r["a_rq"], yi), xytext=(r["a_trP"], yi),
                        arrowprops=dict(arrowstyle="-|>", color=C["ink"],
                                        lw=0.8, shrinkA=0, shrinkB=2),
                        zorder=7)
            ax.annotate(rf"$\alpha_A={r['a_align']:+.2f}$",
                        xy=(0.5 * (r["a_rq"] + r["a_trP"]), yi),
                        xytext=(0, -9.0), textcoords="offset points",
                        fontsize=6.2, color=C["ink"], ha="center", va="top",
                        zorder=7)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{REGIME_LABEL[r['regime']]}\n{ACT_LABEL[r['act']]}"
                        for r in rows], fontsize=6.4)
    ax.set_xlabel(r"width exponent  ($\alpha>0$: decreases with $n$)")
    ax.axvline(0.0, color="#4d4d4d", lw=0.7, zorder=4)
    despine(ax)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.75, 2.95)
    # An empty band above the top bar, so the key never covers a diamond.
    # Room below the bottom row for its alpha_A label, and above the top row
    # for the key, so neither ever lands on a bar.
    ax.set_ylim(-1.05, len(rows) - 1 + 1.85)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C["ink"], alpha=0.45,
                             label=r"$\alpha_{\mathrm{tr}F/P}$  (spectrum)"),
                       Line2D([], [], marker="D", ms=4.4, color=C["ink"],
                              markerfacecolor="white", markeredgewidth=1.0,
                              linestyle="none",
                              label=r"$\alpha_{\mathcal{R}_F}$  (measured)")],
              loc="upper center", ncol=2, fontsize=6.3, handlelength=1.5,
              borderpad=0.35, labelspacing=0.3, columnspacing=1.1,
              framealpha=0.92, frameon=True, edgecolor="#c8c8c8",
              facecolor="white")


def main():
    apply_style()
    rows = cell_rows(rayleigh_canon_cells(need='align_mid_rat'))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 2.75), gridspec_kw={"wspace": 0.42,
                                                "width_ratios": [1.0, 1.0]})
    handles, labels = panel_a(ax_a, rows)
    panel_b(ax_b, rows)
    panel_letter(ax_a, "a")
    panel_letter(ax_b, "b")
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.4,
               frameon=False, handlelength=2.0, columnspacing=1.5,
               handletextpad=0.5, bbox_to_anchor=(0.5, -0.01))
    save(fig, OUT, "figD16_rayleigh_split")

    # Print what was plotted, so a number quoted in the text is checkable.
    al = np.array([r["a_align"] for r in rows])
    tp = np.array([r["a_trP"] for r in rows])
    print(f"n cells = {len(rows)}")
    print(f"  alpha_A    median {np.median(al):+.3f}  "
          f"range [{al.min():+.3f}, {al.max():+.3f}]  "
          f"|alpha_A| <= 0.06 in {(np.abs(al) <= 0.06).sum()}/{len(al)} cells")
    print(f"  alpha_trF/P median {np.median(tp):+.3f}  "
          f"range [{tp.min():+.3f}, {tp.max():+.3f}]")
    for r in rows:
        print(f"  {r['regime']:9s} {r['act']:9s} "
              f"alpha_rq={r['a_rq']:+.3f} = alpha_trP={r['a_trP']:+.3f} "
              f"+ alpha_A={r['a_align']:+.3f}  (A fit R2={r['r2']:.2f}, "
              f"A spans {r['A'].min():.3f}-{r['A'].max():.3f})")


if __name__ == "__main__":
    main()
