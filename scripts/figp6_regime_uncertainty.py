#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp6_regime_uncertainty.py -- Figure 6: Table 1 visualised.

Predictive uncertainty rho* at endpoints and midpoints, and midpoint accuracy,
across the three parameterization regimes.  Colour encodes the regime using the
paper-wide semantic mapping (R3); every bar carries its exact value (R2), so the
figure reads standalone next to the table.

FIXED.  The previous version put all three quantities on ONE unlabelled y axis.
Two of them are a fit residual rho* = (1/N) sum_n ||p_w(x_n) - e_{y_n}||_2, which
lives in [0, sqrt 2]; the third is an accuracy in [0, 1].  Sharing an axis
between them, with no axis label at all, invites the reader to compare a
residual against an accuracy as if the two were the same measurement -- and the
figure's actual claim is a comparison WITHIN each quantity across regimes, never
between quantities.  They are now two panels with their own labelled axes.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fig_style import (apply_style, despine, panel_letter, save, C, REGIMES,
                       REGIME_LABEL)
from fig_data import load_pairs

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"

# Measured, from data/final/*_pairs.csv: the median over cells of the per-cell
# median.  This reproduces Table 1 of the main text exactly
# (NTK 0.239 / 0.727 / 0.589; Standard 0.000 / 0.132 / 0.929;
#  muP 0.000 / 0.047 / 0.977), which is how the loader was checked.
_cells = (load_pairs("final")
          .groupby(["arch", "regime", "act", "width"])
          [["rho_A", "rho_mid", "acc_mid"]].median().reset_index())
_agg = _cells.groupby("regime")[["rho_A", "rho_mid", "acc_mid"]].median()

RHO = {r: [float(_agg.loc[r, "rho_A"]), float(_agg.loc[r, "rho_mid"])]
       for r in REGIMES}
ACC = {r: float(_agg.loc[r, "acc_mid"]) for r in REGIMES}
RHO_GROUPS = ["endpoint", "midpoint"]


def bars(ax, positions, values, regime, width):
    ax.bar(positions, values, width=width, color=C[regime],
           edgecolor="white", linewidth=0.4, zorder=5)
    for p, v in zip(positions, values):
        # A zero bar is invisible; a hairline stub keeps the slot legible.
        if v == 0.0:
            ax.plot([p - width / 2, p + width / 2], [0, 0], color=C[regime],
                    lw=1.4, solid_capstyle="butt", zorder=6)
        ax.annotate(f"{v:.3f}", xy=(p, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.6, color=C["ink"])


def main():
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 2.20),
        gridspec_kw={"width_ratios": [1.95, 1.0], "wspace": 0.26})

    w = 0.26

    # ---------------- (a) predictive uncertainty ------------------------
    for i, regime in enumerate(REGIMES):
        pos = [g + (i - 1) * w for g in range(2)]
        bars(ax_a, pos, RHO[regime], regime, w * 0.92)

    ax_a.axvline(0.5, color="white", lw=1.0, zorder=2)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(RHO_GROUPS)
    ax_a.set_ylim(0, 0.86)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax_a.set_ylabel(r"$\rho^{*}$  (fit residual, $\leq\sqrt{2}$)")
    ax_a.set_xlabel("anchor along $\\gamma_{\\mathrm{lin}}$", labelpad=1.5)
    panel_letter(ax_a, "a", dx=-0.045)
    despine(ax_a)
    ax_a.grid(axis="x", visible=False)
    ax_a.legend(handles=[Rectangle((0, 0), 1, 1, color=C[r],
                                       label=REGIME_LABEL[r]) for r in REGIMES],
                loc="upper left", frameon=False, handlelength=1.2,
                handleheight=0.85, borderaxespad=0.25, labelspacing=0.3)

    # R1: the message lives on the plot
    # Kept clear of the 0.727 bar, and above it in z: the previous placement
    # ran behind the bar and lost its first word.
    ax_a.annotate("separates the regimes\nonly at the midpoint",
                  xy=(1.45, 0.40), fontsize=6.6, style="italic", zorder=8,
                  color=C["ink"], ha="right", va="center", linespacing=1.4)

    # ---------------- (b) midpoint accuracy -----------------------------
    for i, regime in enumerate(REGIMES):
        bars(ax_b, [(i - 1) * w], [ACC[regime]], regime, w * 0.92)

    ax_b.set_xticks([0])
    ax_b.set_xticklabels(["midpoint"])
    ax_b.set_ylim(0, 1.09)
    ax_b.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_b.set_xlim(-0.52, 0.52)
    ax_b.set_ylabel("accuracy")
    panel_letter(ax_b, "b", dx=-0.10)
    despine(ax_b)
    ax_b.grid(axis="x", visible=False)

    save(fig, OUT, "figp6_regime_uncertainty")


if __name__ == "__main__":
    main()
