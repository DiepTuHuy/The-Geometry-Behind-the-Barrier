#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp1_paradox.py -- Figure 1: the central empirical contrast.

Panel (a): interpolation barrier B vs width in the three parameterization
           regimes (collapses in Standard/muP, persists in NTK-lazy).
Panel (b): metric-flattening quantity ||dF||_op decays as a power law in
           EVERY regime.

DATA: measured, from ../data/ (see fig_data.py).  MLP / MNIST, median over the
four smooth activations and all seed pairs; band = interquartile range.  Every
exponent printed on the plot is fitted from the points being drawn, so the
annotation cannot drift away from the curve.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, panel_letter, log_width_axis,
                       legend_table, save, C)
from fig_data import load_pairs, load_dF, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"
REGIMES = ["NTK", "Standard", "muP"]
STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
         "muP": ((0, (1, 2.2)), "^")}


def band(ax, x, y, lo, hi, regime):
    ls, marker = STYLE[regime]
    col = C[regime]
    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(x, y, color=col, lw=1.7, ls=ls, marker=marker, ms=3.5,
            markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.5,
            zorder=5)


def main():
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 2.15), gridspec_kw={"wspace": 0.30})

    pairs = load_pairs("final")
    mlp = pairs[pairs["arch"] == "MLP"]
    dF = load_dF()
    dF_mlp = dF[dF["arch"] == "MLP"]

    # ---------------- panel (a): barrier -------------------------------
    B = by_width(mlp, "B", keys=("regime",))
    exp_a = {}
    for regime in REGIMES:
        s = B[B["regime"] == regime]
        band(ax_a, s["width"], s["med"], s["q1"], s["q3"], regime)
        exp_a[regime] = powerlaw_fit(s["width"], s["med"])[0]

    ax_a.set_yscale("log")
    log_width_axis(ax_a)
    # The old right margin out to 2.6e4 existed only to host the end-of-line
    # labels (R2); the key is a table inside the panel now, and the headroom
    # above the data is what the table occupies.
    ax_a.set_xlim(50, 5.6e3)
    # Panel (a) needs more headroom than (b): its NTK curve RISES, so the key
    # has to clear a band the data is still climbing into at the widest widths.
    ax_a.set_ylim(2.2e-4, 300.0)
    ax_a.set_xlabel("width $n$")
    ax_a.set_ylabel("interpolation barrier $B$")
    # Below the NTK curve, not above it: the "non-monotonic" cell makes the key
    # wide enough to reach this far right, and the gap between the NTK and
    # Standard curves at the widest widths is the clear band here.
    ax_a.annotate("persists", xy=(2600, 0.30), fontsize=8, style="italic",
                  color=C["NTK"], ha="center", va="center")
    ax_a.annotate("collapses", xy=(300, 9e-4), fontsize=8, style="italic",
                  color=C["muP"], ha="center")
    # The Standard barrier is non-monotonic in width, so a single power-law
    # exponent would misdescribe it; its row carries the word instead of a number.
    rows_a = [(dict(color=C[r], ls=STYLE[r][0], marker=STYLE[r][1], lw=1.7,
                    ms=3.5),
               {"NTK": "NTK-lazy", "Standard": "Standard",
                "muP": r"$\mu$P"}[r],
               "non-monotonic" if r == "Standard"
               else rf"$\alpha_B{{=}}{exp_a[r]:+.1f}$")
              for r in REGIMES]
    panel_letter(ax_a, "a")
    despine(ax_a)
    # No table title here, unlike Figures 2 and 3.  Panel (a)'s NTK curve is
    # nearly flat at B ~ 1 across the whole grid, so the key has to clear that
    # band; a fourth row would need the axis to run to ~4000 to stay off the
    # data, which is a panel of mostly empty space.  Both panels are MLP/MNIST,
    # so the caption carries what the title would have said.
    legend_table(ax_a, rows_a, loc="upper left")

    # ---------------- panel (b): metric derivative ---------------------
    D = by_width(dF_mlp, "dF_op", keys=("regime",))
    exp_b = {}
    for regime in REGIMES:
        s = D[D["regime"] == regime]
        band(ax_b, s["width"], s["med"], s["q1"], s["q3"], regime)
        exp_b[regime] = powerlaw_fit(s["width"], s["med"])[0]

    ax_b.set_yscale("log")
    log_width_axis(ax_b)
    ax_b.set_xlim(50, 5.6e3)
    ax_b.set_ylim(1.4e-5, 40.0)
    ax_b.set_xlabel("width $n$")
    ax_b.set_ylabel(r"$\|\partial F\|_{\mathrm{op}}$")
    rows_b = [(dict(color=C[r], ls=STYLE[r][0], marker=STYLE[r][1], lw=1.7,
                    ms=3.5),
               {"NTK": "NTK-lazy", "Standard": "Standard",
                "muP": r"$\mu$P"}[r],
               rf"$\alpha_{{\partial F}}{{=}}{exp_b[r]:.1f}$")
              for r in REGIMES]
    panel_letter(ax_b, "b")
    despine(ax_b)
    legend_table(ax_b, rows_b, loc="upper right")

    save(fig, OUT, "figp1_paradox")
    print("  measured alpha_B      :",
          {k: round(v, 2) for k, v in exp_a.items()})
    print("  measured alpha_dF     :",
          {k: round(v, 2) for k, v in exp_b.items()})


if __name__ == "__main__":
    main()
