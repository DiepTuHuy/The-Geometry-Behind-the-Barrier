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
                       label_at, save, C)
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
    ax_a.set_xlim(50, 2.6e4)           # right margin hosts direct labels (R2)
    ax_a.set_ylim(2.2e-4, 4.5)
    ax_a.set_xlabel("width $n$")
    ax_a.set_ylabel("interpolation barrier $B$")
    ax_a.annotate("persists", xy=(1500, 2.2), fontsize=8, style="italic",
                  color=C["NTK"], ha="center")
    ax_a.annotate("collapses", xy=(300, 9e-4), fontsize=8, style="italic",
                  color=C["muP"], ha="center")
    # The Standard barrier is non-monotonic in width, so a single power-law
    # exponent would misdescribe it; it is labelled without one, as the text does.
    for regime, dy in (("NTK", 0), ("Standard", 3), ("muP", 0)):
        s = B[B["regime"] == regime]
        txt = {"NTK": "NTK-lazy", "Standard": "Standard",
               "muP": r"$\mu$P"}[regime]
        if regime != "Standard":
            txt += rf" ($\alpha_B{{=}}{exp_a[regime]:+.1f}$)"
        else:
            txt += " (non-monotonic)"
        label_at(ax_a, s["width"].iloc[-1], s["med"].iloc[-1], txt, C[regime],
                 dx=5, dy=dy, fontsize=7.0)
    panel_letter(ax_a, "a")
    despine(ax_a)

    # ---------------- panel (b): metric derivative ---------------------
    D = by_width(dF_mlp, "dF_op", keys=("regime",))
    exp_b = {}
    for regime in REGIMES:
        s = D[D["regime"] == regime]
        band(ax_b, s["width"], s["med"], s["q1"], s["q3"], regime)
        exp_b[regime] = powerlaw_fit(s["width"], s["med"])[0]

    ax_b.set_yscale("log")
    log_width_axis(ax_b)
    ax_b.set_xlim(50, 2.6e4)
    ax_b.set_ylim(1.4e-5, 1.0)
    ax_b.set_xlabel("width $n$")
    ax_b.set_ylabel(r"$\|\partial F\|_{\mathrm{op}}$")
    for regime, dy in (("Standard", 4), ("muP", -4), ("NTK", 0)):
        s = D[D["regime"] == regime]
        txt = {"NTK": "NTK-lazy", "Standard": "Standard",
               "muP": r"$\mu$P"}[regime]
        label_at(ax_b, s["width"].iloc[-1], s["med"].iloc[-1],
                 rf"{txt} ($\alpha_{{\partial F}}{{=}}{exp_b[regime]:.1f}$)",
                 C[regime], dx=5, dy=dy, fontsize=7.0)
    panel_letter(ax_b, "b")
    despine(ax_b)

    save(fig, OUT, "figp1_paradox")
    print("  measured alpha_B      :",
          {k: round(v, 2) for k, v in exp_a.items()})
    print("  measured alpha_dF     :",
          {k: round(v, 2) for k, v in exp_b.items()})


if __name__ == "__main__":
    main()
