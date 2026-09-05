#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp3_rayleigh.py -- Figure 3: spectral alignment, and where it stops.

Panel (a): R_F vs width for the MLP, three parameterisations.
Panel (b): R_F vs width in the Standard regime, three architectures.

Why panel (b) exists.  The earlier version showed panel (a) only and its
caption claimed decay is "steep in every parameterisation".  On the plotted
architecture that is not true at large width: the Standard MLP SATURATES from
n = 1024 on (0.0082 -> 0.0070 -> 0.0070), so its fitted exponent comes entirely
from the first half of the grid, and its R^2 = 0.87 is the lowest of the nine
cells while the other eight are >= 0.99.  Panel (b) shows the saturation is a
property of that one cell and not of the Standard regime: the Standard
teacher-student and CNN keep decaying across the whole grid.

The paper's claim survives -- spectral alignment does decay in every regime, so
it cannot be what separates them -- but the reader can now see the one place
where the power law runs out, instead of being told it does not happen.

COLOUR: panel (a) has a parameterisation dimension, so colour carries the
regime; panel (b) is one regime, so colour is free and carries the
architecture (ARCH_COLORS, disjoint from the regime triple).  Each panel says
which.

DATA: measured, from ../data/ via fig_data.load_pairs("geo"); median over the
four smooth activations and all seed pairs, band = interquartile range, each
curve normalised by its own smallest-width value.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, label_at, log_width_axis,
                       panel_letter, save, C, REGIMES, REGIME_LABEL,
                       ARCH_COLORS, ARCH_LABEL, ARCH_MARKERS, WIDTHS)
from fig_data import load_pairs, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"

REG_STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
             "muP": ((0, (1, 2.2)), "^")}
ARCH_LS = {"MLP": "-", "TS": (0, (5.0, 1.7)), "CNN": (0, (1.3, 1.9))}


def curve(ax, s, colour, ls, marker, lw=1.7, ms=3.4):
    s = s.sort_values("width")
    ref = s["med"].iloc[0]
    x = s["width"].to_numpy(float)
    y = (s["med"] / ref).to_numpy(float)
    ax.fill_between(x, s["q1"] / ref, s["q3"] / ref, color=colour,
                    alpha=0.15, lw=0)
    ax.plot(x, y, color=colour, ls=ls, lw=lw, marker=marker, ms=ms,
            markerfacecolor=colour, markeredgecolor="white",
            markeredgewidth=0.5, zorder=5)
    e, r2, _ = powerlaw_fit(x, y)
    return x, y, e, r2


def main():
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 2.30),
                                     gridspec_kw={"wspace": 0.34})
    geo = load_pairs("geo")

    # ---- (a) MLP, three parameterisations --------------------------------
    tabm = by_width(geo[geo["arch"] == "MLP"], "rq_mid", keys=("regime",))
    ends = {}
    for regime in REGIMES:
        ls, marker = REG_STYLE[regime]
        x, y, e, r2 = curve(ax_a, tabm[tabm["regime"] == regime], C[regime],
                            ls, marker)
        ends[regime] = (x, y, e, r2)
    for regime, dy in (("Standard", 0), ("NTK", 4), ("muP", -6)):
        x, y, e, _ = ends[regime]
        label_at(ax_a, x[-1], y[-1],
                 rf"{REGIME_LABEL[regime]}  $n^{{-{e:.1f}}}$", C[regime],
                 dx=6, dy=dy, fontsize=6.9)
    # Mark the one place the power law runs out.
    xs, ys, _, _ = ends["Standard"]
    ax_a.annotate("saturates from $n=1024$\n($R^2=0.87$, lowest of nine)",
                  xy=(xs[-2], ys[-2]), xytext=(0, 38),
                  textcoords="offset points", fontsize=6.2,
                  color=C["Standard"], ha="center", va="bottom",
                  arrowprops=dict(arrowstyle="-", color=C["Standard"],
                                  lw=0.7, shrinkA=1, shrinkB=2))
    ax_a.text(0.035, 0.055, "all three: MLP / MNIST", transform=ax_a.transAxes,
              fontsize=6.9, color=C["ink"], ha="left", va="bottom",
              weight="bold")
    ax_a.set_ylim(3.0e-5, 3.0)
    ax_a.set_ylabel(r"Rayleigh quotient $\mathcal{R}_F$ (normalised)")

    # ---- (b) Standard regime, three architectures ------------------------
    tabs = by_width(geo[geo["regime"] == "Standard"], "rq_mid", keys=("arch",))
    for arch, dy in (("MLP", 9), ("TS", -9), ("CNN", 0)):
        x, y, e, _ = curve(ax_b, tabs[tabs["arch"] == arch], ARCH_COLORS[arch],
                           ARCH_LS[arch], ARCH_MARKERS[arch], ms=3.6)
        label_at(ax_b, x[-1], y[-1], rf"{ARCH_LABEL[arch]}  $n^{{-{e:.2f}}}$",
                 ARCH_COLORS[arch], dx=6, dy=dy, fontsize=6.9)
    ax_b.text(0.035, 0.055, "all three: Standard", transform=ax_b.transAxes,
              fontsize=6.9, color=C["ink"], ha="left", va="bottom",
              weight="bold")
    ax_b.set_ylim(2.5e-3, 3.0)

    for ax in (ax_a, ax_b):
        ax.set_yscale("log")
        log_width_axis(ax)
        ax.set_xlim(WIDTHS[0] * 0.88, WIDTHS[-1] * 8.6)
        ax.set_xlabel("width $n$")
        despine(ax)
    panel_letter(ax_a, "a", dx=-0.08)
    panel_letter(ax_b, "b", dx=-0.05)

    save(fig, OUT, "figp3_rayleigh")
    print("  (a) MLP by regime:", {r: round(v[2], 2) for r, v in ends.items()})
    print("  (b) Standard by arch:", {a: round(powerlaw_fit(
        tabs[tabs["arch"] == a].sort_values("width")["width"].to_numpy(float),
        tabs[tabs["arch"] == a].sort_values("width")["med"].to_numpy(float))[0], 2)
        for a in ("MLP", "TS", "CNN")})


if __name__ == "__main__":
    main()
