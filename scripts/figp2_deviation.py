#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp2_deviation.py -- Figure 2: the geodesic-linear deviation, and its limits.

Panel (a): D_rel vs width in the NTK-lazy regime, three architectures.
Panel (b): D_rel vs width for the MLP, three parameterisations.

Why panel (b) exists.  The earlier version of this figure showed panel (a)
only, which invited the obvious question "why NTK-lazy alone?" and left the
answer in a footnote.  The measured answer is that D_rel decays cleanly ONLY in
the lazy regime: in Standard it is non-monotone (R^2 = 0.00, the fit is
meaningless) and for the Standard CNN it GROWS by 9x.  That is a real boundary
of the Shape diagnostic and belongs on the plot, not in prose.

The reason is methodological rather than a failure of the measurement:
D_rel = sup||xi|| / ||Delta|| is normalised by ||Delta||, which is Theta(sqrt P)
under NTK but shrinks with width under fan-in initialisation (App. F), so the
ratio is not comparable across parameterisations.  The universality claim of
Section 5.2 rests on ||dF||_op (Figure 1b), not on this quantity.

COLOUR: panel (a) has no parameterisation dimension, so colour is free and
carries the architecture (ARCH_COLORS, disjoint from the regime triple);
panel (b) has a parameterisation dimension, so colour carries the regime.
Each panel states which, so a curve can never be misread.

DATA: measured, from ../data/ via fig_data.load_pairs("geo"); median over the
four smooth activations and all seed pairs, band = interquartile range, each
curve normalised by its own smallest-width value.  Every exponent printed is
fitted from the points being drawn.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, label_at, log_width_axis,
                       panel_letter, save, C, REGIMES, REGIME_LABEL,
                       ARCH_COLORS, ARCH_LABEL, ARCH_MARKERS, WIDTHS)
from fig_data import load_pairs, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"

ARCH_LS = {"MLP": "-", "TS": (0, (5.0, 1.7)), "CNN": (0, (1.3, 1.9))}
ARCH_LW = {"MLP": 1.8, "TS": 1.7, "CNN": 1.9}
REG_STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
             "muP": ((0, (1, 2.2)), "^")}


def curve(ax, s, colour, ls, marker, lw=1.7, ms=3.6):
    """One normalised curve + IQR band; returns (x, y, exponent, R^2)."""
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

    # ---- (a) NTK-lazy, three architectures -------------------------------
    tab = by_width(geo[geo["regime"] == "NTK"], "dev_rel", keys=("arch",))
    for arch, dx, dy, ha in (("MLP", 6, 0, "left"),
                             ("TS", 6, 0, "left"),
                             ("CNN", -5, -10, "right")):
        x, y, e, _ = curve(ax_a, tab[tab["arch"] == arch], ARCH_COLORS[arch],
                           ARCH_LS[arch], ARCH_MARKERS[arch], ARCH_LW[arch], 3.8)
        label_at(ax_a, x[-1], y[-1],
                 rf"{ARCH_LABEL[arch]}  $n^{{-{e:.2f}}}$",
                 ARCH_COLORS[arch], dx=dx, dy=dy, fontsize=6.9, ha=ha)
    ax_a.text(0.035, 0.055, "all three: NTK-lazy", transform=ax_a.transAxes,
              fontsize=6.9, color=C["ink"], ha="left", va="bottom",
              weight="bold")
    ax_a.set_ylim(4.4e-2, 2.3)
    ax_a.set_ylabel(r"relative deviation $D_{\mathrm{rel}}$")

    # ---- (b) MLP, three parameterisations --------------------------------
    tabm = by_width(geo[geo["arch"] == "MLP"], "dev_rel", keys=("regime",))
    fits = {}
    for regime in REGIMES:
        ls, marker = REG_STYLE[regime]
        x, y, e, r2 = curve(ax_b, tabm[tabm["regime"] == regime], C[regime],
                            ls, marker)
        fits[regime] = (x[-1], y[-1], e, r2)
    ax_b.axhline(1.0, color=C["ref"], ls=":", lw=0.8, zorder=1)
    for regime, dy in (("NTK", -4), ("Standard", 4), ("muP", 0)):
        xe, ye, e, r2 = fits[regime]
        tag = "no power law" if r2 < 0.5 else rf"$n^{{-{e:.2f}}}$"
        label_at(ax_b, xe, ye, f"{REGIME_LABEL[regime]}  {tag}", C[regime],
                 dx=6, dy=dy, fontsize=6.9)
    ax_b.text(0.035, 0.055, "all three: MLP / MNIST", transform=ax_b.transAxes,
              fontsize=6.9, color=C["ink"], ha="left", va="bottom",
              weight="bold")
    ax_b.set_ylim(6e-2, 3.4)

    for ax in (ax_a, ax_b):
        ax.set_yscale("log")
        log_width_axis(ax)
        ax.set_xlim(WIDTHS[0] * 0.88, WIDTHS[-1] * 8.6)
        ax.set_xlabel("width $n$")
        despine(ax)
    panel_letter(ax_a, "a", dx=-0.08)
    panel_letter(ax_b, "b", dx=-0.05)

    save(fig, OUT, "figp2_deviation")
    print("  (a) NTK by arch:", {a: round(powerlaw_fit(
        tab[tab["arch"] == a].sort_values("width")["width"].to_numpy(float),
        tab[tab["arch"] == a].sort_values("width")["med"].to_numpy(float))[0], 2)
        for a in ("MLP", "TS", "CNN")})
    print("  (b) MLP by regime:", {r: (round(v[2], 2), round(v[3], 2))
                                   for r, v in fits.items()})


if __name__ == "__main__":
    main()
