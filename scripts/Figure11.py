#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure11.py -- Appendix D: the paper's measurement chain, redrawn
on CIFAR-10.

Every width curve in the main text is MNIST, FashionMNIST or a synthetic
teacher.  This redraws the four the scale run can measure, on CIFAR-10, with
the CNN and the grid held at the released CNN's setting:

  (a) barrier B                    Figure 1a
  (b) Fisher length Delta^T F Delta    Figure 4a
  (c) endpoint uncertainty rho*    Figure 4b
  (d) Rayleigh quotient R_F        Figure 3

WHAT IS NOT HERE, AND WHY.  Figure 1b (||dF||_op) and Figure 2 (the
geodesic-linear deviation) are absent because the scale run does not measure
them: it tests the Figure 5 PREDICTION METHOD, which needs the barrier and the
Length-axis quantities and no Christoffel symbols, second or third derivatives,
or geodesic solve.  Adding those would have multiplied the compute for a
dataset check that does not depend on them.  The panels here are the four the
run measures, not a selection from a larger set.

PANEL ORDER FOLLOWS THE ARGUMENT, not the figure numbers: the barrier is the
thing to be explained, (b) and (c) are the two Length-axis controls that
Theorem 4.1 makes the barrier depend on, and (d) is the negative control.

THE KEY CARRIES TWO EXPONENTS PER CURVE.  The CIFAR-10 exponent, then the
released FashionMNIST CNN's in brackets.  Twelve such comparisons is more than
prose can carry and more than a second set of curves can show without putting
six lines in every panel, and the bracket makes each one a glance rather than a
cross-reference.  Both sides are fitted by OLS on seed medians (App. E.1), over
the same four widths, so the pair differs by its dataset and nothing else.

WHAT THE PANELS SAY.  Every quantity keeps its sign, its regime ordering and
its shape, and every one of them is SHALLOWER on CIFAR-10 -- the barrier
collapses at n^-0.35 under muP against n^-0.97, the Fisher length at n^-0.70
against n^-1.51.  The exception is panel (c), and it is a sharp one: rho*
reproduces its exponents to within 0.03 in all three regimes (-0.01/+1.97/+1.44
against -0.02/+1.96/+1.44), so the endpoint gating condition of Theorem 4.1 is
the one part of the chain the harder dataset leaves untouched.

NORMALISATION.  Each curve by its own narrowest-width value, the convention
Figures 1, 3 and 4 share, so all three regimes leave 1 together.  It costs the
levels, and in (c) the levels matter: CIFAR-10 starts at rho* = 0.58 under
NTK-lazy against 0.21 on FashionMNIST, and at 0.29 under Standard against 0.058.
main() prints every anchor it divides out, and the caption carries the rho* row.

X AXIS.  The channel multiplier w_m in {1,2,4,8}, not n = 64 w_m.  figD10 puts
its CNN row on w_m for the reason that applies here too -- the axis is the knob
the run actually turns -- and the relabel changes no exponent, since n and w_m
differ by a constant factor and a power law does not see one.  figD18 is the
deliberate exception: it is read straight across against Figure 3, so it has to
carry Figure 3's axis.

DATA: measured.  CIFAR-10 from ../data/scale/param_scale_cifar10_cnn.csv via
fig_data.load_scale; the bracketed reference from ../data/param_final_cnn.csv
(B, rho_A) and param_geo_cnn.csv (flen_mid, rq_mid) via fig_data.load_pairs --
the same columns Figures 1a, 3, 4a and 4b plot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from fig_style import (apply_style, despine, legend_table, log_width_axis,
                       panel_letter, headroom_for, save, C, REGIMES,
                       REGIME_LABEL)
from fig_data import load_pairs, load_scale, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"

# Figure 1's dash patterns, which Figures 3 and 4 and figD10 all reuse.
REG_STYLE = {"NTK": ("-", "o"),
             "Standard": ((0, (4.5, 1.7)), "s"),
             "muP": ((0, (1.0, 2.2)), "^")}

# (column, y label, panel letter, which released table it comes from).
PANELS = [("B", "barrier", "a", "final"),
          ("flen_mid", r"$\Delta^\top F\Delta$", "b", "geo"),
          ("rho_A", r"endpoint $\rho^*$", "c", "final"),
          ("rq_mid", r"Rayleigh $\mathcal{R}_F$", "d", "geo")]


def tidy_log_y(ax, max_decades=2.0):
    """Plain tick labels when a log panel spans less than two decades.

    Matplotlib then falls back to labelling the MINOR ticks, and prints
    "4 x 10^0, 3 x 10^0, 2 x 10^0, 10^0, 6 x 10^-1" down the axis: five
    superscripted symbols where five plain numbers would do.  In a 2.4 in panel
    that column is the widest thing on the page after the key, and it says
    nothing the numbers do not.  Panels (a) and (b) span well under a decade,
    so both hit this; (c) and (d) span four and are left alone."""
    lo, hi = ax.get_ylim()
    if not np.isfinite(lo) or lo <= 0 or np.log10(hi / lo) >= max_decades:
        return
    ax.yaxis.set_major_locator(
        ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def exponents(tab):
    """Decay exponent per regime, fitted on the medians drawn."""
    out = {}
    for regime in REGIMES:
        s = tab[tab["regime"] == regime].sort_values("width")
        out[regime] = powerlaw_fit(s["width"].to_numpy(float),
                                   s["med"].to_numpy(float))[0]
    return out


def panel(ax, tab_new, tab_ref, ylabel):
    anchors = {}
    lo, hi = np.inf, -np.inf
    for regime in REGIMES:
        ls, marker = REG_STYLE[regime]
        s = tab_new[tab_new["regime"] == regime].sort_values("width")
        ref = s["med"].iloc[0]
        anchors[regime] = ref
        x = s["width"].to_numpy(float)
        y, q1, q3 = (s["med"] / ref).to_numpy(float), \
            (s["q1"] / ref).to_numpy(float), (s["q3"] / ref).to_numpy(float)
        ax.fill_between(x, q1, q3, color=C[regime], alpha=0.15, lw=0, zorder=2)
        ax.plot(x, y, color=C[regime], ls=ls, marker=marker, lw=1.6, ms=3.4,
                markerfacecolor=C[regime], markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        lo, hi = min(lo, q1.min()), max(hi, q3.max())

    a_new, a_ref = exponents(tab_new), exponents(tab_ref)
    rows = []
    for regime in REGIMES:
        ls, marker = REG_STYLE[regime]
        rows.append((dict(color=C[regime], ls=ls, marker=marker, lw=1.6,
                          ms=3.4), REGIME_LABEL[regime],
                     rf"${a_new[regime]:+.2f}\ ({a_ref[regime]:+.2f})$"))

    ax.set_yscale("log")
    # Drawn on the CNN's own channel multiplier, as figD10's CNN row is, so
    # the two appendix CNN figures share an axis.  n = 64 w_m, and a power law
    # is scale-free, so every exponent in the key is unchanged by the relabel.
    log_width_axis(ax, widths=(64, 128, 256, 512),
                   ticks=("1", "2", "4", "8"))
    ax.set_xlim(58, 565)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"channel multiplier $w_m$")
    despine(ax)
    bottom = lo / 1.25
    ax.set_ylim(bottom, hi * 3.0)                  # provisional
    _, _, _, table_h = legend_table(ax, rows, loc="upper right")
    ax.set_ylim(bottom, headroom_for(bottom, hi, table_h, gap=0.018))
    tidy_log_y(ax)          # after the final ylim: the span decides the ticks
    return a_new, a_ref, anchors


def main():
    apply_style()
    sc = load_scale("cifar10", "cnn")
    ref = {"final": load_pairs("final").query("arch == 'CNN'"),
           "geo": load_pairs("geo").query("arch == 'CNN'")}

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.35),
                             gridspec_kw={"wspace": 0.30, "hspace": 0.44})

    printed = {}
    for ax, (col, ylabel, letter, src) in zip(axes.ravel(), PANELS):
        printed[col] = panel(ax,
                             by_width(sc, col, keys=("regime",)),
                             by_width(ref[src], col, keys=("regime",)),
                             ylabel)
        panel_letter(ax, letter, dx=-0.13)

    fig.subplots_adjust(left=0.115, right=0.995, top=0.955, bottom=0.10)
    save(fig, OUT, "Figure11")

    for col, (a_new, a_ref, anchors) in printed.items():
        print(f"  {col}:")
        for regime in REGIMES:
            print(f"    {regime:<9} CIFAR-10 {a_new[regime]:+.2f}   "
                  f"released {a_ref[regime]:+.2f}   "
                  f"(anchor at n=64: {anchors[regime]:.4g})")


if __name__ == "__main__":
    main()
