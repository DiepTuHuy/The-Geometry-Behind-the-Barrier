#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp4_regime_barrier.py -- Figure 4: regime dependence of barrier collapse.

Interpolation barrier normalised by the smallest-width baseline of the same
architecture and regime (dashed line: ratio = 1).  NTK-lazy stays at or above
its baseline; Standard and muP decay by one to two orders of magnitude; the
Standard MLP is non-monotonic at the largest widths.

REDRAWN.  Three defects, all of them in the panel grid rather than in the data:

  1. The CNN panel was drawn on width RATIOS 1, 2, 4, 8 while the other two
     panels were drawn on WIDTHS 64, 256, 1k, 4k.  Three panels of equal
     physical width carrying different x scales is the one thing a small-
     multiples grid may not do -- the reader compares slopes across panels by
     construction, and here that comparison was wrong by a factor of 8 in
     horizontal extent.  All three panels now share one x axis (sharex), the
     CNN curve simply stops at n = 512, which is the fact App. E.1 states.
  2. The muP curve in the MLP panel dipped to 0.0028 and its band to 0.00246,
     below the axis floor of 2.5e-3, so the minimum -- the very point the word
     "collapses" refers to -- was clipped off the figure.
  3. The "Standard" direct label in the Teacher-student panel ran off the right
     edge of the saved figure.  With sharex there is no right-hand margin to
     put labels in, so the three curves are labelled once, inside panel (a),
     the way a small-multiples grid is meant to be keyed.

The dashed ratio = 1 reference is now labelled on the plot rather than only in
the caption (R1/R9).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, panel_letter, log_width_axis,
                       save, C, WIDTHS)
from fig_data import load_pairs, by_width

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"

# Barrier / smallest-width baseline, measured from data/final/*_pairs.csv.
# The CNN grid ends at x8 -- which, on the channel convention of App. E.1, is
# n = 512 -- so its curve simply stops there on the shared axis.
PANELS = [("MLP", "MLP / MNIST", "a"),
          ("CNN", "CNN / Fashion-MNIST", "b"),
          ("TS", "Teacher–student", "c")]
LS = {"NTK": "-", "Standard": (0, (4.5, 1.7)), "muP": (0, (1, 2.2))}
MARKER = {"NTK": "o", "Standard": "s", "muP": "^"}
LABEL = {"NTK": "NTK-lazy", "Standard": "Standard", "muP": r"$\mu$P"}


def main():
    apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(5.5, 2.05), sharey=True, sharex=True,
        gridspec_kw={"wspace": 0.09})

    pairs = load_pairs("final")
    tab = by_width(pairs, "B", keys=("arch", "regime"))

    for ax, (arch, name, letter) in zip(axes, PANELS):
        ax.axhline(1.0, color=C["ref"], ls="--", lw=0.9, zorder=1)
        for regime in ("NTK", "Standard", "muP"):
            s_c = tab[(tab["arch"] == arch) &
                      (tab["regime"] == regime)].sort_values("width")
            ref = s_c["med"].iloc[0]          # smallest-width baseline
            xs = s_c["width"].to_numpy(float)
            y = (s_c["med"] / ref).to_numpy(float)
            color = C[regime]
            ax.fill_between(xs, s_c["q1"] / ref, s_c["q3"] / ref, color=color,
                            alpha=0.15, lw=0)
            ax.plot(xs, y, color=color, ls=LS[regime], lw=1.6,
                    marker=MARKER[regime], ms=3.0, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

        log_width_axis(ax)
        ax.set_xticklabels(["64", "", "256", "", "1k", "", "4k"])
        ax.set_xlim(WIDTHS[0] * 0.80, WIDTHS[-1] * 1.30)
        ax.set_yscale("log")
        ax.set_ylim(4.0e-3, 6.0)
        ax.set_title(name, fontsize=8, pad=3)
        ax.set_xlabel("width $n$", labelpad=1.5)
        panel_letter(ax, letter, dx=-0.04)
        despine(ax)

    axes[0].set_ylabel("barrier / smallest-width\nbaseline")

    # --- key: label the three regimes once, inside panel (a) (R2) ---------
    # Each label sits in a region of panel (a) that its own curve owns.
    axes[0].annotate(LABEL["NTK"], xy=(WIDTHS[3], 1.60), xytext=(0, 5),
                     textcoords="offset points", fontsize=7.0, color=C["NTK"],
                     weight="bold", ha="center", va="bottom")
    # placed in the empty band between the Standard minimum and the NTK curve
    axes[0].annotate(LABEL["Standard"], xy=(WIDTHS[3], 0.44), fontsize=7.0,
                     color=C["Standard"], weight="bold", ha="center",
                     va="center")
    axes[0].annotate(LABEL["muP"], xy=(300.0, 1.5e-2), fontsize=7.0,
                     color=C["muP"], weight="bold", ha="left", va="center")

    # --- R1: the message, and R9: what the reference line means -----------
    axes[0].annotate("persists", xy=(WIDTHS[-1] * 1.20, 4.2), fontsize=7.5,
                     style="italic", color=C["NTK"], ha="right", va="center")
    axes[0].annotate("collapses", xy=(WIDTHS[1] * 1.15, 6.5e-3), fontsize=7.5,
                     style="italic", color=C["muP"], ha="center", va="center")
    # The ratio = 1 reference is keyed in the CNN panel, the only one with
    # empty space beside the line (R8).
    axes[1].annotate(r"ratio $=1$", xy=(WIDTHS[-1] * 1.22, 1.0), xytext=(0, 3),
                     textcoords="offset points", fontsize=6.6, color=C["ref"],
                     ha="right", va="bottom")
    axes[1].annotate(r"CNN grid ends at $\times8$",
                     xy=(WIDTHS[-1] * 1.22, 5.0e-3), fontsize=6.4,
                     color=C["ref"], ha="right", va="bottom")

    save(fig, OUT, "figp4_regime_barrier")


if __name__ == "__main__":
    main()
