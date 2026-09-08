#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp4_regime_barrier.py -- Figure 4: regime dependence of barrier collapse.

Interpolation barrier normalised by the smallest-width baseline of the same
architecture and regime (dashed line: ratio = 1).  NTK-lazy stays at or above
its baseline; Standard and muP decay by one to two orders of magnitude; the
Standard MLP is non-monotonic at the largest widths.

REDRAWN.  Three defects, all of them in the panel grid rather than in the data:

  1. [REVERTED BY AUTHOR REQUEST, 08/09/2026.]  This entry used to record the
     opposite decision, and the reasoning still holds, so it is kept here as a
     caveat rather than deleted.

     Panel (b) is again drawn on the CNN's own channel multiplier w_m in
     {1,2,4,8}; panels (a) and (c) stay on width n in {64..4096}.  Because the
     three panels are equally wide on the page, (b)'s three octaves now occupy
     the width that (a) and (c) give to six, so (b) has DOUBLE the horizontal
     scale.  A power law therefore renders twice as steep in (b) as the same
     power law would in (a) or (c).

     Concretely, for muP: alpha_B is 1.22 (MLP) against 0.97 (CNN), i.e. the
     CNN barrier collapses SLOWER -- but drawn this way the CNN curve reads as
     0.97 x 2 = 1.94, steeper than the MLP's 1.22, which inverts the
     comparison.  Slopes must not be compared across (b) and (a)/(c) by eye;
     the caption says so, and the exponents in App. C are the thing to compare.

     Two mitigations are in the code below: (b)'s x label names the multiplier
     rather than the width, so the tick "8" cannot be read as n = 8, and (b)
     carries the conversion n = 64 w_m on the plot.
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
import matplotlib.ticker as mticker

from fig_style import (apply_style, despine, panel_letter, log_width_axis,
                       legend_table, save, C, WIDTHS)
from fig_data import load_pairs, by_width, CNN_CHANNELS_PER_WM

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"

# Barrier / smallest-width baseline, measured from data/final/*_pairs.csv.
# The CNN grid ends at x8 -- which, on the channel convention of App. E.1, is
# n = 512 -- so its curve simply stops there on the shared axis.
PANELS = [("MLP", "MLP / MNIST", "a"),
          ("CNN", "CNN / Fashion-MNIST", "b"),
          ("TS", "Teacher–student", "c")]
CNN_WM = [1, 2, 4, 8]      # the CNN grid in its own units; n = 64 w_m
LS = {"NTK": "-", "Standard": (0, (4.5, 1.7)), "muP": (0, (1, 2.2))}
MARKER = {"NTK": "o", "Standard": "s", "muP": "^"}
LABEL = {"NTK": "NTK-lazy", "Standard": "Standard", "muP": r"$\mu$P"}


def main():
    apply_style()
    fig, axes = plt.subplots(
        # sharex is gone: panel (b) carries its own scale (see docstring 1).
        1, 3, figsize=(5.5, 2.05), sharey=True, sharex=False,
        gridspec_kw={"wspace": 0.09})

    pairs = load_pairs("final")
    tab = by_width(pairs, "B", keys=("arch", "regime"))

    for ax, (arch, name, letter) in zip(axes, PANELS):
        is_cnn = arch == "CNN"
        ax.axhline(1.0, color=C["ref"], ls="--", lw=0.9, zorder=1)
        for regime in ("NTK", "Standard", "muP"):
            s_c = tab[(tab["arch"] == arch) &
                      (tab["regime"] == regime)].sort_values("width")
            ref = s_c["med"].iloc[0]          # smallest-width baseline
            xs = s_c["width"].to_numpy(float)
            if is_cnn:
                # fig_data multiplies the CSV's w_m up to a width on load; this
                # panel is asked to show the multiplier, so undo that here only.
                xs = xs / CNN_CHANNELS_PER_WM
            y = (s_c["med"] / ref).to_numpy(float)
            color = C[regime]
            ax.fill_between(xs, s_c["q1"] / ref, s_c["q3"] / ref, color=color,
                            alpha=0.15, lw=0)
            ax.plot(xs, y, color=color, ls=LS[regime], lw=1.6,
                    marker=MARKER[regime], ms=3.0, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

        if is_cnn:
            ax.set_xscale("log", base=2)
            ax.set_xticks(CNN_WM)
            ax.set_xticklabels([str(v) for v in CNN_WM])
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.set_xlim(CNN_WM[0] * 0.80, CNN_WM[-1] * 1.30)
        else:
            log_width_axis(ax)
            ax.set_xticklabels(["64", "", "256", "", "1k", "", "4k"])
            ax.set_xlim(WIDTHS[0] * 0.80, WIDTHS[-1] * 1.30)
        ax.set_yscale("log")
        ax.set_ylim(4.0e-3, 6.0)
        ax.set_title(name, fontsize=8, pad=3)
        ax.set_xlabel(r"channel multiplier $w_m$" if is_cnn else "width $n$",
                      labelpad=1.5)
        panel_letter(ax, letter, dx=-0.04)
        despine(ax)

    axes[0].set_ylabel("barrier / smallest-width\nbaseline")

    # --- key: one table for the whole grid, in the space the CNN leaves -----
    # The three panels share one regime encoding, so the grid is keyed once
    # rather than three times.  It goes in panel (b) because the CNN grid stops
    # at n = 512, which leaves that panel's right half genuinely empty -- the
    # only place in this figure where a key covers no data.
    #
    # No third column: this figure plots ratios to a baseline, not fits, so
    # there is no per-regime exponent to put in one.  The exponents for these
    # same curves are in Figure 1.
    legend_table(axes[1], [(dict(color=C[r], ls=LS[r], marker=MARKER[r],
                                 lw=1.6, ms=3.0), LABEL[r], None)
                           for r in ("NTK", "Standard", "muP")],
                 loc="lower right")

    # --- R1: the message, and R9: what the reference line means -----------
    axes[0].annotate("persists", xy=(WIDTHS[-1] * 1.20, 4.2), fontsize=7.5,
                     style="italic", color=C["NTK"], ha="right", va="center")
    axes[0].annotate("collapses", xy=(WIDTHS[1] * 1.15, 6.5e-3), fontsize=7.5,
                     style="italic", color=C["muP"], ha="center", va="center")
    # The ratio = 1 reference is keyed in the CNN panel, the only one with
    # empty space beside the line (R8).
    axes[1].annotate(r"ratio $=1$", xy=(CNN_WM[-1] * 1.22, 1.0), xytext=(0, 3),
                     textcoords="offset points", fontsize=6.6, color=C["ref"],
                     ha="right", va="bottom")
    # Moved up off the floor: the key now sits in the bottom right corner.
    # Above the key, which occupies this panel's bottom right corner.
    axes[1].annotate(r"$n=64\,w_m$  (ends at $n=512$)",
                     xy=(CNN_WM[0] * 0.86, 3.2),
                     fontsize=6.4, color=C["ref"], ha="left", va="center")

    save(fig, OUT, "figp4_regime_barrier")


if __name__ == "__main__":
    main()
