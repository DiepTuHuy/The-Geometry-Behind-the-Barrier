#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure6.py -- Appendix D: Figure 1 for the two architectures the
main text does not show.

Figure 1 makes the paper's central contrast on the MLP alone -- the barrier
behaves in three ways while ||dF||_op decays in every regime -- and a reader is
entitled to ask whether that is a property of the MLP.  This repeats it, panel
for panel and axis for axis, on the teacher--student family and on the CNN.

Rows are architectures, columns the two quantities of Figure 1, so the figure is
read down a column against Figure 1's corresponding panel.  The three
parameterisation curves, their colours, line styles, markers, interquartile
bands and the fitted exponent printed at each curve's end are all as in
Figure 1; nothing is measured differently here.

What carries over and what does not.  The right column carries over intact:
||dF||_op decays as a power law in all six cells, so the flattening result is
not an MLP artefact.  The left column carries over in shape but not in
magnitude: the NTK-lazy barrier fails to collapse in all three architectures,
while the two feature-learning regimes collapse.  Where the architectures differ
is the Standard curve, which is non-monotone only for the MLP -- the
teacher--student and the CNN fall monotonically -- so the reversal Section 5
discusses is a property of that one cell, not of the Standard regime.

DATA: measured, from ../data/ via fig_data, exactly as Figure 1: median over the
four smooth activations and all seed pairs, band = interquartile range, every
exponent fitted from the points drawn.  The CNN grid ends at n = 512 because its
width is the widest layer of a channel multiplier that stops at x8 (App. C).
"""
from pathlib import Path

import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, panel_letter, log_width_axis,
                       legend_table, headroom_for, save, C, REGIME_LABEL)
from fig_data import load_pairs, load_dF, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"
REGIMES = ["NTK", "Standard", "muP"]
STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
         "muP": ((0, (1, 2.2)), "^")}
ARCHS = [("TS", "Teacher\u2013student / synthetic"), ("CNN", "CNN / FashionMNIST")]


def band(ax, x, y, lo, hi, regime):
    ls, marker = STYLE[regime]
    col = C[regime]
    ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(x, y, color=col, lw=1.7, ls=ls, marker=marker, ms=3.5,
            markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.5,
            zorder=5)


def panel(ax, tab, ylab, exp_sym, monotone_ok=True):
    exps = {}
    for regime in REGIMES:
        s = tab[tab["regime"] == regime].sort_values("width")
        band(ax, s["width"], s["med"], s["q1"], s["q3"], regime)
        e, r2, _ = powerlaw_fit(s["width"], s["med"])
        exps[regime] = (s, e, r2)
    ax.set_yscale("log")
    log_width_axis(ax)

    # Limits from the data of THIS panel, not from Figure 1's.  Borrowing the
    # MLP's range left the CNN curves squeezed into the top decade of four,
    # and App. C reports only exponents precisely because absolute levels are
    # not comparable across architectures anyway -- so what has to be legible
    # here is the shape and the slope, and those need the panel's own range.
    lo = min(s["q1"].min() for s, _, _ in exps.values())
    hi = max(s["q3"].max() for s, _, _ in exps.values())
    xmax = max(s["width"].max() for s, _, _ in exps.values())
    # No end-of-line labels any more, so no wide right margin to host them.
    ax.set_xlim(50, xmax * 1.30)
    # Bottom pad tightened from the /3 this figure used before the key existed:
    # the key costs about a third of the panel at the top, and paying for it at
    # both ends leaves the curves squeezed into the middle.
    bottom = lo / 1.8
    ax.set_ylim(bottom, hi * 3.0)       # provisional; the key resets the top
    ax.set_ylabel(ylab)

    # One row per regime, keyed exactly as Figure 1 keys the same three curves.
    # A curve that no power law describes carries the word instead of a number,
    # as Figure 1 does for the Standard MLP.
    rows = []
    for regime in REGIMES:
        s, e, r2 = exps[regime]
        ls, marker = STYLE[regime]
        rows.append((dict(color=C[regime], ls=ls, marker=marker, lw=1.7,
                          ms=3.5), REGIME_LABEL[regime],
                     "non-monotonic" if (not monotone_ok and r2 < 0.5)
                     else rf"${exp_sym}{{=}}{e:+.1f}$"))
    despine(ax)
    _, _, _, table_h = legend_table(ax, rows, loc="upper right")
    # The key is placed in axes fractions, so its height is known before the
    # axis range is: raise the top until the data's own maximum -- band
    # included -- sits clear underneath it.  Every panel here takes its range
    # from its own data (see above), so the headroom is per panel too.
    ax.set_ylim(bottom, headroom_for(bottom, hi, table_h, gap=0.018))
    return {r: (v[1], v[2]) for r, v in exps.items()}


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.15),
                             gridspec_kw={"wspace": 0.34, "hspace": 0.42})
    pairs, dF = load_pairs("final"), load_dF()

    out = {}
    for r, (arch, title) in enumerate(ARCHS):
        ax_a, ax_b = axes[r]
        out[(arch, "B")] = panel(
            ax_a, by_width(pairs[pairs["arch"] == arch], "B", keys=("regime",)),
            "interpolation barrier $B$", r"\alpha_B", monotone_ok=False)
        out[(arch, "dF")] = panel(
            ax_b, by_width(dF[dF["arch"] == arch], "dF_op", keys=("regime",)),
            r"$\|\partial F\|_{\mathrm{op}}$", r"\alpha_{\partial F}")
        for ax in (ax_a, ax_b):
            ax.set_xlabel("width $n$")
        ax_a.text(0.03, 0.055, title, transform=ax_a.transAxes, fontsize=7.0,
                  weight="bold", color=C["ink"], va="bottom")
        panel_letter(ax_a, "ac"[r], dx=-0.10)
        panel_letter(ax_b, "bd"[r], dx=-0.10)

    fig.subplots_adjust(left=0.115, right=0.995, top=0.955, bottom=0.105)
    save(fig, OUT, "Figure6")

    for k, v in out.items():
        print(f"  {k[0]:3s} {k[1]:2s}: " +
              "  ".join(f"{r}={e:+.2f} (R2 {r2:.2f})" for r, (e, r2) in v.items()))


if __name__ == "__main__":
    main()
