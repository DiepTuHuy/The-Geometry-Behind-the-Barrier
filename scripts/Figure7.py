#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure7.py -- Appendix D: Figure 3 for the two architectures the
main text does not show.

Figure 3 reports the Rayleigh quotient R_F against width for the MLP alone, and
reads its decay as the displacement concentrating in F's low-eigenvalue
subspace.  A reader is entitled to ask whether that is a property of the MLP.
This repeats the panel, axis for axis, on the teacher--student family and on the
CNN.

Panels are architectures, so the figure is read straight across against
Figure 3.  The three parameterisation curves, their colours, dash patterns,
markers, interquartile bands, the per-curve normalisation by the smallest-width
value, the in-panel key and the fitted exponent in it are all as in Figure 3;
nothing is measured differently here, and nothing is filtered differently
(smooth activations only, App. E.1).

WHAT CARRIES OVER, AND THE ONE THING THAT DOES NOT.  R_F decays steeply in all
six cells, so the Shape-side reading of Section 5 is not an MLP artefact.  What
does not carry over is the SATURATION: the Standard MLP flattens from n = 1024
on, which is why its Figure 3 exponent is a fit with R^2 = 0.87 while NTK-lazy
and muP sit at 0.99.  Here the Standard curve is a clean power law in both
architectures -- n^-1.31 (R^2 = 0.985) on the teacher--student, n^-1.23
(R^2 = 0.996) on the CNN -- so the saturation belongs to the Standard MLP cell,
not to the Standard regime.  That is the same verdict figD10 reaches for the
barrier reversal, on the same two architectures.

This is what the deleted panel (b) of the original three-panel figp3 was for.
It is restored here rather than in the main text, at full size and with the
teacher--student shown as well, so the comparison is against Figure 3 as
published rather than squeezed beside it.

R^2 IS NOT IN THE KEY, as it is not in Figure 3's.  It was, for one draft: the
panel's point is that these fits are good where the Standard MLP's is not, and
the R^2 column said so on the plot.  It was dropped because a key of three
columns is Figure 3's key and a key of four is a small table, and this figure
has to be read straight across against Figure 3 -- a reader comparing the two
should not first have to work out what the extra column is.  The fit qualities
are quoted in the Appendix D.2 prose instead, where the comparison against the
MLP's 0.87 is being made in words anyway.  Set SHOW_FIT_QUALITY = True to put
the column back.

CNN WIDTH.  Drawn on n = 64 w_m, i.e. 64..512, not on the multiplier -- the
convention fig_data states and Figure 3 uses, so the two panels and Figure 3
share one x axis.  (figD10 puts its CNN row on w_m instead; that figure is read
down a column against Figure 1, this one across against Figure 3.)

DATA: measured, via fig_data.load_pairs("geo") -- the same `rq_mid` column
Figure 3 plots, median over the four smooth activations and all seed pairs,
band = interquartile range, each curve normalised by its own smallest-width
value.  The 2026-09-14 Rayleigh run (data/rayleigh/rayleigh_pairs.csv,
src_rq = 10_rayleigh) re-measures these cells and reproduces every exponent
below to two decimals; it is a cross-check on this figure, not its source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, legend_table, log_width_axis,
                       panel_letter, save, C, REGIMES, REGIME_LABEL,
                       LINEWIDTH_IN)
from fig_data import load_pairs, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"

FIGW, FIGH = LINEWIDTH_IN, 2.45            # \includegraphics at \linewidth
SHOW_FIT_QUALITY = False                   # the R^2 column; see the header

# Figure 3's dash patterns, unchanged: separable in greyscale at the printed
# width, and a reader moving between the two figures must not have to relearn
# which curve is which.
REG_STYLE = {"NTK": ("-", "o"),
             "Standard": ((0, (4.2, 1.6)), "s"),
             "muP": ((0, (1.0, 2.0)), "^")}

# (arch key in fig_data, the key's title, the panel letter).
PANELS = [("TS", "Teacher\u2013student / synthetic", "a"),
          ("CNN", "CNN / FashionMNIST", "b")]


def headroom_beside(ax, series, x0, y0, gap=0.035):
    """`ymax` that clears a key occupying ONE CORNER of the panel.

    Taken from Figure3, for the same reason it exists there: every curve
    falls steeply from the left edge, so the upper-right corner is already
    empty and fig_style.headroom_for's full-width strip would push the top of
    the axis decades above anything drawn.  Reserve only the region the key
    occupies -- the highest curve to the right of the key's left edge,
    including where it crosses that edge -- and put it just under the key's
    bottom.  The key is placed in axes fractions, so one pass is enough."""
    xlo, xhi = ax.get_xlim()
    ylo = ax.get_ylim()[0]
    x_edge = xlo * (xhi / xlo) ** x0
    peak = 0.0
    for x, y in series:
        at_edge = np.exp(np.interp(np.log(np.clip(x_edge, x[0], x[-1])),
                                   np.log(x), np.log(y)))
        peak = max(peak, at_edge, *y[x >= x_edge], 0.0)
    frac = y0 - gap
    return 10.0 ** (np.log10(ylo)
                    + (np.log10(peak) - np.log10(ylo)) / frac)


def curve(ax, s, colour, ls, marker, lw=1.5, ms=3.2):
    """Draw one regime, normalised by its own narrowest width.

    Returns the Line2D property dict alongside the fit, so the key is built
    from the very dict the curve was drawn with and cannot drift away from it."""
    s = s.sort_values("width")
    ref = s["med"].iloc[0]
    x = s["width"].to_numpy(float)
    y = (s["med"] / ref).to_numpy(float)
    lo, hi = (s["q1"] / ref).to_numpy(float), (s["q3"] / ref).to_numpy(float)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0, zorder=2)
    style = dict(color=colour, ls=ls, marker=marker, lw=lw, ms=ms)
    ax.plot(x, y, markerfacecolor=colour, markeredgecolor="white",
            markeredgewidth=0.5, zorder=5, **style)
    exponent, r2, _ = powerlaw_fit(x, y)
    return style, exponent, r2, (x, y, lo, hi)


def panel(ax, pairs, arch, title):
    tab = by_width(pairs.query("arch == @arch"), "rq_mid", keys=("regime",))

    fits, rows, drawn = {}, [], []
    for regime in REGIMES:
        ls, marker = REG_STYLE[regime]
        style, exponent, r2, xy = curve(ax, tab[tab["regime"] == regime],
                                        C[regime], ls, marker)
        fits[regime] = (exponent, r2)
        drawn.append(xy)
        value = rf"$n^{{-{exponent:.2f}}}$"
        if SHOW_FIT_QUALITY:
            value += rf"   $R^2\!=\!{r2:.3f}$"
        rows.append((style, REGIME_LABEL[regime], value))

    widths = np.array(sorted(tab["width"].unique()), float)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # Ticks are the widths this architecture actually has.  The CNN's grid
    # stops at n = 512 (App. C); it stays on the width axis rather than moving
    # to its channel multiplier, so the two panels and Figure 3 read alike.
    log_width_axis(ax, widths=widths,
                   ticks=[f"{int(w)}" if w < 1024 else f"{int(w) // 1024}k"
                          for w in widths])
    # The key lives inside the panel, so the data may run to the frame.
    ax.set_xlim(widths[0] * 0.90, widths[-1] * 1.10)
    ax.set_xlabel(r"width $n$")
    # Labelled on BOTH panels.  They do not share a y range -- each takes its
    # own, as figD10 does and for the same reason -- and an unlabelled right
    # panel beside a labelled left one reads as a shared axis.
    ax.set_ylabel(r"Rayleigh quotient $\mathcal{R}_F$")
    despine(ax)

    # Bands included, so the axis clears the data actually drawn.
    lo = min(band_lo.min() for _, _, band_lo, _ in drawn) / 1.3
    hi = max(band_hi.max() for _, _, _, band_hi in drawn)
    ax.set_ylim(lo, hi)
    x0, y0, _, _ = legend_table(ax, rows, loc="upper right", title=title)
    ax.set_ylim(lo, headroom_beside(ax, [(x, y) for x, y, _, _ in drawn],
                                    x0, y0))
    return fits


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, FIGH),
                             gridspec_kw={"wspace": 0.33})
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.165, top=0.945)

    pairs = load_pairs("geo")
    out = {}
    for ax, (arch, title, letter) in zip(axes, PANELS):
        out[arch] = panel(ax, pairs, arch, title)
        panel_letter(ax, letter, dx=-0.075)

    save(fig, OUT, "Figure7")
    for arch, fits in out.items():
        print(f"  {arch}:")
        for regime in REGIMES:
            exponent, r2 = fits[regime]
            print(f"    {regime:<9} n^-{exponent:.2f}   R^2={r2:.3f}")


if __name__ == "__main__":
    main()
