#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure3.py -- Figure 3: spectral alignment, and what it is made of.

Panel (a): R_F vs width for the MLP, three parameterisations, read at TWO
           anchors -- t = 1/2 and the mean over the path.
Panel (b): R_F vs width in the Standard regime, three architectures (t = 1/2).
Panel (c): R_F against tr F / P -- what the decay in (a) actually is.

WHY PANEL (a) NOW CARRIES TWO ANCHORS.  Every Rayleigh number in the paper is
read at t = 1/2, and that is not a neutral point of the path: R_F(t) has a local
MINIMUM there, between two humps, and the minimum deepens with width (2.2x to
34.5x from n = 64 to 4096 on Standard/GELU).  Sampling one point of such a path
can change the width behaviour qualitatively, and on the Standard regime it
does.  Normalised to n = 64, median over the four smooth activations:

    Standard   at t=1/2   1.000 -> 0.0086 (n=1k) -> 0.0073 (n=4k)   flat
               mean over t 1.000 -> 0.0336 (n=1k) -> 0.0781 (n=4k)   REVERSES

    NTK, muP   both anchors decay monotonically; the pairs stay together.

So what the published figure reports as the Standard curve "saturating" is the
anchor flattening while the path turns back up -- the same non-monotone reversal
the barrier shows in Figure 1 (Sec. 5: Standard falls 6.8x to n = 512 and returns
by n = 4096).  This bears directly on the argument in Sec. 5 that "a predictor
built on R_F alone cannot capture non-monotonicity": that holds for R_F AT THE
MIDPOINT, and the path mean is not monotone.

Only the midpoint curves are fitted.  The path mean is deliberately left
unfitted, for the reason the main text already gives for the barrier: a reversal
is reported, not fitted through.

Why panel (c) exists.  Panels (a) and (b) establish that R_F decays steeply
everywhere, and the text reads that as the displacement concentrating in F's
low-eigenvalue subspace.  That reading was never tested, because a falling R_F
has two causes it cannot tell apart: Delta-hat rotating down a fixed spectrum,
or the whole spectrum shrinking under a Delta-hat that has not moved.

The separator is an identity -- for v uniform on the unit sphere,
E[v'Fv] = tr F / P exactly -- so tr F / P is the value R_F would take if Delta
carried no directional information at all.  Panel (c) plots the two together.
They track each other across four decades, which says the decay in (a) is the
spectrum's, not the direction's.  src/10_rayleigh quantifies it: the alignment
ratio A = R_F/(tr F/P) has |alpha_A| <= 0.06 in five of six measured cells while
alpha_{tr F/P} is 1.2 to 2.4.

Why panel (b) exists.  The earlier version showed panel (a) only and its
caption claimed decay is "steep in every parameterisation".  On the plotted
architecture that is not true at large width: the Standard MLP SATURATES from
n = 1024 on (0.0082 -> 0.0070 -> 0.0070), so its fitted exponent comes entirely
from the first half of the grid, and its R^2 = 0.87 is the lowest of the nine
cells while the other eight are >= 0.985 (the next lowest is the Standard
teacher-student at 0.985; the remaining seven are >= 0.989).  Panel (b) shows the saturation is a
property of that one cell and not of the Standard regime: the Standard
teacher-student and CNN keep decaying across the whole grid.

The paper's claim survives -- spectral alignment does decay in every regime, so
it cannot be what separates them -- but the reader can now see the one place
where the power law runs out, instead of being told it does not happen.

COLOUR: panel (a) has a parameterisation dimension, so colour carries the
regime; panel (b) is one regime, so colour is free and carries the
architecture (ARCH_COLORS, disjoint from the regime triple).  Each panel says
which.

DATA: (a) and (b) are unchanged -- measured, from ../data/ via
fig_data.load_pairs("geo"); median over the four smooth activations and all seed
pairs, band = interquartile range, each curve normalised by its own
smallest-width value.

(a)'s path mean comes from ../data/rayleigh/rayleigh_profile.csv -- the 21-point
R_F(t) grid, MLP only, which is why (b) is NOT redrawn the same way: no R_F(t)
profile exists for the CNN or the teacher-student yet.  `measure_rayleigh.py
audit` lists both as gaps.  The two panels therefore read different points of
the path, and the caption has to say so.

(c) comes from the canonical table ../data/rayleigh/rayleigh_cells.csv
(src/10_rayleigh `merge`), restricted to the MLP and to cells carrying tr F/P.
That is a SEPARATE and smaller measurement than (a)'s: tr F/P exists nowhere
else in the repository, so only the cells it has been measured on can appear --
NTK softplus; Standard GELU and Swish; muP ReLU, Softplus and tanh.
Its activation basis is therefore NOT (a)'s, and the panel is labelled to say
so; the exact cell list is printed by this script for the caption.  What makes
the two comparable at all is that (c)'s R_F reproduces the released rq_mid used
in (a) to a maximum relative deviation of 7e-05, pair by pair -- the ANCHOR
check in data/rayleigh/report_mlp.txt.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, legend_table, log_width_axis,
                       panel_letter, save, C, REGIMES, REGIME_LABEL,
                       ARCH_COLORS, ARCH_LABEL, ARCH_MARKERS, WIDTHS)
from fig_data import (load_pairs, by_width, powerlaw_fit,
                      rayleigh_canon_cells, load_rayleigh_canon)

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


def panel_c(ax):
    """R_F beside tr F / P, both normalised at the smallest width.

    tr F / P is exactly E[v'Fv] for v uniform on the sphere, so it is the curve
    R_F would trace if Delta carried no directional information.  Where the two
    coincide, R_F is reporting the spectrum rather than the direction."""
    cells = rayleigh_canon_cells(need="align_mid_rat")
    cells = cells[cells["arch"] == "MLP"]     # panels (a) and (b) anchor on MLP
    per_regime = {}
    for regime, g in cells.groupby("regime"):
        t = g.groupby("width")[["rq_mid_med", "trP_mid_med"]].median()
        t = t / t.iloc[0]
        per_regime[regime] = (t.index.to_numpy(float),
                              t["rq_mid_med"].to_numpy(float),
                              t["trP_mid_med"].to_numpy(float),
                              sorted(g["act"].unique()))
    # Encoding, and why it is not (a)'s.  In (a) and (b) a pale band of the
    # series colour means the INTERQUARTILE RANGE.  Drawing tr F/P as a pale
    # band of the regime colour would put a second meaning on that same visual
    # inside one figure, and a reader would take the reference for a spread.
    # So this panel drops bands entirely and carries the two quantities on line
    # style alone: solid = measured R_F, dotted = the tr F/P reference.  The
    # regime dashes of REG_STYLE are not reused here either -- muP's is already
    # dotted, which would collide with the reference style.
    for regime in REGIMES:
        if regime not in per_regime:
            continue
        x, rq, tp, _ = per_regime[regime]
        _, marker = REG_STYLE[regime]
        ax.plot(x, tp, color=C[regime], ls=(0, (1.1, 1.5)), lw=1.1, zorder=4)
        ax.plot(x, rq, color=C[regime], ls="-", lw=1.5, marker=marker, ms=3.0,
                markerfacecolor=C[regime], markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
    ax.set_ylabel(r"normalised  ($n=64 \to 1$)")
    ax.set_ylim(2.0e-5, 6.0)
    # R1: the takeaway goes on the plot.  Lower left is the one region every
    # curve has already left by n = 256.
    ax.annotate(r"$\mathcal{R}_F$ tracks" "\n" r"$\mathrm{tr}\,F/P$",
                xy=(0.04, 0.06), xycoords="axes fraction", fontsize=6.4,
                color=C["ink"], ha="left", va="bottom", zorder=8)
    legend_table(ax, [(dict(color=C["ink"], ls="-", marker="o", lw=1.5, ms=3.0),
                       r"$\mathcal{R}_F$", None),
                      (dict(color=C["ink"], ls=(0, (1.1, 1.5)), lw=1.1),
                       r"$\mathrm{tr}\,F/P$", None)],
                 loc="upper right", title="MLP, measured cells", fontsize=6.2)
    return per_regime


def path_mean(arch="MLP", acts=("gelu", "swish", "softplus", "tanh")):
    """Median over activations of the PATH MEAN of R_F, per (regime, width).

    The midpoint is one sample of a path along which R_F varies by more than an
    order of magnitude, and it is not a neutral one: it sits in a local minimum
    that deepens with width.  Averaging over the 21 measured t-points asks what
    the path does rather than what one point of it does.

    Each activation is normalised to its own smallest width before the median,
    so a cell with a large absolute R_F cannot dominate the regime curve."""
    p = load_rayleigh_canon("profile")
    p = p[(p["arch"] == arch) & (p["act"].isin(acts))]
    out = {}
    for regime, g in p.groupby("regime"):
        widths = sorted(g["width"].unique())
        per_act = {}
        for act, ga in g.groupby("act"):
            vals = {}
            for w, gw in ga.groupby("width"):
                m = gw.groupby("t")["rq_t"].median()
                vals[w] = (float(m.loc[0.5]) if 0.5 in m.index else np.nan,
                           float(m.mean()))
            per_act[act] = vals
        mid, mean = [], []
        for w in widths:
            mid.append(np.median([v[w][0] for v in per_act.values() if w in v]))
            mean.append(np.median([v[w][1] for v in per_act.values() if w in v]))
        mid, mean = np.array(mid), np.array(mean)
        out[regime] = (np.array(widths, float), mid / mid[0], mean / mean[0])
    return out


def main():
    apply_style()
    # Panel (a) now carries six curves and two keys, so it gets more of the
    # width; (b) and (c) carry three each.
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(5.5, 2.62),
        gridspec_kw={"wspace": 0.38, "width_ratios": [1.24, 0.88, 0.88]})
    geo = load_pairs("geo")

    # ---- (a) MLP, three parameterisations, at TWO anchors -----------------
    # Colour carries the regime (R3).  Within a regime the two curves are the
    # two anchors, and they are separated by line style and marker fill --
    # NOT by REG_STYLE's per-regime dashes, which would collide (muP's dash is
    # already dotted).  Solid + filled = the midpoint, i.e. exactly what the
    # paper measures; dashed + hollow = the mean over the path.
    tabm = by_width(geo[geo["arch"] == "MLP"], "rq_mid", keys=("regime",))
    pm = path_mean()
    ends, rows_a = {}, []
    for regime in REGIMES:
        _ls, marker = REG_STYLE[regime]
        x, y, e, r2 = curve(ax_a, tabm[tabm["regime"] == regime], C[regime],
                            "-", marker)
        ends[regime] = (x, y, e, r2)
        rows_a.append((dict(color=C[regime], ls="-", marker=marker, lw=1.7,
                            ms=3.4), REGIME_LABEL[regime],
                       rf"$n^{{-{e:.1f}}}$"))
        if regime in pm:
            wm, _mid, mean = pm[regime]
            ax_a.plot(wm, mean, color=C[regime], ls=(0, (3.2, 1.6)), lw=1.15,
                      marker=marker, ms=3.0, markerfacecolor="white",
                      markeredgecolor=C[regime], markeredgewidth=0.9, zorder=4)
    # Mark the one place the power law runs out.  Kept as an annotation rather
    # than folded into the table: it is a caveat on one row, not a column the
    # other two rows could fill.
    #
    # One word, not the former two lines.  Once the key became a table it took
    # the top third of the panel, and the only gap left that clears both the
    # NTK curve above and the muP curve below is about one line tall.  The
    # numbers this used to carry ("from n=1024", "R^2=0.87, lowest of nine")
    # moved into the caption; the plot keeps the pointer, which is the part
    # prose cannot do.
    # The two Standard curves are the point of the panel: at the anchor the
    # quantity flattens, along the path it turns back up -- the same reversal
    # the barrier shows in Figure 1.  Both are labelled, on the same cell.
    wm, _m, mean = pm["Standard"]
    ax_a.annotate("reverses", xy=(wm[-1], mean[-1]),
                  xytext=(-2, 10), textcoords="offset points", fontsize=6.0,
                  color=C["Standard"], ha="right", va="bottom",
                  arrowprops=dict(arrowstyle="-", color=C["Standard"],
                                  lw=0.7, shrinkA=1, shrinkB=2))
    # Headroom for the key.  Every curve is normalised to 1.0 at the smallest
    # width, so nothing this figure measures lives above 1.0.
    ax_a.set_ylim(3.0e-5, 90.0)
    ax_a.set_ylabel(r"Rayleigh quotient $\mathcal{R}_F$ (normalised)")

    # ---- (b) Standard regime, three architectures ------------------------
    tabs = by_width(geo[geo["regime"] == "Standard"], "rq_mid", keys=("arch",))
    rows_b = []
    for arch in ("MLP", "TS", "CNN"):
        x, y, e, _ = curve(ax_b, tabs[tabs["arch"] == arch], ARCH_COLORS[arch],
                           ARCH_LS[arch], ARCH_MARKERS[arch], ms=3.6)
        # Short names here: at this panel width "Teacher-student" alone makes
        # the key wider than the plotting region it sits in.
        rows_b.append((dict(color=ARCH_COLORS[arch], ls=ARCH_LS[arch],
                            marker=ARCH_MARKERS[arch], lw=1.7, ms=3.6),
                       {"MLP": "MLP", "TS": "TS", "CNN": "CNN"}[arch],
                       rf"$n^{{-{e:.2f}}}$"))
    ax_b.set_ylim(2.5e-3, 26.0)

    per_regime = panel_c(ax_c)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_yscale("log")
        log_width_axis(ax)
        # The old right margin (8.6x) hosted the end-of-line labels; the key is
        # now a table inside the panel, so the axis ends where the data does.
        ax.set_xlim(WIDTHS[0] * 0.88, WIDTHS[-1] * 1.16)
        ax.set_xlabel("width $n$")
        despine(ax)
    panel_letter(ax_a, "a", dx=-0.08)
    panel_letter(ax_b, "b", dx=-0.05)
    panel_letter(ax_c, "c", dx=-0.05)

    # Both panels fall steeply left to right, so the top right corner of each
    # is the region its own curves leave empty.
    # Two small keys rather than one tall one: at a third of the text width a
    # five-row table covers the data.  The exponents stay upper right, where
    # the published figure put them; the anchor key goes lower left, which the
    # curves have left empty because they all start at 1.0 on the left edge and
    # descend to the right.
    legend_table(ax_a, rows_a, loc="upper right", title="MLP / MNIST",
                 fontsize=6.0)
    legend_table(ax_a, [
        (dict(color=C["ink"], ls="-", marker="o", lw=1.5, ms=3.0),
         r"$t=\frac{1}{2}$", None),
        (dict(color=C["ink"], ls=(0, (3.2, 1.6)), marker="o", lw=1.15, ms=3.0),
         r"mean over $t$", None)], loc="lower left", fontsize=5.8)
    legend_table(ax_b, rows_b, loc="upper right", title="Standard",
                 fontsize=6.0)

    save(fig, OUT, "Figure3")
    print("  (a) MLP by regime:", {r: round(v[2], 2) for r, v in ends.items()})
    print("  (c) cells:", {r: v[3] for r, v in per_regime.items()})
    for r, (x, rq, tp, _) in per_regime.items():
        span = f"{rq[0]:.2g}->{rq[-1]:.2g}"
        print(f"      {r:9s} R_F {span:>16s}   trF/P "
              f"{tp[0]:.2g}->{tp[-1]:.2g}   ratio stays within "
              f"[{min(rq/tp):.2f}, {max(rq/tp):.2f}]")
    print("  (b) Standard by arch:", {a: round(powerlaw_fit(
        tabs[tabs["arch"] == a].sort_values("width")["width"].to_numpy(float),
        tabs[tabs["arch"] == a].sort_values("width")["med"].to_numpy(float))[0], 2)
        for a in ("MLP", "TS", "CNN")})


if __name__ == "__main__":
    main()
