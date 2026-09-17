#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figD16_rho_trend.py -- Appendix D: rho*(n) for each of the 36 cells, one
small panel per cell, every panel carrying its own axes.

Figure D.7 answers "how big is rho* in this cell" with one number per tile.
This answers the other half -- "and which way does it move across the width
grid" -- which a single number cannot: a cell that falls monotonically and one
that dips and rebounds report the same max/min ratio, and the ratio has no
sign at all.

A tile grid of sparklines was the compact way to show it, but sparklines carry
no axes: the reader sees a shape and cannot say from what to what.  Here each
cell is a real plot with its own ticks, so a descent can be read in decades and
the width it happens at can be read off.  The cost is the page: 36 panels with
legible ticks do not fit beside Figure D.7, so this is its own figure.

Rows are (architecture, parameterisation) in Figure D.7's order, columns the
four smooth activations in its order, so the two figures index the same cell
the same way.  Each panel is ZOOMED to its own cell.  A shared vertical scale spanning the
grid's five decades flattened the 14 cells that move by less than 2x into
straight lines, which defeats a figure about shape.  Heights therefore do NOT
compare across panels, and the two y ticks are the cell's own minimum and
maximum so the reader can see the span a panel was zoomed to -- a trace that
looks dramatic but reads 0.56 to 0.58 says so on its axis.  Levels across
cells are Figure D.7's job.  Colour is the paper's regime
triple (R3).  The dashed rule is rho* = 0.05, the level Figure D.9 splits its
clusters on.

The CNN is swept on a channel multiplier w_m in {1,2,4,8}, so its three rows
carry that axis and fill their panels.  The conversion is n = 64 w_m, which is
stated on the figure.  SLOPES IN THE CNN ROWS ARE NOT COMPARABLE BY EYE with
the six rows above them: three octaves occupy the width the others give to six,
so a given decay renders twice as steep there -- the same caution the main text
attaches to Figure 4(b).

DATA: measured, ../data/param_final_*.csv, column `rho_A`, median over the ten
seed pairs of each (cell, width).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from fig_style import apply_style, save, C, REGIMES, REGIME_LABEL
from fig_data import load_pairs, ACT_LABEL, CNN_CHANNELS_PER_WM

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"
ARCH_ORDER = ["MLP", "TS", "CNN"]
ACT_ORDER = ["gelu", "tanh", "swish", "softplus"]
ROWS = [(a, r) for a in ARCH_ORDER for r in REGIMES]
RHO_SPLIT = 0.05                  # the level Figure D.9 splits its clusters on


def fmt_rho(v):
    """0.22 / 1e-4 / 9e-6 -- Figure D.7's tile convention, so the two figures
    print a value the same way."""
    if v >= 0.01:
        return f"{v:.2f}"
    e = int(np.floor(np.log10(v)))
    return f"{v/10**e:.0f}e{e}"


def curves():
    pf = load_pairs("final")
    med = (pf.groupby(["arch", "regime", "act", "width"])["rho_A"]
             .median().reset_index())
    out = {}
    for arch, regime in ROWS:
        for act in ACT_ORDER:
            s = med[(med["arch"] == arch) & (med["regime"] == regime)
                    & (med["act"] == act)].sort_values("width")
            if not s.empty:
                out[(arch, regime, act)] = (s["width"].to_numpy(float),
                                            s["rho_A"].to_numpy(float))
    return out


def main():
    apply_style()
    cur = curves()
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(4.9, 7.5))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.940, bottom=0.052,
                        wspace=0.34, hspace=0.34)

    for i, (arch, regime) in enumerate(ROWS):
        for j, act in enumerate(ACT_ORDER):
            ax = axes[i][j]
            ax.set_facecolor("#F2F2F2")
            cnn = arch == "CNN"
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            if cnn:
                ax.set_xlim(0.88, 9.1)
                ax.set_xticks([1, 2, 4, 8]); ax.set_xticks([], minor=True)
            else:
                ax.set_xlim(56, 4700)
                ax.set_xticks([64, 256, 1024, 4096])
                ax.set_xticks([128, 512, 2048], minor=True)
            got = cur.get((arch, regime, act))
            if got is None:
                ax.set_yticks([])
            else:
                w, v = got
                if cnn:
                    w = w/CNN_CHANNELS_PER_WM
                # ZOOMED to this cell, not to a shared scale.  Fourteen of the
                # 36 cells move by less than 2x, so on one scale spanning the
                # grid's five decades their trace is a flat line and the shape
                # -- which is the whole point of this figure -- is invisible.
                # The price is that heights no longer compare across panels, so
                # the two y ticks ARE the cell's own minimum and maximum: a
                # panel that looks dramatic but reads 0.56 to 0.58 is saying so.
                lo, hi = float(v.min()), float(v.max())
                # The margin has to be a FRACTION OF THE CELL'S OWN SPAN, not a
                # fixed factor.  A flat 1.35x around a cell that only moves
                # 1.05x opens the axis to 1.92x and leaves the trace filling 8%
                # of the panel -- the two ticks then land on top of each other,
                # which is exactly what a zoomed figure must not do.  In log
                # space a constant fraction makes every cell fill the same 81%.
                span = np.log10(hi/lo)
                m = max(0.12*span, 0.004)
                ylo, yhi = lo/10**m, hi*10**m
                ax.set_ylim(ylo, yhi)
                ticks = [lo, hi] if span > 0.008 else [hi]
                ax.set_yticks(ticks)
                ax.set_yticklabels([fmt_rho(t) for t in ticks])
                ax.set_yticks([], minor=True)
                for t in ticks:
                    ax.axhline(t, color="white", lw=0.45, zorder=1)
                # Only draw the split level where it actually falls inside the
                # zoom; off-range it would sit on an edge and read as an axis.
                if ylo < RHO_SPLIT < yhi:
                    ax.axhline(RHO_SPLIT, color="#a9a9a9", lw=0.6,
                               ls=(0, (2.2, 1.8)), zorder=2)
                ax.plot(w, v, color=C[regime], lw=1.25, zorder=4,
                        solid_capstyle="round")
                ax.plot(w[-1], v[-1], marker="o", ms=2.6, color=C[regime],
                        markeredgecolor="white", markeredgewidth=0.45,
                        zorder=5)
            ax.grid(True, axis="x", which="both", color="white", lw=0.45,
                    zorder=1)
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#c9c9c9")
                sp.set_linewidth(0.55)
            ax.tick_params(labelsize=4.9, length=1.6, width=0.5, pad=1.4)
            ax.tick_params(which="minor", length=0)
            ax.set_xticklabels(["1", "2", "4", "8"] if cnn
                               else ["64", "256", "1k", "4k"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            if i == 0:
                ax.set_title(ACT_LABEL[act], fontsize=7.0, pad=3.5)
            if j == 0:
                ax.set_ylabel(f"{arch} · {REGIME_LABEL[regime]}", fontsize=6.4,
                              rotation=0, ha="right", va="center",
                              labelpad=25)
    # Without this the reader has no way to know the panels are not on a
    # common scale, and would compare heights across cells that are zoomed
    # differently -- the one misreading this figure can produce.
    # \textbf is a LaTeX macro; matplotlib's own mathtext does not know it and
    # prints it literally.  Plain text, emphasised by the wording instead.
    fig.supxlabel(r"width $n$   (CNN rows: channel multiplier "
                  r"$w_m$, $n=64\,w_m$)", fontsize=7.2, y=0.013)
    # No rotated figure-level y label: the row names already own the left
    # margin and the two collided.  The quantity is named once, above them.
    # The horizontal axis is NOT shared any more -- the CNN rows carry the
    # channel multiplier -- so the note claims only what is true.
    save(fig, OUT, "Figure9")

    import numpy as _np
    spans = _np.array([v.max()/v.min() for (w, v) in cur.values()])
    print(f"  {len(cur)} cells, each zoomed to its own range")
    print(f"  span within a cell: min {spans.min():.2f}x, median "
          f"{_np.median(spans):.1f}x, max {spans.max():.0f}x")
    print(f"  cells moving less than 2x (flat on a shared scale): "
          f"{int((spans < 2).sum())}/{len(spans)}")


if __name__ == "__main__":
    main()
