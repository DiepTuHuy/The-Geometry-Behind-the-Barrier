#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure8.py -- Appendix D: endpoint uncertainty for all 36 cells.

A heatmap rather than a table, and rather than the nine small-multiple panels an
earlier draft used.  Against a table it wins because the reader's question here
is "which cells are alike?", which is a grouping question that colour answers at
a glance and a column of numbers does not -- and nothing is given up, since every
tile still carries its value.  Against the small multiples it wins because those
spent nine panels showing that rho* barely moves with width, which is one number
per cell: panel (b) is that number.

(a) rho*(w_A) at the widest width of each cell.
(b) how far rho* travels across that cell's whole width grid, max over min.
    Read together: under NTK-lazy the value is large and FIXED (it moves by less
    than 2x over six octaves), under Standard and muP it is small and still
    falling hard (up to 1370x).  Uncertainty is set by the parameterisation, not
    by the width -- which is the collinearity Figure D.8 has to work around.

COLOUR here is a magnitude on a log scale, with a colour bar, and carries no
regime or architecture meaning: the paper's categorical triple is not used in
this figure at all, so a tile cannot be misread as "NTK-lazy" or "muP".

DATA: measured, ../data/param_final_*.csv, column `rho_A`, median over the ten
seed pairs of each cell.  The endpoint anchor is used because it lies off the
interpolation path and so is independent of the barrier being explained.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fig_style import apply_style, panel_letter, save, C, REGIMES, REGIME_LABEL
from fig_data import load_pairs, ACT_LABEL

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"
ARCH_ORDER = ["MLP", "TS", "CNN"]
ACT_ORDER = ["gelu", "tanh", "swish", "softplus"]
ROWS = [(a, r) for a in ARCH_ORDER for r in REGIMES]


def grids():
    """(rho at widest width, drift factor over the width grid) as 9x4 arrays."""
    pf = load_pairs("final")
    med = (pf.groupby(["arch", "regime", "act", "width"])["rho_A"]
             .median().reset_index())
    last = np.full((len(ROWS), 4), np.nan)
    drift = np.full((len(ROWS), 4), np.nan)
    for i, (arch, regime) in enumerate(ROWS):
        for j, act in enumerate(ACT_ORDER):
            s = med[(med["arch"] == arch) & (med["regime"] == regime)
                    & (med["act"] == act)].sort_values("width")
            if s.empty:
                continue
            v = s["rho_A"].to_numpy(float)
            last[i, j] = v[-1]
            drift[i, j] = v.max()/max(v.min(), 1e-12)
    return last, drift


def fmt_rho(v):
    if not np.isfinite(v): return ""
    if v >= 0.01: return f"{v:.2f}"
    e = int(np.floor(np.log10(v)))
    return f"{v/10**e:.0f}e{e}"


def panel(ax, M, cmap, norm, fmt, title, cbar_label):
    # The shared style paints white gridlines on every axis, which on an
    # imshow reads as cuts through the tiles rather than as a grid.
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(4)); ax.set_yticks(range(len(ROWS)))
    ax.set_xticklabels([ACT_LABEL[a] for a in ACT_ORDER], fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    # A tile's own label has to stay legible on its own fill, so the text
    # flips to white once the fill is dark enough.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]): continue
            rgba = cmap(norm(M[i, j]))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=6.1,
                    color="white" if lum < 0.55 else "#1a1a1a")
    # rule between architecture blocks
    for y in (2.5, 5.5):
        ax.axhline(y, color="white", lw=1.6)
    ax.set_title(title, fontsize=7.4, pad=4)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.048, pad=0.025)
    cb.ax.tick_params(labelsize=6.0, length=1.8)
    cb.outline.set_visible(False)
    cb.set_label(cbar_label, fontsize=6.4, labelpad=2)
    return im


def main():
    apply_style()
    last, drift = grids()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 2.75),
                                     gridspec_kw={"wspace": 0.26})

    cmap = plt.get_cmap("viridis")
    panel(ax_a, last, cmap, LogNorm(vmin=5e-6, vmax=1.0), fmt_rho,
          r"(endpoint $\rho^*$ at the widest $n$)", r"$\rho^*$")
    panel(ax_b, drift, plt.get_cmap("cividis"), LogNorm(vmin=1.0, vmax=2000.0),
          lambda v: (rf"$\times${v:.0f}" if v >= 10 else rf"$\times${v:.1f}"),
          r"(how far $\rho^*$ moves across the width grid)",
          r"max $/$ min")

    ax_a.set_yticklabels([f"{a} · {REGIME_LABEL[r]}" for a, r in ROWS],
                         fontsize=6.6)
    ax_b.set_yticklabels([])
    panel_letter(ax_a, "a", dx=-0.02, dy=1.06)
    panel_letter(ax_b, "b", dx=-0.02, dy=1.06)
    fig.subplots_adjust(left=0.150, right=0.945, top=0.86, bottom=0.10)
    save(fig, OUT, "Figure8")

    ntk = np.array([drift[i] for i, (a, r) in enumerate(ROWS) if r == "NTK"])
    oth = np.array([drift[i] for i, (a, r) in enumerate(ROWS) if r != "NTK"])
    print(f"  drift under NTK-lazy: max {np.nanmax(ntk):.1f}x ; "
          f"elsewhere: min {np.nanmin(oth):.0f}x, max {np.nanmax(oth):.0f}x")
    print(f"  rho* at widest n: NTK [{np.nanmin([last[i] for i,(a,r) in enumerate(ROWS) if r=='NTK']):.3f}"
          f", {np.nanmax([last[i] for i,(a,r) in enumerate(ROWS) if r=='NTK']):.3f}]")


if __name__ == "__main__":
    main()
