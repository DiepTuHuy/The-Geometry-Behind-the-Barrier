#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figE_gridcheck.py -- Appendix E.1: is the T=41 barrier grid dense enough?

The claim in E.1 is a claim of fact, so it is settled by measurement rather than
argued.  B is recomputed on a 401-point grid that CONTAINS the original 41 as a
subset, so the two numbers are the same quantity at two densities and differ
only by what the coarse grid missed.

(a) B(41) against B(401) on log-log, one point per seed pair.  Everything sits
    on the diagonal across three decades of barrier height.
(b) The relative gap, as a histogram, against the 0.5% line no pair reaches.

DATA: ../data/regrid/summary.csv -- 128 seed pairs, 32 cells, all three
parameterisations, recomputed from the released MLP checkpoints.  The
pipeline that produced it reproduces data/final/*_pairs.csv to six digits.
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from fig_style import (apply_style, despine, panel_letter, save, C,
                       REGIME_LABEL)

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"
CSV = Path(__file__).resolve().parent.parent / "data" / "regrid" / "summary.csv"
REG_FROM = {"ntk": "NTK", "sp": "Standard", "mup": "muP"}
MARK = {"NTK": "o", "Standard": "s", "muP": "^"}


def main():
    apply_style()
    d = pd.read_csv(CSV)
    d["regime"] = d.cell.str.split("_").str[0].map(REG_FROM)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 2.25),
                                     gridspec_kw={"wspace": 0.30})

    # ---- (a) B(41) vs B(401) --------------------------------------------
    lo = min(d.B_paper_grid41.min(), d.B_grid401.min())*0.55
    hi = max(d.B_paper_grid41.max(), d.B_grid401.max())*1.8
    ax_a.plot([lo, hi], [lo, hi], color=C["ref"], ls=":", lw=0.9, zorder=1)
    for reg in ("NTK", "Standard", "muP"):
        s = d[d.regime == reg]
        ax_a.scatter(s.B_paper_grid41, s.B_grid401, s=13, marker=MARK[reg],
                     facecolor=C[reg], edgecolor="white", linewidth=0.35,
                     zorder=5, label=REGIME_LABEL[reg])
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(lo, hi); ax_a.set_ylim(lo, hi)
    ax_a.set_xlabel(r"$B$ on the reported grid, $T=41$")
    ax_a.set_ylabel(r"$B$ on $T=401$")
    ax_a.annotate(r"$y=x$", xy=(0.055, 0.10), xycoords="axes fraction",
                  fontsize=6.8, color=C["ref"], rotation=45)
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.6,
                handletextpad=0.25, borderpad=0.1, labelspacing=0.22)

    # ---- (b) relative gap ------------------------------------------------
    g = d.rel_pct.to_numpy(float)
    ax_b.hist(g, bins=np.linspace(0, 0.55, 23), color=C["ink"], alpha=0.78,
              edgecolor="white", linewidth=0.4)
    ax_b.axvline(0.5, color=C["muP"], ls="--", lw=1.1, zorder=6)
    ax_b.annotate("no pair reaches 0.5 percent", xy=(0.5, ax_b.get_ylim()[1]*0.80),
                  xytext=(-5, 0), textcoords="offset points", fontsize=6.6,
                  color=C["muP"], ha="right", va="center", weight="bold")
    ax_b.annotate(f"median {np.median(g):.3f}\nmax {g.max():.3f}",
                  xy=(0.97, 0.95), xycoords="axes fraction", fontsize=6.8,
                  ha="right", va="top")
    ax_b.set_xlabel(r"$(B_{401}-B_{41})\,/\,B_{41}$   (percent)")
    ax_b.set_ylabel("seed pairs")
    ax_b.set_xlim(0, 0.55)

    for ax in (ax_a, ax_b): despine(ax)
    panel_letter(ax_a, "a", dx=-0.09); panel_letter(ax_b, "b", dx=-0.07)
    save(fig, OUT, "figE_gridcheck")
    print(f"  n={len(d)} cap | median {np.median(g):.3f}% | max {g.max():.3f}% "
          f"| >0.5%: {(g>0.5).sum()}")


if __name__ == "__main__":
    main()
