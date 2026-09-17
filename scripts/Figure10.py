#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure10.py -- Appendix D: the summary claim in one plane.

The claim names two conditions -- the Fisher length must fall, and the endpoint
uncertainty must be controlled -- and the barrier is the outcome.  All three fit
on one axis pair: alpha_LF across, alpha_B up, rho* in the colour.  Plotting the
outcome against ONE condition at a time, which an earlier draft did, is what hid
the problem: each marginal view looks supportive while the cell that would decide
the question is missing.

What the plane shows.

  * Nothing collapses left of alpha_LF = 0.  Length reduction is necessary; it is
    not sufficient, since two cells reduce the length and still do not collapse.
  * The bright points -- large rho* -- all sit in the left third.  The corner that
    would separate the two conditions, a steep length reduction held together with
    a large rho*, contains no cell: cells with rho* > 0.05 reach alpha_LF = 0.53
    at most, while those with rho* < 0.05 run to 2.26.
  * So the colour and the x position are not independent in this data, and no
    re-analysis makes them so.  src/07_uncertainty/mlp_uncertainty_sweep.py in the
    code release is the run that fills the gap, by moving rho* with label
    smoothing inside a fixed parameterisation.

COLOUR is rho* on a log scale, on the SAME viridis scale as
Figure D.7(a), so a point here and a tile there are directly comparable.  It
carries no regime meaning -- the paper's categorical triple is not used in this
figure -- and marker shape carries the architecture, as elsewhere.

DATA: measured.  Barrier and rho* from param_final_*.csv, endpoint Fisher length
from param_geo_*.csv, joined per seed pair; all 2160 pairs match and ||Delta||_2
agrees between the rounds to 2e-6 relative.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fig_style import apply_style, despine, save, C, ARCH_MARKERS
from fig_data import load_pairs, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"
ARCHS = ["MLP", "TS", "CNN"]
ARCH_LEG = {"MLP": "MLP", "TS": "Teacher–student", "CNN": "CNN"}
KEY = ["arch", "regime", "act", "width", "seedA", "seedB"]
RHO_HI = 0.05


def cells36() -> pd.DataFrame:
    pf, geo = load_pairs("final"), load_pairs("geo")
    m = pf[KEY + ["B", "rho_A"]].merge(geo[KEY + ["flen_A"]], on=KEY)
    rows = []
    for k, sub in m.groupby(["arch", "regime", "act"]):
        med = sub.groupby("width").median(numeric_only=True)
        w = med.index.values.astype(float)
        rows.append(dict(zip(["arch", "regime", "act"], k)) | {
            "a_B": powerlaw_fit(w, med["B"].values)[0],
            "a_flen": powerlaw_fit(w, med["flen_A"].values)[0],
            "rho": float(med["rho_A"].iloc[-1])})
    return pd.DataFrame(rows)


def main():
    apply_style()
    c = cells36()
    hi, lo = c[c["rho"] > RHO_HI], c[c["rho"] <= RHO_HI]
    x_stop = float(hi["a_flen"].max())

    fig, ax = plt.subplots(figsize=(5.5, 2.95))
    cmap = plt.get_cmap("viridis")
    norm = LogNorm(vmin=5e-6, vmax=1.0)

    # No shading of the region a large rho* never reaches: it holds most of the
    # data, so tinting it reads as "nothing here" when the point is only that
    # nothing BRIGHT is here.  The colour already says that; the arrow names it.
    ax.axvline(0.0, color=C["ref"], ls=":", lw=0.9, zorder=1)
    ax.axhline(0.0, color=C["ref"], ls=":", lw=0.9, zorder=1)

    for arch in ARCHS:
        s = c[c["arch"] == arch]
        ax.scatter(s["a_flen"], s["a_B"], c=s["rho"], cmap=cmap, norm=norm,
                   s=34, marker=ARCH_MARKERS[arch], edgecolor="#3a3a3a",
                   linewidth=0.5, zorder=5)

    ax.set_xlim(-0.42, 2.55)
    ax.set_ylim(-0.62, 1.58)
    ax.set_xlabel(r"$\alpha_{\mathcal{L}_F}$   (endpoint Fisher length, $>0$ = falls with width)")
    ax.set_ylabel(r"$\alpha_B$   ($>0$ = barrier collapses)")

    ax.annotate("no cell collapses\nwithout length reduction",
                xy=(-0.38, 1.42), fontsize=6.6, color="#5f5f5f",
                ha="left", va="top", style="italic")
    ax.annotate(f"no cell with $\\rho^*>0.05$ beyond "
                f"$\\alpha_{{\\mathcal{{L}}_F}}={x_stop:.2f}$",
                xy=(0.5*(x_stop + 2.55), -0.50), fontsize=6.8, color="#4a4a4a",
                ha="center", va="center", style="italic")
    ax.annotate("", xy=(2.50, -0.36), xytext=(x_stop + 0.03, -0.36),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#8a8a8a"))

    despine(ax)
    ax.grid(axis="x", visible=True, color="#ffffff", lw=0.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.042, pad=0.02)
    cb.set_label(r"endpoint uncertainty $\rho^*(w_A)$", fontsize=7.2, labelpad=3)
    cb.ax.tick_params(labelsize=6.4, length=1.8)
    cb.outline.set_visible(False)
    cb.ax.axhline(RHO_HI, color="#d0021b", lw=0.9)
    cb.ax.annotate(r"$0.05$", xy=(1.02, RHO_HI), xycoords=("axes fraction", "data"),
                   xytext=(3, 0), textcoords="offset points", fontsize=6.0,
                   color="#d0021b", va="center", ha="left")

    handles = [plt.Line2D([], [], color="#666666", marker=ARCH_MARKERS[a],
                          ls="none", markersize=4.8, label=ARCH_LEG[a])
               for a in ARCHS]
    ax.legend(handles=handles, ncol=1, loc="upper left",
              bbox_to_anchor=(0.015, 0.80), columnspacing=1.2,
              handletextpad=0.4, fontsize=7, labelspacing=0.35)
    fig.subplots_adjust(left=0.105, right=0.845, top=0.965, bottom=0.175)
    save(fig, OUT, "Figure10")

    n_corner = int(((c["a_flen"] > x_stop) & (c["rho"] > RHO_HI)).sum())
    print(f"  high-rho cells (n={len(hi)}): a_flen <= {x_stop:.2f}")
    print(f"  low-rho  cells (n={len(lo)}): a_flen <= {lo['a_flen'].max():.2f}")
    print(f"  cells in the corner beyond {x_stop:.2f} with rho*>{RHO_HI}: {n_corner}")
    print(f"  collapse left of a_flen=0: "
          f"{int((c[c['a_flen'] <= 0]['a_B'] > 0).sum())}/{int((c['a_flen'] <= 0).sum())}")


if __name__ == "__main__":
    main()
