#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figD9_quadrants.py -- Appendix D: the two conditions as the two axes.

The claim names two conditions, so the plane whose axes ARE those two conditions
is where the four cases live: each is a quadrant.  The two that decide the claim
are the off-diagonal ones -- a large rho* while the Fisher length still falls
(top right), and a small rho* while it does not (bottom left) -- and the barrier
is what the colour reports in each.

An earlier version of this figure drew the four cases as a 2x2 of barrier-versus-
width panels.  That put the two conditions in the panel LABELS and two other
quantities on the axes, so a reader met a grid captioned "Fisher length" and
"rho*" whose axes were neither.  Position is the channel a reader trusts, so the
two conditions belong on it.

Reading.  Left of the vertical line nothing is green: no cell collapses without
length reduction, whatever rho* does.  Crossing that line turns the points green
in both rows, so the length condition gates the outcome and the uncertainty one
does not.  The top-right quadrant is green too, which rules out the strong
reading of the claim -- a large rho* does not prevent collapse.  What it does not
rule out is a weaker one: those cells are a paler green, but they also sit at
smaller alpha_LF than the bottom-right ones, so the two quadrants differ in both
coordinates and the shade cannot be credited to rho* alone.

Note on the x axis.  It is an EXPONENT, not the Fisher length: alpha_LF > 0 means
the length FALLS with width, and larger means it falls faster.  Nothing here has
a growing Fisher length together with a collapsing barrier -- the nine cells
whose length does not fall are the nine purple points, all of them.

DATA: measured.  Barrier and rho* from param_final_*.csv, endpoint Fisher length
from param_geo_*.csv, joined per seed pair.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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
    fig, ax = plt.subplots(figsize=(5.5, 2.95))

    cmap = plt.get_cmap("PRGn")
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=1.30)

    ax.axvline(0.0, color="#8a8a8a", lw=0.9, zorder=1)
    ax.axhline(RHO_HI, color="#8a8a8a", lw=0.9, zorder=1)
    for arch in ARCHS:
        s = c[c["arch"] == arch]
        ax.scatter(s["a_flen"], s["rho"], c=s["a_B"], cmap=cmap, norm=norm,
                   s=40, marker=ARCH_MARKERS[arch], edgecolor="#3a3a3a",
                   linewidth=0.5, zorder=5)

    ax.set_xscale("linear"); ax.set_yscale("log")
    ax.set_xlim(-0.45, 2.55); ax.set_ylim(4e-6, 22.0)
    ax.set_xlabel(r"$\alpha_{\mathcal{L}_F}$ larger $\rightarrow$ endpoint Fisher length falls faster with width")
    ax.set_ylabel(r"$\rho^*(w_A)$, widest $n$")

    # one label per quadrant: how many cells, and what the barrier does there
    boxes = [(False, True,  0.02, 0.955, "left",  "top"),
             (True,  True,  0.98, 0.955, "right", "top"),
             (False, False, 0.02, 0.035, "left",  "bottom"),
             (True,  False, 0.98, 0.035, "right", "bottom")]
    for falls, hi, fx, fy, ha, va in boxes:
        g = c[((c["a_flen"] > 0) == falls) & ((c["rho"] > RHO_HI) == hi)]
        ax.text(fx, fy, ("one cell" if len(g) == 1 else f"{len(g)} cells")
                + "\n" + rf"median $\alpha_B={np.median(g['a_B']):+.2f}$",
                transform=ax.transAxes, fontsize=6.8, ha=ha, va=va,
                weight="bold", color=C["ink"], linespacing=1.35)

    ax.annotate(r"$\rho^*$ large", xy=(0.005, RHO_HI), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=6.4,
                color="#7a7a7a", ha="left", va="bottom", style="italic")
    ax.annotate("length falls", xy=(0.0, 4.4e-6), xytext=(4, 0),
                textcoords="offset points", fontsize=6.4, color="#7a7a7a",
                ha="left", va="bottom", style="italic")
    ax.annotate("both conditions met", xy=(0.98, 0.155), xycoords="axes fraction",
                fontsize=6.6, color="#4a7a4a", ha="right", va="bottom",
                style="italic")

    despine(ax)
    ax.grid(axis="x", visible=True, color="#ffffff", lw=0.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.042, pad=0.02)
    cb.set_label(r"$\alpha_B$   (green: barrier collapses)", fontsize=7.2,
                 labelpad=3)
    cb.ax.tick_params(labelsize=6.4, length=1.8)
    cb.outline.set_visible(False)
    cb.ax.axhline(0.0, color="#1a1a1a", lw=0.9)

    handles = [plt.Line2D([], [], color="#666666", marker=ARCH_MARKERS[a],
                          ls="none", markersize=4.8, label=ARCH_LEG[a])
               for a in ARCHS]
    # Parked in the empty band just under the rho* divider, the one region of
    # the plane no cell occupies.
    ax.legend(handles=handles, ncol=3, loc="center", bbox_to_anchor=(0.47, 0.55),
              columnspacing=1.2, handletextpad=0.4, fontsize=7)
    fig.subplots_adjust(left=0.105, right=0.845, top=0.965, bottom=0.175)
    save(fig, OUT, "figD9_quadrants")

    for falls in (True, False):
        for hi in (False, True):
            g = c[((c["a_flen"] > 0) == falls) & ((c["rho"] > RHO_HI) == hi)]
            print(f"  falls={str(falls):5s} rho_large={str(hi):5s}: n={len(g):2d} "
                  f"median a_B={np.median(g['a_B']):+.3f}  a_flen in "
                  f"[{g['a_flen'].min():+.2f},{g['a_flen'].max():+.2f}]")


if __name__ == "__main__":
    main()
