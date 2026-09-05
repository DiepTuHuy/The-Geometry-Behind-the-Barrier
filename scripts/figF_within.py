#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figF_within.py -- Appendix F: does the Fisher-length / barrier relation survive
splitting, and does the Rayleigh quotient?

Table F.7 is seven rows of "R^2, slope" for two predictors -- twenty-eight
numbers whose entire message is a comparison across splits and a sign.  That is
the table type Gelman, Pasarica & Dodhia (2002) and Kastellec & Leoni (2007)
say to draw, and the sign reversal in particular is invisible in a table and
unmissable on an axis.

(a) R^2 within each sub-population.  Fisher length stays high everywhere;
    the Rayleigh quotient is high in two regimes and near zero in two
    architectures -- it is inconsistent, not uniformly weak.
(b) Fitted slope.  Fisher length keeps one sign; the Rayleigh slope CROSSES
    ZERO between architectures, which is why the two cannot be pooled.

COLOUR: this figure has no regime or architecture dimension -- it compares two
PREDICTORS -- so it uses the two term colours reserved in fig_style for exactly
this (Okabe-Ito, disjoint from the regime triple).

DATA: measured, one exponent triple per cell via figp5.measured_cells().
"""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from scipy import stats

from fig_style import apply_style, despine, panel_letter, save, C
from figp5_length_predicts import measured_cells

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"
LF_C, RF_C = C["term_fisher"], C["term_resid"]

ROWS = [("All 36 cells",        lambda r: True),
        ("NTK-lazy only",       lambda r: r[0] == "NTK"),
        ("Standard only",       lambda r: r[0] == "Standard"),
        (r"$\mu$P only",        lambda r: r[0] == "muP"),
        ("MLP only",            lambda r: r[1] == "MLP"),
        ("Teacher–student only", lambda r: r[1] == "TS"),
        ("CNN only",            lambda r: r[1] == "CNN")]


def main():
    apply_style()
    rows = measured_cells()                      # [regime, arch, a_LF, a_B, a_RF]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 2.30), sharey=True,
                                     gridspec_kw={"wspace": 0.10})
    ys = np.arange(len(ROWS))[::-1]

    stat = []
    for (name, keep) in ROWS:
        sel = [r for r in rows if keep(r)]
        y = np.array([r[3] for r in sel])
        f = stats.linregress(np.array([r[2] for r in sel]), y)
        g = stats.linregress(np.array([r[4] for r in sel]), y)
        stat.append((name, len(sel), f, g))

    for k, (name, n, f, g) in enumerate(stat):
        y = ys[k]
        for ax, vf, vg in ((ax_a, f.rvalue**2, g.rvalue**2),
                           (ax_b, f.slope,     g.slope)):
            ax.plot([vg, vf], [y, y], color="#c9c9c9", lw=1.0, zorder=1)
            ax.scatter([vf], [y], s=26, marker="o", facecolor=LF_C,
                       edgecolor="white", linewidth=0.5, zorder=5)
            ax.scatter([vg], [y], s=26, marker="D", facecolor=RF_C,
                       edgecolor="white", linewidth=0.5, zorder=5)

    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([f"{n}  ({c})" for (n, c, _, _) in stat], fontsize=6.8)
    ax_a.set_ylim(-0.7, len(ROWS) - 0.3)
    ax_a.set_xlim(-0.04, 1.04); ax_a.set_xlabel(r"$R^2$")
    ax_b.axvline(0.0, color=C["muP"], ls="--", lw=1.0, zorder=2)
    ax_b.set_xlim(-0.45, 1.62); ax_b.set_xlabel("fitted slope")
    ax_b.annotate("Rayleigh slope\ncrosses zero", xy=(-0.03, ys[6]),
                  xytext=(6, 16), textcoords="offset points", fontsize=6.4,
                  color=C["muP"], ha="left", va="center", weight="bold")

    h = [plt.Line2D([], [], color=LF_C, marker="o", ls="none", ms=5,
                    label=r"Fisher length $\mathcal{L}_F$"),
         plt.Line2D([], [], color=RF_C, marker="D", ls="none", ms=5,
                    label=r"Rayleigh quotient $\mathcal{R}_F$")]
    fig.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.57, 0.005),
               frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.235)

    for ax in (ax_a, ax_b): despine(ax)
    panel_letter(ax_a, "a", dx=-0.02); panel_letter(ax_b, "b", dx=-0.02)
    save(fig, OUT, "figF_within")
    for name, n, f, g in stat:
        print(f"  {name:<22} n={n:<3} LF R2={f.rvalue**2:.2f} slope={f.slope:+.2f} | "
              f"RF R2={g.rvalue**2:.2f} slope={g.slope:+.2f}")


if __name__ == "__main__":
    main()
