#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figp5_length_predicts.py -- Figure 5: Fisher length predicts the barrier.

Barrier exponents alpha_B versus Fisher-length exponents alpha_{L_F} (a)
and Rayleigh-quotient exponents alpha_{R_F} (b), across the 36 cells
(3 architectures x 3 parameterizations x 4 smooth activations).
Colour encodes parameterization, marker encodes architecture.
Reported: (a) R^2 = 0.91, slope 1.12;  (b) R^2 = 0.02.

DATA: measured, from ../data/ -- one exponent per cell, fitted by OLS on seed
medians (App. E.1) via measured_cells() below.  The R^2 and slope printed on
the figure are COMPUTED from the plotted points (linregress), so the
annotation is always consistent with the data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from fig_style import (apply_style, despine, panel_letter, save, C,
                       REGIMES, ARCH_MARKERS)
from fig_data import load_pairs, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"

ARCHS = ["MLP", "CNN", "TS"]


def measured_cells():
    """One (alpha_LF, alpha_B, alpha_RF) triple per cell, from measurement.

    36 cells = 3 architectures x 3 parameterisations x 4 smooth activations.
    Each exponent is an OLS fit of log(quantity) on log(width) over the seed
    medians of that cell -- the procedure App. E.1 specifies.  The barrier comes
    from data/final/*_pairs.csv, the Fisher length and the Rayleigh quotient
    from data/geodesic/*_pairs.csv."""
    pf, geo = load_pairs("final"), load_pairs("geo")
    key = ["arch", "regime", "act"]

    def per_cell(df, col):
        rows = []
        for k, sub in df.dropna(subset=[col]).groupby(key):
            med = sub.groupby("width")[col].median()
            rows.append(dict(zip(key, k))
                        | {col: powerlaw_fit(med.index.values, med.values)[0]})
        return pd.DataFrame(rows)

    m = (per_cell(pf, "B")
         .merge(per_cell(geo, "flen_mid"), on=key)
         .merge(per_cell(geo, "rq_mid"), on=key)
         .dropna())
    return [[r.regime, r.arch, r.flen_mid, r.B, r.rq_mid] for r in m.itertuples()]


def fit_with_band(ax, x, y, color="#777777"):
    """OLS fit line + 95% bootstrap CI band."""
    res = stats.linregress(x, y)
    xf = np.linspace(x.min(), x.max(), 50)
    ax.plot(xf, res.intercept + res.slope * xf, color=color, ls="--", lw=1.0)
    rng = np.random.default_rng(0)
    idx = np.arange(x.size)
    slopes, inter = [], []
    for _ in range(400):
        j = rng.choice(idx, idx.size, replace=True)
        r = stats.linregress(x[j], y[j])
        slopes.append(r.slope)
        inter.append(r.intercept)
    # Pointwise 95% bootstrap band: percentiles of the PREDICTIONS, not an
    # envelope built from the extreme intercept and the extreme slope (which is
    # what the previous version drew, and is far wider than 95%).
    preds = np.array([i + sl * xf for i, sl in zip(inter, slopes)])
    ax.fill_between(xf, np.percentile(preds, 2.5, axis=0),
                    np.percentile(preds, 97.5, axis=0),
                    color="#999999", alpha=0.22, lw=0)
    return res


def main():
    apply_style()
    rows = measured_cells()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 2.5), sharey=True,
        gridspec_kw={"wspace": 0.16})

    for regime in REGIMES:
        for arch in ARCHS:
            sel = [(r[2], r[3], r[4]) for r in rows
                   if r[0] == regime and r[1] == arch]
            a_lf = np.array([s[0] for s in sel])
            a_b = np.array([s[1] for s in sel])
            a_rf = np.array([s[2] for s in sel])
            for ax, xx in ((ax_a, a_lf), (ax_b, a_rf)):
                ax.scatter(xx, a_b, s=16, marker=ARCH_MARKERS[arch],
                           facecolor=C[regime], edgecolor="white",
                           linewidth=0.4, zorder=5,
                           label=f"_{regime}{arch}")  # dummy: legend via fig

    res_a = fit_with_band(ax_a, np.array([r[2] for r in rows]),
                          np.array([r[3] for r in rows]))
    res_b = fit_with_band(ax_b, np.array([r[4] for r in rows]),
                          np.array([r[3] for r in rows]))

    # (a) y = x diagonal reference, labelled at its empty top-right end
    lims = [-0.6, 2.0]
    ax_a.plot(lims, lims, color=C["ref"], ls=":", lw=0.8)
    ax_a.annotate("diagonal $y=x$", xy=(-0.55, -0.42), fontsize=6.8,
                  color=C["ref"], ha="left", rotation=33,
                  rotation_mode="anchor")

    ax_a.annotate(f"$R^2 = {res_a.rvalue**2:.2f}$,   slope $= {res_a.slope:.2f}$",
                  xy=(0.04, 0.965), xycoords="axes fraction", fontsize=7.6,
                  va="top", weight="bold")
    ax_b.annotate(f"$R^2 = {res_b.rvalue**2:.2f}$   (no relationship)",
                  xy=(0.04, 0.965), xycoords="axes fraction", fontsize=7.6,
                  va="top", weight="bold")

    ax_a.set_xlabel(r"$\alpha_{\Delta^\top F \Delta}$  (Fisher length)")
    ax_a.set_ylabel(r"$\alpha_B$  (barrier, $>0$ = collapse)")
    ax_b.set_xlabel(r"$\alpha_{R_F}$  (Rayleigh quotient)")
    ax_a.set_title("predictor: Fisher length", fontsize=8, pad=3)
    ax_b.set_title("predictor: Rayleigh quotient", fontsize=8, pad=3)
    panel_letter(ax_a, "a", dx=-0.05)
    panel_letter(ax_b, "b", dx=-0.03)
    ax_a.set_ylim(-0.80, 2.08)
    ax_a.set_xlim(lims)
    ax_b.set_xlim(0.2, 3.4)
    for ax in (ax_a, ax_b):
        despine(ax)
        ax.grid(True, ls="-", lw=0.6, color="#ffffff")  # both axes: scatter

    # Shared two-row legend below the panels (R2 compromise for scatter)
    handles = [plt.Line2D([], [], color=C[r], lw=3,
                          label={"NTK": "NTK-lazy", "Standard": "Standard",
                                 "muP": r"$\mu$P"}[r]) for r in REGIMES]
    handles += [plt.Line2D([], [], color="#666666", marker=ARCH_MARKERS[a],
                           ls="none", markersize=4.5,
                           label={"MLP": "MLP", "CNN": "CNN",
                                  "TS": "Teacher\u2013student"}[a])
                for a in ARCHS]
    fig.legend(handles=handles, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
               handletextpad=0.4, fontsize=7)
    fig.subplots_adjust(bottom=0.24)

    save(fig, OUT, "figp5_length_predicts")
    print(f"  n cells = {len(rows)} | Fisher length R2 = "
          f"{res_a.rvalue**2:.3f}, slope = {res_a.slope:.2f}"
          f" | Rayleigh R2 = {res_b.rvalue**2:.3f}")


if __name__ == "__main__":
    main()
