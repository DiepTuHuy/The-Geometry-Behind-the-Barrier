#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure12.py -- Appendix D: does the Figure 5 prediction method survive a
harder dataset?

Figure 5 fits the barrier exponent alpha_B against the Fisher-length exponent
alpha_{L_F} over 36 cells, and against the Rayleigh exponent alpha_{R_F} as the
negative control.  Every one of those cells is MNIST, FashionMNIST or a
synthetic teacher.  This repeats BOTH panels on CIFAR-10.

THE COMPARISON IS MATCHED, AND THAT IS THE WHOLE DESIGN.  The scale run holds
everything but the dataset fixed at the released CNN's setting -- three
parameterisations, four smooth activations, channel multipliers w_m in
{1,2,4,8}, weight-matching alignment, exponents by OLS on seed medians -- so
each panel can show the 12 CIFAR-10 cells against the 12 released
FashionMNIST CNN cells and nothing but the dataset differs between them.

The alternative, plotting CIFAR-10 against Figure 5's pooled 36, would have
confounded the dataset with the architecture, and badly: refitting the released
cells one architecture at a time gives slopes of 1.43 (MLP), 1.16 (TS) and 0.81
(CNN), so a CIFAR-10 slope below Figure 5's pooled 1.13 would have been read as
the harder dataset weakening the relationship when it is the CNN doing it.
Against the matched CNN baseline the slope moves 0.81 -> 0.74, which is the
honest size of the dataset effect.

The Rayleigh control needs the same matching for the opposite reason.  Pooled,
it is R^2 = 0.026; CNN-only it is 0.111, and MLP-only it is 0.630 -- the control
draws most of its power from disagreement ACROSS architectures, so a
single-architecture run cannot reproduce 0.026 and would be misread as the
control failing.  Panel (b) therefore states the CNN-only baseline, not the
pooled one.

ENCODING.  Colour is the parameterisation, as everywhere in the paper (R3).
Marker shape stays the architecture's -- a square, ARCH_MARKERS["CNN"] -- since
both sets of cells are CNNs; the dataset is the FILL, hollow for the released
FashionMNIST reference and solid for CIFAR-10.  Nothing here needed a new hue.

DATA: measured.  CIFAR-10 from ../data/scale/param_scale_cifar10_cnn.csv via
fig_data.load_scale; FashionMNIST from ../data/param_final_cnn.csv and
param_geo_cnn.csv via fig_data.load_pairs, i.e. the very rows Figure 5 plots as
its CNN squares.  Both sides are fitted by the same per_cell() below, so the
two numbers in each panel differ by their data and by nothing else.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from fig_style import (apply_style, despine, panel_letter, save, C,
                       REGIMES, REGIME_LABEL, ARCH_MARKERS)
from fig_data import load_pairs, load_scale, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"

MARKER = ARCH_MARKERS["CNN"]        # both datasets are the same CNN
REF_LABEL = "FashionMNIST (Fig. 5)"
NEW_LABEL = "CIFAR-10"


def per_cell(df, value, keys=("regime", "act")):
    """One exponent per cell, by OLS on seed medians -- the App. E.1 procedure.

    Identical to figp5's own per_cell, deliberately: if this figure fitted the
    two datasets by even slightly different rules, the comparison it exists to
    make would not be a comparison of datasets."""
    keys = list(keys)
    rows = []
    for k, sub in df.dropna(subset=[value]).groupby(keys):
        med = sub.groupby("width")[value].median()
        ok = med.size >= 2 and np.all(np.isfinite(med.values)) \
            and np.all(med.values > 0)
        if not ok:
            warnings.warn(f"{value}: cell {k} has <2 widths or non-positive "
                          f"medians -> exponent undefined, cell dropped")
            alpha = np.nan
        else:
            alpha = powerlaw_fit(med.index.values.astype(float), med.values)[0]
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)))
                    | {value: alpha})
    return pd.DataFrame(rows)


def cells_released():
    """The 12 released CNN cells -- exactly Figure 5's CNN squares."""
    pf = load_pairs("final")
    geo = load_pairs("geo")
    pf, geo = pf[pf["arch"] == "CNN"], geo[geo["arch"] == "CNN"]
    return (per_cell(pf, "B")
            .merge(per_cell(geo, "flen_mid"), on=["regime", "act"])
            .merge(per_cell(geo, "rq_mid"), on=["regime", "act"])
            .dropna())


def cells_scale():
    """The 12 CIFAR-10 cells.  One table: the run measures B and the geometry
    from the same seed pairs, so no merge across files is needed."""
    s = load_scale("cifar10", "cnn")
    return (per_cell(s, "B")
            .merge(per_cell(s, "flen_mid"), on=["regime", "act"])
            .merge(per_cell(s, "rq_mid"), on=["regime", "act"])
            .dropna())


def scatter(ax, tab, xcol, filled):
    for regime in REGIMES:
        sel = tab[tab["regime"] == regime]
        if sel.empty:
            continue
        ax.scatter(sel[xcol], sel["B"], s=19, marker=MARKER,
                   facecolor=C[regime] if filled else "white",
                   edgecolor=C[regime] if not filled else "white",
                   linewidth=0.9 if not filled else 0.4,
                   zorder=6 if filled else 5)


def fit_line(ax, x, y, colour, ls, lw, band=False, zorder=3):
    """OLS fit over the span of `x`, optionally with a 95% bootstrap band."""
    res = stats.linregress(x, y)
    xf = np.linspace(float(np.min(x)), float(np.max(x)), 50)
    if band:
        rng = np.random.default_rng(0)
        idx = np.arange(len(x))
        preds = []
        for _ in range(400):
            j = rng.choice(idx, idx.size, replace=True)
            r = stats.linregress(np.asarray(x)[j], np.asarray(y)[j])
            preds.append(r.intercept + r.slope * xf)
        preds = np.array(preds)
        ax.fill_between(xf, np.percentile(preds, 2.5, axis=0),
                        np.percentile(preds, 97.5, axis=0),
                        color="#999999", alpha=0.20, lw=0, zorder=2)
    ax.plot(xf, res.intercept + res.slope * xf, color=colour, ls=ls, lw=lw,
            zorder=zorder)
    return res


def panel(ax, ref, new, xcol, xlabel, title, diagonal=False):
    scatter(ax, ref, xcol, filled=False)
    scatter(ax, new, xcol, filled=True)

    # The released fit first and quietly: it is the baseline, not the result.
    # A long dash, NOT the dotted pattern: panel (a) also carries a dotted
    # y = x rule in this same grey, and two grey dotted lines meaning
    # different things is the one thing this panel cannot afford.
    res_ref = fit_line(ax, ref[xcol], ref["B"], C["ref"], (0, (5.0, 1.8)), 1.0)
    res_new = fit_line(ax, new[xcol], new["B"], "#4d4d4d", "--", 1.15,
                       band=True, zorder=4)

    xs = np.concatenate([ref[xcol].to_numpy(float), new[xcol].to_numpy(float)])
    ys = np.concatenate([ref["B"].to_numpy(float), new["B"].to_numpy(float)])
    xpad, ypad = 0.12 * np.ptp(xs), 0.12 * np.ptp(ys)
    x0, x1 = xs.min() - xpad, xs.max() + xpad
    y0, y1 = ys.min() - ypad, ys.max() + 2.6 * ypad
    # BOTH limits before the diagonal is drawn: its label is anchored to the
    # point where it leaves the panel, which is not known until then.  Anchored
    # to (x1, x1) instead, it lands above the top of the axis and is clipped.
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if diagonal:
        # Figure 5 draws y = x in its Fisher panel; a slope of one is what
        # "the exponent transfers" would look like, so the reader needs it here
        # more than there.
        ax.plot([x0, x1], [x0, x1], color="#b4b4b4", ls=":", lw=0.8, zorder=1)
        edge = min(x1, y1)
        ax.annotate("$y=x$", xy=(edge, edge), xytext=(-3, -7),
                    textcoords="offset points", fontsize=6.5,
                    color="#9a9a9a", ha="right", va="top")

    # Both fits, stacked, new one first and in the darker ink.

    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8, pad=3)
    despine(ax)
    ax.grid(True, ls="-", lw=0.6, color="#ffffff")   # both axes: it is a scatter
    return res_ref, res_new


def main():
    apply_style()
    ref, new = cells_released(), cells_scale()
    for name, tab in (("released CNN", ref), ("CIFAR-10", new)):
        if len(tab) != len(REGIMES) * 4:
            warnings.warn(f"{name}: expected 12 cells, got {len(tab)}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 2.62), sharey=True,
                                     gridspec_kw={"wspace": 0.14})

    out_a = panel(ax_a, ref, new, "flen_mid",
                  r"$\alpha_{\Delta^\top F \Delta}$  (Fisher length)",
                  "Fisher length", diagonal=True)
    out_b = panel(ax_b, ref, new, "rq_mid",
                  r"$\alpha_{\mathcal{R}_F}$  (Rayleigh quotient)",
                  "Rayleigh quotient")
    ax_a.set_ylabel(r"$\alpha_B$  (barrier, $>0$ = collapse)")
    panel_letter(ax_a, "a", dx=-0.05)
    panel_letter(ax_b, "b", dx=-0.03)

    handles = [plt.Line2D([], [], color=C[r], lw=3, label=REGIME_LABEL[r])
               for r in REGIMES]
    handles += [
        plt.Line2D([], [], color="#666666", marker=MARKER, ls="none",
                   markersize=4.5, markerfacecolor="white",
                   markeredgecolor="#666666", label=REF_LABEL),
        plt.Line2D([], [], color="#666666", marker=MARKER, ls="none",
                   markersize=4.5, markerfacecolor="#666666",
                   markeredgecolor="white", label=NEW_LABEL)]
    fig.legend(handles=handles, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.1,
               handletextpad=0.4, fontsize=7)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.915, bottom=0.255)

    save(fig, OUT, "Figure12")
    for name, (rr, rn) in (("Fisher length", out_a), ("Rayleigh", out_b)):
        print(f"  {name:14s} FashionMNIST R2={rr.rvalue**2:.3f} "
              f"slope={rr.slope:+.2f} p={rr.pvalue:.3f}  |  "
              f"CIFAR-10 R2={rn.rvalue**2:.3f} slope={rn.slope:+.2f} "
              f"p={rn.pvalue:.3f}")


if __name__ == "__main__":
    main()
