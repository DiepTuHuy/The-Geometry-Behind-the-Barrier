#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure2.py -- panel (b) of Figure 2, redrawn from the
fully-converged re-measurement.

WHY THIS EXISTS.  Panel (b) of `figp2_deviation` is the one curve family in the
figure that the released run could not solve to tolerance.  Splitting the
released MLP pairs by whether conjugate gradients met its 1e-6 relative
residual:

        arch/regime     NTK      Standard     muP
        MLP             0.0%      23.2%       0.0%
        TS              0.0%      63.2%      35.0%
        CNN             0.0%      15.6%       0.0%

So exactly one curve in the whole figure is contaminated -- Standard in panel
(b), where 65 of 280 pairs never converged.  Panel (a) uses NTK only and is
clean at 0% everywhere, which is why it is NOT redrawn here.

THE RE-MEASUREMENT.  `data/param_geo_mlp_cg1000.csv`: the same grid (3 regimes
x 4 smooth activations x 7 widths x 10 seed pairs = 840 rows), same batch, same
damping (lam_rel = 1e-2), same Green quadrature -- only the CG iteration cap
raised from 300 to 1000.  Result: 840/840 rows converged, max 536 iterations
(Standard at n=4096), zero rows above tolerance.

WHAT CHANGES, AND WHAT DOES NOT.  The trend is the same curve:

                       64    128    256    512     1k     2k     4k
        released     1.000  0.467  0.299  0.242  0.244  0.396  1.226
        re-measured  1.000  0.477  0.310  0.260  0.292  0.504  1.548

Minimum at n=512 in both; descent n^-0.68 vs n^-0.65; ascent n^+0.77 vs
n^+0.85; both cross back above 1.0 at n=4096; log-log shape correlation 0.988;
and the interquartile bands overlap at all seven widths.  The exponents move in
the last printed digit (NTK -0.35 -> -0.34, muP -0.57 -> -0.56) and Standard
still admits no power law.  What the re-measurement buys is not a different
conclusion but an unimpeachable one: the Standard upturn SURVIVES full CG
convergence, and is in fact stronger (1.23 -> 1.55), so it cannot be read as a
solver artifact.

CAVEAT WORTH KEEPING.  Both runs are float32.  Measured against a float64
reference on the same code, ||Gamma(Delta,Delta)|| carries a width-dependent
error of 0.04%-18% at n=128-256 -- comparable to the gap between the two runs.
That ceiling is common to both and is not a reason to prefer either, but it is
a reason not to trust the second decimal of a fitted exponent.

COLOUR: this panel has a parameterisation dimension, so colour carries the
regime (C[...]), identical to the released figure.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from fig_style import (apply_style, despine, legend_table, log_width_axis,
                       save, C, REGIMES, REGIME_LABEL, WIDTHS)
from fig_data import DATA, SMOOTH_ACTS, REGIME_FROM_CSV, by_width, powerlaw_fit

OUT = Path(__file__).resolve().parent.parent / "figures" / "main"
SRC = DATA / "param_geo_mlp_cg1000.csv"

REG_STYLE = {"NTK": ("-", "o"), "Standard": ((0, (4.5, 1.7)), "s"),
             "muP": ((0, (1, 2.2)), "^")}


def load_cg1000() -> pd.DataFrame:
    """The re-measured MLP rows, canonicalised exactly as `fig_data` does.

    Same three conventions the released loader applies: the four smooth
    activations only (relu is not C^3, App. E.1), CSV regime names mapped to
    the paper's, and rows whose status is not `ok` dropped."""
    df = pd.read_csv(SRC)
    df = df[df["act"].isin(SMOOTH_ACTS)]
    df["regime"] = df["regime"].map(REGIME_FROM_CSV)
    df = df[df["regime"].notna()]
    df = df[df["status"].astype(str).str.startswith("ok")]
    return df


def curve(ax, s, colour, ls, marker, lw=1.7, ms=3.6):
    """One normalised curve + IQR band; returns (x, y, exponent, R^2).

    Identical to `figp2_deviation.curve` -- each curve is normalised by its own
    smallest-width value, band = interquartile range."""
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


def main():
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(2.90, 2.30))

    geo = load_cg1000()
    tab = by_width(geo, "dev_rel", keys=("regime",))
    fits, rows = {}, []
    for regime in REGIMES:
        ls, marker = REG_STYLE[regime]
        x, y, e, r2 = curve(ax, tab[tab["regime"] == regime], C[regime],
                            ls, marker)
        fits[regime] = (e, r2, y)
        tag = "no power law" if r2 < 0.5 else rf"$n^{{-{e:.2f}}}$"
        rows.append((dict(color=C[regime], ls=ls, marker=marker, lw=1.7,
                          ms=3.6), REGIME_LABEL[regime], tag))

    ax.axhline(1.0, color=C["ref"], ls=":", lw=0.8, zorder=1)
    # Same headroom as the released panel: every series starts at 1.0 by
    # construction, so the band above it carries no data and hosts the key.
    ax.set_ylim(6e-2, 11.0)
    ax.set_yscale("log")
    log_width_axis(ax)
    ax.set_xlim(WIDTHS[0] * 0.88, WIDTHS[-1] * 1.16)
    ax.set_xlabel("width $n$")
    ax.set_ylabel(r"relative deviation $D_{\mathrm{rel}}$")
    despine(ax)
    # Standard turns back up at the right, so the key goes top left -- the same
    # corner the released panel (b) uses.
    legend_table(ax, rows, loc="upper left", title="all three: MLP / MNIST")

    save(fig, OUT, "Figure2")
    for r, (e, r2, y) in fits.items():
        tag = "no power law" if r2 < 0.5 else f"n^{-e:+.4f}"
        print(f"  {r:9s} {tag:14s} R2={r2:.3f}  "
              + " ".join(f"{v:.3f}" for v in y))


if __name__ == "__main__":
    main()
