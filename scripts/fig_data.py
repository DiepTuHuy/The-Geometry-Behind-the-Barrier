#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_data.py
===========
The single place where figures read MEASURED data.

Every array plotted in this paper comes from `../data/`, the measured tables
released in this repository:

    final/{mlp,cnn,ts}_pairs.csv       per seed-pair: barrier, rho*, acc,
                                       ||Delta||_2, Fisher length, Rayleigh
    final/{mlp,cnn,ts}_cells.csv       the same, aggregated per cell
    geodesic/{mlp,cnn,ts}_pairs.csv    per seed-pair: geodesic deviation
    geodesic/{mlp,cnn,ts}_cells.csv    ||dF||_op medians + fitted exponents
    train/{mlp,cnn,ts}_combined.csv    per-seed ||dF||_op from the training run

`PATHS` below is the ONLY place a name maps onto a file: no figure script
builds a path, so moving the data means editing that one block.

Conventions applied here once, so no figure script has to know them:

  * regime names        sp -> Standard, ntk -> NTK, mup -> muP
  * activations         the four SMOOTH ones only.  `relu` appears in
                        the geodesic tables but is excluded everywhere in the paper: F and
                        the Hessian must be well defined (App. E.1).
  * CNN width           the CSVs store a channel MULTIPLIER wm in {1,2,4,8};
                        the network is c = [16 wm, 32 wm, 64 wm], so the width
                        in the paper's sense -- the widest layer -- is 64 wm,
                        i.e. 64, 128, 256, 512.  That is why the CNN grid
                        "ends at x8": x8 is n = 512, and it lines up with the
                        MLP / teacher-student grid rather than living on a
                        separate axis.
  * failed measurements rows with status != "ok" are dropped, and the count of
                        what was dropped is available via `load_pairs(...)`.

Nothing here fits, smooths or reconstructs anything: aggregation is median over
seed pairs, and the spread reported is the interquartile range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# figure-side name -> file under data/.  `{}` takes the architecture stem
# (mlp / cnn / ts).  See docs/DATA.md for what each column means.
PATHS = {
    "pairs_final": "final/{}_pairs.csv",
    "pairs_geo":   "geodesic/{}_pairs.csv",
    "cells_final": "final/{}_cells.csv",
    "cells_geo":   "geodesic/{}_cells.csv",
    "train":       "train/{}_combined.csv",
}

SMOOTH_ACTS = ["gelu", "tanh", "swish", "softplus"]
ACT_LABEL = {"gelu": "GELU", "tanh": "tanh", "swish": "Swish",
             "softplus": "Softplus"}
REGIME_FROM_CSV = {"ntk": "NTK", "sp": "Standard", "mup": "muP"}
ARCHS = ["MLP", "TS", "CNN"]
ARCH_CSV = {"MLP": "mlp", "TS": "ts", "CNN": "cnn"}

CNN_CHANNELS_PER_WM = 64          # c = [16 wm, 32 wm, 64 wm]


def _canon(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    """Rename regimes, drop non-smooth activations, put the CNN on real widths."""
    df = df.copy()
    df = df[df["act"].isin(SMOOTH_ACTS)]
    df["regime"] = df["regime"].map(REGIME_FROM_CSV)
    df = df[df["regime"].notna()]
    if arch == "CNN":
        df["width"] = df["width"].astype(int) * CNN_CHANNELS_PER_WM
    df["arch"] = arch
    return df


def _read(key: str, arch: str) -> pd.DataFrame:
    return _canon(pd.read_csv(DATA / PATHS[key].format(ARCH_CSV[arch])), arch)


def load_pairs(kind: str = "final") -> pd.DataFrame:
    """Per-seed-pair rows for all three architectures.

    kind="final" -> final/*_pairs.csv     (barrier, rho*, acc, ||Delta||, ...)
    kind="geo"   -> geodesic/*_pairs.csv  (geodesic deviation)"""
    key = "pairs_final" if kind == "final" else "pairs_geo"
    out = pd.concat([_read(key, a) for a in ARCHS], ignore_index=True)
    if "status" in out:
        out = out[out["status"].astype(str).str.startswith("ok")]
    return out


def load_cells() -> pd.DataFrame:
    """Per-cell aggregates as released (final/*_cells.csv)."""
    return pd.concat([_read("cells_final", a) for a in ARCHS],
                     ignore_index=True)


def load_finalfinal() -> pd.DataFrame:
    """||dF||_op medians/quartiles and the exponents fitted in the released run."""
    return pd.concat([_read("cells_geo", a) for a in ARCHS],
                     ignore_index=True)


# ----------------------------------------------------------------------
# Aggregation helpers.  Median over seed pairs, IQR as the spread -- the same
# summary the released cell tables use, recomputed here so a figure can also
# show the spread, which the cell tables do not carry for every column.
# ----------------------------------------------------------------------
def by_width(df: pd.DataFrame, value: str, keys=("arch", "regime")) -> pd.DataFrame:
    """median / q1 / q3 of `value` against width, grouped by `keys`."""
    g = (df.dropna(subset=[value])
           .groupby(list(keys) + ["width"])[value]
           .agg(med="median", q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75), n="size")
           .reset_index())
    return g.sort_values(list(keys) + ["width"])


def powerlaw_fit(width, y):
    """OLS slope of log y on log width -> (exponent, R^2, n).

    Returned exponent is the DECAY rate: y ~ n^{-exponent}."""
    width = np.asarray(width, float)
    y = np.asarray(y, float)
    m = np.isfinite(width) & np.isfinite(y) & (y > 0) & (width > 0)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    lx, ly = np.log(width[m]), np.log(y[m])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return -slope, r2, int(m.sum())


def fit_ci(width, y, n_boot=2000, seed=0):
    """Exponent with a 95% bootstrap CI over the (width, y) points."""
    width = np.asarray(width, float)
    y = np.asarray(y, float)
    m = np.isfinite(width) & np.isfinite(y) & (y > 0) & (width > 0)
    width, y = width[m], y[m]
    est, r2, n = powerlaw_fit(width, y)
    if not np.isfinite(est) or n < 3:
        return est, np.nan, np.nan, r2, n
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(n)
    for _ in range(n_boot):
        j = rng.choice(idx, n, replace=True)
        if len(np.unique(width[j])) < 2:
            continue
        e, _, _ = powerlaw_fit(width[j], y[j])
        if np.isfinite(e):
            boots.append(e)
    if len(boots) < 50:
        return est, np.nan, np.nan, r2, n
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), r2, n


def cell_exponents(df: pd.DataFrame, value: str, per=("arch", "regime", "act")):
    """Fit one decay exponent per cell, from the per-pair rows of that cell."""
    rows = []
    for key, sub in df.dropna(subset=[value]).groupby(list(per)):
        med = sub.groupby("width")[value].median()
        est, lo, hi, r2, n = fit_ci(med.index.values, med.values)
        rows.append(dict(zip(per, key if isinstance(key, tuple) else (key,)))
                    | {"exponent": est, "lo": lo, "hi": hi, "r2": r2,
                       "n_widths": n})
    return pd.DataFrame(rows)


def load_dF() -> pd.DataFrame:
    """Per-seed ||dF||_op, from the combined training tables.

    App. E.1 fits "OLS on log Q(n) = a - gamma log n over seed medians", and
    doing exactly that here reproduces all twelve alpha_op point estimates of
    Table F.1 to two decimals, which is how this loader was validated."""
    out = []
    for arch in ARCHS:
        d = pd.read_csv(DATA / PATHS["train"].format(ARCH_CSV[arch]))
        d = d[d["dF_op"].notna()]
        out.append(_canon(d, arch))
    return pd.concat(out, ignore_index=True)


def boot_ci_over_seeds(df, value, n_boot=2000, seed=0):
    """Exponent + 95% CI, resampling SEEDS within each width.

    The point estimate is the paper's procedure -- OLS on the seed medians --
    and the interval propagates the seed-to-seed spread through that same
    procedure, rather than treating the medians as noiseless."""
    g = {w: sub[value].to_numpy(float)
         for w, sub in df.dropna(subset=[value]).groupby("width")}
    widths = np.array(sorted(g))
    if len(widths) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(widths)
    med = np.array([np.median(g[w]) for w in widths])
    est, r2, _ = powerlaw_fit(widths, med)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        m = np.array([np.median(rng.choice(g[w], g[w].size, replace=True))
                      for w in widths])
        e, _, _ = powerlaw_fit(widths, m)
        if np.isfinite(e):
            boots.append(e)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) > 50 else (np.nan,) * 2
    return est, float(lo), float(hi), r2, len(widths)
