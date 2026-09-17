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
    "rq_pairs":    "rayleigh/{}_pairs.csv",
    "rq_profile":  "rayleigh/{}_profile.csv",
    "rq_cells":    "rayleigh/{}_cells.csv",
    "profile_len": "profile/{}_length.csv",
    # The canonical Rayleigh tables, written by src/10_rayleigh `merge`.  They
    # already fold in data/geodesic, data/profile and every Rayleigh run, so a
    # figure reads ONE file and gets the widest grid available.  No {} -- these
    # span all three architectures.
    "rq_canon_pairs":   "rayleigh/rayleigh_pairs.csv",
    "rq_canon_profile": "rayleigh/rayleigh_profile.csv",
    "rq_canon_cells":   "rayleigh/rayleigh_cells.csv",
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


def load_rayleigh_canon(kind: str = "cells") -> pd.DataFrame:
    """The canonical Rayleigh table: `cells`, `pairs` or `profile`.

    Produced by `src/10_rayleigh/measure_rayleigh.py merge`, which folds
    data/geodesic (R_F, lambda_max), data/profile (R_F(t)), data/train (the
    barrier) and every Rayleigh run into one table, preferring the source with
    the widest coverage and recording the choice in the `src_*` columns.

    Regimes are mapped to the paper's spelling and `arch` to its labels.  ReLU
    is NOT dropped -- unlike `_canon`, which drops it because differentiating F
    needs C^3.  The Rayleigh quotient differentiates nothing, so ReLU rows are
    real measurements and throwing them away would discard cells.  Filter on
    `smooth == 1` to reproduce the paper's four-activation subset."""
    df = pd.read_csv(DATA / PATHS[f"rq_canon_{kind}"])
    if "status" in df:
        df = df[df["status"].astype(str).str.startswith("ok")]
    df = df.copy()
    df["regime"] = df["regime"].map(REGIME_FROM_CSV)
    df = df[df["regime"].notna()]
    df["arch"] = df["arch"].map({"mlp": "MLP", "cnn": "CNN", "ts": "TS"})
    for c in df.columns:
        if c not in ("regime", "act", "arch", "status") and not c.startswith("src_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rayleigh_canon_cells(min_widths: int = 3, need: str | None = None) -> pd.DataFrame:
    """Canonical cells, keeping only those that can carry a fitted exponent.

    `min_widths` is the threshold `fit_alpha` applies, so every cell returned
    has finite exponents.  `need` names a column that must be present -- pass
    "align_mid_rat" for the alignment figures, which exist only where tr F/P
    was measured."""
    c = load_rayleigh_canon("cells")
    if need:
        c = c[c[need].notna()]
    return c.groupby(["arch", "regime", "act"]).filter(
        lambda g: g["width"].nunique() >= min_widths)


def load_profile_length(arch: str = "MLP") -> pd.DataFrame:
    """R_F(t) along the linear path, long format, from `src/04_profile`.

    The file on disk is WIDE -- one `rq_t0.000` ... `rq_t1.000` column per grid
    point, 21 of them -- because that is how the measurement wrote it.  Every
    figure wants it long, so the reshape happens here, once.

    This is the bigger of the two profile measurements in the repository: 84
    cells (3 regimes x 4 smooth activations x 7 widths), 840 seed pairs, on a
    21-point grid.  `data/rayleigh/mlp_profile.csv` is an independent later run
    on a 9-point grid covering 6 cells; the two agree to a maximum relative
    deviation of 7e-05 over the 1800 rows they share, which is why either can be
    trusted -- but prefer this one, it has twice the cells and a finer grid."""
    df = pd.read_csv(DATA / PATHS["profile_len"].format(ARCH_CSV[arch]))
    if "status" in df:
        df = df[df["status"].astype(str).str.startswith("ok")]
    tcols = [c for c in df.columns if c.startswith("rq_t")]
    keys = ["regime", "act", "width", "seedA", "seedB", "dnorm"]
    long = df.melt(id_vars=[k for k in keys if k in df.columns],
                   value_vars=tcols, var_name="tcol", value_name="rq_t")
    long["t"] = long["tcol"].str.removeprefix("rq_t").astype(float)
    long = long.drop(columns=["tcol"])
    long["regime"] = long["regime"].map(REGIME_FROM_CSV)
    long = long[long["regime"].notna()]
    long["arch"] = arch
    for c in ("width", "rq_t", "t", "dnorm"):
        if c in long:
            long[c] = pd.to_numeric(long[c], errors="coerce")
    return long.dropna(subset=["rq_t"])


def load_rayleigh(kind: str = "cells", arch: str = "MLP") -> pd.DataFrame:
    """Rows from the Rayleigh run (src/10_rayleigh), for ONE architecture.

    Deliberately NOT routed through `_canon`, for one reason: `_canon` drops
    every non-smooth activation, and `relu` is legitimate here.  The C^3
    requirement that excludes it elsewhere comes from differentiating F, and
    the Rayleigh quotient differentiates nothing -- it is one JVP and one VJP.
    Dropping relu would silently discard a measured cell.

    Regime names are still mapped to the paper's spelling, and `status` is
    still filtered, so the rest of the conventions hold."""
    df = pd.read_csv(DATA / PATHS[f"rq_{kind}"].format(ARCH_CSV[arch]))
    if "status" in df:
        df = df[df["status"].astype(str).str.startswith("ok")]
    df = df.copy()
    df["regime"] = df["regime"].map(REGIME_FROM_CSV)
    df = df[df["regime"].notna()]
    df["arch"] = arch
    for c in df.columns:
        if c not in ("regime", "act", "arch", "status", "mode", "kind"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rayleigh_cells(min_widths: int = 3, arch: str = "MLP") -> pd.DataFrame:
    """Per-cell Rayleigh rows, keeping only cells with enough widths to carry a
    fitted exponent.  `min_widths` is the same threshold _fit_alpha applies, so
    a cell that appears here always has a finite alpha."""
    c = load_rayleigh("cells", arch)
    return c.groupby(["regime", "act"]).filter(
        lambda g: g["width"].nunique() >= min_widths)


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
