#!/usr/bin/env python3
"""Check every headline number in the paper against the released measurements.

    python verify.py

No GPU, no torch, no dataset download, no training. The script reads the CSVs
committed under data/ and recomputes each number the paper states, then prints
the two side by side. It exits 0 only if every check passes.

It does not reimplement anything: the loaders and the fitting routine are the
ones the figure scripts use (scripts/fig_data.py), and the 36-cell table is
built by Figure5.measured_cells(), so a number that passes here is the same
number the figure draws.

Every exponent is a decay rate: alpha is reported for Q(n) ~ n^-alpha, fitted
by OLS of log Q on log n over the seed medians of a cell, which is the
procedure the appendix specifies.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np
    import pandas as pd  # noqa: F401  (needed by fig_data)
    from scipy import stats
except ImportError as e:
    sys.exit(f"missing dependency: {e.name}\n"
             f"install with:  pip install -r requirements-figures.txt")

try:
    import fig_data as FD
    from Figure5 import measured_cells
except Exception as e:
    sys.exit(f"could not import the figure pipeline: {e!r}\n"
             f"run this script from the repository root.")


# --------------------------------------------------------------- reporting
GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = OFF = ""

results = []


def check(label, paper, measured, tol, note=""):
    """Record one comparison. `paper` may be a value or an (lo, hi) range."""
    if isinstance(paper, tuple):
        ok = paper[0] - tol <= measured <= paper[1] + tol
        shown = f"{paper[0]:g} to {paper[1]:g}"
    else:
        ok = abs(measured - paper) <= tol
        shown = f"{paper:+.2f}" if isinstance(paper, float) else str(paper)
    results.append((label, shown, measured, ok, note))
    return ok


def table():
    w = max(len(r[0]) for r in results)
    print(f"\n{'':<{w}}   {'paper':>12}   {'measured':>10}")
    print("-" * (w + 28))
    for label, shown, measured, ok, note in results:
        mark = f"{GREEN}ok{OFF}" if ok else f"{RED}MISMATCH{OFF}"
        val = f"{measured:+.2f}" if isinstance(measured, float) else str(measured)
        print(f"{label:<{w}}   {shown:>12}   {val:>10}   {mark}"
              + (f"   {DIM}{note}{OFF}" if note else ""))


# --------------------------------------------------------------- the checks
def alpha(df, col, arch, regime):
    """Decay exponent of `col` in width, for one cell family."""
    sub = df[(df.arch == arch) & (df.regime == regime)].dropna(subset=[col])
    if sub.empty:
        return float("nan")
    med = sub.groupby("width")[col].median()
    return FD.powerlaw_fit(med.index.values, med.values)[0]


def main():
    print(__doc__.strip().splitlines()[0])
    print(f"{DIM}reading the CSVs under {ROOT / 'data'}{OFF}")

    pairs_final = FD.load_pairs("final")
    pairs_geo = FD.load_pairs("geo")
    dF = FD.load_dF()

    # -- the main result: Fisher length predicts the barrier, the Rayleigh
    #    quotient does not.  Paper: R^2 = 0.90 slope 1.13, against R^2 = 0.03.
    cells = measured_cells()
    a_len = np.array([c[2] for c in cells])
    a_bar = np.array([c[3] for c in cells])
    a_ray = np.array([c[4] for c in cells])
    fit_len = stats.linregress(a_len, a_bar)
    fit_ray = stats.linregress(a_ray, a_bar)

    print(f"\n{DIM}Figure 5 -- barrier exponent against each predictor, "
          f"one point per cell{OFF}")
    check("cells behind Figure 5", 36, len(cells), 0)
    check("Fisher length  R^2", 0.90, fit_len.rvalue ** 2, 0.02)
    check("Fisher length  slope", 1.13, fit_len.slope, 0.05)
    check("Rayleigh       R^2", 0.03, fit_ray.rvalue ** 2, 0.02,
          "the negative control")

    # -- Figure 1b: the metric flattens in every regime (alpha ~ 0.9 to 1.3)
    print(f"\n{DIM}Figure 1b -- ||dF||_op decays as a power law in every "
          f"regime (MLP){OFF}")
    for regime in ("NTK", "Standard", "muP"):
        check(f"  |dF| exponent, {regime}", (0.9, 1.3),
              alpha(dF, "dF_op", "MLP", regime), 0.05)

    # -- Figure 1a: the barrier does not follow it
    print(f"\n{DIM}Figure 1a -- the barrier does not follow the metric{OFF}")
    check("  barrier exponent, NTK", -0.20, alpha(pairs_final, "B", "MLP", "NTK"),
          0.05, "grows with width")
    check("  barrier exponent, muP", +1.20, alpha(pairs_final, "B", "MLP", "muP"),
          0.05, "collapses")

    # -- Figure 2: geodesic-linear deviation
    print(f"\n{DIM}Figure 2 -- relative geodesic-linear deviation (MLP){OFF}")
    check("  D_rel exponent, NTK", 0.34, alpha(pairs_geo, "dev_rel", "MLP", "NTK"), 0.05)
    check("  D_rel exponent, muP", 0.56, alpha(pairs_geo, "dev_rel", "MLP", "muP"), 0.05)

    # -- Figure 3: the Rayleigh quotient falls steeply everywhere, which is
    #    exactly why it cannot separate the regimes
    print(f"\n{DIM}Figure 3 -- Rayleigh quotient decays steeply in all three "
          f"regimes (MLP){OFF}")
    for regime, expect in (("NTK", 1.74), ("Standard", 1.21), ("muP", 2.32)):
        check(f"  R_F exponent, {regime}", expect,
              alpha(pairs_geo, "rq_mid", "MLP", regime), 0.05)

    # -- grid coverage.  Five activations are trained; every Fisher quantity
    #    uses the four smooth ones only, because relu is not C^3.  So the
    #    trained grid is larger than the reported one, and both are checked.
    print(f"\n{DIM}Grid coverage{OFF}")
    # read the training tables raw: load_dF() drops relu, and the point here is
    # to count the grid before that filter
    raw = pd.concat(
        [pd.read_csv(FD.DATA / FD.PATHS["train"].format(FD.ARCH_CSV[a])).assign(arch=a)
         for a in FD.ARCHS],
        ignore_index=True)
    n_trained = raw.groupby(["arch", "regime", "act", "width"]).ngroups
    n_reported = pairs_final.groupby(["arch", "regime", "act", "width"]).ngroups
    check("cells trained (5 activations)", 270, n_trained, 0)
    check("cells reported (4 smooth activations)", 216, n_reported, 0,
          "relu is not C^3")
    check("aligned pairs behind the barrier", 2160, len(pairs_final), 0,
          "216 cells x 10 seed pairs")

    table()

    bad = [r for r in results if not r[3]]
    print()
    if bad:
        print(f"{RED}{len(bad)} of {len(results)} checks did not match.{OFF}")
        return 1
    print(f"{GREEN}All {len(results)} checks match the paper.{OFF}")
    print(f"{DIM}Figures: bash scripts/make_all.sh{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
