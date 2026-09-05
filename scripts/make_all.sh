#!/usr/bin/env bash
# make_all.sh -- ONE command that redraws EVERY figure in the paper.
#
#   bash scripts/make_all.sh
#
# Every script below READS CSV ONLY. None of them trains or re-measures
# anything, so running this any number of times is safe and takes seconds.
# Output goes to figures/main/ and figures/appendix/ as .pdf (for LaTeX) and,
# where a raster is useful, .png.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Main text (figures/main) ==="
python3 figp1_paradox.py              # Fig 1: the paradox -- barrier grows, curvature flattens
python3 figp2_deviation.py            # Fig 2: geodesic deviation xi(t) across width and regime
python3 figp3_rayleigh.py             # Fig 3: Rayleigh quotient -- direction alone is not enough
python3 figp4_regime_barrier.py       # Fig 4: barrier by parameterisation
python3 figp5_length_predicts.py      # Fig 5: Fisher length is the predictor that works
python3 figp6_regime_uncertainty.py   # Fig 6: spread within a regime

echo
echo "=== Appendix (figures/appendix) ==="
python3 figR_roadmap.py               # roadmap: dependency graph of the appendix results
python3 figB_scaling.py               # B.2: ||Delta||_2 = Theta(sqrt P)
python3 figC_geodesic.py              # C.4-C.5: deviation inside the tube, and where the theorem stops
python3 figD_vacuous.py               # D: where the bound goes vacuous
python3 fig_counterexample_D.py       # D: the counterexample
python3 figE_pipeline.py              # E: the measurement pipeline, end to end
python3 figE_gridcheck.py             # E: B(41) vs B(401) -- the quadrature grid is converged
python3 figF_exponents.py             # F: fitted width exponents, all cells
python3 figF_within.py                # F: within-cell spread

echo
echo "Done. Figures are in figures/main/ and figures/appendix/"
