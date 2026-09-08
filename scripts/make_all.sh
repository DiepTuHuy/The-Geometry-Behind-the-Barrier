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
python3 figp6_regime_uncertainty.py   # spread within a regime (not \includegraphics'd)

echo
echo "=== Appendix (figures/appendix) ==="
python3 figD7_rho_heatmap.py          # D: endpoint rho* across all cells
python3 figD8_joint.py                # D: the claim in one plane
python3 figD9_quadrants.py            # D: the two conditions as two axes
python3 figD10_paradox_arch.py        # D: Figure 1 on the other two architectures
python3 figR_roadmap.py               # roadmap: dependency graph (not \includegraphics'd)

echo
echo "Done. Figures are in figures/main/ and figures/appendix/"
